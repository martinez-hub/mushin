# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from inspect import getattr_static
from pathlib import Path
from typing import (
    Any,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

import numpy as np
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.utils import JobReturn, JobStatus
from hydra_zen import hydra_list, launch, load_from_yaml, make_config, multirun, zen
from hydra_zen._compatibility import HYDRA_VERSION
from hydra_zen._launch import _NotSet
from typing_extensions import Self

from ._validate import value_check

LoadedValue: TypeAlias = str | int | float | bool | list[Any] | dict[str, Any]

__all__ = [
    "BaseWorkflow",
    "RobustnessCurve",
    "MultiRunMetricsWorkflow",
]


T = TypeVar("T", list[Any], tuple[Any])
T1 = TypeVar("T1")


_VERSION_BASE_DEFAULT = _NotSet if HYDRA_VERSION < (1, 2, 0) else "1.1"


def _sort_x_by_k(x: T, k: Iterable[Any]) -> T:
    k = tuple(k)
    assert len(x) == len(k)
    sorted_, _ = zip(*sorted(zip(x, k, strict=True), key=lambda x: x[1]), strict=True)
    return type(x)(sorted_)


def _identity(x: T1) -> T1:
    return x


def _to_override_element(item: Any) -> str:
    """Serialize a single multirun element to Hydra override-grammar syntax.

    Booleans/numbers are emitted unquoted; strings are single-quoted (with
    embedded single-quotes escaped) so that values containing commas, spaces,
    or '=' round-trip exactly through Hydra's override parser; lists/tuples are
    emitted recursively in Hydra's bracketed list syntax.
    """
    # Unwrap a NumPy/torch 0-d scalar (e.g. from ``np.arange`` or RobustnessCurve's
    # epsilon array) to its Python scalar first, so a swept number is emitted
    # unquoted rather than falling through to the quoted-string fallback.
    if hasattr(item, "item") and getattr(item, "ndim", None) == 0:
        item = item.item()

    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, (int, float)):
        return str(item)
    if isinstance(item, str):
        # Escape backslashes first (Hydra's quoted-string grammar treats '\' as
        # an escape char), then single-quotes, so the value round-trips.
        return "'" + item.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(item, (list, tuple)):
        return "[" + ",".join(_to_override_element(x) for x in item) + "]"
    # Fall back to a quoted string representation for any other type.
    return "'" + str(item).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _scalar_repr(item: Any) -> Any:
    """A stable, type-aware key for detecting duplicate sweep-axis values.

    Unwraps 0-d numpy/torch scalars (as :func:`_to_override_element` does) and
    tags the value with its type so ``1`` and ``"1"`` (distinct coordinates)
    do not compare equal while ``1`` and ``1`` do.
    """
    if hasattr(item, "item") and getattr(item, "ndim", None) == 0:
        item = item.item()
    # repr keeps the key hashable (list-valued elements) and type-aware.
    return (type(item).__name__, repr(item))


def _cfg_missing_or_none(cfg, key: str) -> bool:
    """True if ``key`` (possibly a dotted path) is absent from ``cfg`` or None.

    Drives the Hydra ``+`` append prefix. A dotted path like ``model.width``
    must be resolved as a config *path*, not a literal attribute — a plain
    ``hasattr(cfg, "model.width")`` is always False, which would wrongly
    ``+``-append onto an existing nested field and make Hydra raise.
    """
    if "." not in key:
        return not hasattr(cfg, key) or getattr(cfg, key) is None
    from omegaconf import OmegaConf

    _missing = object()
    try:
        node = OmegaConf.structured(cfg) if not OmegaConf.is_config(cfg) else cfg
        val = OmegaConf.select(node, key, default=_missing)
    except Exception:  # noqa: BLE001 - unresolvable path -> treat as append
        return True
    return val is _missing or val is None


class _FailedRun:
    """Sentinel returned by a fail-soft task wrapper in place of a raised
    exception. Hydra's basic sweeper re-raises the first FAILED job's exception
    from inside ``launch`` (``BasicSweeper.sweep`` accesses ``r.return_value``),
    so a failed job never reaches ``jobs_post_process``. Under ``on_error='nan'``
    we instead catch the exception and return this sentinel: the Hydra job then
    completes normally and ``jobs_post_process`` can detect the failure and
    NaN-fill its grid cell."""

    __slots__ = ("exception",)

    def __init__(self, exception: BaseException) -> None:
        self.exception = exception


class _PriorCells:
    """What a resuming `_TaskRunner` needs from the prior sweep, and nothing
    more: the sweep root, the swept param names, and the completed
    {combo_key: dir} map. Picklable and small — failed/pending cells and their
    error strings never ship to workers."""

    __slots__ = ("root", "params", "completed")

    def __init__(self, root: str, params: tuple, completed: dict):
        self.root = root
        self.params = params
        self.completed = completed

    @classmethod
    def from_manifest(cls, m) -> "_PriorCells":
        completed = {
            k: v["dir"]
            for k, v in m.cells.items()
            if v.get("status") == "completed" and v.get("dir")
        }
        return cls(str(m.root), tuple(m.params), completed)


_MISSING_SWEPT = object()


def _read_swept_value(cfg, name: str):
    """Read swept param ``name`` from a job config. A literal flat key wins;
    otherwise ``name`` is treated as a nested config *path* (``model.width``),
    which a plain ``cfg[name]`` subscription would reject."""
    try:
        return cfg[name]
    except Exception:  # noqa: BLE001 - fall through to path lookup
        from omegaconf import OmegaConf

        val = OmegaConf.select(cfg, name, default=_MISSING_SWEPT)
        if val is _MISSING_SWEPT:
            raise KeyError(
                f"swept override {name!r} not found in the job config, neither "
                "as a literal key nor as a nested path"
            ) from None
        return val


def _current_group_choice(name: str) -> str | None:
    """The Hydra config-group option chosen for ``name`` in the current job
    (from ``runtime.choices``), or None when ``name`` is not a group axis or
    no Hydra context is active."""
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            choices = HydraConfig.get().runtime.choices
            if name in choices and choices[name] is not None:
                return str(choices[name])
    except Exception:  # noqa: BLE001 - best-effort; fall back to the raw value
        pass
    return None


def _group_choices_of_job(job) -> dict | None:
    """``runtime.choices`` recorded in a job's hydra config, or None."""
    try:
        choices = job.hydra_cfg.hydra.runtime.choices
        return None if choices is None else dict(choices)
    except Exception:  # noqa: BLE001 - jobs loaded without hydra_cfg
        return None


class _TaskRunner:
    """Picklable per-cell dispatch. Collapses the old closure chain
    (_task_calls / _instrument_task / _fail_soft / _resume_short_circuit) into one
    object so out-of-process launchers (joblib/submitit) can pickle it. Holds only
    picklable state; imports module-level helpers inside __call__ (never captures
    the _CURRENT_RESUME contextvar). Behavior mirrors the previous chain exactly —
    see the spec's Semantics Mapping."""

    def __init__(
        self,
        *,
        task,
        pre_task,
        swept_names,
        base_provenance,
        on_error,
        inject_resume,
        prior_manifest,
        code_hash=None,
    ):
        self.task = task  # zen(_ResumeInjector(fn)) or zen(fn)
        self.pre_task = pre_task  # zen(self.pre_task)
        self.swept_names = tuple(swept_names)
        self.base_provenance = base_provenance
        self.on_error = on_error
        self.inject_resume = inject_resume
        self.prior_manifest = prior_manifest
        self.code_hash = code_hash  # hash of the task source (resume guard)

    @staticmethod
    def _combo(cfg, names):
        from omegaconf import OmegaConf

        combo = {}
        for n in names:
            val = _read_swept_value(cfg, n)
            if OmegaConf.is_config(val):
                # A config-GROUP axis (model=small,big): the coordinate is the
                # chosen option name, not the composed sub-config. Inside a job
                # Hydra records the choice in runtime.choices.
                choice = _current_group_choice(n)
                if choice is not None:
                    val = choice
                else:
                    val = OmegaConf.to_container(val, resolve=True)
            combo[n] = MultiRunMetricsWorkflow._sanitize_coordinate_for_xarray(
                _unwrap_scalar(val)
            )
        return combo

    def __call__(self, cfg):
        import warnings
        from pathlib import Path

        from ._provenance import write_provenance
        from ._resume import (
            _CURRENT_RESUME,
            build_resume_context,
            config_fingerprint,
            read_cell_status,
            write_cell_status,
        )
        from ._sweep_io import combo_key, read_metrics_sidecar, write_metrics_sidecar

        chash = config_fingerprint(cfg)

        # (1) resume short-circuit — before pre_task/instrument, only when resuming
        if self.prior_manifest is not None:
            # Key the lookup on the CURRENT swept names, not the prior sweep's:
            # if the grid shape changed (an axis added/removed), the current
            # cell's key won't match any completed cell, so it re-runs instead
            # of projecting onto the old params and reusing the wrong cell.
            sc = self._combo(cfg, self.swept_names)
            key = combo_key(sc)
            dir_ = self.prior_manifest.completed.get(key)
            if dir_ is not None:
                cell_dir = Path(self.prior_manifest.root) / dir_
                prior = read_cell_status(cell_dir) or {}
                prior_config = prior.get("config_hash")
                prior_code = prior.get("code_hash")
                # A completed cell is reused only if BOTH the resolved config
                # and the task source still match. A missing prior hash (a
                # pre-fingerprint sweep) can't be checked, so it isn't a reason
                # to re-run — legacy sweeps still resume.
                config_changed = (
                    prior_config is not None
                    and chash is not None
                    and prior_config != chash
                )
                code_changed = (
                    prior_code is not None
                    and self.code_hash is not None
                    and prior_code != self.code_hash
                )
                if config_changed or code_changed:
                    what = (
                        "config and task code"
                        if config_changed and code_changed
                        else "config"
                        if config_changed
                        else "task code"
                    )
                    warnings.warn(
                        f"resume: cell {key!r} was completed under a different "
                        f"{what} (fingerprint mismatch); re-running it instead "
                        "of reusing the cached result.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    cached = read_metrics_sidecar(cell_dir)
                    if cached is not None:
                        # Refresh THIS job dir so its metrics/status match the
                        # config Hydra just wrote here. Otherwise, when a resume
                        # reuses a numeric dir that a prior sweep filled with a
                        # DIFFERENT cell, the dir keeps stale metrics -> the
                        # manifest and an offline load_from_dir mis-key the cell.
                        # A no-op when the cached cell is already this dir.
                        cwd = Path.cwd()
                        write_metrics_sidecar(cwd, cached)
                        write_cell_status(
                            cwd,
                            status="completed",
                            combo=sc,
                            attempt=1,
                            config_hash=chash,
                            code_hash=self.code_hash,
                        )
                        return cached

        # (2) fail-soft wraps everything below, only when on_error == "nan"
        try:
            self.pre_task(cfg)
            cwd = Path.cwd()
            combo = self._combo(cfg, self.swept_names)
            rc = build_resume_context(cwd, combo)
            try:
                write_provenance(cwd, cfg, base=self.base_provenance)
            except Exception:  # noqa: BLE001 - provenance is best-effort
                pass
            write_cell_status(
                cwd,
                status="running",
                combo=combo,
                attempt=rc.attempt,
                config_hash=chash,
                code_hash=self.code_hash,
            )
            # Set the contextvar AFTER the (fallible) status write, so a failed
            # write can't leave the token set with no matching reset -> a stale
            # ResumeContext leaking into later in-process cells.
            token = _CURRENT_RESUME.set(rc) if self.inject_resume else None
            try:
                result = self.task(cfg)
            except Exception:  # noqa: BLE001 - durable failed status, then re-raise
                write_cell_status(
                    cwd,
                    status="failed",
                    combo=combo,
                    attempt=rc.attempt,
                    config_hash=chash,
                    code_hash=self.code_hash,
                )
                raise
            finally:
                if token is not None:
                    _CURRENT_RESUME.reset(token)
            if isinstance(result, dict):
                write_metrics_sidecar(cwd, result)
            write_cell_status(
                cwd,
                status="completed",
                combo=combo,
                attempt=rc.attempt,
                config_hash=chash,
                code_hash=self.code_hash,
            )
            return result
        except Exception as exc:  # noqa: BLE001 - fail-soft sentinel or re-raise
            if self.on_error == "nan":
                # The sentinel keeps only repr(exc); persist the full stack in
                # the cell dir so a fail-soft sweep stays debuggable. (Hydra's
                # job cwd IS the cell dir here.)
                import traceback

                try:
                    Path("mushin_error.txt").write_text(traceback.format_exc())
                except OSError:
                    pass  # a full disk must not break fail-soft itself
                return _FailedRun(exc)
            raise


class _ResumeInjector:
    """Picklable replacement for the old `_bind_resume_kwarg` closure. Hides a
    task's `mushin_resume` parameter from hydra-zen's `zen` (via a stripped
    __signature__, so zen never tries to resolve it from config) and injects the
    current cell's ResumeContext from a contextvar at call time. Unlike a closure,
    an instance holding a module-level task + a Signature is stdlib-picklable."""

    def __init__(self, task):
        import inspect

        self._task = task
        sig = inspect.signature(task)
        self._sig = sig.replace(
            parameters=[p for n, p in sig.parameters.items() if n != "mushin_resume"]
        )

    @property
    def __signature__(self):  # inspect.signature / zen read this
        return self._sig

    def __call__(self, *args, **kwargs):
        from ._resume import current_resume

        return self._task(*args, **kwargs, mushin_resume=current_resume())


def _prepare_task(task):
    """If `task` declares a `mushin_resume` parameter, wrap it in a picklable
    `_ResumeInjector`; otherwise return it unchanged. Returns `(prepared, wants)`."""
    import inspect

    if "mushin_resume" in inspect.signature(task).parameters:
        return _ResumeInjector(task), True
    return task, False


def _write_env_snapshot(working_dir: Path) -> None:
    """Best-effort environment snapshot written once to <working_dir>/mushin_env.txt.

    Tries `uv export`, then `uv pip freeze`; falls back to an
    importlib.metadata dump so a record always lands even without `uv`."""
    import subprocess

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    out = working_dir / "mushin_env.txt"
    # Never overwrite a prior run's snapshot: a resume may execute in a
    # different environment, and clobbering would misattribute the env the
    # already-completed cells actually ran in. Later runs land beside it.
    if out.exists():
        n = 2
        while (working_dir / f"mushin_env.{n}.txt").exists():
            n += 1
        out = working_dir / f"mushin_env.{n}.txt"

    for cmd in (["uv", "export"], ["uv", "pip", "freeze"]):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except Exception:  # noqa: BLE001 - uv may be absent
            continue
        if res.returncode == 0 and res.stdout.strip():
            out.write_text(res.stdout)
            return

    # Fallback: dump installed distributions via importlib.metadata.
    try:
        from importlib.metadata import distributions

        lines = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in distributions()
            if d.metadata["Name"]
        )
        out.write_text("\n".join(lines) + "\n")
    except Exception:  # noqa: BLE001 - env capture is best-effort
        pass


class BaseWorkflow:
    """Provides an interface for creating a reusable workflow: encapsulated
    "boilerplate" for running, aggregating, and analyzing one or more Hydra jobs.

    Attributes
    ----------
    cfgs : List[Any]
        List of configurations for each Hydra job.

    metrics : Dict[str, List[Any]]
        Dictionary of metrics for across all jobs.

    workflow_overrides : Dict[str, Any]
        Present for backward compatibility; not populated. The swept parameters
        of the last run are exposed by the ``multirun_task_overrides`` property.

    jobs : List[Any]
        List of jobs returned for each experiment within the workflow.

    working_dir: Optional[pathlib.Path]
        The working directory of the experiment defined by Hydra's sweep directory
        (`hydra.sweep.dir`).
    """

    _REQUIRED_STATIC_METHODS = ("task", "pre_task")

    cfgs: list[Any]
    metrics: dict[str, list[Any]]
    workflow_overrides: dict[str, Any]
    jobs: list[JobReturn] | list[Any] | JobReturn

    def __init__(self, eval_task_cfg=None) -> None:
        """Workflows and experiments using Hydra.

        Parameters
        ----------
        eval_task_cfg: Mapping | None (default: None)
            The workflow configuration object.

        """
        # we can do validation checks here
        self.eval_task_cfg = (
            eval_task_cfg if eval_task_cfg is not None else make_config()
        )

        # initialize attributes
        self.cfgs = []
        self.metrics = {}
        self.workflow_overrides = {}
        self._multirun_task_overrides = {}
        self.jobs = []
        self._working_dir = None
        # combo_key(sanitized swept-param combo) -> that cell's metrics dict.
        # Populated for resilient, config-keyed grid assembly in `to_xarray`.
        self._metrics_by_combo: dict[str, Any] = {}

    @property
    def working_dir(self) -> Path:
        if self._working_dir is None:
            raise ValueError("`self.working_dir` must be set.")

        return self._working_dir

    @working_dir.setter
    def working_dir(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        value_check("path", path, type_=Path)
        path = path.resolve()

        if not path.is_dir():
            raise FileNotFoundError(
                f"`path` must point to an existing directory, got {path}"
            )

        self._working_dir = path

    @staticmethod
    def _parse_overrides(
        overrides,
    ) -> dict[str, LoadedValue | Sequence[LoadedValue]]:
        parser = OverridesParser.create()
        parsed_overrides = parser.parse_overrides(overrides=overrides)

        output = {}

        for override in parsed_overrides:
            param_name = override.get_key_element()
            val = override.value()
            if override.is_sweep_override():
                if hasattr(val, "list"):  # ChoiceSweep (also glob results)
                    val = multirun(val.list)  # type: ignore
                elif hasattr(val, "range"):  # RangeSweep: discrete -> a grid axis
                    val = multirun(list(val.range()))  # type: ignore
                else:
                    # e.g. IntervalSweep: continuous Bayesian-sweeper syntax
                    # cannot form the Cartesian grid to_xarray assembles.
                    raise ValueError(
                        f"unsupported sweep override for {param_name!r}: "
                        f"{type(val).__name__}. mushin assembles a Cartesian "
                        "grid, so sweep axes must be discrete — use "
                        "`param=a,b,c`, `choice(...)`, or `range(...)`; "
                        "`interval(...)`/adaptive-sweeper syntax is not "
                        "supported."
                    )

            param_name = param_name.split("+")[-1]
            output[param_name] = val

        return output

    @property
    def multirun_task_overrides(
        self,
    ) -> dict[str, LoadedValue | Sequence[LoadedValue]]:
        """Returns override param-name -> value.

        A sequence of overrides associated with a multirun will
        be stored in a `mushin.multirun` list. This
        enables one to distinguish this from an override whose sole
        value was a list of values.

        Returns
        -------
        multirun_task_overrides: Dict[str, LoadedValue | Sequence[LoadedValue]]

        Examples
        --------
        >>> from mushin import multirun, hydra_list
        >>>
        >>> class WorkFlow(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def task(*args, **kwargs):
        ...         return None
        >>>
        >>> wf = WorkFlow()
        >>> wf.run(foo=hydra_list(["val"]), bar=multirun(["a", "b"]), apple=1)
        >>> wf.multirun_task_overrides
        {'foo': ['val'], 'bar': multirun(['a', 'b']), 'apple': 1}
        """
        if not self._multirun_task_overrides:
            overrides = load_from_yaml(
                self.working_dir / "multirun.yaml"
            ).hydra.overrides.task

            output = self._parse_overrides(overrides)
            self._multirun_task_overrides = output

        return self._multirun_task_overrides

    @staticmethod
    def pre_task(*args: Any, **kwargs: Any) -> None:
        """Called prior to `task`

        This can be useful for doing things like setting random seeds,
        which must occur prior to instantiating objects for the evaluation
        task.

        Notes
        -----
        This function is automatically wrapped by `zen`, which is responsible
        for parsing the function's signature and then extracting and instantiating
        the corresponding fields from a Hydra config object – passing them to the
        function. This behavior can be modified by `self.run(pre_task_fn_wrapper=...)`
        """

    @staticmethod
    def task(*args: Any, **kwargs: Any) -> Any:
        """User-defined task that is run by the workflow. This should be
        a static method.

        Arguments will be instantiated configuration variables.  For example,
        if the the workflow configuration is structured as::

            ├── eval_task_cfg
            │    ├── trainer
            |    ├── module
            |    ├── another_config

        The inputs to `task` can be any of the three configurations:
        `trainer`, `module`, or `another_config` such as::

            @staticmethod
            def task(trainer: Trainer, module: LightningModule) -> None:
                trainer.fit(module)

        Notes
        -----
        This function is automatically wrapped by `zen`, which is responsible
        for parsing the function's signature and then extracting and instantiating
        the corresponding fields from a Hydra config object – passing them to the
        function. This behavior can be modified by `self.run(task_fn_wrapper=...)`
        """
        raise NotImplementedError()

    def validate(self, include_pre_task: bool = True):
        """Validates that the configuration will execute with the user-defined evaluation task"""
        if include_pre_task:
            zen(self.pre_task).validate(self.eval_task_cfg)

        zen(self.task).validate(self.eval_task_cfg)

    def run(
        self,
        *,
        working_dir: str | None = None,
        sweeper: str | None = None,
        launcher: str | None = None,
        overrides: list[str] | None = None,
        task_fn_wrapper: Callable[[Callable[..., T1]], Callable[[Any], T1]]
        | None = zen,
        pre_task_fn_wrapper: Callable[[Callable[..., None]], Callable[[Any], None]]
        | None = zen,
        version_base: str | type[_NotSet] | None = _VERSION_BASE_DEFAULT,
        to_dictconfig: bool = False,
        config_name: str = "mushin_workflow",
        job_name: str = "mushin_workflow",
        with_log_configuration: bool = True,
        on_error: str = "raise",
        resume: bool = False,
        capture_env: bool = False,
        **workflow_overrides: str | int | float | bool | multirun | hydra_list,
    ):
        """Run the experiment.

        Individual workflows can explicitly define `workflow_overrides` to improve
        readability and undstanding of what parameters are expected for a particular
        workflow.

        Parameters
        ----------
        task_fn_wrapper: Callable[[Callable[..., T1]], Callable[[Any], T1]] | None, optional (default=hydra_zen.zen)
            A wrapper applied to `self.task` prior to launching the task.
            The default wrapper is `hydra_zen.zen`. Specify `None` for no
            wrapper to be applied.

        working_dir: str (default: None, the Hydra default will be used)
            The directory to run the experiment in.  This value is used for
            setting `hydra.sweep.dir`.

        sweeper: str | None (default: None)
            The configuration name of the Hydra Sweeper to use (i.e., the override for
            `hydra/sweeper=sweeper`)

        launcher: str | None (default: None)
            The configuration name of the Hydra Launcher to use (i.e., the override for
            `hydra/launcher=launcher`)

        overrides: List[str] | None (default: None)
            Parameter overrides not considered part of the swept workflow
            parameter set (which are exposed via `multirun_task_overrides`).

        version_base : Optional[str], optional (default=1.1)
            Available starting with Hydra 1.2.0.
            - If the `version_base parameter` is not specified, Hydra 1.x will use defaults compatible with version 1.1. Also in this case, a warning is issued to indicate an explicit version_base is preferred.
            - If the `version_base parameter` is `None`, then the defaults are chosen for the current minor Hydra version. For example for Hydra 1.2, then would imply `config_path=None` and `hydra.job.chdir=False`.
            - If the `version_base` parameter is an explicit version string like "1.1", then the defaults appropriate to that version are used.

        to_dictconfig: bool (default: False)
            If ``True``, convert a ``dataclasses.dataclass`` to a ``omegaconf.DictConfig``. Note, this
            will remove Hydra's capability for validation with structured configurations.

        config_name : str (default: "mushin_workflow")
            Name of the stored configuration in Hydra's ConfigStore API.

        job_name : str (default: "mushin_workflow")
            Name of job for logging.

        with_log_configuration : bool (default: True)
            If ``True``, enables the configuration of the logging subsystem from the loaded config.

        **workflow_overrides: str | int | float | bool | multirun | hydra_list
            These parameters represent the values for configurations to use for the
            experiment.

            Passing `param=multirun([1, 2, 3])` will perform a multirun over those
            three param values, whereas passing `param=hydra_list([1, 2, 3])` will
            pass the entire list as a single input.

            These values will be appended to the `overrides` for the Hydra job.

        on_error : str (default: "raise")
            Failure policy for the sweep. ``"raise"`` (the default) preserves the
            existing behavior: a failing job aborts the sweep and the exception
            propagates. ``"nan"`` enables fail-soft: a failing job is recorded
            (in ``self.failures`` and the on-disk sweep manifest), its grid cell
            becomes NaN, a ``UserWarning`` is emitted, and the sweep completes.
        """
        if on_error not in {"raise", "nan"}:
            raise ValueError(
                f"`on_error` must be one of {{'raise', 'nan'}}, got {on_error!r}"
            )
        self._on_error = on_error

        if resume and working_dir is None:
            raise ValueError("`resume=True` requires `working_dir` to be provided.")

        launch_overrides = []

        if overrides is not None:
            launch_overrides.extend(overrides)

        if working_dir is not None:
            launch_overrides.append(f"hydra.sweep.dir={working_dir}")

        if sweeper is not None:
            launch_overrides.append(f"hydra/sweeper={sweeper}")

        if launcher is not None:
            launch_overrides.append(f"hydra/launcher={launcher}")

        # Under version_base "1.1" Hydra changes the working directory per job at
        # runtime, and the workflow depends on it: each job writes and reads its
        # metrics sidecar in its own chdir'd directory. Set `hydra.job.chdir`
        # explicitly to preserve that behavior AND silence Hydra's 1.1->1.2
        # deprecation warning, which fires only while the setting is left
        # implicit. A caller who passes their own `hydra.job.chdir` override wins.
        if not any(o.split("=", 1)[0] == "hydra.job.chdir" for o in launch_overrides):
            launch_overrides.append("hydra.job.chdir=True")

        for k, v in workflow_overrides.items():
            # A bare list/tuple is the most common new-user slip (forgetting to
            # wrap sweep values). Give an actionable error instead of the generic
            # type-list one — `multirun`/`hydra_list` are UserList, not list/tuple,
            # so a real sweep argument is not caught here.
            if isinstance(v, (list, tuple)):
                raise TypeError(
                    f"`{k}` was given a bare {type(v).__name__}, {list(v)!r}. To "
                    f"sweep over these values, wrap them: "
                    f"`{k}=mushin.multirun({list(v)!r})`. To pass the list as a "
                    f"single (non-swept) argument value, use "
                    f"`mushin.hydra_list({list(v)!r})`."
                )
            value_check(k, v, type_=(int, float, bool, str, multirun, hydra_list))

            prefix = "+" if _cfg_missing_or_none(self.eval_task_cfg, k) else ""

            if isinstance(v, multirun):
                # A repeated axis value cannot form a well-defined coordinate:
                # two cells would share a combo key (silently collapsing to the
                # last, and making `.sel` ambiguous). Repeat trials belong on a
                # distinct axis (e.g. seed), not as duplicate values.
                keys = [_scalar_repr(item) for item in v]
                if len(set(keys)) != len(keys):
                    dupes = sorted(
                        {
                            repr(item)
                            for item, kk in zip(v, keys, strict=True)
                            if keys.count(kk) > 1
                        }
                    )
                    raise ValueError(
                        f"`{k}` has duplicate sweep values {', '.join(dupes)}; a "
                        "repeated axis value collapses to a single cell. Use "
                        "distinct values, or add a separate axis (e.g. seed) for "
                        "repeat trials."
                    )
                # Build a Hydra `choice(...)` sweep rather than a comma-joined
                # string. `choice(...)` always yields a sweep – even for a
                # single element (so length-1 multiruns remain swept dimensions)
                # – and each element is serialized/quoted so that values
                # containing commas, spaces, or '=' are not silently re-split by
                # Hydra's override parser.
                choices = ",".join(_to_override_element(item) for item in v)
                launch_overrides.append(f"{prefix}{k}=choice({choices})")
            elif isinstance(v, hydra_list):
                # hydra_list serializes itself to Hydra list syntax; a single
                # non-swept list value passed to one job.
                launch_overrides.append(f"{prefix}{k}={v}")
            else:
                # Scalar value: serialize like a multirun element so a fixed
                # string containing ',' / '=' stays a single string (not an
                # accidental sweep) and a string like "true"/"1" is not coerced
                # to a bool/int.
                launch_overrides.append(f"{prefix}{k}={_to_override_element(v)}")

        for _name in self._REQUIRED_STATIC_METHODS:
            if _name == "task" and hasattr(self, "evaluation_task"):
                # TODO: remove when evaluation_task support is removed
                _name = "evaluation_task"
            if not isinstance(getattr_static(self, _name), staticmethod):
                raise TypeError(
                    f"{type(self).__name__}.{_name} must be a static method"
                )

        if task_fn_wrapper is None:
            task_fn_wrapper = _identity

        if pre_task_fn_wrapper is None:
            pre_task_fn_wrapper = _identity

        # Swept dimension names for the per-cell combo (unchanged).
        _swept_names = tuple(
            k
            for k, v in self._parse_overrides(launch_overrides).items()
            if isinstance(v, multirun)
        )
        # BaseWorkflow (no sanitizer) records an empty combo, as before.
        _runner_swept = (
            _swept_names if hasattr(self, "_sanitize_coordinate_for_xarray") else ()
        )

        from ._provenance import capture_base
        from ._sweep_io import Manifest

        _base_provenance = capture_base()

        # Kill-durable resume state (only when resuming) — built here so the
        # runner can hold it. Slimmed to the completed {combo_key: dir} map:
        # out-of-process launchers pickle the runner to EVERY worker, so
        # shipping the full N-cell manifest would cost O(N^2) serialized bytes
        # across an N-cell sweep.
        _prior_manifest = None
        if resume:
            _prior_manifest = _PriorCells.from_manifest(
                Manifest.from_cell_status(
                    Path(working_dir).resolve(), list(_swept_names)
                )
            )
            # A resume whose grid shape differs from the prior sweep (an axis
            # added or removed) cannot reuse the old cells — their combo keys
            # no longer correspond. Warn once so the full re-run isn't a
            # surprise (the per-cell lookup already refuses the mismatched
            # cells; this is the user-facing signal).
            if _prior_manifest.completed and set(_prior_manifest.params) != set(
                _swept_names
            ):
                import warnings

                warnings.warn(
                    f"resume: the sweep grid changed (was over "
                    f"{sorted(_prior_manifest.params)}, now over "
                    f"{sorted(_swept_names)}); prior cells cannot be reused and "
                    "the full grid will be re-run.",
                    UserWarning,
                    stacklevel=2,
                )

        from ._resume import code_fingerprint

        _task_fn, _wants_resume = _prepare_task(self.task)
        task_call = _TaskRunner(
            task=task_fn_wrapper(_task_fn),
            pre_task=pre_task_fn_wrapper(self.pre_task),
            swept_names=_runner_swept,
            base_provenance=_base_provenance,
            on_error=on_error,
            inject_resume=_wants_resume,
            prior_manifest=_prior_manifest,
            code_hash=code_fingerprint(self.task),
        )

        self._capture_env = capture_env

        # Run a Multirun over epsilons
        jobs = launch(
            self.eval_task_cfg,
            task_call,
            overrides=launch_overrides,
            multirun=True,
            version_base=version_base,
            to_dictconfig=to_dictconfig,
            config_name=config_name,
            job_name=job_name,
            with_log_configuration=with_log_configuration,
        )

        if isinstance(jobs, list):
            # Hydra returns a list of per-batch job lists (one batch for the
            # basic sweeper, several when a sweeper/launcher batches — e.g.
            # `hydra.sweeper.max_batch_size`). Flatten to a single JobReturn
            # list; the old `len == 1` special case silently mishandled the
            # multi-batch shape (AssertionError after all jobs had run).
            flat = []
            for batch in jobs:
                flat.extend(batch) if isinstance(batch, list) else flat.append(batch)
            jobs = flat
            _job_nums = [j.hydra_cfg.hydra.job.num for j in jobs]
            # ensure jobs are always sorted by job-num
            jobs = _sort_x_by_k(jobs, _job_nums)

        self.jobs = jobs
        self.jobs_post_process()

        if getattr(self, "_capture_env", False):
            try:
                _write_env_snapshot(self.working_dir)
            except Exception:  # noqa: BLE001 - env capture is best-effort
                pass

    def jobs_post_process(self):  # pragma: no cover
        """Method to extract attributes and metrics relevant to the workflow."""
        raise NotImplementedError()

    def plot(self, **kwargs) -> None:  # pragma: no cover
        """Plot workflow metrics."""
        raise NotImplementedError()

    def to_xarray(self):  # pragma: no cover
        """Convert workflow data to xArray Dataset or DataArray."""
        raise NotImplementedError()


def _non_str_sequence(x: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(x, Sequence) and not isinstance(x, str)


def _coerce_list_of_arraylikes(v: list[Any]):
    if v and hasattr(v[0], "__array__"):
        return [np.asarray(i) for i in v]
    return v


def _unwrap_scalar(item: Any) -> Any:
    """Unwrap a NumPy/torch 0-d scalar (e.g. from ``np.arange``) to its Python
    scalar. Mirrors the unwrapping in ``_to_override_element`` so that a swept
    value read back from a config keys identically to its override element."""
    if hasattr(item, "item") and getattr(item, "ndim", None) == 0:
        return item.item()
    return item


class MultiRunMetricsWorkflow(BaseWorkflow):
    """Abstract class for workflows that record metrics using Hydra multirun.

    This workflow creates subdirectories of multirun experiments using Hydra.  These directories
    contain the Hydra YAML configuration and any saved metrics file (defined by the evaluationf task)::

        ├── working_dir
        │    ├── <experiment directory name: 0>
        │    |    ├── <hydra output subdirectory: (default: .hydra)>
        |    |    |    ├── config.yaml
        |    |    |    ├── hydra.yaml
        |    |    |    ├── overrides.yaml
        │    |    ├── <metrics_filename>
        │    ├── <experiment directory name: 1>
        |    |    ...

    The evaluation task is expected to return a dictionary that maps
    `metric-name (str) -> value (number | Sequence[number])`

    Examples
    --------
    Let's create a simple workflow where we perform a multirun over a parameter,
    `epsilon`, and evaluate a task function that computes an accuracy and loss based on
    that `epsilon` value and a specified `scale`.

    >>> from mushin.workflows import MultiRunMetricsWorkflow
    >>> from mushin import multirun

    >>> class LocalRobustness(MultiRunMetricsWorkflow):
    ...     @staticmethod
    ...     def task(epsilon: float, scale: float) -> dict:
    ...         epsilon *= scale
    ...         val = 100 - epsilon**2
    ...         result = dict(accuracies=val+2, loss=epsilon**2)
    ...         tr.save(result, "test_metrics.pt")
    ...         return result

    We'll run this workflow for six total configurations of three `epsilon` values and
    two `scale` values. This will launch a Hydra multirun job and aggregate the results.

    >>> wf = LocalRobustness()
    >>> wf.run(epsilon=multirun([1.0, 2.0, 3.0]), scale=multirun([0.1, 1.0]))
    [2022-05-02 11:57:59,219][HYDRA] Launching 6 jobs locally
    [2022-05-02 11:57:59,220][HYDRA] 	#0 : +epsilon=1.0 +scale=0.1
    [2022-05-02 11:57:59,312][HYDRA] 	#1 : +epsilon=1.0 +scale=1.0
    [2022-05-02 11:57:59,405][HYDRA] 	#2 : +epsilon=2.0 +scale=0.1
    [2022-05-02 11:57:59,498][HYDRA] 	#3 : +epsilon=2.0 +scale=1.0
    [2022-05-02 11:57:59,590][HYDRA] 	#4 : +epsilon=3.0 +scale=0.1
    [2022-05-02 11:57:59,683][HYDRA] 	#5 : +epsilon=3.0 +scale=1.0

    Now that this workflow has run, we can view the results as an xarray-dataset whose
    coordinates reflect the multirun parameters that were varied, and whose
    data-variables are our recorded metrics: "accuracies" and "loss".

    >>> ds = wf.to_xarray()
    >>> ds
    <xarray.Dataset>
    Dimensions:     (epsilon: 3, scale: 2)
    Coordinates:
    * epsilon     (epsilon) float64 1.0 2.0 3.0
    * scale       (scale) float64 0.1 1.0
    Data variables:
        accuracies  (epsilon, scale) float64 102.0 101.0 102.0 98.0 101.9 93.0
        loss        (epsilon, scale) float64 0.01 1.0 0.04 4.0 0.09 9.0

    We can also load this workflow by providing the working directory where it was run.

    >>> loaded = LocalRobustness().load_from_dir(wf.working_dir)
    >>> loaded.to_xarray()
    <xarray.Dataset>
    Dimensions:     (epsilon: 3, scale: 2)
    Coordinates:
    * epsilon     (epsilon) float64 1.0 2.0 3.0
    * scale       (scale) float64 0.1 1.0
    Data variables:
        accuracies  (epsilon, scale) float64 102.0 101.0 102.0 98.0 101.9 93.0
        loss        (epsilon, scale) float64 0.01 1.0 0.04 4.0 0.09 9.0
    """

    def __init__(self, eval_task_cfg=None, working_dir: Path | None = None) -> None:
        super().__init__(eval_task_cfg)
        self._working_dir = working_dir
        # Per-cell failure records ({"combo", "exception", "working_dir"}) and the
        # on-disk sweep manifest, populated by `jobs_post_process` under fail-soft.
        self.failures: list[dict[str, Any]] = []
        self._manifest: Any = None

        if self._working_dir is not None:
            self.load_from_dir(self.working_dir, metrics_filename=None)

    @property
    def is_complete(self) -> bool:
        """Whether every requested grid cell completed. ``True`` when no manifest
        exists yet (e.g. a workflow loaded from disk that never swept)."""
        if self._manifest is not None:
            return self._manifest.is_complete()
        return True

    @property
    def provenance(self) -> dict | None:
        """Best-effort per-run provenance (git/versions/config) read from one
        job's ``mushin_provenance.json``. ``None`` if no record is found."""
        import json

        dirs = self.multirun_working_dirs or []
        for d in dirs:
            p = Path(d) / "mushin_provenance.json"
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except Exception:  # noqa: BLE001 - best-effort
                    return None
        return None

    # TODO: add target_job_dirs example
    #      Document .swap_dims({"job_dir": <...>}) and .set_index(job_dir=[...]).unstack("job_dir")
    #      for re-indexing based on overrides values

    _JOBDIR_NAME: str = "job_dir"
    _target_dir_multirun_overrides: defaultdict[str, list[Any]] | None = None
    output_subdir: str | None = None

    # List of all the dirs that the multirun writes to; sorted by job-num
    multirun_working_dirs: list[Path] | None = None

    @staticmethod
    def task(*args: Any, **kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover
        """Abstract `staticmethod` for users to define the task that is configured and
        launched by the workflow"""
        raise NotImplementedError()

    @staticmethod
    def metric_load_fn(file_path: Path) -> Mapping[str, Any]:
        """Loads a metric file and returns a dictionary of metric-name -> metric-value
        mappings.

        The default loader sniffs the file: a JSON metrics sidecar (the
        ``mushin_metrics.json`` a task writes by returning a dict) is read with
        ``json``; anything else with ``torch.load`` (the ``MetricsCallback``
        ``.pt`` path). Override this method for a custom format.

        Parameters
        ----------
        file_path : Path

        Returns
        -------
        named_metrics : Mapping[str, Any]
            metric-name -> metric-value(s)

        Examples
        --------
        Designing a workflow that uses the `pickle` module to save and load
        metrics

        >>> from mushin import MultiRunMetricsWorkflow, multirun
        >>> import pickle
        >>>
        >>> class PickledWorkFlow(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def metric_load_fn(file_path: Path):
        ...         with file_path.open("rb") as f:
        ...             return pickle.load(f)
        ...
        ...     @staticmethod
        ...     def task(a, b):
        ...         with open("./metrics.pkl", "wb") as f:
        ...             pickle.dump(dict(a=a, b=b), f)
        >>>
        >>> wf = PickledWorkFlow()
        >>> wf.run(a=multirun([1, 2, 3]), b=False)
        >>> wf.load_metrics("metrics.pkl")
        >>> wf.metrics
        dict(a=[1, 2, 3], b=[False, False, False])"""
        file_path = Path(file_path)
        # Sniff the content (not the extension): a JSON metrics sidecar starts
        # with `{`/`[`; a torch pickle/zip starts with other bytes. This lets
        # the default loader read the JSON sidecar written by dict-returning
        # tasks without the caller overriding metric_load_fn.
        try:
            with open(file_path, "rb") as f:
                head = f.read(64).lstrip()
        except OSError:
            head = b""
        if head[:1] in (b"{", b"["):
            import json

            with open(file_path) as f:
                return json.load(f)
        # weights_only=False: workflow-produced metrics files are trusted and
        # contain numpy arrays/dicts. torch 2.6 flipped this default to True.
        import torch as tr

        return tr.load(file_path, weights_only=False)

    def run(
        self,
        *,
        task_fn_wrapper: Callable[[Callable[..., T1]], Callable[[Any], T1]]
        | None = zen,
        pre_task_fn_wrapper: Callable[[Callable[..., None]], Callable[[Any], None]]
        | None = zen,
        working_dir: str | None = None,
        sweeper: str | None = None,
        launcher: str | None = None,
        overrides: list[str] | None = None,
        version_base: str | type[_NotSet] | None = _VERSION_BASE_DEFAULT,
        target_job_dirs: Sequence[str | Path] | None = None,
        to_dictconfig: bool = False,
        config_name: str = "mushin_workflow",
        job_name: str = "mushin_workflow",
        with_log_configuration: bool = True,
        on_error: str = "raise",
        resume: bool = False,
        capture_env: bool = False,
        **workflow_overrides: str | int | float | bool | multirun | hydra_list,
    ):
        """Run the sweep: one Hydra job per grid cell, metrics collected per cell.

        Extends :meth:`BaseWorkflow.run` (see it for the launcher/sweeper/
        Hydra parameters) with metrics-workflow behavior:

        Parameters
        ----------
        target_job_dirs : Sequence[str | Path] | None (default: None)
            Existing job directories to evaluate over: each directory becomes
            one cell of a ``job_dir`` sweep dimension, and the task receives
            that cell's directory as its ``job_dir`` argument. Use this to
            post-process completed runs (e.g. evaluate saved checkpoints)
            instead of sweeping parameter values.
        on_error : str (default: "raise")
            ``"raise"`` propagates the first failing cell. ``"nan"`` records
            the failure (``self.failures``, the sweep manifest, and a
            ``mushin_error.txt`` traceback in the cell dir), fills that cell's
            metrics with NaN, and keeps sweeping.
        resume : bool (default: False)
            Skip cells already recorded as completed in ``working_dir`` (their
            metrics are read from the sidecar) and re-run the rest. A completed
            cell is reused only if its swept-parameter combination AND
            fingerprints of both its resolved config and its task source still
            match; a changed non-swept value or an edited task body re-runs that
            cell (with a warning). The fingerprint does not cover helper
            functions the task calls, module-level constants, or the
            environment — for a larger refactor, re-run from a fresh
            ``working_dir`` rather than resuming.
        capture_env : bool (default: False)
            After the sweep, snapshot the environment (``uv export``, falling
            back to ``uv pip freeze`` then an ``importlib.metadata`` dump) to
            ``working_dir/mushin_env.txt`` — or ``mushin_env.<n>.txt`` if a
            snapshot already exists there (a resume never overwrites the
            original run's snapshot).
        **workflow_overrides
            The sweep itself: ``param=value`` fixes a value,
            ``param=multirun([...])`` makes a grid dimension. Nested config
            paths (``**{"model.width": multirun([4, 8])}``) and config groups
            are supported — see the workflows guide's "Sweep-axis support"
            section.
        """
        if target_job_dirs is not None:
            if isinstance(target_job_dirs, str):
                raise TypeError(
                    f"`target_job_dirs` must be a sequence of pathlike objects, got: {target_job_dirs}"
                )
            value_check("target_job_dirs", target_job_dirs, type_=Sequence)

            target_job_dirs = [Path(s).resolve() for s in target_job_dirs]
            for d in target_job_dirs:
                if not d.is_dir() or not d.exists():
                    raise FileNotFoundError(
                        f"The specified target directory – {d} – does not exist."
                    )
            target_job_dirs = multirun([str(s) for s in target_job_dirs])
            workflow_overrides[self._JOBDIR_NAME] = target_job_dirs

        return super().run(
            working_dir=working_dir,
            sweeper=sweeper,
            launcher=launcher,
            overrides=overrides,
            task_fn_wrapper=task_fn_wrapper,
            pre_task_fn_wrapper=pre_task_fn_wrapper,
            version_base=version_base,
            to_dictconfig=to_dictconfig,
            config_name=config_name,
            job_name=job_name,
            with_log_configuration=with_log_configuration,
            on_error=on_error,
            resume=resume,
            capture_env=capture_env,
            **workflow_overrides,
        )

    @property
    def target_dir_multirun_overrides(self) -> dict[str, list[Any]]:
        """
        For a multirun that sweeps over the target directories of a
        previous multirun, `target_dir_multirun_overrides` provides
        the flattened overrides for that previous run.

        Examples
        --------
        >>> class A(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def task(value: float, scale: float):
        ...         pass
        ...

        >>> class B(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def task():
        ...         pass

        >>> a = A()
        >>> a.run(value=multirun([-1.0, 0.0, 1.0]), scale=multirun([11.0, 9.0]))
        [2022-05-13 17:19:51,497][HYDRA] Launching 6 jobs locally
        [2022-05-13 17:19:51,497][HYDRA] 	#0 : +value=-1.0 +scale=11.0
        [2022-05-13 17:19:51,555][HYDRA] 	#1 : +value=-1.0 +scale=9.0
        [2022-05-13 17:19:51,729][HYDRA] 	#2 : +value=1.0 +scale=11.0
        [2022-05-13 17:19:51,787][HYDRA] 	#3 : +value=1.0 +scale=9.0

        >>> b = B()
        >>> b.run(target_job_dirs=a.multirun_working_dirs)
        [2022-05-13 17:19:59,900][HYDRA] Launching 6 jobs locally
        [2022-05-13 17:19:59,900][HYDRA] 	#0 : +job_dir=/home/scratch/multirun/0
        [2022-05-13 17:19:59,958][HYDRA] 	#1 : +job_dir=/home/scratch/multirun/1
        [2022-05-13 17:20:00,015][HYDRA] 	#2 : +job_dir=/home/scratch/multirun/2
        [2022-05-13 17:20:00,073][HYDRA] 	#3 : +job_dir=/home/scratch/multirun/3

        >>> b.target_dir_multirun_overrides
        {'value': [-1.0, -1.0, 1.0, 1.0],
         'scale': [11.0, 9.0, 11.0, 9.0]}"""
        if self._target_dir_multirun_overrides is not None:
            return dict(self._target_dir_multirun_overrides)
        assert self.output_subdir is not None

        multirun_cfg = self.working_dir / "multirun.yaml"
        self._target_dir_multirun_overrides = defaultdict(list)

        overrides = load_from_yaml(multirun_cfg).hydra.overrides.task
        self.overrides = overrides

        # Use Hydra's override parser (rather than naive string splitting) so
        # that the job-dir sweep – e.g. `+job_dir=choice('a','b')` – and any
        # quoted values containing '=' / commas are parsed correctly.
        parsed = self._parse_overrides(list(overrides))
        jobdir_value = parsed.get(self._JOBDIR_NAME, [])
        dirs = list(jobdir_value) if _non_str_sequence(jobdir_value) else []

        for d in dirs:
            overrides: list[str] = list(
                load_from_yaml(Path(d) / f"{self.output_subdir}/overrides.yaml")
            )
            output = self._parse_overrides(overrides)

            for ko, vo in output.items():
                self._target_dir_multirun_overrides[ko].append(vo)
        return dict(self._target_dir_multirun_overrides)

    def jobs_post_process(self):
        assert len(self.jobs) > 0
        # TODO: Make protocol type for JobReturn
        assert isinstance(self.jobs[0], JobReturn)
        self.jobs: list[JobReturn]

        self.multirun_working_dirs = []

        for job in self.jobs:
            _hydra_cfg = job.hydra_cfg
            assert _hydra_cfg is not None
            assert job.working_dir is not None
            _cwd = _hydra_cfg.hydra.runtime.cwd
            working_dir = Path(_cwd) / job.working_dir
            self.multirun_working_dirs.append(working_dir)

        # set working directory of this workflow
        self.working_dir = self.multirun_working_dirs[0].parent

        hydra_cfg = self.jobs[0].hydra_cfg
        assert hydra_cfg is not None
        self.output_subdir = hydra_cfg.hydra.output_subdir

        # extract configs
        self.cfgs = [j.cfg for j in self.jobs]

        # Status-aware collection: build the combo-keyed metrics map (so
        # `to_xarray` can assemble the grid by config combination, resilient to
        # missing/failed cells), record failures, and persist a sweep manifest.
        from ._sweep_io import Manifest, combo_key

        on_error = getattr(self, "_on_error", "raise")
        swept_names = self._swept_param_names()
        # Scope the manifest to the CURRENT grid: constructing a fresh Manifest
        # (rather than load_or_new) drops stale cells left by a prior sweep that
        # reused this dir with a different/narrowed grid, so `is_complete()` does
        # not AND over cells absent from the current grid. Every current-grid
        # combo is re-marked in the loop below, and resume reads prior state from
        # its own `prior_manifest` (loaded in `run()` before this file is rewritten).
        manifest = Manifest(self.working_dir, swept_names)

        self._metrics_by_combo = {}
        self.failures = []
        job_metrics: list[Any] = []
        for job, cfg, wdir in zip(
            self.jobs, self.cfgs, self.multirun_working_dirs, strict=True
        ):
            # Project onto the swept params only, matching `to_xarray`'s
            # `_lookup_key` so `_metrics_by_combo` and manifest keys align.
            combo = self._combo_of_cfg(cfg, choices=_group_choices_of_job(job))
            swept_combo = {n: combo[n] for n in swept_names}
            key = combo_key(swept_combo)

            # Determine the raw return value / failure WITHOUT triggering a
            # re-raise: a real Hydra FAILED job stores its exception in
            # `_return_value`; under fail-soft, a completed job may carry a
            # `_FailedRun` sentinel wrapping the caught exception.
            if job.status == JobStatus.COMPLETED:
                rv = job.return_value
                exc = rv.exception if isinstance(rv, _FailedRun) else None
            else:
                rv = None
                exc = job._return_value

            if exc is not None:
                if on_error == "raise":
                    raise exc
                self.failures.append(
                    {
                        "combo": key,
                        "exception": repr(exc),
                        "working_dir": str(wdir),
                    }
                )
                manifest.mark(
                    swept_combo, dir=wdir.name, status="failed", error=repr(exc)
                )
                job_metrics.append(None)
            else:
                job_metrics.append(rv)
                if rv is not None:
                    self._metrics_by_combo[key] = rv
                manifest.mark(swept_combo, dir=wdir.name, status="completed")

        self.metrics = self._process_metrics(job_metrics)

        manifest.save()
        self._manifest = manifest

        if self.failures:
            import warnings

            warnings.warn(
                f"{len(self.failures)} run(s) failed: "
                f"{[f['combo'] for f in self.failures]}; grid cells set to NaN.",
                UserWarning,
                stacklevel=2,
            )

    def _swept_param_names(self) -> list[str]:
        """Names of the multirun (grid-dimension) parameters."""
        return [
            k
            for k, v in self.multirun_task_overrides.items()
            if isinstance(v, multirun)
        ]

    def _combo_of_cfg(self, cfg: Any, choices: dict | None = None) -> dict[str, Any]:
        """The sanitized swept-parameter combination for a single job config.

        Values are unwrapped (0-d scalars), converted out of OmegaConf
        containers, and passed through `_sanitize_coordinate_for_xarray` so
        they key identically to the (already-sanitized) coordinate values used
        to build the grid in `to_xarray`. Swept names may be nested config
        paths (``model.width``). ``choices`` is the job's
        ``hydra.runtime.choices``; a config-GROUP axis keys by its chosen
        option name (matching the grid coordinate), not the composed
        sub-config."""
        from omegaconf import OmegaConf

        combo = {}
        for n in self._swept_param_names():
            val = _read_swept_value(cfg, n)
            if OmegaConf.is_config(val):
                if choices and choices.get(n) is not None:
                    val = str(choices[n])
                else:
                    val = OmegaConf.to_container(val, resolve=True)
            combo[n] = self._sanitize_coordinate_for_xarray(_unwrap_scalar(val))
        return combo

    def _job_choices(self, i: int) -> dict | None:
        """``runtime.choices`` for job ``i``, when jobs align with cfgs."""
        jobs = getattr(self, "jobs", None)
        if jobs and len(jobs) == len(self.cfgs):
            return _group_choices_of_job(jobs[i])
        return None

    @staticmethod
    def _process_metrics(job_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = defaultdict(list)
        for task_metrics in job_metrics:
            if task_metrics is None:
                continue
            for k, v in task_metrics.items():
                # get item if it's a single element array
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]

                metrics[k].append(v)
        return metrics

    def load_from_dir(
        self: Self,
        working_dir: Path | str,
        metrics_filename: str | Sequence[str] | None,
    ) -> Self:
        """Loading workflow job data from a given working directory. The workflow
        is loaded in-place and "self" is returned by this method.

        Parameters
        ----------
        working_dir: str | Path
            The base working directory of the experiment. It is expected
            that subdirectories within this working directory will contain
            individual Hydra jobs data (yaml configurations) and saved metrics files.

        metrics_filename: str | Sequence[str] | None
            The filename(s) or glob-pattern(s) uses to load the metrics.
            If `None`, the metrics stored in `self.metrics` is used.

        Returns
        -------
        loaded_workflow : Self
        """
        # Reset memoized caches so a reused workflow object does not return
        # stale overrides/coordinates from a previously loaded directory.
        self._multirun_task_overrides = {}
        self._target_dir_multirun_overrides = None

        self.working_dir = Path(working_dir)
        self.output_subdir = load_from_yaml(
            self.working_dir / "multirun.yaml"
        ).hydra.output_subdir

        self.multirun_working_dirs = list(
            x.parent for x in self.working_dir.glob(f"**/*/{self.output_subdir}")
        )

        # ensure working dirs are sorted by job num
        _job_nums = (
            load_from_yaml(dir_ / f"{self.output_subdir}/hydra.yaml").hydra.job.num
            for dir_ in self.multirun_working_dirs
        )

        self.multirun_working_dirs = _sort_x_by_k(self.multirun_working_dirs, _job_nums)
        self.cfgs = []

        for dir_ in self.multirun_working_dirs:
            # Ensure we load saved YAML configurations for each job (in hydra.job.output_subdir)
            cfg_file = dir_ / f"{self.output_subdir}/config.yaml"
            if not cfg_file.exists():
                raise FileNotFoundError(
                    f"{cfg_file} not found; the sweep directory is incomplete or "
                    "was cleaned. Re-run the sweep, or point load_from_dir at a "
                    "directory whose cells still have their .hydra/config.yaml."
                )
            self.cfgs.append(load_from_yaml(cfg_file))

        if metrics_filename is not None:
            self.load_metrics(metrics_filename)

        return self

    def load_metrics(
        self, metrics_filename: str | Sequence[str]
    ) -> dict[str, list[Any]]:
        """Loads and aggregates across all multirun working dirs, and stores
        the metrics in `self.metrics`.

        `self.metric_load_fn` is used to load each job's metric file(s).

        Parameters
        ----------
        metrics_filename : str | Sequence[str]
            The filename(s) or glob-pattern(s) used to load the metrics.
            Required (unlike ``to_xarray``/``load_from_dir``, this method does
            not accept ``None``).

        Returns
        -------
        metrics : Dict[str, List[Any]]

        Examples
        --------
        Creating a workflow that saves named metrics using `torch.save`

        >>> from mushin.workflows import MultiRunMetricsWorkflow, multirun
        >>> import torch as tr
        >>>
        ... class TorchWorkFlow(MultiRunMetricsWorkflow):
        ...     @staticmethod
        ...     def task(a, b):
        ...         tr.save(dict(a=a, b=b), "metrics.pt")
        ...
        >>> wf = TorchWorkFlow()
        >>> wf.run(a=multirun([1, 2, 3]), b=False)
        [2022-06-01 12:35:51,650][HYDRA] Launching 3 jobs locally
        [2022-06-01 12:35:51,650][HYDRA] 	#0 : +a=1 +b=False
        [2022-06-01 12:35:51,715][HYDRA] 	#1 : +a=2 +b=False
        [2022-06-01 12:35:51,780][HYDRA] 	#2 : +a=3 +b=False

        `~MultiRunMetricsWorkflow.metric_load_fn` reads the JSON metrics sidecar
        or a torch file by default (refer to it to change this behavior).

        >>> wf.load_metrics("metrics.pt")
        defaultdict(list, {'a': [1, 2, 3], 'b': [False, False, False]})
        >>> wf.metrics
        defaultdict(list, {'a': [1, 2, 3], 'b': [False, False, False]})
        """
        if self.multirun_working_dirs is None:
            self.load_from_dir(self.working_dir, metrics_filename=None)
            assert self.multirun_working_dirs is not None

        if isinstance(metrics_filename, str):
            metrics_filename = [metrics_filename]

        job_metrics = []
        for dir_ in self.multirun_working_dirs:
            _metrics = {}
            for name in metrics_filename:
                files = sorted(dir_.glob(name))
                if not files:
                    raise FileNotFoundError(
                        f"No files with the path/pattern {dir_ / name} were found"
                    )

                for f_ in files:
                    _metrics.update(self.metric_load_fn(f_))
            job_metrics.append(_metrics)

        self.metrics = self._process_metrics(job_metrics)

        return self.metrics

    @staticmethod
    def _sanitize_coordinate_for_xarray(
        value: LoadedValue | Sequence[LoadedValue],
    ) -> str | int | float | bool | list[str | int | float | bool]:
        """Nested sequences are not permitted for xarray coordinates. This
        Returns a list of scalars when `value` is a multi-run or a scalar.

        Inner sequences are converted to strings"""
        if _non_str_sequence(value):
            if isinstance(value, multirun):
                _seq: Sequence[LoadedValue] = value
                return [str(_v) if _non_str_sequence(_v) else _v for _v in _seq]
            return str(value)
        return value  # type: ignore

    def to_dataframe(self, **to_xarray_kwargs):
        """The sweep results as a tidy long-form :class:`pandas.DataFrame`.

        One row per sweep cell (times any extra metric dimensions), with the
        sweep parameters and metrics as plain columns — the pandas view of
        :meth:`to_xarray` (``to_xarray(...).to_dataframe().reset_index()``),
        for when you'd rather not touch xarray at all. Keyword arguments
        forward to :meth:`to_xarray`. Dataset-level ``attrs`` (provenance,
        failure records) live only on the xarray form.
        """
        return self.to_xarray(**to_xarray_kwargs).to_dataframe().reset_index()

    def to_xarray(
        self,
        include_working_subdirs_as_data_var: bool = False,
        coord_from_metrics: str | None = None,
        non_multirun_params_as_singleton_dims: bool = False,
        metrics_filename: str | Sequence[str] | None = None,
    ):
        """Convert workflow data to xarray Dataset.

        Parameters
        ----------
        include_working_subdirs_as_data_var : bool, optional (default=False)
            If `True` then the data-variable "working_subdir" will be included in the
            xarray. This data variable is used to lookup the working sub-dir path
            (a string) by multirun coordinate.

        coord_from_metrics : str | None (default: None)
            If not `None` defines the metric key to use as a coordinate
            in the `Dataset`.  This function assumes that this coordinate
            represents the leading dimension for all data-variables.

        non_multirun_params_as_singleton_dims : bool, optional (default=False)
            If `True` then non-multirun entries from `workflow_overrides` will be
            included as length-1 dimensions in the xarray. Useful for merging/
            concatenation with other Datasets

        metrics_filename: Optional[str]
            The filename or glob-pattern uses to load the metrics.
            If `None`, the metrics stored in `self.metrics` is used.

        Returns
        -------
        results : xarray.Dataset
            A dataset whose dimensions and coordinate-values are determined by the
            quantities over which the multi-run was performed. The data variables
            correspond to the named results returned by the jobs."""
        import xarray as xr

        if metrics_filename is not None:
            if self.multirun_working_dirs is None:
                self.load_from_dir(self.working_dir, metrics_filename=metrics_filename)
            else:
                self.load_metrics(metrics_filename)

        # all overrides containing non-multirun lists must be converted to
        # strings so that xarray treats that list value as a "scalar"
        #
        # stores: override-name -> value
        # where value is either a scalar (i.e. int|float|bool|str) or a list of scalars
        # A list of scalars indicates a multirun
        cast_overrides = {
            k: self._sanitize_coordinate_for_xarray(value)
            for k, value in self.multirun_task_overrides.items()
        }

        orig_coords = {
            k: (v if _non_str_sequence(v) else [v])
            for k, v in cast_overrides.items()
            if non_multirun_params_as_singleton_dims or _non_str_sequence(v)
        }

        metric_coords = {}
        if coord_from_metrics:
            if coord_from_metrics not in self.metrics:
                raise ValueError(
                    f"key `{coord_from_metrics}` not in metrics (available: "
                    f"{list(self.metrics.keys())})"
                )

            v = _coerce_list_of_arraylikes(self.metrics[coord_from_metrics])
            v = np.asarray(v)

            if v.ndim > 1:  # pragma: no cover
                # assume this coord was repeated across experiments, e.g., "epochs"
                v = v[0]
            metric_coords[coord_from_metrics] = v

        attrs = {k: v for k, v in cast_overrides.items() if not _non_str_sequence(v)}

        # we will add additional coordinates as-needed for multi-dim metrics
        coords: dict[str, Any] = orig_coords.copy()
        shape = tuple(len(v) for v in coords.values())

        # Assemble the metric data in *grid order*, keyed by each cell's config
        # combination, so that a missing/failed cell yields a full-shaped array
        # with NaN rather than mis-sizing a job-ordered reshape.
        import itertools

        from ._sweep_io import combo_key

        # The grid is keyed *only* on the swept (multirun) params. Under
        # `non_multirun_params_as_singleton_dims` `orig_coords` also carries
        # length-1 non-multirun dims; those must be projected out of the lookup
        # key so it matches `_combo_of_cfg` (which uses only swept names).
        swept_names = self._swept_param_names()

        def _lookup_key(combo: dict[str, Any]) -> str:
            return combo_key({n: combo[n] for n in swept_names})

        # If `_metrics_by_combo` was not populated by `jobs_post_process` (e.g. a
        # workflow loaded from disk), reconstruct it from the job-aligned
        # `self.metrics` columns and `self.cfgs`. This index-based mapping is only
        # sound when every column aligns 1:1 with `self.cfgs`; `_process_metrics`
        # drops None jobs / absent keys, so a ragged column would mis-map. Guard
        # against that and only reconstruct when all columns are full-length.
        if not self._metrics_by_combo and self.cfgs:
            n_jobs = len(self.cfgs)
            if all(len(col) == n_jobs for col in self.metrics.values()):
                for i, cfg in enumerate(self.cfgs):
                    per_job = {k: col[i] for k, col in self.metrics.items()}
                    if per_job:
                        combo = self._combo_of_cfg(cfg, choices=self._job_choices(i))
                        self._metrics_by_combo[combo_key(combo)] = per_job

        grid_names = list(orig_coords)
        grid_combos = [
            dict(zip(grid_names, vals, strict=True))
            for vals in itertools.product(*[orig_coords[n] for n in grid_names])
        ]

        # Union of metric keys across all present cells, preserving first-seen order.
        metric_keys: list[str] = []
        _seen: set[str] = set()
        for m in self._metrics_by_combo.values():
            if not isinstance(m, Mapping):
                continue
            for k in m:
                if k not in _seen:
                    _seen.add(k)
                    metric_keys.append(k)

        def _grid_column(key: str) -> list[Any]:
            col: list[Any] = []
            missing: list[int] = []
            for i, combo in enumerate(grid_combos):
                m = self._metrics_by_combo.get(_lookup_key(combo))
                if m is not None and key in m:
                    val = m[key]
                    # match `_process_metrics`: unwrap length-1 lists to a scalar
                    if isinstance(val, list) and len(val) == 1:
                        val = val[0]
                    col.append(val)
                else:
                    col.append(None)
                    missing.append(i)
            # NaN-fill missing cells, broadcasting to the metric's per-cell shape
            missing_set = set(missing)
            per_cell_shape: tuple[int, ...] = ()
            for i, c in enumerate(col):
                if i not in missing_set:
                    per_cell_shape = np.asarray(c).shape
                    break
            fill = np.nan if per_cell_shape == () else np.full(per_cell_shape, np.nan)
            for i in missing:
                col[i] = fill
            return col

        metrics_to_add = {k: _grid_column(k) for k in metric_keys}
        if (
            include_working_subdirs_as_data_var
            and self.multirun_working_dirs is not None
        ):
            # Emit `working_subdir` in the SAME grid order as the metric vars so
            # it cannot transpose relative to the data. Map each job's combo to
            # its working dir (cfgs and working dirs are job-aligned), then look
            # up per grid cell; a missing cell gets an empty string.
            workdir_by_combo = {
                combo_key(self._combo_of_cfg(cfg, choices=self._job_choices(i))): str(
                    wd
                )
                for i, (cfg, wd) in enumerate(
                    zip(self.cfgs, self.multirun_working_dirs, strict=True)
                )
            }
            metrics_to_add["working_subdir"] = [
                workdir_by_combo.get(_lookup_key(combo), "") for combo in grid_combos
            ]

        data = {}
        for k, v in metrics_to_add.items():
            if coord_from_metrics and k == coord_from_metrics:
                continue

            v = _coerce_list_of_arraylikes(v)

            datum = np.asarray(v).reshape(shape + np.asarray(v[0]).shape)

            k_coords = list(orig_coords)
            for n in range(datum.ndim - len(orig_coords)):
                if coord_from_metrics and n < len(metric_coords):
                    # Assume the first coordinate of the metric is the metric coordinate dimension
                    k_coords += list(metric_coords.keys())
                    for mk, mv in metric_coords.items():
                        coords[mk] = mv
                else:
                    # Create additional arbitrary coordinates as-needed for non-scalar
                    # metrics
                    k_coords += [f"{k}_dim{n}"]
                    coords[f"{k}_dim{n}"] = np.arange(datum.shape[len(orig_coords) + n])

            data[k] = (k_coords, datum)

        coords.update(metric_coords)
        import json as _json

        # Record fail-soft failures (combo keys of NaN-filled cells) on the
        # dataset. Only set when a fail-soft run actually recorded failures so a
        # fully-completed sweep's `attrs` is unchanged from prior behavior.
        # Stored as a JSON string: a raw list of strings is not a portable
        # netCDF attr (the scipy engine rejects it, and netCDF4 collapses a
        # 1-element list to a bare str on reload).
        if self.failures:
            attrs["mushin_failures"] = _json.dumps([f["combo"] for f in self.failures])
        # Attach best-effort per-run provenance (git/versions/config) from one
        # job's sidecar, so a saved/re-loaded dataset carries its lineage. Stored
        # as a JSON string (nested dicts are not valid netCDF-serializable attrs).
        prov = self.provenance
        if prov is not None:
            attrs["provenance"] = _json.dumps(prov)
        out = xr.Dataset(coords=coords, data_vars=data, attrs=attrs)

        if self._JOBDIR_NAME in set(out.coords):
            exp_dir = out.coords[self._JOBDIR_NAME]
            coords = {}
            for k, v in self.target_dir_multirun_overrides.items():
                if len(v) == len(exp_dir):
                    if (
                        len(set(np.unique(v))) > 1
                        or non_multirun_params_as_singleton_dims
                    ):
                        coords[k] = (
                            [self._JOBDIR_NAME],
                            [self._sanitize_coordinate_for_xarray(item) for item in v],
                        )
            out = out.assign_coords(coords)
        return out


class RobustnessCurve(MultiRunMetricsWorkflow):
    """Abstract class for workflows that measure performance for different perturbation
    values.

    This workflow requires and uses parameter `epsilon` as the configuration option for
    varying the perturbation.

    See Also
    --------
    MultiRunMetricsWorkflow
    """

    def run(
        self,
        *,
        epsilon: str | Sequence[float],
        task_fn_wrapper: Callable[[Callable[..., T1]], Callable[[Any], T1]]
        | None = zen,
        pre_task_fn_wrapper: Callable[[Callable[..., None]], Callable[[Any], None]]
        | None = zen,
        target_job_dirs: Sequence[str | Path] | None = None,
        version_base: str | type[_NotSet] | None = _VERSION_BASE_DEFAULT,
        working_dir: str | None = None,
        sweeper: str | None = None,
        launcher: str | None = None,
        overrides: list[str] | None = None,
        to_dictconfig: bool = False,
        config_name: str = "mushin_workflow",
        job_name: str = "mushin_workflow",
        with_log_configuration: bool = True,
        on_error: str = "raise",
        resume: bool = False,
        capture_env: bool = False,
        **workflow_overrides: str | int | float | bool | multirun | hydra_list,
    ):
        """Run the experiment for varying value `epsilon`.

        Parameters
        ----------
        epsilon: str | Sequence[float]
            The configuration parameter for the perturbation.  Unlike Hydra overrides,
            this parameter can be a list of floats that will be converted into a
            multirun sequence override for Hydra.

        task_fn_wrapper: Callable[[Callable[..., T1]], Callable[[Any], T1]] | None, optional (default=hydra_zen.zen)
            A wrapper applied to `self.task` prior to launching the task.
            The default wrapper is `hydra_zen.zen`. Specify `None` for no
            wrapper to be applied.

        working_dir: str (default: None, the Hydra default will be used)
            The directory to run the experiment in.  This value is used for
            setting `hydra.sweep.dir`.

        sweeper: str | None (default: None)
            The configuration name of the Hydra Sweeper to use (i.e., the override for
            `hydra/sweeper=sweeper`)

        launcher: str | None (default: None)
            The configuration name of the Hydra Launcher to use (i.e., the override for
            `hydra/launcher=launcher`)

        overrides: List[str] | None (default: None)
            Parameter overrides not considered part of the swept workflow
            parameter set (which are exposed via `multirun_task_overrides`).

        **workflow_overrides: str | int | float | bool | multirun | hydra_list
            These parameters represent the values for configurations to use for the
            experiment.

            These values will be appended to the `overrides` for the Hydra job.
        """

        if isinstance(epsilon, str):
            # Convenience string form, e.g. "0,1,2,3": parse the comma-separated
            # values into a multirun explicitly. (Fixed scalar strings no longer
            # auto-split into sweeps in the general override path, so this axis
            # must be built here rather than relying on that behavior.)
            def _parse(tok: str):
                tok = tok.strip()
                for cast in (int, float):
                    try:
                        return cast(tok)
                    except ValueError:
                        continue
                return tok

            epsilon = multirun([_parse(t) for t in epsilon.split(",")])
        else:
            epsilon = multirun(epsilon)

        return super().run(
            task_fn_wrapper=task_fn_wrapper,
            pre_task_fn_wrapper=pre_task_fn_wrapper,
            working_dir=working_dir,
            sweeper=sweeper,
            launcher=launcher,
            version_base=version_base,
            overrides=overrides,
            to_dictconfig=to_dictconfig,
            config_name=config_name,
            job_name=job_name,
            with_log_configuration=with_log_configuration,
            on_error=on_error,
            resume=resume,
            capture_env=capture_env,
            **workflow_overrides,
            # for multiple multi-run params, epsilon should fastest-varying param;
            # i.e. epsilon should be the trailing dim in the multi-dim array of results
            target_job_dirs=target_job_dirs,
            epsilon=epsilon,
        )

    def to_xarray(
        self,
        include_working_subdirs_as_data_var: bool = False,
        coord_from_metrics: str | None = None,
        non_multirun_params_as_singleton_dims: bool = False,
        metrics_filename: str | Sequence[str] | None = None,
    ):
        """Convert workflow data to xarray Dataset.

        Parameters
        ----------
        include_working_subdirs_as_data_var : bool, optional (default=False)
            If `True` then the data-variable "working_subdir" will be included in the
            xarray. This data variable is used to lookup the working sub-dir path
            (a string) by multirun coordinate.

        coord_from_metrics : str | None (default: None)
            If not `None` defines the metric key to use as a coordinate
            in the `Dataset`.  This function assumes that this coordinate
            represents the leading dimension for all data-variables.

        non_multirun_params_as_singleton_dims : bool, optional (default=False)
            If `True` then non-multirun entries from `workflow_overrides` will be
            included as length-1 dimensions in the xarray. Useful for merging/
            concatenation with other Datasets

        metrics_filename: Optional[str]
            The filename or glob-pattern uses to load the metrics.
            If `None`, the metrics stored in `self.metrics` is used.

        Returns
        -------
        results : xarray.Dataset
            A dataset whose dimensions and coordinate-values are determined by the
            quantities over which the multi-run was performed. The data variables
            correspond to the named results returned by the jobs."""
        return (
            super()
            .to_xarray(
                include_working_subdirs_as_data_var=include_working_subdirs_as_data_var,
                coord_from_metrics=coord_from_metrics,
                non_multirun_params_as_singleton_dims=non_multirun_params_as_singleton_dims,
                metrics_filename=metrics_filename,
            )
            .sortby("epsilon")
        )

    def plot(
        self,
        metric: str,
        ax: Any = None,
        group: str | None = None,
        save_filename: str | None = None,
        non_multirun_params_as_singleton_dims: bool = False,
        **kwargs,
    ) -> Any:
        """Plot metrics versus `epsilon`.

        Using the `xarray.Dataset` from `to_xarray`, plot the metrics
        against the workflow perturbation parameters.

        Parameters
        ----------
        metric: str
            The metric saved

        ax: Axes | None (default: None)
            If not `None`, the matplotlib.Axes to use for plotting.

        group: str | None (default: None)
            Needed if other parameters besides `epsilon` were varied.

        save_filename: str | None (default: None)
            If not `None` save figure to the filename provided.

        non_multirun_params_as_singleton_dims : bool, optional (default=False)
            If `True` then non-multirun entries from `workflow_overrides` will be
            included as length-1 dimensions in the xarray. Useful for merging/
            concatenation with other Datasets

        **kwargs: Any
            Additional arguments passed to `xarray.plot`.
        """
        import matplotlib.pyplot as plt

        created_fig = ax is None
        if ax is None:
            _, ax = plt.subplots()

        try:
            xdata = self.to_xarray(
                non_multirun_params_as_singleton_dims=non_multirun_params_as_singleton_dims
            )
            if group is None:
                plots = xdata[metric].plot.line(x="epsilon", ax=ax, **kwargs)
            else:
                # TODO: xarray.groupby doesn't support multidimensional grouping
                dg = xdata.groupby(group)
                plots = [
                    grp[metric].plot(x="epsilon", label=name, ax=ax, **kwargs)
                    for name, grp in dg
                ]
        except Exception:
            # Don't leak the figure we created if assembling/plotting fails.
            if created_fig:
                plt.close(ax.figure)
            raise

        if save_filename is not None:
            plt.savefig(save_filename)

        if created_fig and save_filename is not None:
            # We created the figure and the caller asked for a file, not a live
            # figure: close it so repeated plotting in one process doesn't
            # accumulate open figures (matplotlib warns past 20).
            plt.close(ax.figure)

        return plots
