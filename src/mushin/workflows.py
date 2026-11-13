# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import copy
import dataclasses
import functools
import json
import math
import uuid
from collections import UserList, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from inspect import getattr_static
from pathlib import Path
from typing import (
    Any,
    Self,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

import numpy as np
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import Quote, QuotedString
from hydra.core.utils import JobReturn, JobStatus
from hydra_zen import hydra_list, launch, load_from_yaml, make_config, multirun, zen
from hydra_zen._launch import _NotSet

from ._console import echo
from ._validate import value_check

LoadedValue: TypeAlias = str | int | float | bool | list[Any] | dict[str, Any]

__all__ = [
    "BaseWorkflow",
    "RobustnessCurve",
    "MultiRunMetricsWorkflow",
]


T = TypeVar("T", list[Any], tuple[Any])
T1 = TypeVar("T1")


# "1.1"-compat is deprecated (Hydra 1.4 removes it); the workflow keeps its
# per-job chdir behavior by setting `hydra.job.chdir=True` explicitly in run().
_VERSION_BASE_DEFAULT = "1.3"


def _mtime(path: Path) -> float:
    """``path``'s modification time, or infinity if it cannot be read (so it
    never counts as older than anything)."""
    try:
        return path.stat().st_mtime
    except OSError:
        return float("inf")


def _sort_x_by_k(x: T, k: Iterable[Any]) -> T:
    k = tuple(k)
    assert len(x) == len(k)
    sorted_, _ = zip(*sorted(zip(x, k, strict=True), key=lambda x: x[1]), strict=True)
    return type(x)(sorted_)


def _identity(x: T1) -> T1:
    return x


def _to_override_element(item: Any) -> str:
    """Serialize a single multirun element to Hydra override-grammar syntax.

    Booleans/numbers are emitted unquoted; strings are single-quoted so that
    values containing commas, spaces, or '=' round-trip exactly through Hydra's
    override parser; lists/tuples are emitted recursively in Hydra's bracketed
    list syntax.
    """
    # Unwrap a NumPy/torch 0-d scalar (e.g. from ``np.arange`` or RobustnessCurve's
    # epsilon array) to its Python scalar first, so a swept number is emitted
    # unquoted rather than falling through to the quoted-string fallback.
    if hasattr(item, "item") and getattr(item, "ndim", None) == 0:
        item = item.item()

    if item is None:
        # Hydra null — the string 'None' would e.g. submit partition="None".
        return "null"
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, (int, float)):
        return str(item)
    if isinstance(item, str):
        return _quoted(item)
    if hasattr(item, "tolist") and getattr(item, "ndim", 0) >= 1:
        # A NumPy array (or a tensor): its elements, not its text (#321).
        return _to_override_element(item.tolist())
    if _is_list_config(item):
        # Its elements as written, an interpolation (`${a}`) unresolved, for
        # the job to resolve: iterating would resolve it against the list.
        from omegaconf import OmegaConf

        item = OmegaConf.to_container(item, resolve=False)
    if isinstance(item, _LIST_TYPES):
        # A list, a tuple, a range, and what main wrote as a list through its
        # repr: a nested `hydra_list`/`multirun` (UserList) or an OmegaConf
        # ListConfig (#321). Named, not `Sequence`: a UserString or a
        # bytearray is one too, and must keep the text it had.
        return "[" + ",".join(_to_override_element(x) for x in item) + "]"
    if isinstance(item, Mapping):
        # Hydra dict grammar, so e.g. a launcher's
        # additional_parameters={"requeue": True} reaches the structured config
        # as a real dict rather than a quoted Python repr that fails validation.
        return (
            "{"
            + ",".join(
                f"{_dict_key(k)}:{_to_override_element(v)}" for k, v in item.items()
            )
            + "}"
        )
    # Fall back to a quoted string representation for any other type.
    return _quoted(str(item))


#: The sequences written as a Hydra list (``_to_override_element``).
_LIST_TYPES = (list, tuple, range, UserList)


def _is_list_config(item: Any) -> bool:
    from omegaconf import ListConfig

    return isinstance(item, ListConfig)


@functools.lru_cache(maxsize=1)
def _override_parser():
    return OverridesParser.create()


def _is_nan(x: Any) -> bool:
    try:
        return math.isnan(x)
    except TypeError:
        return False


@functools.lru_cache(maxsize=4096, typed=True)
def _dict_key_written(key: Any) -> tuple[str | None, str]:
    """``(refusal, text)`` for a dict key written for Hydra's override
    grammar: ``refusal`` is None when the key reads back as itself through
    both readers, else what it would become. Hydra has no quoted key, so a
    key is read back as its text parses (``'1'`` as ``1``; a leading or
    trailing space is stripped, an inner one stays), and one holding ``=`` or
    ``:`` does not parse; OmegaConf's merge takes no ``None`` key. The probe
    must keep its shape (one dict of one key): a key holding ``:0,`` would
    otherwise rewrite it. A str key is written as its value, as ``_quoted``
    writes a str subclass (an Enum mixing in str). Typed, so equal keys of
    different types (``0.5`` and ``Fraction(1, 2)``) are judged apart."""
    from hydra.errors import OverrideParseException
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    text = str.__str__(key) if isinstance(key, str) else _to_override_element(key)
    try:
        value = _override_parser().parse_overrides([f"k={{{text}:0}}"])[0].value()
    except OverrideParseException:
        return "an override Hydra cannot parse", text
    if not isinstance(value, dict) or len(value) != 1:
        return "a different override", text
    (read,) = value
    try:
        OmegaConf.create({read: 0})
    except OmegaConfBaseException:
        return f"{read!r}, a key OmegaConf does not take", text
    # Equal is enough: a str never equals what reads back as a number, a bool
    # or null, and a NumPy integer key reads back as the int it equals; two
    # NaNs are the same key, though never equal.
    if read == key or (_is_nan(read) and _is_nan(key)):
        return None, text
    return repr(read), text


def _dict_key(key: Any) -> str:
    """A dict key as Hydra's override grammar reads it back as itself, or a
    ``ValueError`` before launch naming it (#321): it would reach the task
    changed, or fail inside the launch (``_dict_key_written``)."""
    refusal, text = _dict_key_written(key)
    if refusal is not None:
        raise ValueError(
            f"the dict key {key!r} cannot be written as a Hydra override: Hydra "
            f"has no quoted dict keys, so it would become {refusal}. Use a key "
            "Hydra writes as itself (a name, a number, a bool), or pass the "
            "values as scalar parameters and build the dict in the task."
        )
    return text


#: What ``_swept_strings_read`` gives for a job override that does not parse.
_UNPARSED = object()
#: What ``_job_value`` gives for a value the job's config does not take.
_NOT_TAKEN = object()


def _swept_strings_read(written: list[str]) -> list[tuple[str, Any]]:
    """For each value of a ``choice(...)`` sweep written as a quoted string,
    ``(text, read)``: the string a fixed value of it gives, and what its
    job's override reads back as (``_UNPARSED`` when it does not parse).

    Hydra's ``choice()`` drops a top-level element's quotes, and the basic
    sweeper writes each job's value as that bare text, which the job parses
    again (#319): ``'true'`` reads back as ``True``, ``'1'`` as ``1``,
    ``"''"`` as ``''``, and ``"it's"`` or a non-ASCII string does not parse.
    This asks Hydra's own sweeper and parser rather than a copy of the
    grammar: the job's text is what ``sweep_string_iterator`` gives the basic
    sweeper. ``text`` is the element's text, not the Python one: a ``${``
    string keeps main's escapes in both. One parse for the axis, then one
    per job text not read before."""
    sweep = _override_parser().parse_overrides([f"x=choice({','.join(written)})"])[0]
    return [
        (text, _job_read(job))
        for text, job in zip(
            sweep.value().list, sweep.sweep_string_iterator(), strict=True
        )
    ]


@functools.lru_cache(maxsize=4096)
def _job_read(job: str) -> Any:
    """What a job's override value ``job`` reads back as, or ``_UNPARSED``.
    Cached: what it returns is a str, a number, a bool or None."""
    from hydra.errors import OverrideParseException

    try:
        return _override_parser().parse_overrides([f"x={job}"])[0].value()
    except OverrideParseException:
        return _UNPARSED


#: The sweeper whose way of writing each job's value the check models.
_BASIC_SWEEPER = "hydra._internal.core_plugins.basic_sweeper.BasicSweeper"


def _composed_for_launch(
    cfg: Any,
    overrides: list[str],
    names: list[str],
    *,
    to_dictconfig: bool,
    config_name: str,
    job_name: str,
    version_base: Any,
) -> tuple[Any, set[str], set[str]]:
    """The config Hydra composes for this sweep before its sweeper writes the
    jobs, composed as hydra-zen's ``launch`` composes it; which of ``names``
    are config groups; and the config paths of the groups the sweep sweeps
    (#319).

    The config is where a job's override lands: it holds the fields' types
    (a ``str`` field from a defaults-list group's config included, and none
    under ``to_dictconfig``) and names the sweeper. A config group's
    override names an option, not a value, and a swept group picks each
    job's schema for what lies under it, while this config holds only its
    default's. ``cfg`` is stored under
    ``config_name``, as ``launch`` is about to store it; ``initialize``
    clears Hydra's global state on its way out. A compose that fails here
    fails in ``launch`` too, from the same inputs, so it is not caught."""
    from hydra import initialize
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from hydra.types import RunMode
    from omegaconf import OmegaConf

    if to_dictconfig and dataclasses.is_dataclass(cfg):
        cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.structured(cfg)))
    ConfigStore.instance().store(name=config_name, node=cfg)
    with initialize(
        config_path=None,
        job_name=job_name,
        **({} if version_base is _NotSet else {"version_base": version_base}),
    ):
        hydra = GlobalHydra.instance()
        master = hydra.hydra.compose_config(
            config_name=config_name,
            overrides=list(overrides),
            run_mode=RunMode.MULTIRUN,
            from_shell=False,
            with_log_configuration=False,
        )
        loader = hydra.config_loader()
        groups = {name for name in names if loader.get_group_options(name)}
        swept_groups = {
            o.key_or_group.replace("/", ".")
            for o in _override_parser().parse_overrides(list(overrides))
            if o.is_sweep_override() and loader.get_group_options(o.key_or_group)
        }
    return master, groups, swept_groups


def _job_value(cfg: Any, key: str, value: Any) -> Any:
    """``value`` as the job's config holds it at ``key``, or ``_NOT_TAKEN``.

    The job's config is the second reader (#319): a typed field converts what
    it is given (a ``str`` field turns ``1`` into ``'1'``, an ``int`` field
    ``'1'`` into ``1``), and a key the config lacks takes it as it is.
    Written as Hydra applies an override, into ``cfg``."""
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    try:
        OmegaConf.update(cfg, key, value, merge=True, force_add=True)
        return OmegaConf.select(cfg, key)
    except OmegaConfBaseException:
        return _NOT_TAKEN


class _TargetDirs(multirun):
    """The job directories ``target_job_dirs=`` sweeps: an axis mushin builds
    itself, so the check on swept strings words its refusal for a directory
    (#319)."""


def _swept_strings_refusal(
    misread: list[tuple[str, str, list[str], bool, list[tuple[Any, str, bool]]]],
) -> str:
    """The refusal for swept strings that would not reach the task as
    written (#319). ``misread`` holds, per axis, its name, its override key,
    every value as written, whether ``target_job_dirs=`` built it, and each
    misread value with how it would arrive and whether it would arrive as a
    number, a bool or None.

    It names each axis with its first such values, then the ways out: the
    raw comma sweep that keeps the quotes, written out whole, for an axis of
    two values or more; the type a value reads as, when the task would get
    one; and a fixed value for a one-value axis, which no comma sweep keeps
    swept. A target job directory is told to move or be renamed."""
    shown = 5

    def listed(entries: list[str]) -> str:
        more = len(entries) - shown
        return ", ".join(entries[:shown]) + (f", and {more} more" if more > 0 else "")

    axes = [m for m in misread if not m[3]]
    parts = []
    if axes:
        named = " ".join(
            f"`{name}`: " + listed([f"{item!r} {how}" for item, how, _ in found]) + "."
            for name, _, _, _, found in axes
        )
        parts.append(
            f"these swept strings would not reach the task as written. {named} "
            "multirun() sweeps through Hydra's choice(), which writes each "
            "job's value without its quotes."
        )
        raws = [
            f"{key}=" + ",".join(written)
            for _, key, written, _, _ in axes
            if len(written) > 1
        ]
        if raws:
            parts.append(
                "Sweep the strings through a raw comma sweep instead, whose "
                f"values keep their quotes: add {raws!r} to run()'s overrides= "
                "list in place of the multirun argument."
            )
        if any(typed for *_, found in axes for _, _, typed in found):
            parts.append(
                "A value meant as a number, a bool or null can be passed as that type."
            )
        single = [f"`{name}`" for name, _, written, _, _ in axes if len(written) == 1]
        if single:
            parts.append(
                f"A one-value sweep ({', '.join(single)}) has no such form: pass "
                "the value fixed, which keeps its quotes (it is then not swept)."
            )
    for *_, targets, found in misread:
        if targets:
            parts.extend(
                f"The target job directory {item!r} {how}: Hydra writes each "
                "job's directory into its override without quotes. Move or "
                "rename it so its path holds no character Hydra cannot write "
                "unquoted (such as '#', a quote or a non-ASCII letter)."
                for item, how, _ in found
            )
    return " ".join(parts)


def _hydra_quoted(text: str) -> str:
    """``text`` as Hydra's own writer quotes a string for an override: it
    escapes a quote and the backslashes before one, which is all Hydra's
    grammar unescapes."""
    return QuotedString(text=text, quote=Quote.single).with_quotes()


def _quoted(text: str) -> str:
    """``text`` as a single-quoted Hydra string for an override.

    Hydra's grammar unescapes a backslash only where it precedes a quote, so
    doubling every backslash (as this did) reached the task doubled (#319):
    ``C:\\data`` arrived as ``C:\\\\data``. Hydra's own writer escapes the
    quotes and only the backslashes that precede one.

    A string holding ``${`` is an interpolation, which OmegaConf parses again
    with backslash escapes of its own: a backslash before ``${`` escapes it,
    and inside one a backslash escapes the next character. The doubling undid
    most of those, so a Windows
    ``outputs\\${now:%Y}`` or a ``${oc.env:X,\\\\server\\share}`` default
    resolved as written. Such a string keeps that form, exactly as before; the
    task gets what it always did.

    So in such a string, a backslash that is neither right before the ``${``
    nor inside the interpolation may arrive doubled (#321): write such a
    path with ``/``, or put the value in the config. Writing only where
    OmegaConf unescapes would be a third imitation of its escapes, the
    kind #320 found wrong twice."""
    if "${" in text:
        return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return _hydra_quoted(text)


def _scalar_repr(item: Any) -> Any:
    """A stable, type-aware key for detecting duplicate sweep-axis values.

    Unwraps 0-d numpy/torch scalars (as ``_to_override_element`` does) and
    tags the value with its type so ``1`` and ``"1"`` do not compare equal
    while ``1`` and ``1`` do. Those two still key one cell; the check over the
    parsed grid in ``run`` refuses them.
    """
    if hasattr(item, "item") and getattr(item, "ndim", None) == 0:
        item = item.item()
    # repr keeps the key hashable (list-valued elements) and type-aware.
    return (type(item).__name__, repr(item))


def _unquote(value: Any) -> Any:
    """``value`` with every Hydra ``QuotedString`` in it replaced by its text.

    A sweep keeps the quoted strings it holds: ``choice(['x'])`` gives
    ``[QuotedString('x')]``, and a comma sweep ``'x','y'`` quotes even its
    top-level values. The grid's coordinates come from those values while each
    cell's config holds the plain strings, so no cell matched its grid key
    (#308). ``Override.value()`` does the same conversion for a value that is
    not a sweep; its helper, ``Override._convert_value``, is private, hence
    this copy of it.
    """
    if isinstance(value, QuotedString):
        return value.text
    if isinstance(value, list):
        return [_unquote(v) for v in value]
    if isinstance(value, dict):
        return {_unquote(k): _unquote(v) for k, v in value.items()}
    return value


def _cfg_missing_or_none(cfg, key: str) -> bool:
    """True if ``key`` (possibly a dotted path) is absent from ``cfg`` or None.

    Drives the Hydra ``+`` append prefix. A dotted path like ``model.width``
    must be resolved as a config *path*, not a literal attribute: a plain
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


def _format_dry_run(summary: Mapping[str, Any]) -> str:
    """Render a `dry_run=True` summary as a human-readable preview: the cell
    count, each swept axis with its values (a range typo shows up here as an
    unexpectedly wide axis), and the fixed parameters."""
    axes = summary.get("axes", {})
    fixed = summary.get("fixed", {})
    n = summary.get("num_cells", 0)
    lines = [f"Dry run: {n} cell{'s' if n != 1 else ''}, no jobs launched."]
    wd = summary.get("working_dir")
    lines.append(f"  working_dir: {wd or '(Hydra default)'}")
    if "sample" in summary:
        lines.append(f"  sample: only {summary['sample']} cell(s) would compute")
    if axes:
        shape = " x ".join(str(len(v)) for v in axes.values())
        width = max(len(k) for k in axes)
        lines.append(f"  swept axes ({shape}):")
        for k, v in axes.items():
            lines.append(f"    {k:<{width}} = {list(v)!r}")
    else:
        lines.append("  swept axes: none (a single cell)")
    if fixed:
        width = max(len(k) for k in fixed)
        lines.append("  fixed:")
        for k, v in fixed.items():
            lines.append(f"    {k:<{width}} = {v!r}")
    return "\n".join(lines)


def _required_params(fn) -> set[str]:
    """Named parameters of ``fn`` that have no default: the ones a run must
    supply. Ignores ``*args``/``**kwargs`` and the injected ``mushin_resume``."""
    import inspect

    try:
        params = _task_signature(fn)[0].parameters
    except (TypeError, ValueError):
        return set()
    empty = inspect.Parameter.empty
    skip = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    return {
        name
        for name, p in params.items()
        if p.kind not in skip and p.default is empty and name != "mushin_resume"
    }


def _config_field_names(cfg) -> set[str]:
    """Top-level field names of a config (a ``make_config`` dataclass type, an
    instance, or a ``DictConfig``), or an empty set if it can't be read."""
    try:
        from omegaconf import OmegaConf

        return {str(k) for k in OmegaConf.structured(cfg).keys()}
    except Exception:  # noqa: BLE001 - unreadable cfg -> no fields known
        import dataclasses

        if dataclasses.is_dataclass(cfg):
            return {f.name for f in dataclasses.fields(cfg)}
        return set()


def _override_name_targets(task_fn) -> set[str] | None:
    """The top-level override names a task's signature accepts, or ``None`` to
    allow any name (the task declares ``**kwargs``, or its signature can't be
    read). ``*args`` is ignored; every explicitly-named parameter is a target.

    Drives up-front override validation: a name that is neither a task parameter
    nor already in the base config is almost always a typo (``lrate=`` for an
    ``lr`` parameter), which Hydra would otherwise ``+``-append as a bogus config
    key and mushin would surface as a phantom constant sweep dimension.
    """
    import inspect

    try:
        params = _task_signature(task_fn)[0].parameters
    except (TypeError, ValueError):
        return None
    names: set[str] = set()
    for name, p in params.items():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None  # **kwargs -> any override name is valid
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        names.add(name)
    return names


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


class _DeadJob(_FailedRun):
    """A cell whose launcher job ended without a result: killed out of
    memory, on a failed node, by ``scancel`` or at a time limit before Python
    could act, exited outside Python, or raised outside the task's own error
    handling, as when it cannot import the task (#345).
    Under ``on_error="nan"`` mushin's submitit launchers return it in place of
    the result the job never sent, so ``jobs_post_process`` takes the cell as
    failed (under ``"raise"`` they return the job's error as a FAILED job, and
    a cell ``sample=`` left out is skipped; see
    ``_TaskRunner.record_dead_cell``). ``summary`` is the one line the
    manifest and the failures list keep; the exception carries submitit's
    message, with the job's error stream (its start and end when long),
    which goes to the cell's mushin_error.txt. ``keepable`` says a completed
    record the job's runner wrote can stand for the job's result: the job
    had one task, and gave no result rather than an error it pickled in its
    place (it could not save its result, say)."""

    __slots__ = ("keepable", "summary")

    def __init__(
        self, exception: BaseException, summary: str, keepable: bool = False
    ) -> None:
        super().__init__(exception)
        self.summary = summary
        self.keepable = keepable


class _SkippedRun:
    """Sentinel returned in place of a cell that was intentionally not run: the
    wall-clock budget (``max_total_seconds``) was exhausted, or the cell was not
    in a ``sample=`` subset. Like ``_FailedRun`` it lets the Hydra job complete
    normally, so ``jobs_post_process`` can NaN-fill and record the skipped cell;
    unlike it, no exception occurred: the cell simply never ran."""

    __slots__ = ("reason",)

    def __init__(self, reason: str = "budget") -> None:
        self.reason = reason


class _PriorCells:
    """What a resuming `_TaskRunner` needs from the prior sweep, and nothing
    more: the sweep root, the swept param names, and the completed
    {combo_key: dir} map. It is picklable and small: failed/pending cells and their
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


def _override_texts(overrides) -> list[str]:
    """A sweep's override strings, read back from its YAML (#351).

    Each is read as before, its interpolations resolved, except a fixed
    value whose resolution fails, which is read unresolved, as the file
    holds it. An override such as ``+out='${hydra:runtime.output_dir}'``
    cannot be resolved once the jobs have run: there is no ``HydraConfig``
    for the ``hydra`` resolver to read (in a new process, no such resolver).
    It stopped ``run()`` after every cell had completed, and stopped a
    reload. A sweep's own values (a ``choice`` in its ``multirun.yaml``) key
    the grid's cells, which each job resolved for itself, so one that cannot
    be resolved still raises (#353)."""
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    texts = []
    for i in range(len(overrides)):
        try:
            texts.append(overrides[i])
        except InterpolationResolutionError:
            text = OmegaConf.to_container(overrides, resolve=False)[i]  # type: ignore[index]
            try:
                swept = _override_parser().parse_override(text).is_sweep_override()
            except Exception:  # noqa: BLE001 - one Hydra cannot parse fails as before
                swept = True
            if swept:
                raise
            texts.append(text)
    return texts


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


def _swept_combo(
    cfg,
    names,
    choice_of: Callable[[str], str | None],
    sanitize: Callable[[Any], Any],
) -> dict:
    """The sanitized swept-parameter combination of the job config ``cfg``.

    Values are unwrapped (0-d scalars), converted out of OmegaConf containers,
    and passed through ``sanitize`` (a workflow's
    `_sanitize_coordinate_for_xarray`), so they key identically to the
    (already-sanitized) coordinate values `to_xarray` builds the grid from.
    Swept names may be nested config paths
    (``model.width``). A config-GROUP axis (model=small,big) keys by its
    chosen option name, ``choice_of(name)``, not the composed sub-config:
    inside a job Hydra records the choice in runtime.choices, and a job loaded
    from disk records it in its sidecar or hydra.yaml."""
    from omegaconf import OmegaConf

    combo = {}
    for n in names:
        val = _read_swept_value(cfg, n)
        if OmegaConf.is_config(val):
            choice = choice_of(n)
            if choice is not None:
                val = choice
            else:
                val = OmegaConf.to_container(val, resolve=True)
        combo[n] = sanitize(_unwrap_scalar(val))
    return combo


def _current_job_fields() -> tuple[int | None, dict | None]:
    """The current Hydra job's number and ``runtime.choices``, as its
    hydra.yaml records them, each None when unknown: outside a Hydra job (a
    task called directly), in a single run (whose job has no number), or when
    Hydra records no choices. The number alone still drives ``sample=``; the
    status sidecar records the two only together."""
    try:
        from hydra.core.hydra_config import HydraConfig

        if not HydraConfig.initialized():
            return None, None
        hydra = HydraConfig.get()
        num = hydra.job.num
    except Exception:  # noqa: BLE001 - no job number
        return None, None
    if type(num) is not int:
        return None, None
    from ._hydra_yaml import hydra_choices

    return num, hydra_choices(hydra)


def _current_job_id() -> str | None:
    """The current Hydra job's ``job.id``, or None outside a Hydra job. A
    submitit launcher sets it to the scheduler's job id (an array task's is
    ``<array>_<task>``); other launchers set an id of their own."""
    try:
        from hydra.core.hydra_config import HydraConfig

        if not HydraConfig.initialized():
            return None
        return _job_id_or_none(HydraConfig.get().job.id)
    except Exception:  # noqa: BLE001 - no job id
        return None


# What to `pip install` per Hydra launcher name, for the missing-plugin install
# hint. The submitit launchers come with mushin's `slurm` extra, which also
# holds the plugin's floor (#344); quoted, as a shell needs `[...]` to be.
_LAUNCHER_PLUGINS: dict[str, str] = {
    "joblib": "hydra-joblib-launcher",
    "submitit_slurm": '"mushin-py[slurm]"',
    "submitit_local": '"mushin-py[slurm]"',
    "ray": "hydra-ray-launcher",
    "ray_aws": "hydra-ray-launcher",
    "rq": "hydra-rq-launcher",
}


def _with_launcher_hint(error: Exception, launcher: str | None) -> Exception | None:
    """``error`` amended with the pip install hint when it is Hydra's error
    for a ``launcher`` plugin that is not installed, else None.

    A missing launcher plugin is the FIRST error every cluster user hits;
    Hydra's raw message lists options but never names the pip package. Both
    places Hydra composes the sweep's config raise it: ``launch``, and the
    check on swept strings before it (#319)."""
    from hydra.errors import MissingConfigException

    if not (
        isinstance(error, MissingConfigException)
        and launcher is not None
        and f"hydra/launcher/{launcher}" in str(error)
    ):
        return None
    # mushin's own submitit launchers come with the plugin's extra.
    plain = {v: k for k, v in _MUSHIN_SUBMITIT.items()}.get(launcher, launcher)
    pkg = _LAUNCHER_PLUGINS.get(plain)
    hint = (
        f"pip install {pkg}"
        if pkg
        else f"install the Hydra launcher plugin providing {launcher!r}"
    )
    return MissingConfigException(
        f"{error}\n\nThe {launcher!r} launcher comes from a Hydra "
        f"plugin that is not installed here: `{hint}` and re-run.",
        getattr(error, "missing_cfg_file", None),
    )


def _group_choices_of_job(job) -> dict | None:
    """``runtime.choices`` recorded in a job's hydra config, or None."""
    from ._hydra_yaml import hydra_choices

    try:
        hydra = job.hydra_cfg.hydra
    except Exception:  # noqa: BLE001 - jobs loaded without hydra_cfg
        return None
    return hydra_choices(hydra)


# The submitit plugin's launcher names, and mushin's launchers that run in
# their place (`hydra_plugins/mushin_submitit/launcher.py`, #345).
_MUSHIN_SUBMITIT = {
    "submitit_slurm": "mushin_submitit_slurm",
    "submitit_local": "mushin_submitit_local",
}


def _route_to_mushin_launchers(overrides: list[str]) -> list[str]:
    """``overrides`` with the submitit plugin's launcher choice swapped for
    mushin's launcher of the same kind, which waits on every job and returns a
    job that ended without a result as a failed cell (#345). ``launcher=``
    and a raw ``hydra/launcher=`` override are routed alike. A launcher a
    config's defaults list chooses is not: seeing it would take composing
    the config for every sweep. Unchanged when mushin's launchers are not
    there to run (no `slurm` extra), so Hydra reports the name the user
    wrote, with the install hint."""
    routed = []
    for o in overrides:
        key, sep, name = o.partition("=")
        name = name.strip()  # Hydra reads `hydra/launcher= submitit_slurm` too
        if sep and key.lstrip("+") == "hydra/launcher" and name in _MUSHIN_SUBMITIT:
            if not _mushin_launchers_available():
                return overrides
            o = f"{key}={_MUSHIN_SUBMITIT[name]}"
        routed.append(o)
    return routed


def _record_dead_cell(
    ret: JobReturn,
    dead: _DeadJob,
    *,
    combo: dict | None,
    status: str,
    on_error: str,
    prior: dict,
    run_id: str | None,
) -> None:
    """Record in its cell that the cell's job ended without a result (#345).

    ``ret`` is the job as mushin's submitit launcher composed it in the
    driver, and ``prior`` the cell's record as the runner found it:
    'running', or none (the job died before the runner wrote anything,
    perhaps before Hydra made the cell's dir).

    The cell gets the ``status`` the runner would have recorded: 'skipped'
    for a cell ``sample=`` left out, else 'failed', as for a task that
    raised, with this run's ``run_id`` and the job's id. A cell without
    Hydra's ``.hydra/`` snapshot gets the one ``run_job`` would have written,
    so a reload finds it. A metrics file a prior attempt left is removed, and
    a failed cell under ``on_error="nan"`` gets mushin_error.txt: the
    one-line summary, which names the job's log, then submitit's message
    (under ``"raise"`` the raised error carries submitit's message alone).
    Best-effort throughout: the run records the failure through the job's
    return whatever happens here."""
    from omegaconf import OmegaConf

    from ._resume import write_cell_status
    from ._sweep_io import METRICS_FILE, remove_sidecar

    try:
        cell_dir = _job_dir(ret)
        hydra = ret.hydra_cfg.hydra
        # Written before the status, which must be the newer of the two
        # (`sidecar_job_fields`). As `run_job` saves them, unresolved.
        if hydra.output_subdir is not None:
            snapshot = cell_dir / str(hydra.output_subdir)
            if not (snapshot / "hydra.yaml").exists():
                snapshot.mkdir(parents=True, exist_ok=True)
                for name, node in (
                    ("config.yaml", ret.cfg),
                    ("hydra.yaml", ret.hydra_cfg),
                    ("overrides.yaml", OmegaConf.create(list(ret.overrides))),
                ):
                    (snapshot / name).write_text(
                        OmegaConf.to_yaml(node), encoding="utf-8"
                    )
        # Creates the cell's dir if nothing above did (no `output_subdir`).
        write_cell_status(
            cell_dir,
            status=status,
            # This sweep's combo (None: it could not be computed); a prior
            # record's may be an earlier sweep's.
            combo=prior.get("combo", {}) if combo is None else combo,
            attempt=prior.get("attempt", 1),
            config_hash=prior.get("config_hash"),
            code_hash=prior.get("code_hash"),
            job_num=hydra.job.num,
            choices=_group_choices_of_job(ret),
            run_id=run_id,
            job_id=_job_id_or_none(OmegaConf.select(hydra, "job.id")),
        )
        # After the record: a metrics file that cannot be removed must not keep
        # the cell's end from being recorded.
        remove_sidecar(cell_dir / METRICS_FILE)
        if status == "failed" and on_error == "nan":
            # UTF-8 whatever the locale: the error stream is the job's output.
            (cell_dir / "mushin_error.txt").write_text(
                f"{dead.summary}\n\n{dead.exception}\n",
                encoding="utf-8",
                errors="backslashreplace",
            )
    except Exception:  # noqa: BLE001 - see the docstring: never fails the run
        pass


def _job_id_or_none(value: Any) -> str | None:
    """A Hydra ``job.id`` worth recording: a non-empty string."""
    return value if isinstance(value, str) and value else None


def _mushin_launchers_available() -> bool:
    try:
        from hydra_plugins.mushin_submitit import launcher
    except ImportError:
        return False
    return getattr(launcher, "MushinSlurmLauncher", None) is not None


# Prefix of the file, in the sweep root, through which a parallel launcher's
# workers share a run's `max_total_seconds` start; the run's
# `_TaskRunner.budget_id` completes the name. Hidden: it means nothing once its
# run is over.
_BUDGET_START_PREFIX = ".mushin_budget_start."

# The start record is self-validating: its creator writes `repr(start)` and a
# newline in one write, so a record that ends with the newline is complete.
# The file is named by the run's uuid, so only this run wrote it, and a
# complete record is trusted when it parses as finite and falls inside this
# range: after 2020-01-01 UTC and no more than a day ahead of the reader's
# clock, which only rejects a value no clock could have produced. The file's
# mtime is no check on it: on NFS the mtime comes from the server's clock, and
# the server's skew would become a budget error. (Publishing the record with
# `os.link` from a temp file would close the incomplete-read window instead,
# but hard links fail on FAT and many SMB shares, and an NFS link whose reply
# is lost can report EEXIST to the process that made it.)
_BUDGET_START_FLOOR = 1577836800.0
_BUDGET_START_LEAD_S = 86400.0

# How many times a reader reads an incomplete start record (its creator
# between the create and the write), 2 ms apart, before the mtime stands in.
_BUDGET_START_READS = 5

# This process's monotonic deadline per run (`budget_id`), for the runner
# run() built and for every copy a parallel launcher unpickles: joblib
# unpickles a fresh copy for every job inside the same worker process, so a
# deadline kept on the copy would restart with each job. loky keeps its
# workers across run() calls, so only the `_BUDGET_DEADLINES_KEPT` most recent
# runs are kept, far more than a process runs cells of at once.
_BUDGET_DEADLINES: dict[str, float] = {}
_BUDGET_DEADLINES_KEPT = 64


def _remember_deadline(budget_id: str, deadline: float) -> None:
    """Record this process's deadline for a run, as its most recent entry."""
    _BUDGET_DEADLINES.pop(budget_id, None)
    _BUDGET_DEADLINES[budget_id] = deadline
    while len(_BUDGET_DEADLINES) > _BUDGET_DEADLINES_KEPT:
        del _BUDGET_DEADLINES[next(iter(_BUDGET_DEADLINES))]


def _subdir_depth(hydra) -> int:
    """How many levels below the sweep root `hydra.sweep.subdir` puts a job's
    dir: 1 for Hydra's default (``${hydra.job.num}``), 2 for ``a_${a}/b_${b}``.

    Counted on the resolved subdir when the config resolves it: a running
    job's live config does, while a launched job's detached config cannot
    resolve the job's own keys (``${a}``), so there the unresolved text is
    counted instead. Climbing from a job dir needs no read of
    `hydra.sweep.dir`, whose default holds a ``${now:...}``."""
    try:
        return max(len(Path(str(hydra.sweep.subdir)).parts), 1)
    except Exception:  # noqa: BLE001 - not resolvable here: count the text
        pass
    try:
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(hydra.sweep, resolve=False)["subdir"]
        return max(len(Path(str(raw)).parts), 1)
    except Exception:  # noqa: BLE001 - no subdir at all: Hydra's default
        return 1


def _job_dir(job) -> Path:
    """A launched job's dir: the `working_dir` Hydra reports for it, resolved
    against the original cwd."""
    return Path(job.hydra_cfg.hydra.runtime.cwd) / job.working_dir


def _cell_sweep_root() -> Path:
    """The sweep root of the cell running in this process: its dir, the cwd
    (Hydra chdirs into it, and every sidecar write in the runner relies on
    that), climbed `_subdir_depth` levels. Outside a Hydra job (a runner
    called directly), the cwd's parent."""
    depth = 1
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            depth = _subdir_depth(HydraConfig.get())
    except Exception:  # noqa: BLE001 - no usable job config: the default depth
        pass
    cwd = Path.cwd()
    parents = cwd.parents
    return parents[min(depth, len(parents)) - 1] if parents else cwd


def _read_budget_start(path: Path) -> float | None:
    """The start recorded in the budget start file ``path``, as a
    ``time.time()`` value, or None when the file is gone.

    A complete record (it ends with the newline its creator writes) is
    trusted when it parses as a finite float in the range `_BUDGET_START_FLOOR`
    describes. The creator may be between its create and its write, so an
    incomplete read is retried for a few milliseconds; a file that stays
    incomplete or unparsable stands for its creation time, its mtime, which,
    kept at 1 s resolution, would shorten a sub-second budget.
    """
    import math
    import time

    text = ""
    for attempt in range(_BUDGET_START_READS):
        try:
            text = path.read_text(encoding="ascii")
        except FileNotFoundError:
            return None  # gone: no retry brings it back
        except ValueError:
            text = ""
            break  # undecodable, so not a record: the mtime stands in
        except OSError:
            text = ""
        if text.endswith("\n") or attempt == _BUDGET_START_READS - 1:
            break
        time.sleep(0.002)
    value = math.nan
    if text.endswith("\n"):
        try:
            value = float(text)
        except ValueError:
            pass
    if (
        math.isfinite(value)
        and _BUDGET_START_FLOOR <= value <= time.time() + _BUDGET_START_LEAD_S
    ):
        return value
    mtime = _mtime(path)  # infinity when the file is gone
    return mtime if math.isfinite(mtime) else None


def _remove_budget_start(runner, jobs, working_dir) -> None:
    """Delete the start file of the run that ``runner`` dispatched, and this
    process's deadline for it, once its launch has returned normally.

    Only a runner a launcher pickled has workers that write the file, so the
    sequential launcher's run looks for none. The file sits in the sweep root,
    `_subdir_depth` levels above every job dir; it is looked for by climbing
    from each distinct job-dir parent, at most one level further than that
    depth and never past ``working_dir``, rather than at a derived root, and
    the search stops at the first file found: its name holds the run's uuid,
    so no other file can match. Best-effort."""
    if runner.budget_id is None:
        return
    _BUDGET_DEADLINES.pop(runner.budget_id, None)
    if not runner._shipped:
        return
    import os

    stop = None if working_dir is None else Path(os.getcwd(), working_dir)
    from ._sweep_io import remove_sidecar

    name = f"{_BUDGET_START_PREFIX}{runner.budget_id}"
    seen: set[Path] = set()
    for job in jobs:
        try:
            parent = _job_dir(job).parent
            depth = _subdir_depth(job.hydra_cfg.hydra)
        except Exception:  # noqa: BLE001 - a job without a hydra config
            continue
        if parent in seen:
            continue
        seen.add(parent)
        d = parent
        for _ in range(depth + 1):
            try:
                if (d / name).exists():
                    remove_sidecar(d / name)
                    return
            except OSError:
                pass
            if d == stop or d.parent == d:
                break
            d = d.parent


class _TaskRunner:
    """Picklable per-cell dispatch. Collapses the old closure chain
    (_task_calls / _instrument_task / _fail_soft / _resume_short_circuit) into one
    object so out-of-process launchers (joblib/submitit) can pickle it. Holds only
    picklable state; imports module-level helpers inside __call__ (never captures
    the _CURRENT_RESUME contextvar). Behavior mirrors the previous chain
    exactly; see the spec's Semantics Mapping."""

    # mushin's submitit launchers turn a job that ended without a result into a
    # `_DeadJob` only for a task function that says it handles one (#345); any
    # other Hydra application on those launchers sees the plugin's error.
    _mushin_collects_dead_jobs = True

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
        max_total_seconds=None,
        sampled_indices=None,
        cache_dir=None,
        code_roots=(),
        code_body_roots=(),
        cache_ok=True,
        script_entries=None,
    ):
        self.task = task  # zen(_ResumeInjector(fn)) or zen(fn)
        self.pre_task = pre_task  # zen(self.pre_task)
        self.swept_names = tuple(swept_names)
        self.base_provenance = base_provenance
        self.on_error = on_error
        self.inject_resume = inject_resume
        self.prior_manifest = prior_manifest
        self.code_hash = code_hash  # hash of the task source (resume guard)
        # The roots of the project-code record (the task, Study's training
        # functions, pre_task): the project files their modules import. The
        # record is taken in each cell's own process (cell_project_record),
        # since a parallel worker imports the code it runs from disk, and the
        # code the driver loaded may not be what ran. The token keys the
        # per-process cache of the roots' record and is shared by pickled
        # copies.
        self.code_roots = tuple(code_roots)
        self.code_body_roots = tuple(code_body_roots)
        self._code_token = uuid.uuid4().hex
        # False for a partial task: never cached, as on main, since a key
        # cannot see into the data it binds (see partial_task_cacheable).
        self.cache_ok = cache_ok
        # The driver script's entries, taken by the driver when the run
        # started (driver_script_entries): a worker runs the driver's pickled
        # script functions, so this, not the file as it is when the worker
        # starts, is what its cells run.
        self.script_entries = dict(script_entries or {})
        # Grid subsampling: the set of Hydra job numbers to actually run (None =
        # run every cell). A cell whose job.num is not selected is skipped
        # (NaN), keyed on the deterministic job index so it's launcher- and
        # axis-type-agnostic and identical across a resume.
        self.sampled_indices = sampled_indices
        # Content-addressed cross-working_dir cache: a directory keyed on
        # (config fingerprint, code fingerprint, project-code fingerprints). A
        # cell whose config, task source AND the project code the task reaches
        # match a cached entry reuses it instead of recomputing.
        self.cache_dir = cache_dir
        # Wall-clock budget: one deadline for the whole sweep, however many
        # processes run its cells. It starts at the first COMPUTED cell in any
        # process (`_start_budget`), so launch/import overhead and resume cache
        # hits don't consume it. Each process keeps its deadline for the run in
        # `_BUDGET_DEADLINES`, keyed by `budget_id`. The runner run() built,
        # which the default sequential launcher calls directly, times the
        # budget there on its own and writes no file. A parallel launcher
        # pickles the runner to its workers (joblib once per job), so the
        # copies share the start through a file in the sweep root that the
        # first computing cell creates; `budget_id` names it. Every copy
        # carries the same id, so they share one start; the next run() (a
        # resume) builds a new runner with a new id, so a start that an
        # earlier run left behind is never read.
        self.max_total_seconds = max_total_seconds
        self.budget_id = uuid.uuid4().hex if max_total_seconds is not None else None
        # Names this run in every status record its cells write, and is the
        # same in every pickled copy: a job that ended without a result is
        # judged by whether its cell's record is this run's (#345).
        self.run_id = uuid.uuid4().hex
        # True only in an unpickled copy (`__setstate__`).
        self._shared_budget = False
        # True once a launcher has pickled this runner (`__getstate__`): only
        # then can a worker have written a start file for run() to remove.
        self._shipped = False
        self._budget_rank_warned = False

    def __getstate__(self):
        self._shipped = True
        return self.__dict__.copy()

    def __setstate__(self, state):
        # Being unpickled means a launcher shipped this runner to a worker, so
        # the deadline comes from the sweep's shared start. A runner pickled by
        # an earlier mushin (a submitit array queued before an upgrade) has no
        # `budget_id` to share a start by, and an old per-copy `_deadline`
        # that means nothing here: it gets an id of its own and times a budget
        # of its own, as that version did. The budget fields are normalized
        # here, once, so every later read can trust them.
        self.__dict__.update(state)
        self.__dict__.pop("_deadline", None)
        self._shared_budget = "budget_id" in state
        if not self._shared_budget:
            self.budget_id = (
                uuid.uuid4().hex if state.get("max_total_seconds") is not None else None
            )
        self.__dict__.setdefault("_shipped", False)
        # Such a runner has no code roots either: it records no project code
        # and its cells are compared as that version compared them.
        self.__dict__.setdefault("code_roots", ())
        self.__dict__.setdefault("code_body_roots", ())
        self.__dict__.setdefault("_code_token", "")
        self.__dict__.setdefault("cache_ok", True)
        self.__dict__.setdefault("script_entries", {})
        # Such a runner names no run: its records never count as this run's.
        self.__dict__.setdefault("run_id", None)

    def budget_start_path(self, root) -> Path:
        """The file in sweep root ``root`` that holds this run's budget start."""
        return Path(root) / f"{_BUDGET_START_PREFIX}{self.budget_id}"

    def _known_deadline(self) -> float | None:
        """This process's monotonic deadline for the run, or None before
        `_start_budget` has fixed it."""
        return _BUDGET_DEADLINES.get(self.budget_id)

    def _start_budget(self) -> bool:
        """Fix this process's deadline for the cell about to compute: the one
        place the clock starts. Returns True when this cell fixed it, so it
        runs whatever the clock says: it started the sweep's clock, or, when no
        start can be shared, this process's own. A no-op returning False once
        the deadline is known.

        The runner run() built times the budget from this cell on the
        monotonic clock, so a wall-clock step (NTP) mid-sweep cannot move it.
        An unpickled copy takes the sweep's start from the start file in the
        sweep root, creating the file, and so starting the clock, if no cell
        has, and turns that start into a deadline on its own process's
        monotonic clock. When no start can be shared, the process times the
        budget from this cell, with a warning (`_budget_fallback`).
        """
        import time

        if self._known_deadline() is not None:
            return False
        if not self._shared_budget:
            _remember_deadline(
                self.budget_id, time.monotonic() + self.max_total_seconds
            )
            return True
        shared = self._shared_budget_start(_cell_sweep_root())
        if shared is None:
            return True  # `_budget_fallback` started this process's clock here
        start, created = shared
        # Both clocks are read after the file I/O (the read's retries, a slow
        # NFS stat), so its time is not taken off the budget.
        wall, now = time.time(), time.monotonic()
        _remember_deadline(self.budget_id, now + start + self.max_total_seconds - wall)
        return created

    def _budget_expired(self) -> bool:
        """Whether this process's deadline for the run has passed. Pure: False
        before `_start_budget` has fixed a deadline, so a stray call cannot
        start the clock."""
        import time

        deadline = self._known_deadline()
        return deadline is not None and time.monotonic() >= deadline

    def _budget_fallback(self, reason: str, *, stacklevel: int) -> None:
        """Time the budget from this process's first cell when the sweep's
        start cannot be shared, and warn that it may not hold across
        processes. ``stacklevel`` is the one `warnings.warn` would take at the
        call site, counted so the warning names the caller of `__call__`.
        """
        import time
        import warnings

        # The deadline first, so a warning that raises cannot leave the
        # process without one.
        if self._known_deadline() is None:
            _remember_deadline(
                self.budget_id, time.monotonic() + self.max_total_seconds
            )
        # Raised in the worker process that runs this cell, so it reaches that
        # process's stderr (the console under joblib, the job's log under
        # submitit), never the warning filters of the process that called
        # run(); the deadline above makes it once per process and run. A
        # filter that turns it into an error (PYTHONWARNINGS=error, which the
        # workers inherit) must not fail the cell over a degraded budget, so
        # the error is dropped.
        try:
            warnings.warn(
                f"`max_total_seconds`: {reason}; this worker process times the "
                "budget from its own first cell, so the budget may not hold "
                "across worker processes.",
                UserWarning,
                stacklevel=stacklevel + 1,
            )
        except Warning:
            pass

    def _shared_budget_start(self, root: Path) -> tuple[float, bool] | None:
        """``(start, created)``: the sweep's budget start as a ``time.time()``
        value, and whether this call started the clock by creating the start
        file (the first cell to compute in any process). None when no start
        can be shared: `_budget_fallback` has then fixed this process's own.

        ``os.O_EXCL`` makes the creation first-writer-wins across processes
        (and across nodes on a shared filesystem, as submitit needs anyway).
        The start is a ``time.time()`` value because ``time.monotonic`` is per
        process; the wall clock is the only clock that worker processes, and
        nodes, share.
        """
        import os
        import time

        path = self.budget_start_path(root)
        start = time.time()
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            read = _read_budget_start(path)  # another cell started the clock
            if read is None:
                # warn <- here <- _start_budget <- __call__ <- its caller
                self._budget_fallback(
                    f"the budget start file {str(path)!r} vanished before it "
                    "could be read",
                    stacklevel=4,
                )
                return None
            return read, False
        except OSError as exc:
            self._budget_fallback(
                f"could not create the budget start file {str(path)!r} ({exc})",
                stacklevel=4,
            )
            return None
        # One write of the whole record: its newline tells a reader it is
        # complete (`_read_budget_start`). A failed write or close leaves an
        # incomplete record, and readers fall back to the file's mtime.
        try:
            os.write(fd, f"{start!r}\n".encode("ascii"))
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        return start, True

    def _began_under_this_budget(self, cell_dir) -> bool:
        """Whether this cell already started under this run's budget: its own
        status sidecar says 'running' with this run's `budget_id`. A cell that
        a scheduler preempted and requeued (SLURM `requeue`) enters again in a
        new process, possibly after the deadline; it started before the
        deadline, and a running cell is never cut short, so it runs on."""
        from ._resume import read_cell_status

        status = read_cell_status(cell_dir) or {}
        return (
            status.get("status") == "running"
            and status.get("budget_id") == self.budget_id
        )

    def _cache_key(self, config_hash, deps) -> str | None:
        """The content-address for this cell: a hash of its resolved-config and
        task-source fingerprints and of the project code the cell reaches
        (``deps``, its project-code record), or None if the first two are
        unavailable (so an unresolvable cell is simply never cached). A cell
        that reaches no project code keeps the two-part key, so its entries
        from before the third part are still found."""
        if (
            not self.cache_dir
            or not self.cache_ok
            or config_hash is None
            or self.code_hash is None
        ):
            return None
        if deps is None:
            if self.code_roots:
                # The project code could not be fingerprinted: no key would
                # tell this code era from another, so the cache is not used.
                return None
            # A runner pickled by an earlier mushin walks nothing: its key is
            # the two-part one that version used.
            deps = {}
        import hashlib

        payload = f"{config_hash}:{self.code_hash}"
        if deps.get("code"):
            # The code only: its digests read alike on every Python version
            # (for one exception, see `changed_project_code`).
            payload += ":" + json.dumps(deps["code"], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    @staticmethod
    def _multi_rank_world() -> bool:
        """True when this process is one rank of an EXTERNAL multi-rank launch
        (SLURM/submitit set SLURM_NTASKS; torchrun sets WORLD_SIZE + RANK,
        both before the process starts). Requires a per-rank marker alongside
        WORLD_SIZE so mushin's own single-node launcher (which exports
        WORLD_SIZE in rank 0 but never RANK/SLURM_PROCID) is not mistaken for
        an external rank."""
        import os

        def _gt1(name: str) -> bool:
            v = os.environ.get(name)
            try:
                return v is not None and int(v) > 1
            except ValueError:
                return False

        if _gt1("SLURM_NTASKS") and "SLURM_PROCID" in os.environ:
            # SLURM_NTASKS is allocation-wide; SLURM_PROCID is set per task by
            # srun — a sequential driver inside an salloc shell has no PROCID
            # and is NOT a rank.
            return True
        return _gt1("WORLD_SIZE") and (
            "RANK" in os.environ or "SLURM_PROCID" in os.environ
        )

    @staticmethod
    def _combo(cfg, names):
        return _swept_combo(
            cfg,
            names,
            _current_group_choice,
            MultiRunMetricsWorkflow._sanitize_coordinate_for_xarray,
        )

    def record_dead_cell(self, ret: JobReturn, dead: _DeadJob) -> tuple[Any, Any]:
        """Record a cell whose job ended without a result (#345), and return
        the status and value of the JobReturn mushin's submitit launcher
        sends for it. Called in the driver, with HydraConfig set to the
        job's config, from which the combo reads config-group choices.

        The cell's record says how far the job got. It is the job's own when
        this run wrote it for this cell (its ``run_id`` and combo). A
        completed one whose attempt wrote a metrics file holding the result
        whole (its ``metrics``, that file's digest) gives the cell's result
        from that file, while the file is still that one, as a resume would, when the job had one task and did not pickle an error
        in place of its result (`_DeadJob.keepable`): it finished, but its
        result did not reach the driver, as when it came after submitit
        stopped waiting. A skipped one is skipped. Any other terminal record
        of the job's own is kept as it is, and the cell fails.

        Otherwise, over a 'running' record or none, the job died before its
        runner recorded an end, and `_record_dead_cell` records it. Any
        other terminal record (an earlier run's, which a resume may reuse,
        or another cell's in a reused dir, #355) is left as it is, and the
        cell fails for this run.

        A cell ``sample=`` left out is skipped, as this runner skips it. A
        failed cell is, under ``on_error="raise"``, a FAILED return with the
        job's error, as ``run_job`` returns a task that raised (so the
        sweeper raises it after the batch); under ``"nan"`` a COMPLETED
        return with the `_DeadJob`, which ``jobs_post_process`` takes as a
        failed cell (a FAILED return is raised whatever ``on_error`` says)."""
        from ._resume import read_cell_status
        from ._sweep_io import read_metrics_sidecar_written

        num = ret.hydra_cfg.hydra.job.num
        sampled = getattr(self, "sampled_indices", None)
        skipped = sampled is not None and num is not None and num not in sampled
        try:
            combo = self._combo(ret.cfg, self.swept_names)
        except Exception:  # noqa: BLE001 - the combo itself may not compute
            combo = None
        try:
            cell_dir = _job_dir(ret)
            prior = read_cell_status(cell_dir) or {}
        except Exception:  # noqa: BLE001 - no record to read
            cell_dir, prior = None, {}
        status = prior.get("status")
        this_cell = combo is not None and prior.get("combo") == combo
        own = (
            this_cell and self.run_id is not None and prior.get("run_id") == self.run_id
        )
        written = prior.get("metrics")
        if own and status == "completed" and dead.keepable and isinstance(written, str):
            # Only the file this record names: another job in a shared dir
            # may have written its own since.
            metrics = read_metrics_sidecar_written(cell_dir, written)
            if metrics is not None:
                return JobStatus.COMPLETED, metrics
        if own and status == "skipped":
            return JobStatus.COMPLETED, _SkippedRun("recorded by its job")
        if status in (None, "running"):
            _record_dead_cell(
                ret,
                dead,
                combo=combo,
                status="skipped" if skipped else "failed",
                on_error=self.on_error,
                prior=prior,
                run_id=self.run_id,
            )
        if skipped:
            return JobStatus.COMPLETED, _SkippedRun("not sampled")
        if self.on_error == "raise":
            return JobStatus.FAILED, dead.exception
        return JobStatus.COMPLETED, dead

    def __call__(self, cfg):
        import warnings
        from pathlib import Path

        from ._provenance import NOT_WRITTEN, record_accelerator, write_provenance
        from ._resume import (
            _CURRENT_RESUME,
            build_resume_context,
            cell_project_record,
            changed_project_code,
            discover_last_ckpt,
            merge_project_records,
            read_cell_status,
            resolved_config,
            resolved_config_hash,
            write_cell_status,
        )
        from ._sweep_io import (
            METRICS_FILE,
            combo_key,
            read_metrics_sidecar,
            remove_sidecar,
            write_metrics_sidecar,
        )

        # The config is resolved once, for its fingerprint and for the
        # `_target_`s the project-code record follows.
        resolved = resolved_config(cfg)
        chash = resolved_config_hash(resolved)
        # Hydra's job number and config-group choices go into every status
        # write below, so a reload of this sweep sorts and keys its cells from
        # mushin's own sidecar instead of parsing hydra.yaml (`load_from_dir`).
        # The project-code record goes only into the 'completed' writes (each
        # passes it): a resume reads it from completed cells alone, and at a
        # project's file cap it is most of a sidecar's size.
        job_num, choices = _current_job_fields()
        write_status = functools.partial(
            write_cell_status,
            config_hash=chash,
            code_hash=self.code_hash,
            job_num=job_num,
            choices=choices,
            run_id=self.run_id,
            job_id=_current_job_id(),
        )

        # (0) sampling skip — BEFORE resume/budget/compute. A cell whose Hydra
        # job number is not in the selected sample never runs (NaN); keying on
        # the job index (not the combo) makes the selection launcher- and
        # axis-type-agnostic and identical across a resume. An un-sampled cell is
        # recorded 'skipped', so a later resume WITHOUT `sample` fills it in.
        # No job number -> don't skip.
        if self.sampled_indices is not None:
            if job_num is not None and job_num not in self.sampled_indices:
                cwd = Path.cwd()
                combo = self._combo(cfg, self.swept_names)
                write_status(cwd, status="skipped", combo=combo, attempt=1)
                # A skipped cell has no valid metrics; drop any sidecar a prior
                # sweep left in this reused dir so show/export don't list old
                # metrics on a 'skipped' row while to_xarray NaN-fills it.
                remove_sidecar(cwd / METRICS_FILE)
                return _SkippedRun("not sampled")

        # This cell's project-code record, taken here, in the process that will
        # run (or reuse) it (see cell_project_record), and only by a cell that
        # can compare or store it: an unsampled cell above never does.
        deps = cell_project_record(
            self._code_token,
            self.code_roots,
            self.code_body_roots,
            resolved,
            scripts=self.script_entries,
        )

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
                prior_deps = prior.get("code_deps")
                # A completed cell is reused only if the resolved config, the
                # task source AND the project code the task reaches all still
                # match. A missing prior hash (a pre-fingerprint sweep) can't be
                # checked, so it isn't a reason to re-run — legacy sweeps still
                # resume.
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
                # Judged on the files both records have, on any Python version
                # (for one exception, see its docstring);
                # None when they cannot be compared (a sidecar from before the
                # field, an unknown record).
                changed_deps = changed_project_code(prior_deps, deps)
                deps_changed = bool(changed_deps)
                if config_changed or code_changed or deps_changed:
                    parts = []
                    if config_changed:
                        parts.append("config")
                    if code_changed:
                        parts.append("task code")
                    if deps_changed:
                        # Name where the change is, so the user sees which
                        # helper or constant the edit was in.
                        parts.append(
                            f"code the task reaches ({', '.join(changed_deps)})"
                        )
                    what = " and ".join(parts)
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
                        written = write_metrics_sidecar(cwd, cached)
                        # Keep the reused cell's prior record (see
                        # merge_project_records): writing this run's keys would
                        # declare it checked against code it was not compared
                        # with, and dropping the record would let a later resume
                        # reuse the cell unchecked.
                        write_status(
                            cwd,
                            status="completed",
                            combo=sc,
                            attempt=1,
                            code_deps=merge_project_records(prior_deps, deps),
                            metrics=written,
                        )
                        return cached

        # (1.4) content-addressed cache: reuse a cell computed under the same
        # resolved config, task source AND project code in ANY working_dir. Keyed on the
        # fingerprints (not the combo/dir), so it survives a fresh working_dir.
        # The cache hit refreshes this job dir (metrics + completed status) so
        # the manifest and an offline load see a normal completed cell.
        cache_key = self._cache_key(chash, deps)
        if cache_key is not None:
            cache_cell = Path(self.cache_dir) / cache_key
            cached = read_metrics_sidecar(cache_cell)
            if cached is not None:
                cwd = Path.cwd()
                combo = self._combo(cfg, self.swept_names)
                written = write_metrics_sidecar(cwd, cached)
                write_status(
                    cwd,
                    status="completed",
                    combo=combo,
                    attempt=1,
                    code_deps=deps,
                    metrics=written,
                )
                return cached

        # (1.5) wall-clock budget: once the deadline passes, skip the remaining
        # cells (no compute) so a long sweep ends gracefully instead of running
        # to completion. The clock starts at the sweep's first computed cell, in
        # whichever process runs it (cache hits above returned before here), so
        # at least one cell always runs. A skipped cell is 'skipped', not
        # 'completed', so a later resume with more time finishes it. A cell
        # already running is never interrupted.
        if self.max_total_seconds is not None and self._multi_rank_world():
            # Under an external multi-rank launch (submitit DDP/FSDP: every
            # rank runs this task and checks the deadline at its own moment),
            # a budget that expires on one rank but not its siblings would
            # leave the training ranks hanging at NCCL rendezvous. Disable the
            # budget rather than risk the hang; the scheduler's own time limit
            # still bounds the job.
            if not self._budget_rank_warned:
                warnings.warn(
                    "`max_total_seconds` is disabled for this cell: it runs "
                    "under a multi-rank launch (WORLD_SIZE/SLURM_NTASKS > 1), "
                    "where ranks checking the deadline at different moments "
                    "could diverge and hang DDP at rendezvous. Use the "
                    "scheduler's time limit (e.g. submitit's `timeout_min`) "
                    "to bound multi-rank jobs.",
                    UserWarning,
                    stacklevel=2,
                )
                self._budget_rank_warned = True
        elif self.max_total_seconds is not None:
            # Start the clock first (a no-op once this process knows the
            # deadline): the cell that starts it always runs. A cell that a
            # scheduler requeued after the deadline started before it, so it
            # runs on (its 'running' record below carries this run's id).
            try:
                skip = (
                    not self._start_budget()
                    and self._budget_expired()
                    and not self._began_under_this_budget(Path.cwd())
                )
            except Exception as exc:  # noqa: BLE001 - the budget never fails a cell
                # This step runs before the fail-soft guard below, so an
                # unexpected error here would escape on_error="nan": time the
                # budget per process instead, and run this cell.
                # warn <- here (__call__) <- its caller
                self._budget_fallback(
                    f"the shared budget failed ({exc!r})", stacklevel=2
                )
                skip = False
            if skip:
                cwd = Path.cwd()
                combo = self._combo(cfg, self.swept_names)
                write_status(cwd, status="skipped", combo=combo, attempt=1)
                # Same reused-dir hygiene as the sampling skip above.
                remove_sidecar(cwd / METRICS_FILE)
                return _SkippedRun()

        # (2) fail-soft wraps everything below, only when on_error == "nan"
        failed_recorded = False
        try:
            cwd = Path.cwd()
            combo = self._combo(cfg, self.swept_names)
            # The prior sidecar (attempt, is_resume) is read before the write
            # below replaces it.
            rc = build_resume_context(cwd, combo)
            # 'running' is written before the hooks that take time (pre_task,
            # provenance capture), so a job killed in them leaves its own record
            # in the dir, not a prior sweep's: a reload would take that one's
            # 'completed' status and metrics for this cell's, and its job
            # number and choices (`load_from_dir` checks their age against
            # hydra.yaml, but a record of this job is what makes it certain).
            # It names this run's budget, so a requeued cell started under it
            # is not skipped (`_began_under_this_budget`).
            write_status(
                cwd,
                status="running",
                combo=combo,
                attempt=rc.attempt,
                budget_id=self.budget_id,
            )
            self.pre_task(cfg)
            if rc.is_resume:
                # pre_task may stage a checkpoint into the cell dir (the
                # resilience guide's pattern), so the checkpoint is looked for
                # after it, as it was when the hook ran first.
                rc = dataclasses.replace(rc, last_ckpt=discover_last_ckpt(cwd))
            written = NOT_WRITTEN
            try:
                written = write_provenance(cwd, cfg, base=self.base_provenance)
            except Exception:  # noqa: BLE001 - provenance is best-effort
                pass
            # Set the contextvar AFTER the (fallible) status write, so a failed
            # write can't leave the token set with no matching reset -> a stale
            # ResumeContext leaking into later in-process cells.
            token = _CURRENT_RESUME.set(rc) if self.inject_resume else None
            try:
                result = self.task(cfg)
            except Exception:  # noqa: BLE001 - durable failed status, then re-raise
                try:
                    write_status(cwd, status="failed", combo=combo, attempt=rc.attempt)
                except OSError:
                    pass  # best-effort: never mask the task's own failure
                # A metrics sidecar left by a prior successful attempt no longer
                # describes this cell — remove it so offline tools (show/best/
                # diff/load) cannot present stale metrics as current results.
                remove_sidecar(cwd / METRICS_FILE)
                failed_recorded = True
                raise
            finally:
                if token is not None:
                    _CURRENT_RESUME.reset(token)
                # The accelerator comes from this process, after the task that
                # may have imported torch or initialized CUDA has returned or
                # raised; the record written above carries this process's view
                # from before the task.
                record_accelerator(cwd, written)
            written = None
            if isinstance(result, dict):
                written = write_metrics_sidecar(cwd, result)
                # Store the freshly-computed cell in the content-addressed cache
                # so a future sweep with the same config+code reuses it. Keyed on
                # the fingerprints; best-effort (a cache-write failure must never
                # fail the cell itself).
                cache_key = self._cache_key(chash, deps)
                if cache_key is not None:
                    try:
                        cache_cell = Path(self.cache_dir) / cache_key
                        cache_cell.mkdir(parents=True, exist_ok=True)
                        write_metrics_sidecar(cache_cell, result)
                    except OSError:
                        pass
            else:
                # No metrics: a file an earlier sweep or attempt left in this
                # reused dir is not this cell's result (#355).
                remove_sidecar(cwd / METRICS_FILE)
            write_status(
                cwd,
                status="completed",
                combo=combo,
                attempt=rc.attempt,
                code_deps=deps,
                # The digest of this attempt's metrics file, when it holds the
                # result whole: a launcher may take the file for the job's
                # result if the job's own return is lost, while it is still
                # this one (#345).
                metrics=written,
            )
            return result
        except Exception as exc:  # noqa: BLE001 - fail-soft sentinel or re-raise
            if not failed_recorded:
                # A SETUP-phase failure (combo extraction, the 'running' status
                # write, pre_task) never reached the task-body handler above, so
                # the cell dir carries this job's 'running' status, or still a
                # prior sweep's 'completed' status + metrics sidecar. Record the
                # failure and remove the stale sidecar so offline tools cannot
                # present the old metrics as this sweep's results. Best-effort:
                # the combo itself may be what failed to compute.
                cwd = Path.cwd()
                try:
                    write_status(
                        cwd,
                        status="failed",
                        combo=self._combo(cfg, self.swept_names),
                        attempt=1,
                    )
                except Exception:  # noqa: BLE001 - never mask the real failure
                    pass
                # Apart from the status write: a metrics file that cannot be
                # removed must not keep the failure from being recorded.
                remove_sidecar(cwd / METRICS_FILE)
            if self.on_error == "nan":
                # The sentinel keeps only repr(exc); persist the full stack in
                # the cell dir so a fail-soft sweep stays debuggable. (Hydra's
                # job cwd IS the cell dir here.)
                import traceback

                try:
                    # UTF-8, not the locale's encoding: under cp1252 a message
                    # with a character the code page lacks raised
                    # UnicodeEncodeError, which no OSError handler catches, and
                    # ended the fail-soft sweep. backslashreplace takes the one
                    # thing UTF-8 cannot encode, a lone surrogate.
                    Path("mushin_error.txt").write_text(
                        traceback.format_exc(),
                        encoding="utf-8",
                        errors="backslashreplace",
                    )
                except OSError:
                    pass  # a full disk must not break fail-soft itself
                return _FailedRun(exc)
            raise


def _task_signature(fn):
    """``(signature, as_text)``: ``inspect.signature(fn)``, or on Python 3.14+,
    when evaluating ``fn``'s annotations raises (a name imported only under
    ``TYPE_CHECKING``, which PEP 649 lets a definition name, raises
    ``NameError``; a submodule imported so, ``AttributeError``; a string
    stand-in in ``X | None``, ``TypeError``; #330), its signature with every
    annotation kept as its text, which never evaluates them. Text, not
    ``ForwardRef``: hydra-zen's ``zen`` keeps the parameters, and a
    ``ForwardRef`` pickles its module's globals, which a parallel run cannot
    send when they hold a module. When the text cannot be read either, the
    evaluation's own error is raised (with the text read's as its context);
    an error that is no annotation's (no signature at all) is raised as it
    was, since the text read fails the same way."""
    import inspect
    import sys

    try:
        return inspect.signature(fn), False
    except Exception as error:
        if sys.version_info < (3, 14):
            raise
        import annotationlib

        try:
            text = annotationlib.Format.STRING
            return inspect.signature(fn, annotation_format=text), True
        except Exception:  # noqa: BLE001 - the evaluation's own error is the one to see
            raise error  # noqa: B904 - its context is the text read's failure


class _SignedTask:
    """A callable for hydra-zen's ``zen`` that reads its task's signature
    without evaluating its annotations (``_task_signature``): a task whose
    annotations name what is not defined when the run starts (3.14+, #330).
    It carries the task's names and docstring for a custom wrapper and for
    zen's messages, copied one by one: ``functools.update_wrapper`` would copy
    3.14's ``__annotate__`` too, and set ``__wrapped__``. Picklable: it holds
    the module-level task and a ``Signature`` whose annotations are text."""

    def __init__(self, task, sig):
        import types

        self._task = task
        self._sig = sig
        # A function's own; another callable's would be its class's (a
        # partial's module is `functools`), so `__repr__` names it instead.
        if type(task) is types.FunctionType:
            for name in ("__name__", "__qualname__", "__module__", "__doc__"):
                setattr(self, name, getattr(task, name))

    @property
    def __signature__(self):  # inspect.signature / zen read this
        return self._sig

    def __repr__(self) -> str:  # zen's messages name the task
        return repr(self._task)

    def __call__(self, /, *args, **kwargs):  # a task may take a `self`
        return self._task(*args, **kwargs)


class _ResumeInjector(_SignedTask):
    """Picklable replacement for the old `_bind_resume_kwarg` closure. Hides a
    task's `mushin_resume` parameter from hydra-zen's `zen` (via a stripped
    __signature__, so zen never tries to resolve it from config) and injects the
    current cell's ResumeContext from a contextvar at call time. Unlike a closure,
    an instance holding a module-level task + a Signature is stdlib-picklable."""

    def __init__(self, task, sig):
        super().__init__(task, sig)
        self._sig = self._sig.replace(
            parameters=[
                p for n, p in self._sig.parameters.items() if n != "mushin_resume"
            ]
        )

    def __call__(self, /, *args, **kwargs):
        from ._resume import current_resume

        return self._task(*args, **kwargs, mushin_resume=current_resume())


def _prepare_task(task):
    """If `task` declares a `mushin_resume` parameter, wrap it in a picklable
    `_ResumeInjector`; one whose annotations cannot be evaluated yet in a
    `_SignedTask` (`_signed`); otherwise return it unchanged. Returns
    `(prepared, wants)`."""
    sig, as_text = _task_signature(task)
    if "mushin_resume" in sig.parameters:
        return _ResumeInjector(task, sig), True
    return _signed(task, (sig, as_text)), False


def _check_sent_by_value(fns) -> None:
    """Raise before launch when a parallel launcher could not send one of
    ``fns`` (#330). On 3.14+, a function whose annotations do not evaluate
    (``_task_signature``) is sent by value when the launcher cannot import
    it by name (a script's or a notebook's), and cloudpickle, which joblib
    and submitit use, reads its annotations to send it, which raises inside
    the launch. Whether it is sent by value is cloudpickle's to say, so it is
    asked: submitit's, or the copy joblib bundles; without either there is
    nothing to ask."""
    import sys

    if sys.version_info < (3, 14):
        return
    try:
        import cloudpickle
    except ImportError:
        try:
            from joblib.externals import cloudpickle
        except ImportError:
            return
    for fn in fns:
        try:
            as_text = _task_signature(fn)[1]
        except (TypeError, ValueError):
            continue
        if not as_text:
            continue
        try:
            cloudpickle.dumps(fn)
        except Exception as error:
            name = getattr(fn, "__qualname__", None) or repr(fn)
            raise ValueError(
                f"a parallel launcher cannot send {name!r}: its annotations name "
                "something that is not defined when it runs (an import made only "
                "under TYPE_CHECKING), and a function the launcher cannot import "
                "by name (one defined in a script or a notebook) is sent with its "
                "annotations evaluated. Define it in an importable module, add "
                "`from __future__ import annotations` to its file, or run without "
                "a parallel launcher."
            ) from error


def _signed(fn, read=None):
    """``fn``, or a `_SignedTask` of it when its annotations cannot be
    evaluated yet (3.14+, #330), for hydra-zen's ``zen``. ``read`` is
    ``_task_signature(fn)`` when the caller has it."""
    if read is None:
        try:
            read = _task_signature(fn)
        except (TypeError, ValueError):  # no signature to read: zen says so
            return fn
    sig, as_text = read
    return _SignedTask(fn, sig) if as_text else fn


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

    # uv writes UTF-8, so its output is decoded as such ("replace" for any
    # byte that is not) and the snapshot is written in UTF-8. Earlier, it was
    # decoded and re-encoded in the locale's codec: where that decode worked
    # the file kept uv's bytes, but where it raised (any non-ASCII byte under
    # an ASCII locale; under cp1252 a UTF-8 byte the code page leaves
    # undefined, 0x81, 0x8D, 0x8F, 0x90 or 0x9D, as in "A" or "I" with an
    # acute accent and some CJK characters) the snapshot silently fell back
    # to the bare package list below.
    for cmd in (["uv", "export"], ["uv", "pip", "freeze"]):
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except Exception:  # noqa: BLE001 - uv may be absent
            continue
        if res.returncode == 0 and res.stdout.strip():
            out.write_text(res.stdout, encoding="utf-8")
            return

    # Fallback: dump installed distributions via importlib.metadata.
    try:
        from importlib.metadata import distributions

        lines = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in distributions()
            if d.metadata["Name"]
        )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
            param_name = override.get_key_element().split("+")[-1]
            val = override.value()
            if override.is_sweep_override():
                if hasattr(val, "list"):  # ChoiceSweep (a Glob has no .list)
                    val = multirun(_unquote(val.list))  # type: ignore
                elif hasattr(val, "range"):  # RangeSweep: discrete -> a grid axis
                    val = multirun(list(val.range()))  # type: ignore
                else:
                    # An IntervalSweep (continuous, for Bayesian sweepers)
                    # cannot form the Cartesian grid to_xarray assembles, and
                    # a Glob picks config-group options this parser cannot see.
                    raise ValueError(
                        f"unsupported sweep override for {param_name!r}: "
                        f"{type(val).__name__}. mushin assembles a Cartesian "
                        "grid from the values an override lists: use "
                        "`param=a,b,c`, `choice(...)`, or `range(...)`; "
                        "`glob(...)`, `interval(...)` and adaptive-sweeper "
                        "syntax are not supported."
                    )

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
            overrides = _override_texts(
                load_from_yaml(self.working_dir / "multirun.yaml").hydra.overrides.task
            )

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
            zen(_signed(self.pre_task)).validate(self.eval_task_cfg)

        # As run() prepares it: `mushin_resume` is injected, never a field.
        try:
            task = _prepare_task(self.task)[0]
        except (TypeError, ValueError):  # no signature to read: zen says so
            task = self.task
        zen(task).validate(self.eval_task_cfg)

    def run(
        self,
        *,
        working_dir: str | None = None,
        sweeper: str | None = None,
        launcher: str | None = None,
        launcher_config: Mapping[str, Any] | None = None,
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
        dry_run: bool = False,
        confirm_above: int | None = None,
        max_total_seconds: float | None = None,
        sample: int | None = None,
        sample_seed: int = 0,
        cache_dir: str | None = None,
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

        launcher_config: Mapping[str, Any] | None (default: None)
            Fields for the selected ``launcher``, applied as
            ``hydra.launcher.<key>=<value>`` overrides: no hand-rolled
            strings. Pair with `mushin.slurm` for cells that each run as
            one SLURM job, e.g. ``run(launcher="submitit_slurm",
            launcher_config=mushin.slurm(gpus=1, partition="gpu"))``, or
            with `mushin.submitit_slurm_config` for DDP cells, a task per GPU.
            Requires ``launcher=``.

        overrides: List[str] | None (default: None)
            Parameter overrides not considered part of the swept workflow
            parameter set (which are exposed via `multirun_task_overrides`).

        version_base : Optional[str], optional (default="1.3")
            The Hydra compatibility version governing default behaviors.
            - An explicit version string like "1.3" selects the defaults of
              that Hydra version.
            - `None` selects the defaults of the installed Hydra version.
            Regardless of `version_base`, `run` sets ``hydra.job.chdir=True``
            (unless you override it); each job runs in its own working
            directory, which the workflow's metrics sidecars depend on.

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

        dry_run : bool (default: False)
            Preview the grid (cell count + per-axis values) and return a summary
            dict without launching any job.

        confirm_above : int | None (default: None)
            Refuse to launch a sweep with more than this many cells (raising a
            ``ValueError``). ``MUSHIN_MAX_CELLS`` supplies a default ceiling.

        max_total_seconds : float | None (default: None)
            Graceful wall-clock budget: once exhausted, remaining cells are
            skipped (NaN, ``self.skipped``) and a later ``resume=True`` finishes
            them. One deadline for the whole sweep, timed from its first
            computed cell: under a parallel launcher (joblib, submitit) every
            worker stops starting cells once it passes. Disabled (with a
            warning) for cells running under an external multi-rank launch,
            because ranks checking it at different moments could diverge and
            hang DDP.

        sample : int | None (default: None)
            Run a random ``sample``-cell subset of the grid (rest NaN) for fast
            exploration; deterministic and resume-safe. ``sample_seed`` varies
            the subset.

        cache_dir : str | None (default: None)
            Content-addressed cache of completed cells shared across
            ``working_dir``\\ s: a cell whose config, task source and the
            project code the task can run match a cached entry reuses it
            instead of recomputing.

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

        if max_total_seconds is not None and max_total_seconds <= 0:
            raise ValueError(
                f"`max_total_seconds` must be a positive number of seconds, got "
                f"{max_total_seconds!r}."
            )

        if sample is not None and sample <= 0:
            raise ValueError(
                f"`sample` must be a positive number of cells, got {sample!r}."
            )

        if resume and sample is not None:
            raise ValueError(
                "`resume` and `sample` cannot be combined: sampling skips cells "
                "by job index, which would overwrite already-completed cells on a "
                "resume. Run WITHOUT `sample` and `resume=True` to fill in the "
                "remaining cells."
            )

        # Normalize cache_dir to an absolute path NOW (before Hydra chdirs each
        # job into its own numeric dir). A relative/`~` cache_dir would otherwise
        # be dereferenced inside every cell's job dir, so cross-cell reuse would
        # silently never hit.
        if cache_dir is not None:
            cache_dir = str(Path(cache_dir).expanduser().resolve())

        launch_overrides = []
        # Each multirun axis: its name, its override key, its values, how each
        # is written, and whether `target_job_dirs=` built it, for the check
        # on swept strings below (#319).
        swept: list[tuple[str, str, list[Any], list[str], bool]] = []

        if overrides is not None:
            launch_overrides.extend(overrides)

        if working_dir is not None:
            # Quote the path: dirs named after params (`lr=0.1`) or containing
            # commas would otherwise be re-split by Hydra's override parser.
            launch_overrides.append(
                f"hydra.sweep.dir={_to_override_element(str(working_dir))}"
            )

        if sweeper is not None:
            launch_overrides.append(f"hydra/sweeper={sweeper}")

        if launcher is not None:
            launch_overrides.append(f"hydra/launcher={launcher}")

        if launcher_config is not None:
            if launcher is None:
                raise ValueError(
                    "`launcher_config` requires `launcher=`: its fields "
                    "configure that launcher (e.g. launcher='submitit_slurm' "
                    "with mushin.slurm(...), or mushin.submitit_slurm_config(...) "
                    "for DDP)."
                )
            # Emit the `hydra.launcher.*` overrides so callers never hand-roll
            # them (values serialized like sweep elements: numbers/bools bare,
            # strings quoted so ',' or '=' inside a value cannot re-split).
            launch_overrides.extend(
                f"hydra.launcher.{k}={_to_override_element(v)}"
                for k, v in launcher_config.items()
            )

        # The workflow depends on Hydra changing the working directory per job
        # at runtime: each job writes and reads its metrics sidecar in its own
        # chdir'd directory. Under version_base >= 1.2 that is off by default,
        # so set `hydra.job.chdir=True` explicitly. A caller who passes their
        # own `hydra.job.chdir` override wins.
        if not any(o.split("=", 1)[0] == "hydra.job.chdir" for o in launch_overrides):
            launch_overrides.append("hydra.job.chdir=True")

        # Catch a typo'd override name BEFORE launching. mushin deliberately
        # allows overrides beyond the task's own parameters (values consumed by
        # `pre_task`, config-group choices, interpolations), so an unknown name is
        # not by itself an error. But a name that is a near-miss of a real target
        # (`lrate` for `lr`) is almost always a typo — left unchecked it is
        # `+`-appended as a config key nothing reads and surfaces as a phantom
        # constant sweep dimension. Only the head of a dotted path is checked
        # (`model.width` -> `model`). A task with `**kwargs` opts out entirely.
        jobdir = getattr(self, "_JOBDIR_NAME", None)
        task_targets = _override_name_targets(getattr(self, "task", None))
        if task_targets is not None:
            # Explicit pre_task params are valid targets too; the default
            # `pre_task(**kwargs)` contributes nothing (targets is None -> set()).
            pre_targets = _override_name_targets(getattr(self, "pre_task", None))
            cfg_names: set[str] = set()
            try:
                from omegaconf import OmegaConf

                container = (
                    OmegaConf.to_container(self.eval_task_cfg, resolve=False)
                    if OmegaConf.is_config(self.eval_task_cfg)
                    else self.eval_task_cfg
                )
                if isinstance(container, Mapping):
                    cfg_names = {str(key) for key in container}
            except Exception:  # noqa: BLE001 - unreadable cfg -> signature only
                cfg_names = set()
            real_targets = (
                task_targets
                | (pre_targets or set())
                | cfg_names
                | ({jobdir} if jobdir else set())
            )
            import difflib

            for k in workflow_overrides:
                head = k.split(".", 1)[0]
                if head in real_targets:
                    continue
                # cutoff 0.5 fires on typos (`lrate`~`lr` .57, `epsilonn`~`epsilon`
                # .93) but clears deliberate extra names (`list_vals`~`epsilon` .13).
                near = difflib.get_close_matches(
                    head, sorted(real_targets), n=1, cutoff=0.5
                )
                if near:
                    raise ValueError(
                        f"override `{head}` does not match any task parameter; "
                        f"did you mean {near[0]!r}? A mismatched name is appended "
                        "as a config key nothing reads and would appear as a "
                        "phantom constant sweep dimension. Rename it, or use "
                        f"`{near[0]}=` if that was intended."
                    )

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
                written = [_to_override_element(item) for item in v]
                launch_overrides.append(f"{prefix}{k}=choice({','.join(written)})")
                swept.append(
                    (k, f"{prefix}{k}", list(v), written, isinstance(v, _TargetDirs))
                )
            elif isinstance(v, hydra_list):
                # A single non-swept list value passed to one job, each element
                # written as a sweep element is (#321): its own str is Python's
                # list repr, which doubled a backslash and wrote None as 'None'.
                launch_overrides.append(f"{prefix}{k}={_to_override_element(list(v))}")
            else:
                # Scalar value: serialize like a multirun element so a fixed
                # string containing ',' / '=' stays a single string (not an
                # accidental sweep) and a string like "true"/"1" is not coerced
                # to a bool/int.
                launch_overrides.append(f"{prefix}{k}={_to_override_element(v)}")

        for _name in self._REQUIRED_STATIC_METHODS:
            if not isinstance(getattr_static(self, _name), staticmethod):
                raise TypeError(
                    f"{type(self).__name__}.{_name} must be a static method"
                )

        # Validate up front that every required task/pre_task parameter is
        # satisfied — by the base config or an override — BEFORE launching. A
        # genuinely missing parameter otherwise surfaces as an opaque per-job
        # Hydra error (ConfigAttributeError) after the sweep has started, once
        # per cell. Only the head of a dotted override counts (`model.width`
        # satisfies a `model` parameter). A parameter supplied via override is
        # valid even when the base config is empty (the common decorator case).
        required = _required_params(getattr(self, "task", None))
        required |= _required_params(getattr(self, "pre_task", None))
        provided = _config_field_names(self.eval_task_cfg) | {
            k.split(".", 1)[0] for k in workflow_overrides
        }
        # Raw Hydra override strings (`overrides=["+seed=0,1,2"]`) are another way
        # to supply a parameter; strip any leading +/~/++ and dotted tail.
        for o in overrides or []:
            provided.add(o.split("=", 1)[0].lstrip("+~").split(".", 1)[0])
        if jobdir:
            provided.add(jobdir)
        missing = sorted(required - provided)
        if missing:
            names = ", ".join(repr(m) for m in missing)
            raise ValueError(
                f"the task requires parameter(s) {names} that are not provided "
                "by the base config or any override. Pass them to run() (e.g. "
                f"`{missing[0]}=...` or `{missing[0]}=multirun([...])`), or set "
                "them in the workflow's eval_task_cfg."
            )

        # On 3.14+, a parallel launcher must be able to send what the runner
        # holds before anything launches (#330).
        launcher_set = (launcher is not None and launcher != "basic") or any(
            o.lstrip("+~").startswith("hydra/launcher=")
            and o.split("=", 1)[1] != "basic"
            for o in overrides or []
        )
        if launcher_set:
            _check_sent_by_value(
                (self.task, *getattr(self, "_mushin_code_roots", ()), self.pre_task)
            )

        # The grid axes and cell count drive the dry-run preview, the pre-launch
        # gate, and `sample=`. Derived from the FULLY parsed launch overrides so a
        # sweep axis counts whether it came through a `param=` kwarg OR the raw
        # `overrides=[...]` list; the `hydra.*` plumbing is excluded.
        parsed = self._parse_overrides(launch_overrides)
        _grid = {
            k: v
            for k, v in parsed.items()
            # Exclude only real Hydra plumbing (`hydra.sweep.dir`, `hydra/launcher`)
            # — a USER axis merely named with that prefix (`hydraulic`) counts.
            if not (k == "hydra" or k.startswith(("hydra.", "hydra/")))
        }
        axes = {k: list(v) for k, v in _grid.items() if isinstance(v, multirun)}
        fixed = {k: v for k, v in _grid.items() if not isinstance(v, multirun)}
        num_cells = 1
        for v in axes.values():
            num_cells *= len(v)

        # Duplicate-value guard over the PARSED grid, so a sweep supplied via raw
        # `overrides=["+a=1,1"]` is rejected like a duplicate multirun kwarg is
        # (two cells sharing a combo key silently collapse to the last). Values
        # are keyed by `combo_key`, which holds their text, so two that differ
        # only in type, such as '1' and 1, are caught too. On parsed values it
        # gives the key of the base `_sanitize_coordinate_for_xarray`; that is
        # not called here, as `BaseWorkflow` does not define it.
        from ._sweep_io import combo_key

        for k, vals in axes.items():
            keys = [combo_key({k: v}) for v in vals]
            if len(set(keys)) != len(keys):
                dupes = sorted(
                    {
                        repr(item)
                        for item, kk in zip(vals, keys, strict=True)
                        if keys.count(kk) > 1
                    }
                )
                raise ValueError(
                    f"`{k}` has duplicate sweep values {', '.join(dupes)}; a "
                    "repeated axis value collapses to a single cell. Use "
                    "distinct values, or add a separate axis (e.g. seed) for "
                    "repeat trials."
                )

        # A string swept through `choice(...)` reaches its job without its
        # quotes: one that reads as another value ('true', '1', 'null')
        # reached the task as that value, and one Hydra cannot write unquoted
        # ("it's", a non-ASCII name) failed inside the launch (#319). Both are
        # refused here, after the duplicate checks (whose message names a
        # value two cells share) and before the dry-run preview or any write.
        # Only a value written quoted can be misread, so an axis of numbers
        # parses nothing, and most sweeps stop at Hydra's parser. What reads
        # back differently is then asked of the config Hydra composes for the
        # sweep, as `launch` composes it: whether the basic sweeper writes the
        # jobs (another writes them its own way), whether the key is a config
        # group (whose override must name an option), and what its field
        # makes of the value beside the string a fixed value gives (a `str`
        # field turns 1 back into '1'). A raw `overrides=[...]` sweep is the
        # user's own Hydra syntax and is left to Hydra.
        odd = []
        for name, key, items, written, targets in swept:
            quoted = [
                (item, text)
                for item, text in zip(items, written, strict=True)
                if text.startswith("'")
            ]
            if not quoted:
                continue
            reads = _swept_strings_read([text for _, text in quoted])
            values = [
                (item, text, read)
                for (item, _), (text, read) in zip(quoted, reads, strict=True)
                if read is _UNPARSED or read != text
            ]
            if values:
                odd.append((name, key, written, targets, values))
        misread = []
        if odd:
            try:
                master, groups, swept_groups = _composed_for_launch(
                    self.eval_task_cfg,
                    launch_overrides,
                    [name for name, *_ in odd],
                    to_dictconfig=to_dictconfig,
                    config_name=config_name,
                    job_name=job_name,
                    version_base=version_base,
                )
            except Exception as e:
                # It fails where `launch` would, and says so as launch does.
                amended = _with_launcher_hint(e, launcher)
                if amended is not None:
                    raise amended from e
                raise
            if master.hydra.sweeper._target_ != _BASIC_SWEEPER:
                odd = []
        for name, key, written, targets, values in odd:
            # A copy per axis: a key's value written here must not reach
            # another axis's key below it (`model` and `model.width`).
            job_cfg = copy.deepcopy(master)
            found = []
            for item, text, read in values:
                if read is _UNPARSED:
                    found.append((item, "would not parse in its job", False))
                    continue
                if name in groups:
                    found.append((item, "would fail in its job", False))
                    continue
                if any(name.startswith(f"{group}.") for group in swept_groups):
                    # Each job's option types what lies under a swept group,
                    # and the composed config holds only its default's: the
                    # field is not judged, so nothing that ran is refused.
                    continue
                as_swept = _job_value(job_cfg, name, read)
                as_fixed = _job_value(job_cfg, name, text)
                # A string the field does not take fails fixed too: the sweep
                # is not what changes it.
                if (
                    as_fixed is _NOT_TAKEN
                    or as_swept == as_fixed
                    or (_is_nan(as_swept) and _is_nan(as_fixed))
                ):
                    continue
                if as_swept is _NOT_TAKEN:
                    found.append((item, "would fail in its job", False))
                else:
                    found.append(
                        (
                            item,
                            f"would reach the task as {as_swept!r}",
                            not isinstance(as_swept, str),
                        )
                    )
            if found:
                misread.append((name, key, written, targets, found))
        if misread:
            raise ValueError(_swept_strings_refusal(misread))

        # `sample=` bounds what actually COMPUTES; the dry-run preview reports
        # that. The gate below still fires on the full grid: Hydra enumerates
        # and launches every cell regardless of sampling (a job dir, composed
        # config, and — under submitit — a scheduler submission per cell), so
        # the typo'd-grid rail must cap launch cost, not just compute.
        effective_cells = num_cells if sample is None else min(sample, num_cells)

        if dry_run:
            # Preview the grid and return WITHOUT launching (also the intended
            # escape hatch for previewing an over-limit sweep — the gate below is
            # skipped on a dry run).
            summary = {
                "num_cells": num_cells,
                "axes": axes,
                "fixed": fixed,
                "working_dir": working_dir,
            }
            if sample is not None:
                summary["sample"] = effective_cells
            # Axis values, fixed values and the path are user data: echo
            # escapes what the console cannot encode instead of crashing.
            echo(_format_dry_run(summary))
            return summary

        # Pre-launch cell-count gate: refuse an over-large sweep before it
        # launches. `confirm_above=` is the per-call ceiling; the MUSHIN_MAX_CELLS
        # environment variable is the default used when the call doesn't set one
        # (so a shared cluster/CI profile can cap everything). Preview an
        # over-limit sweep with dry_run=True, or raise the ceiling to proceed.
        limit = confirm_above
        limit_src = "confirm_above"
        if limit is None:
            import os

            env = os.environ.get("MUSHIN_MAX_CELLS")
            if env:
                try:
                    limit = int(env)
                    limit_src = "MUSHIN_MAX_CELLS"
                except ValueError:
                    limit = None  # malformed env -> no gate
        if limit is not None and num_cells > limit:
            sample_note = (
                ""
                if sample is None
                else f" `sample={sample}` limits compute but does not reduce the "
                f"number of launched Hydra jobs: all {num_cells} cells would "
                "still be launched."
            )
            raise ValueError(
                f"this sweep has {num_cells} cells, above the {limit_src} limit "
                f"of {limit}.{sample_note} Preview it with `dry_run=True`, or raise "
                f"the ceiling (e.g. `confirm_above={num_cells}`) to run it."
            )

        if task_fn_wrapper is None:
            task_fn_wrapper = _identity

        if pre_task_fn_wrapper is None:
            pre_task_fn_wrapper = _identity

        # Swept dimension names for the per-cell combo: from the full parse,
        # `hydra.*` keys included (unlike `_grid`, which drops them).
        _swept_names = tuple(k for k, v in parsed.items() if isinstance(v, multirun))
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

        from ._resume import (
            _unwrap_task,
            code_fingerprint,
            driver_script_entries,
            partial_task_cacheable,
        )

        # Grid subsampling: pick a deterministic random subset of the job indices
        # 0..num_cells-1. Seeding on (num_cells, sample, sample_seed) makes the
        # selection reproducible AND identical across a resume, and the runner
        # skips any cell whose job.num is not selected. `sample >= num_cells`
        # runs the whole grid (nothing to subsample).
        sampled_indices = None
        if sample is not None and sample < num_cells:
            import random as _random

            _rng = _random.Random(f"{num_cells}:{sample}:{sample_seed}")
            sampled_indices = frozenset(_rng.sample(range(num_cells), sample))

        _task_fn, _wants_resume = _prepare_task(self.task)
        _code_roots = (self.task, *getattr(self, "_mushin_code_roots", ()))
        task_call = _TaskRunner(
            task=task_fn_wrapper(_task_fn),
            pre_task=pre_task_fn_wrapper(_signed(self.pre_task)),
            swept_names=_runner_swept,
            base_provenance=_base_provenance,
            on_error=on_error,
            inject_resume=_wants_resume,
            prior_manifest=_prior_manifest,
            # A partial task has no source of its own; its function's body is
            # what runs (main recorded None for it, so older sweeps still
            # resume). Its bound arguments are not compared, as on main.
            code_hash=code_fingerprint(_unwrap_task(self.task)),
            max_total_seconds=max_total_seconds,
            sampled_indices=sampled_indices,
            cache_dir=cache_dir,
            # Further roots a subclass names (Study's training functions,
            # which its task calls but does not import), and pre_task: the
            # project files their modules import are recorded like the task's.
            code_roots=_code_roots,
            code_body_roots=(self.pre_task,),
            cache_ok=partial_task_cacheable(self.task),
            script_entries=driver_script_entries((*_code_roots, self.pre_task)),
        )

        self._capture_env = capture_env

        # Reset memoized override caches and loaded-from-disk state so a SECOND
        # run() on the same instance keys its grid on this sweep, not the
        # previous one (mirrors the reset in `load_from_dir` for reused loader
        # objects). Deliberately placed HERE — after every validation raise and
        # the dry_run early-return — so a preview or a rejected call on a
        # loaded workflow does not wipe state its next to_xarray() needs.
        self._multirun_task_overrides = {}
        self._target_dir_multirun_overrides = None
        self._loaded_job_choices = None
        self._job_metrics_by_index = None

        # Run a Multirun over epsilons
        try:
            jobs = launch(
                self.eval_task_cfg,
                task_call,
                overrides=_route_to_mushin_launchers(launch_overrides),
                multirun=True,
                version_base=version_base,
                to_dictconfig=to_dictconfig,
                config_name=config_name,
                job_name=job_name,
                with_log_configuration=with_log_configuration,
            )
        except Exception as e:
            amended = _with_launcher_hint(e, launcher)
            if amended is not None:
                raise amended from e
            raise

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

        # launch() returned, so every cell of this run is done with its budget
        # start: remove it from the sweep root. Only on a normal return of the
        # launch: when launch() raises (a failed cell under on_error="raise",
        # Ctrl-C, a submitit timeout), jobs a scheduler still holds can start
        # after this process is gone, and they must find the same start, not
        # create a fresh one. A start left behind is inert: no later run has
        # its id. Before the post-processing, which can raise on its own.
        _remove_budget_start(
            task_call, jobs if isinstance(jobs, list) else [jobs], working_dir
        )

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
        # For a sweep loaded from disk, `load_metrics` rebuilds `failures` and
        # `skipped`; there `exception` is None when no manifest describes the
        # attempt (the traceback is in the cell's mushin_error.txt).
        self.failures: list[dict[str, Any]] = []
        # Cells that were not run ({"combo", "working_dir"}): sampled out, or
        # skipped by an exhausted max_total_seconds budget. A loaded sweep also
        # lists cells that never finished; `working_dir` is None for one that
        # never started.
        self.skipped: list[dict[str, Any]] = []
        self._manifest: Any = None

        if self._working_dir is not None:
            self.load_from_dir(self.working_dir, metrics_filename=None)

    @property
    def is_complete(self) -> bool:
        """Whether every requested grid cell completed. For a workflow loaded
        from disk, whether its record (``failures``, ``skipped``) is empty: the
        record is rebuilt on each load, from the cells' statuses and, once
        metrics are loaded, their metrics. ``True`` for a workflow that has
        neither swept nor loaded."""
        if self._manifest is not None:
            return self._manifest.is_complete()
        return not (self.failures or self.skipped)

    @property
    def notes(self) -> str | None:
        """Free-form note recorded for this sweep via ``run(notes=...)`` (or read
        from the manifest of a loaded sweep); ``None`` if none was set."""
        if self._manifest is not None:
            return self._manifest.notes
        return getattr(self, "_notes", None)

    @property
    def tags(self) -> list[str]:
        """Tags recorded for this sweep via ``run(tags=[...])`` (or read from the
        manifest of a loaded sweep); an empty list if none were set."""
        if self._manifest is not None:
            return list(self._manifest.tags)
        return list(getattr(self, "_tags", None) or [])

    @property
    def provenance(self) -> dict | None:
        """Best-effort per-run provenance (git/versions/config) read from one
        job's ``mushin_provenance.json``: the first that has an accelerator,
        else the first. ``None`` if no record is found."""
        from ._provenance import representative_record

        dirs = self.multirun_working_dirs or []
        return representative_record(Path(d) / "mushin_provenance.json" for d in dirs)

    # TODO: add target_job_dirs example
    #      Document .swap_dims({"job_dir": <...>}) and .set_index(job_dir=[...]).unstack("job_dir")
    #      for re-indexing based on overrides values

    _JOBDIR_NAME: str = "job_dir"
    _target_dir_multirun_overrides: defaultdict[str, list[Any]] | None = None
    # Per-job `runtime.choices` recovered by `load_from_dir` (job-num order),
    # so offline loads key config-group axes by choice name like in-process.
    _loaded_job_choices: list[dict | None] | None = None
    # Job-aligned metrics from `load_metrics` (None = failed/skipped cell), so
    # `to_xarray` can rebuild the combo map with holes NaN-filled.
    _job_metrics_by_index: list[Any] | None = None
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
        # tasks without the caller overriding metric_load_fn. The first bytes
        # are decoded before the check (the encoding json detects consumes a
        # byte-order mark), so a JSON file with one or in UTF-16/32, as
        # Windows tools write, is not taken for a pickle.
        try:
            with open(file_path, "rb") as f:
                head = f.read(64)
        except OSError:
            head = b""
        # "replace", not "ignore": a byte that is not text must not vanish
        # and leave a "{" behind it at the start.
        start = head.decode(json.detect_encoding(head), errors="replace")
        if start.lstrip()[:1] in ("{", "["):
            from ._sweep_io import _restore_nonfinite, decode_legacy

            with open(file_path, "rb") as f:
                data = f.read()

            # mushin's sidecar is UTF-8 (ASCII, in fact), but this loader
            # also reads a JSON file the user names (load_metrics("my.json")),
            # which an earlier release decoded in the locale's encoding: the
            # encoding json detects (UTF-8, UTF-16 or UTF-32, with or without
            # a byte-order mark) first, then the locale's, strictly, so a file
            # in neither raises rather than load with replaced characters.
            loaded = json.loads(
                decode_legacy(data, encoding=json.detect_encoding(data))
            )
            # Decode the sidecar's non-finite encodings (NaN as null, ±Inf
            # as a tagged marker) so an offline reload yields the same
            # float values the in-process run produced.
            return _restore_nonfinite(loaded)
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
        launcher_config: Mapping[str, Any] | None = None,
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
        dry_run: bool = False,
        confirm_above: int | None = None,
        max_total_seconds: float | None = None,
        sample: int | None = None,
        sample_seed: int = 0,
        cache_dir: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        **workflow_overrides: str | int | float | bool | multirun | hydra_list,
    ):
        """Run the sweep: one Hydra job per grid cell, metrics collected per cell.

        Extends ``BaseWorkflow.run`` (see it for the launcher/sweeper/
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
            metrics with NaN, and keeps sweeping. On the submitit launchers, a
            cell whose job ends without returning (an out-of-memory kill, a
            failed node, ``scancel``, or an error outside the task, such as
            the job failing to import it) fails the same way, and its
            ``mushin_error.txt`` names the job's log. A time limit usually
            reaches the task as an error submitit raises in it, which fails
            the cell as any task error does.
        resume : bool (default: False)
            Skip cells already recorded as completed in ``working_dir`` (their
            metrics are read from the sidecar) and re-run the rest. A completed
            cell is reused only if its swept-parameter combination AND
            fingerprints of its resolved config, its task source and the
            project code the task can run still match. That project code is
            every project file the task's module imports (and ``pre_task``'s),
            directly or through other project files, at any depth (imports
            inside functions included), and the modules of the classes the
            cell's config names (``_target_``, or hydra-zen's ``just()`` and
            ``zen_meta``), each with the ``__init__.py`` of the packages it is
            in, which importing it runs. Each file is compared as a whole, by its syntax tree
            without its main block and docstrings: an edit anywhere in it
            re-runs the cells, and a comment, a docstring or formatting does
            not (so neither does a prompt a decorator reads from a
            docstring). The task function's own source is compared as text,
            as before, so any edit to it re-runs its cells, a comment
            included. A task defined in the driver script is compared by what
            its definition can reach by name in the script (a helper, a
            constant, a star import, a definition a project decorator
            registers, in the main block too), not by the run call, other
            top-level calls or prints, or what only they read (``GRID =
            dict(...)``), so extending the grid or adding a print is not an
            edit. The script is read when the run starts (a ``_target_``
            defined in it, when the first cell naming it runs), so editing it
            while the sweep runs does not change what its cells record. A task
            defined in a notebook is compared by the definitions its code
            reaches by name in the kernel as the run starts (each function
            and class its cells define, through decorators, a ``partial``, a
            bound method or an instance) and by every project module its
            cells import, so re-running an edited cell and resuming re-runs
            the cells; with ``cache_dir``, importing another project module
            in a cell can make them compute again. A class whose ``class``
            statement cannot be told from another kept version of its cell
            (or, on 3.11/3.12, from another statement of it its cell runs)
            can make a resume re-run them, and ``cache_dir`` compute them,
            when its cell runs again or after a restart. A class that a cell
            run after ``import mushin`` makes without stating it (in a
            ``%%time`` cell, by ``exec``, ``%run`` or a ``global`` factory, or
            a copy such as ``dataclass(slots=True)(C)``) is not read, but
            re-runs them whenever that cell runs again and after a restart. A
            resume on another Python version can also re-run the cells of a
            class stated more than once in its cell. No value is compared (a
            number, a string, a dict config): pass one as a swept parameter,
            or use a new ``working_dir`` after editing it. A re-run
            warns, naming the file (``helpers (file)``) or the notebook name
            (``helper (notebook)``). Files are compared as
            they are on disk: if one changes after this process loaded it and
            is not reloaded, the old code keeps running while the cells
            computed now are recorded as the new code's, so reload it (or
            restart) before resuming. mushin warns when it had recorded that
            file earlier in the same process. A run can import a module
            loaded through ``importlib.util.LazyLoader`` that has not run
            yet, even one the task does not use, so import such a module
            yourself before ``run()``. Not
            covered: installed packages other than editable ones, data files,
            the environment, in a notebook its values, a function reached
            only through a container (a registry), a closure or a default, a
            name one of its functions rebinds (``global helper``), a class's
            attribute values beyond its text, possibly a module loaded other
            than by a cell's ``import`` statement (``importlib.import_module``,
            a plugin loader that fills a registry, an import in a ``%%time``
            cell), possibly a class no ``class`` statement makes (``type(...)``,
            ``make_dataclass``) or one of which a cell holds more than one
            version (two ``class`` statements run, or a statement and a class
            bound from elsewhere), a ``_target_`` defined in the notebook,
            code whose text the kernel does not keep (a ``%%time`` cell,
            ``exec``, a ``%run`` file; a class it makes re-runs instead),
            possibly a cell holding a form feed, a vertical tab or another
            character that ends a line for ``str.splitlines`` (not a newline
            or a carriage return) that the shell's history does not hold and
            mushin did not see run (such as one ``%run`` runs from a notebook
            or an ``.ipy`` file, a silent one, or one run inside another cell
            before ``import mushin``; a cell holding one can re-run the
            cells instead), what an IPython input transformer
            changes in a cell, possibly a class a background thread or a
            callback makes while a cell runs (taken for that cell's), a
            top-level statement of the driver script that changes an object in
            place (``CFG["x"] = 2``, ``parser.add_argument(...)``) or a call
            in its main block that rebinds a global (``setup()``), a script
            definition reached only through an ``__init_subclass__`` or
            metaclass registry, a definition registered inside a block (``if
            ENABLED: @register class ...``), a class body reading the module's
            name it shadows (``SCALE = SCALE``), and code reached only through
            some other string. The script rule follows names, so state
            reached another way (``globals()``, ``exec``) can hide an edit:
            keep helpers in an imported module, which is compared as a whole
            file, when that matters. In a notebook, a name rebound through
            ``globals()`` or ``exec``, or by a function the notebook's names
            do not lead to (one kept only in an object's attributes), is
            still compared, so once a run rebinds it in the kernel, a resume
            re-runs the cells; the resilience guide lists where mushin looks.
            A very large project is covered up to a fixed number of files
            (with a warning). After upgrading a dependency, changing the data
            or editing such code, re-run from a fresh ``working_dir`` rather
            than resuming.
        capture_env : bool (default: False)
            After the sweep, snapshot the environment (``uv export``, falling
            back to ``uv pip freeze`` then an ``importlib.metadata`` dump) to
            ``working_dir/mushin_env.txt``, or ``mushin_env.<n>.txt`` if a
            snapshot already exists there (a resume never overwrites the
            original run's snapshot).
        dry_run : bool (default: False)
            Preview the sweep instead of running it: print the cell count and
            each swept axis with its values (a range typo shows up as an
            unexpectedly wide axis), then return a summary dict
            (``num_cells``/``axes``/``fixed``/``working_dir``) without launching
            any job.
        confirm_above : int | None (default: None)
            Refuse to launch a sweep with more than this many grid cells,
            raising a ``ValueError`` instead: a guard against an accidentally
            huge grid. The ``MUSHIN_MAX_CELLS`` environment variable supplies a
            default ceiling when this is not set (an explicit value wins). Use
            ``dry_run=True`` to preview an over-limit sweep. ``sample=`` does
            not lower this gate: Hydra still launches every grid cell (sampling
            saves compute, not launch overhead).
        max_total_seconds : float | None (default: None)
            A graceful wall-clock budget. Once it is exhausted the remaining
            cells are skipped (recorded ``'skipped'``, NaN in the dataset,
            surfaced in ``self.skipped``) instead of the sweep running to the
            end. The clock starts when the sweep's first computed cell starts,
            so at least one cell always runs and resume cache hits do not
            consume it; a cell already running is never interrupted, so the
            sweep can end up to one cell's duration past the deadline (a cell
            that a scheduler preempts and requeues after the deadline still
            runs, since it started before it). Skipped cells are not
            completed, so a later ``resume=True`` with more time finishes
            them; pass ``max_total_seconds`` again to time-box that run too,
            and its clock starts afresh. The budget is one deadline for the
            whole sweep, whatever the launcher, as long as every worker sees
            the same sweep directory (a filesystem they share, as submitit
            needs anyway) and, on several nodes, synchronized clocks: under a
            parallel launcher (``launcher="joblib"``, submitit) no worker
            starts a cell after it passes, so it bounds elapsed time, not
            compute summed over the workers, and each worker can finish one
            cell it started just before it. The default sequential launcher
            keeps the deadline in memory; a parallel launcher's workers share
            the start through a hidden file in the sweep directory
            (``.mushin_budget_start.<id>``), compared across processes on the
            wall clock. ``run`` removes the file when its launch returns
            normally; a launch that raises leaves it, so jobs a scheduler
            still holds keep the same deadline, and it is inert afterwards
            because each run has its own id. If no start can be shared (the
            file cannot be created, or is gone when read) or the budget step
            fails, each worker process times the budget from its own first
            cell, which runs, with a warning raised in that worker (on its
            stderr, not through the caller's warning filters). Disabled (with
            a warning) for cells running under a multi-rank launch (submitit
            DDP/FSDP): ranks checking the deadline at different moments could
            diverge and hang at rendezvous; use the scheduler's time limit for
            those jobs instead.
        sample : int | None (default: None)
            Run only a random subset of ``sample`` cells from the full grid for
            fast exploration; the remaining cells are skipped (NaN, and
            ``self.skipped``). The subset is chosen deterministically (see
            ``sample_seed``) so it is reproducible and identical across a resume;
            resuming WITHOUT ``sample`` fills in the remaining cells. The full
            grid is still composed by Hydra (only the sampled cells run), so this
            saves compute, not launch overhead. ``sample >= n_cells`` runs
            everything. A sampled sweep is intentionally incomplete
            (``is_complete`` is False).
        sample_seed : int (default: 0)
            Seeds the ``sample`` selection; change it to draw a different
            reproducible subset of the same grid.
        cache_dir : str | None (default: None)
            A content-addressed cache of completed cells, shared across
            ``working_dir``\\ s. A cell whose resolved config, task source and
            the project code the task reaches match a cached entry (keyed on the
            same fingerprints as resume) reuses that result instead of
            recomputing, so a cell computed in one sweep is free in another.
            Newly-computed cells are stored there. Complements ``resume`` (which
            reuses within a single ``working_dir``); a changed config value, an
            edited task body or an edited project file is a cache miss. An
            entry an earlier mushin wrote is recomputed once. Files are read
            as they are on disk (see ``resume``): after editing a module in a
            running process, reload it (or restart) before using the cache, or
            the old code's results are stored under the new code's key. When
            the project code cannot be fingerprinted (a warning says so), the
            cache is not used for that cell, and a partial task never uses the
            cache, as on main: the key cannot see into the data it binds.
        notes : str | None (default: None)
            A free-form note recorded for the sweep, in the manifest, on
            ``wf.notes``, and as the dataset attr ``mushin_notes``. Lineage for
            "why did I run this?". A resume that does not re-pass it keeps the
            original run's note.
        tags : list[str] | None (default: None)
            Tags recorded for the sweep, in the manifest, on ``wf.tags``, and as
            the dataset attr ``mushin_tags`` (preserved across a resume).
        **workflow_overrides
            The sweep itself: ``param=value`` fixes a value,
            ``param=multirun([...])`` makes a grid dimension. Nested config
            paths (``**{"model.width": multirun([4, 8])}``) and config groups
            are supported; see the workflows guide's "Sweep-axis support"
            section.
        """
        if notes is not None and not isinstance(notes, str):
            raise ValueError(
                f"`notes` must be a string or None, got {type(notes).__name__}"
            )
        if tags is not None and (
            not isinstance(tags, (list, tuple))
            or not all(isinstance(t, str) for t in tags)
        ):
            raise ValueError(f"`tags` must be a list of strings, got {tags!r}")
        # Stashed for jobs_post_process to record on the sweep manifest (run() and
        # jobs_post_process are separate methods; the manifest is built there).
        # `None` means "not passed" — distinct from an explicit `tags=[]`, which
        # is a request to clear — and the prior-manifest fallback only applies
        # on a resume (a fresh run reusing a dir is a NEW sweep, not lineage).
        self._notes = notes
        self._tags = list(tags) if tags is not None else None
        self._lineage_from_prior = resume

        if target_job_dirs is not None:
            if isinstance(target_job_dirs, str):
                raise TypeError(
                    f"`target_job_dirs` must be a sequence of pathlike objects, got: {target_job_dirs}"
                )
            value_check("target_job_dirs", target_job_dirs, type_=Sequence)

            target_job_dirs = [Path(s).resolve() for s in target_job_dirs]
            for d in target_job_dirs:
                if not d.is_dir():
                    # FileNotFoundError for both, as before, so a caller that
                    # catches it still does; the message tells them apart.
                    why = "is not a directory" if d.exists() else "does not exist"
                    raise FileNotFoundError(
                        f"The specified target directory {why}: {d}"
                    )
            target_job_dirs = _TargetDirs([str(s) for s in target_job_dirs])
            workflow_overrides[self._JOBDIR_NAME] = target_job_dirs

        return super().run(
            working_dir=working_dir,
            sweeper=sweeper,
            launcher=launcher,
            launcher_config=launcher_config,
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
            dry_run=dry_run,
            confirm_above=confirm_above,
            max_total_seconds=max_total_seconds,
            sample=sample,
            sample_seed=sample_seed,
            cache_dir=cache_dir,
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
        # Cached only once every dir has been read: a read that raises
        # partway must not leave a part of the dirs to be returned next time.
        found: defaultdict[str, list[Any]] = defaultdict(list)

        self.overrides = load_from_yaml(multirun_cfg).hydra.overrides.task

        # Use Hydra's override parser (rather than naive string splitting) so
        # that the job-dir sweep – e.g. `+job_dir=choice('a','b')` – and any
        # quoted values containing '=' / commas are parsed correctly.
        parsed = self._parse_overrides(_override_texts(self.overrides))
        jobdir_value = parsed.get(self._JOBDIR_NAME, [])
        dirs = list(jobdir_value) if _non_str_sequence(jobdir_value) else []

        for d in dirs:
            overrides = _override_texts(
                load_from_yaml(Path(d) / f"{self.output_subdir}/overrides.yaml")
            )
            output = self._parse_overrides(overrides)

            for ko, vo in output.items():
                found[ko].append(vo)
        self._target_dir_multirun_overrides = found
        return dict(found)

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
            self.multirun_working_dirs.append(_job_dir(job))

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
        # Lineage (notes/tags) from THIS run, falling back to the prior manifest
        # ONLY on a resume that did not re-pass them — so a resume preserves the
        # original run's lineage, while a fresh run reusing the dir starts clean
        # and an explicit `tags=[]` clears. (The fresh Manifest below is
        # deliberately not load_or_new — that is only to drop stale *cells*.)
        _notes = getattr(self, "_notes", None)
        _tags = getattr(self, "_tags", None)
        if getattr(self, "_lineage_from_prior", False) and (
            _notes is None or _tags is None
        ):
            _prev = Manifest.load_or_new(self.working_dir, [])
            if _notes is None:
                _notes = _prev.notes
            if _tags is None:
                _tags = _prev.tags
        manifest = Manifest(
            self.working_dir, swept_names, notes=_notes, tags=_tags or []
        )

        self._metrics_by_combo = {}
        self.failures = []
        self.skipped = []
        job_metrics: list[Any] = []
        for job, cfg, wdir in zip(
            self.jobs, self.cfgs, self.multirun_working_dirs, strict=True
        ):
            # Keyed on the swept params only (all `_combo_of_cfg` returns), as
            # `to_xarray`'s `_lookup_key` is, so `_metrics_by_combo` and manifest
            # keys align.
            swept_combo = self._combo_of_cfg(cfg, choices=_group_choices_of_job(job))
            key = combo_key(swept_combo)

            # A budget-skipped cell completed the Hydra job with a `_SkippedRun`
            # sentinel: it never ran, so NaN-fill it (like a failure) but record
            # it as 'skipped' rather than 'failed' — no error occurred, and a
            # resume with more time will finish it.
            if job.status == JobStatus.COMPLETED and isinstance(
                job.return_value, _SkippedRun
            ):
                self.skipped.append({"combo": key, "working_dir": str(wdir)})
                manifest.mark(swept_combo, dir=wdir.name, status="skipped")
                job_metrics.append(None)
                continue

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
                # A dead job's error carries its error stream; keep one line,
                # shaped like a repr (#345).
                error = rv.summary if isinstance(rv, _DeadJob) else repr(exc)
                self.failures.append(
                    {
                        "combo": key,
                        "exception": error,
                        "working_dir": str(wdir),
                    }
                )
                manifest.mark(swept_combo, dir=wdir.name, status="failed", error=error)
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

        if self.skipped:
            import warnings

            warnings.warn(
                f"{len(self.skipped)} cell(s) were not run (skipped); their grid "
                "cells are set to NaN. Resume to run them (a budget-exhausted "
                "sweep needs more time; a sampled sweep needs a resume without "
                "`sample`).",
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
        """`_swept_combo` of one job config; ``choices`` is the job's
        ``hydra.runtime.choices``."""

        def choice_of(name: str) -> str | None:
            choice = (choices or {}).get(name)
            return None if choice is None else str(choice)

        return _swept_combo(
            cfg,
            self._swept_param_names(),
            choice_of,
            self._sanitize_coordinate_for_xarray,
        )

    def _cell_record(self, i: int) -> tuple[dict | None, bool]:
        """Job dir ``i``'s status record, and whether it is another cell's.

        Hydra writes a job's ``.hydra/`` snapshot before its task runs, and
        mushin's runner writes the record after it, so a record older than
        the hydra.yaml beside it was written by an earlier job in a reused
        directory, and this job died before its first write (#355). Such a
        record is still this cell's when its combo and its config hash match
        this job's config, or cannot be compared with it (a sweep copied
        without its files' times also leaves a record older than its
        hydra.yaml). Otherwise it is another cell's, and the cell has no
        record and no result: ``(None, True)``."""
        from ._resume import (
            STATUS_FILE,
            read_cell_status,
            resolved_config,
            resolved_config_hash,
        )
        from ._sweep_io import combo_key

        dir_ = self.multirun_working_dirs[i]
        record = read_cell_status(dir_)
        cfgs = getattr(self, "cfgs", None)
        if record is None or self.output_subdir is None or not cfgs:
            return record, False
        if len(cfgs) != len(self.multirun_working_dirs):
            return record, False
        try:
            written = (dir_ / STATUS_FILE).stat().st_mtime_ns
            snapshot = (dir_ / self.output_subdir / "hydra.yaml").stat().st_mtime_ns
        except OSError:
            return record, False
        if snapshot <= written:
            return record, False
        cfg = cfgs[i]
        try:
            combo = self._combo_of_cfg(cfg, choices=self._job_choices(i))
            if combo_key(record["combo"]) != combo_key(combo):
                return None, True
        except Exception:  # noqa: BLE001 - not comparable: the record stands
            pass
        prior = record.get("config_hash")
        if isinstance(prior, str):
            current = resolved_config_hash(resolved_config(cfg))
            if current is not None and current != prior:
                return None, True
        return record, False

    def _job_choices(self, i: int) -> dict | None:
        """``runtime.choices`` for job ``i``, when jobs align with cfgs.

        Falls back to the choices `load_from_dir` recovered for each cell (from
        its status sidecar, else its hydra.yaml), so offline loads key
        config-group axes correctly."""
        jobs = getattr(self, "jobs", None)
        if jobs and len(jobs) == len(self.cfgs):
            return _group_choices_of_job(jobs[i])
        loaded = getattr(self, "_loaded_job_choices", None)
        if loaded is not None and len(loaded) == len(self.cfgs):
            return loaded[i]
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
        # Point at the new directory first: the setter refuses one that does
        # not exist, and then this workflow is left as it was.
        self.working_dir = Path(working_dir)
        # Reset memoized caches so a reused workflow object does not return
        # stale overrides/coordinates from a previously loaded directory.
        self._multirun_task_overrides = {}
        self._target_dir_multirun_overrides = None
        self._loaded_job_choices = None
        self._job_metrics_by_index = None
        # Unset until the job directories and their configs are read:
        # `to_xarray` and `load_metrics` reload when it is None, so a load that
        # fails partway is retried, not half-reused.
        self.multirun_working_dirs = None

        self.output_subdir = load_from_yaml(
            self.working_dir / "multirun.yaml"
        ).hydra.output_subdir

        job_dirs = [
            x.parent for x in self.working_dir.glob(f"**/*/{self.output_subdir}")
        ]
        if not job_dirs:
            if self.output_subdir is None:
                why = (
                    "it ran with hydra.output_subdir=null, so its jobs kept no "
                    "record to reload from"
                )
            else:
                why = (
                    "no job has started yet (a sweep still queued on a cluster), "
                    "the sweep was killed before its first job started, or its "
                    "job directories were removed"
                )
            raise FileNotFoundError(
                f"no job directories (**/*/{self.output_subdir}) under "
                f"{self.working_dir}: {why}."
            )

        # ensure working dirs are sorted by job num; alongside, recover each
        # job's `runtime.choices` so a config-GROUP axis keys by its chosen
        # option name offline exactly as it does in-process (without this,
        # `_combo_of_cfg` falls back to the composed sub-config and no combo
        # ever matches the choice-name grid coordinates). Both come from the
        # cell's status sidecar when `run` recorded them there, unless Hydra
        # wrote the cell's hydra.yaml after it (a reused directory whose new
        # job never reached its first status write); otherwise, as for a sweep
        # from before they were recorded, from hydra.yaml, whose parse is most
        # of a reload's time.
        from ._hydra_yaml import hydra_job_fields, load_hydra_record
        from ._resume import sidecar_job_fields

        hydra_yamls = [d / f"{self.output_subdir}/hydra.yaml" for d in job_dirs]
        fields = [
            sidecar_job_fields(d, y) for d, y in zip(job_dirs, hydra_yamls, strict=True)
        ]
        from_yaml = [i for i, f in enumerate(fields) if f is None]
        # Every file is loaded before any field is read, so a sweep with
        # several broken files reports the error it always did.
        _records = [load_hydra_record(hydra_yamls[i]) for i in from_yaml]
        for i, record in zip(from_yaml, _records, strict=True):
            fields[i] = hydra_job_fields(record)
        _job_nums, _choices = zip(*fields, strict=True)
        job_dirs = _sort_x_by_k(job_dirs, _job_nums)
        cfgs = []
        for dir_ in job_dirs:
            # Ensure we load saved YAML configurations for each job (in hydra.job.output_subdir)
            cfg_file = dir_ / f"{self.output_subdir}/config.yaml"
            if not cfg_file.exists():
                raise FileNotFoundError(
                    f"{cfg_file} not found; the sweep directory is incomplete or "
                    "was cleaned. Re-run the sweep, or point load_from_dir at a "
                    "directory whose cells still have their .hydra/config.yaml."
                )
            cfgs.append(load_from_yaml(cfg_file))
        self.multirun_working_dirs = job_dirs
        self._loaded_job_choices = _sort_x_by_k(list(_choices), _job_nums)
        self.cfgs = cfgs

        if metrics_filename is not None:
            self.load_metrics(metrics_filename)
        elif not self.jobs:
            # No metrics to load, so record the cells without a result from
            # their statuses, so `is_complete` holds before any `to_xarray`.
            self._record_cells_without_result(None)

        # Surface any sweep-level lineage (notes/tags) recorded in the manifest,
        # so a loaded sweep reports the same wf.notes/wf.tags it ran with. The
        # manifest itself is not attached: `is_complete` reads the cell record
        # `load_metrics` rebuilds.
        try:
            from ._sweep_io import Manifest

            _m = Manifest.load_or_new(self.working_dir, [])
            self._notes = _m.notes
            self._tags = list(_m.tags)
        except Exception:  # noqa: BLE001 - lineage is best-effort
            pass

        return self

    def _grid_keys(self) -> list[str]:
        """The key of every cell of the sweep's grid, in grid order, as
        `to_xarray` looks each cell up."""
        import itertools

        from ._sweep_io import combo_key

        swept = self._swept_param_names()
        overrides = self.multirun_task_overrides
        values = [self._sanitize_coordinate_for_xarray(overrides[n]) for n in swept]
        return [
            combo_key(dict(zip(swept, combo, strict=True)))
            for combo in itertools.product(*values)
        ]

    def _record_cells_without_result(self, job_metrics: list[Any] | None) -> None:
        """Rebuild `failures` and `skipped` for a sweep loaded from disk.

        `.run()` records them as it collects each job, and `to_xarray` turns them
        into the `mushin_failures` / `mushin_skipped` attrs that
        `compare_methods` checks. A loaded workflow has no jobs, so derive them
        from the data `to_xarray` assembles: a cell of the sweep's grid that no
        job dir completed or gives metrics has no result. It failed if its job
        dir's status says so; otherwise it was not run (skipped, still running,
        or never started, as in a sweep killed between cells).

        A loaded metrics file is a result, even an empty one. With
        ``job_metrics=None`` (loaded without metrics) only the statuses are read.
        A job dir without a status sidecar (from before sidecars) takes its
        status from the sweep manifest when the manifest lists that dir, and
        otherwise counts as completed, as `load_metrics` would load it.
        """
        from ._resume import STATUS_FILE
        from ._sweep_io import MANIFEST_FILE, Manifest, combo_key

        try:
            manifest_cells = Manifest.load_or_new(self.working_dir, []).cells
            manifest_mtime = (self.working_dir / MANIFEST_FILE).stat().st_mtime
        except Exception:  # noqa: BLE001 - the manifest is best-effort here
            manifest_cells, manifest_mtime = {}, 0.0

        with_result: set[str] = set()
        dirs: dict[str, Path] = {}
        failed: set[str] = set()
        if job_metrics is None:
            job_metrics = [None] * len(self.cfgs)
            no_sidecar_counts = True
        else:
            no_sidecar_counts = False
        for i, (dir_, cfg, metrics) in enumerate(
            zip(self.multirun_working_dirs, self.cfgs, job_metrics, strict=True)
        ):
            try:
                key = combo_key(self._combo_of_cfg(cfg, choices=self._job_choices(i)))
            except Exception:  # noqa: BLE001 - e.g. an earlier sweep's dir, over
                continue  # other axes; its grid cell then counts as without result
            sidecar, foreign = self._cell_record(i)
            if foreign:  # its job died before its first record: not run
                dirs[key] = dir_
                continue
            status = (sidecar or {}).get("status")
            if sidecar is None:  # a sweep from before sidecars: ask the manifest
                entry = manifest_cells.get(key, {})
                if entry.get("dir") == dir_.name:
                    status = entry.get("status")
                elif no_sidecar_counts:
                    status = "completed"
            # A loaded metrics file (even `{}`) is a result, as `.run()` counts it.
            if metrics is not None or status == "completed":
                with_result.add(key)
                continue
            dirs[key] = dir_
            if status == "failed":
                failed.add(key)

        try:
            grid = self._grid_keys()
        except Exception:  # noqa: BLE001 - a grid mushin cannot enumerate: record
            grid = [*dirs]  # only the cells whose job dirs it found
        self.failures, self.skipped = [], []
        for key in grid:
            if key in with_result:
                continue
            dir_ = dirs.get(key)
            wdir = None if dir_ is None else str(dir_.absolute())
            if key in failed:
                # The manifest's error text, if it describes this attempt: it
                # lists this dir, and was written after the cell's status (a
                # resume killed before its own manifest leaves the old one).
                entry = manifest_cells.get(key, {})
                current = (
                    entry.get("dir") == dir_.name
                    and _mtime(dir_ / STATUS_FILE) <= manifest_mtime
                )
                self.failures.append(
                    {
                        "combo": key,
                        "exception": entry.get("error") if current else None,
                        "working_dir": wdir,
                    }
                )
            else:
                self.skipped.append({"combo": key, "working_dir": wdir})

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

        job_metrics: list[Any] = []
        for i, dir_ in enumerate(self.multirun_working_dirs):
            # A cell recorded 'failed' or 'skipped' (fail-soft, `sample=`, or an
            # exhausted `max_total_seconds` budget) deliberately has no valid
            # metrics: NaN-fill it exactly like the in-process path — never load
            # a stale sidecar a prior attempt left behind, and never raise for
            # its (expected) missing files. A cell still 'running' or 'pending'
            # (a sweep reloaded mid-run, or killed mid-cell) has no result yet
            # either; `_record_cells_without_result` records it as not run. So
            # has a cell whose dir holds another cell's record (`_cell_record`).
            record, foreign = self._cell_record(i)
            status = (record or {}).get("status")
            if foreign or status in ("failed", "skipped", "running", "pending"):
                job_metrics.append(None)
                continue
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

        # Keep the job-aligned view (None = failed/skipped cell) so `to_xarray`
        # can rebuild `_metrics_by_combo` per cell even when some cells are
        # absent — the column-aligned `self.metrics` cannot express holes.
        self._job_metrics_by_index = job_metrics
        self.metrics = self._process_metrics(job_metrics)
        if not self.jobs:
            # A loaded workflow: rebuild the record of cells without a result,
            # and drop data cached from an earlier load so `to_xarray` rebuilds
            # it from these metrics and the two agree.
            self._metrics_by_combo = {}
            self._record_cells_without_result(job_metrics)

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
        """The sweep results as a tidy long-form ``pandas.DataFrame``.

        One row per sweep cell (times any extra metric dimensions), with the
        sweep parameters and metrics as plain columns: the pandas view of
        ``to_xarray`` (``to_xarray(...).to_dataframe().reset_index()``),
        for when you'd rather not touch xarray at all. Keyword arguments
        forward to ``to_xarray``. Dataset-level ``attrs`` (provenance,
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
        # workflow loaded from disk), reconstruct it per cell. Prefer the
        # job-aligned `_job_metrics_by_index` from `load_metrics` (it expresses
        # holes: None marks a failed/skipped cell, which simply NaN-fills).
        # Otherwise fall back to inverting the column-aligned `self.metrics`,
        # which is only sound when every column aligns 1:1 with `self.cfgs` —
        # `_process_metrics` drops None jobs / absent keys, so a ragged column
        # would mis-map; guard and only reconstruct full-length columns.
        if not self._metrics_by_combo and self.cfgs:
            n_jobs = len(self.cfgs)
            aligned = getattr(self, "_job_metrics_by_index", None)
            if aligned is not None and len(aligned) == n_jobs:
                for i, (cfg, per_job) in enumerate(
                    zip(self.cfgs, aligned, strict=True)
                ):
                    if per_job:
                        combo = self._combo_of_cfg(cfg, choices=self._job_choices(i))
                        self._metrics_by_combo[combo_key(combo)] = per_job
            elif all(len(col) == n_jobs for col in self.metrics.values()):
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
                    coords.update(metric_coords)
                else:
                    # Create additional arbitrary coordinates as-needed for non-scalar
                    # metrics
                    k_coords += [f"{k}_dim{n}"]
                    coords[f"{k}_dim{n}"] = np.arange(datum.shape[len(orig_coords) + n])

            data[k] = (k_coords, datum)

        coords.update(metric_coords)

        # Record fail-soft failures (combo keys of NaN-filled cells) on the
        # dataset. Only set when a fail-soft run actually recorded failures so a
        # fully-completed sweep's `attrs` is unchanged from prior behavior.
        # Stored as a JSON string: a raw list of strings is not a portable
        # netCDF attr (the scipy engine rejects it, and netCDF4 collapses a
        # 1-element list to a bare str on reload).
        if self.failures:
            attrs["mushin_failures"] = json.dumps([f["combo"] for f in self.failures])
        # Budget-skipped cells (NaN-filled, not errors) carry their own signal so
        # a reloaded dataset knows the sweep is incomplete-by-budget, distinct
        # from failures.
        if getattr(self, "skipped", None):
            attrs["mushin_skipped"] = json.dumps([s["combo"] for s in self.skipped])
        # Lineage annotations travel with the dataset (tags as a JSON string:
        # netCDF attrs cannot portably hold a list of strings).
        if self.notes is not None:
            attrs["mushin_notes"] = self.notes
        if self.tags:
            attrs["mushin_tags"] = json.dumps(self.tags)
        # Attach best-effort per-run provenance (git/versions/config) from one
        # job's sidecar, so a saved/re-loaded dataset carries its lineage. Stored
        # as a JSON string (nested dicts are not valid netCDF-serializable attrs).
        prov = self.provenance
        if prov is not None:
            attrs["provenance"] = json.dumps(prov)
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
        launcher_config: Mapping[str, Any] | None = None,
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

        launcher_config: Mapping[str, Any] | None (default: None)
            Fields for the selected ``launcher``, applied as
            ``hydra.launcher.<key>=<value>`` overrides: no hand-rolled
            strings. Pair with `mushin.slurm` for cells that each run as
            one SLURM job, e.g. ``run(launcher="submitit_slurm",
            launcher_config=mushin.slurm(gpus=1, partition="gpu"))``, or
            with `mushin.submitit_slurm_config` for DDP cells, a task per GPU.
            Requires ``launcher=``.

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
            launcher_config=launcher_config,
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
            # Save the figure `ax` belongs to — `plt.savefig` targets whatever
            # figure is current, which need not be this one when the caller
            # passed their own `ax`.
            ax.figure.savefig(save_filename)

        if created_fig and save_filename is not None:
            # We created the figure and the caller asked for a file, not a live
            # figure: close it so repeated plotting in one process doesn't
            # accumulate open figures (matplotlib warns past 20).
            plt.close(ax.figure)

        return plots
