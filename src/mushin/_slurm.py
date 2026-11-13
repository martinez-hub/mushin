# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.slurm`: the resources of a sweep's SLURM jobs, one cell per job (#346).

Standard library only: `import mushin` loads this module, and the check below
runs only when the helper is called.
"""

from __future__ import annotations

import math
import numbers
import os
import re
import shutil
import subprocess
import warnings
from typing import Any

#: How long the check waits for `scontrol` before leaving the request unjudged.
_SCONTROL_TIMEOUT_S = 10

#: Fields the helper sets from its own arguments, or SLURM's other ways to
#: ask for the same, which sbatch refuses beside them (`--cpus-per-gpu` with
#: `--cpus-per-task`, `--mem-per-*` with `--mem`): given in ``extra`` they
#: would override the arguments unchecked. Its named arguments cannot reach
#: ``extra`` at all.
_OWN_FIELDS = {
    "nodes": "a job of one task has one node",
    "tasks_per_node": "a job of one task",
    "cpus_per_task": "cpus=",
    "gpus_per_node": "gpus=",
    "cpus_per_gpu": "cpus=",
    "gpus_per_task": "gpus=",
    "mem_per_cpu": "mem_gb=",
    "mem_per_gpu": "mem_gb=",
    "gres": "gpus=",
}

#: sbatch options in ``additional_parameters`` (written last into the job
#: script, so they win) that would redo the helper's fields, give a job
#: several tasks (any `ntasks...`), or send it where the check does not look.
_REDONE = ("nodes", "cpus", "gpus", "gres", "mem", "account")
_ELSEWHERE = ("partition", "time", "qos", "clusters", "cluster")

#: sbatch reads these from its environment over the job script's #SBATCH
#: lines, so they decide where and how long a job runs (sbatch's INPUT
#: ENVIRONMENT VARIABLES).
_ENV = {
    "partition": "SBATCH_PARTITION",
    "account": "SBATCH_ACCOUNT",
    "qos": "SBATCH_QOS",
    "time": "SBATCH_TIMELIMIT",
}

#: Resolvers a setup line may call: Hydra registers these when a sweep
#: starts (`setup_globals`), so they may not be registered yet here.
_HYDRA_RESOLVERS = ("hydra", "now", "python_version")


def slurm(
    *,
    gpus: int = 0,
    cpus: int | None = None,
    mem_gb: int | None = None,
    timeout_min: int = 60,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    setup: list[str] | None = None,
    check: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """The ``launcher_config`` of a sweep that runs each cell as one SLURM job.

    Pass it with ``launcher="submitit_slurm"`` (the ``slurm`` extra)::

        experiment.run(
            lr=mushin.multirun([0.01, 0.1]),
            launcher="submitit_slurm",
            launcher_config=mushin.slurm(gpus=1, timeout_min=120, partition="gpu"),
        )

    Each job runs one task, with ``gpus`` GPUs and ``cpus`` CPUs, so a cell
    runs once (a job of several tasks runs the cell once per task). A cell
    that trains with DDP, on one node or several, runs a task per GPU: build
    its config with ``mushin.submitit_slurm_config`` instead.

    Parameters
    ----------
    gpus : int
        GPUs per job (``--gpus-per-node``); 0, the default, requests none.
    cpus : int, optional
        CPUs for the job's task (``--cpus-per-task``); unset leaves SLURM's
        default (which can depend on the GPUs asked for).
    mem_gb : int, optional
        Memory per job, in whole GB (``--mem``); unset leaves SLURM's default.
    timeout_min : int
        The job's time limit, in minutes (``--time``).
    partition, account, qos : str, optional
        Written into each job's script; unset leaves SLURM's defaults. An
        ``SBATCH_PARTITION``, ``SBATCH_ACCOUNT``, ``SBATCH_QOS`` or
        ``SBATCH_TIMELIMIT`` in the environment that submits the jobs
        overrides the script (sbatch reads them first); one that overrides an
        argument given here warns.
    setup : list of str, optional
        Shell lines each job runs before the cell, such as ``module load
        cuda`` or ``source .venv/bin/activate``. A shell variable is written
        ``$HOME``, not ``${HOME}``: Hydra reads every ``${...}`` as its own
        interpolation, so only its resolvers (``${oc.env:HOME}``,
        ``${hydra:runtime.cwd}``) may appear, in a line without a backslash
        (Hydra's override writer doubles it there).
    check : bool
        Check the request against the cluster when this is called: see below.
    **extra
        Other fields of hydra-submitit-launcher's SLURM config, passed through
        (``array_parallelism``, ``constraint``, ``additional_parameters``,
        ...), but none that redoes an argument above.

    Returns
    -------
    dict
        The launcher config; nothing is submitted.

    Notes
    -----
    With ``check=True``, on a machine with ``scontrol`` (where ``run()`` will
    submit from), it refuses a request SLURM would reject or never start, as
    ``scontrol`` states it: a partition that does not exist, or a time limit
    past the partition's ``MaxTime``. It names the partitions whose time limit
    fits, and never picks one for you. It judges only a partition named here
    or in ``SBATCH_PARTITION`` (a job that names none can be routed by the
    cluster), and no time limit under a QoS, which can lift it (one that the
    account's default QoS lifts is not seen: pass ``check=False``). GPUs and
    accounts are not judged: ``scontrol`` lists a partition's GPUs only where
    the cluster accounts for them, and an account may run in a partition
    through its parent account. Nothing is judged when ``scontrol`` is
    missing, slow or unreadable, when ``additional_parameters`` sets the
    partition, time, QoS or cluster, or inside a job. The answer is kept for
    the process, so the cluster is asked once.
    """
    for key, instead in _OWN_FIELDS.items():
        if key in extra:
            raise ValueError(
                f"mushin.slurm sets {key} itself ({instead}). For multi-node or "
                "DDP jobs, where every GPU is a task, use "
                "mushin.submitit_slurm_config."
            )
    # sbatch's options, as submitit writes them: `ntasks-per-node` and
    # `ntasks_per_node` are one option.
    sbatch_options = {
        str(key).replace("-", "_") for key in extra.get("additional_parameters") or {}
    }
    redone = sorted(
        key
        for key in sbatch_options
        if key.startswith("ntasks") or key.startswith(_REDONE)
    )
    if redone:
        raise ValueError(
            f"additional_parameters {redone!r} would redo what mushin.slurm sets "
            "(a job of one task, and its gpus=, cpus=, mem_gb= and account=), "
            "written after it in the job script. For DDP, where every GPU is a "
            "task, use mushin.submitit_slurm_config."
        )
    gpus = _whole("gpus", gpus, 0)
    timeout_min = _whole("timeout_min", timeout_min, 1)
    config: dict[str, Any] = {
        "nodes": 1,
        "tasks_per_node": 1,
        "timeout_min": timeout_min,
    }
    if cpus is not None:
        config["cpus_per_task"] = _whole("cpus", cpus, 1)
    if gpus > 0:
        config["gpus_per_node"] = gpus
    if mem_gb is not None:
        # Whole GB, at least one: `--mem=0` asks SLURM for a whole node's.
        config["mem_gb"] = _whole("mem_gb", mem_gb, 1)
    named = {"partition": partition, "account": account, "qos": qos}
    for key, value in named.items():
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a non-empty string, got {value!r}")
            config[key] = value
    if setup is not None:
        config["setup"] = _setup_lines(setup)
    config.update(extra)
    if os.environ.get("SUBMITIT_EXECUTOR"):
        # Inside a job: a module that builds its config at import is imported
        # by every job, which must neither query the controller nor warn.
        return config
    given = {**named, "time": timeout_min}
    environ = {key: os.environ.get(name) for key, name in _ENV.items()}
    for key, value in given.items():
        if value is not None and environ[key] and environ[key] != str(value):
            warnings.warn(
                f"{_ENV[key]}={environ[key]!r} in this environment overrides "
                f"{'timeout_min' if key == 'time' else key}={value!r}: sbatch "
                "reads its environment over the job script.",
                UserWarning,
                stacklevel=2,
            )
    elsewhere = sbatch_options & set(_ELSEWHERE) or os.environ.get("SBATCH_CLUSTERS")
    if check and not elsewhere:
        _check(
            environ["partition"] or partition,
            _minutes(environ["time"]) if environ["time"] else timeout_min,
            environ["qos"] or qos,
            source=_ENV["partition"] if environ["partition"] else None,
        )
    return config


def _whole(name: str, value: Any, least: int) -> int:
    """``value`` as an int of at least ``least``: an integer (NumPy's too) or
    a whole float; a fraction or an infinity is refused, not truncated."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if isinstance(value, numbers.Integral):
        whole = int(value)
    elif (
        isinstance(value, numbers.Real) and math.isfinite(value) and value == int(value)
    ):
        whole = int(value)
    else:
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if whole < least:
        raise ValueError(f"{name} must be a whole number >= {least}, got {value!r}")
    return whole


def _setup_lines(setup: Any) -> list[str]:
    if not isinstance(setup, (list, tuple)) or not all(
        isinstance(line, str) for line in setup
    ):
        raise ValueError(f"setup must be a list of shell lines, got {setup!r}")
    for line in setup:
        # run() writes launcher_config as Hydra overrides, and OmegaConf takes
        # every `${...}` for its own interpolation; no escape survives
        # (`_quoted` doubles a backslash before it, and every other backslash
        # in such a line).
        for head in _INTERPOLATION_HEAD.findall(line):
            if head not in _HYDRA_RESOLVERS and not _has_resolver(head):
                raise ValueError(
                    f"setup line {line!r} holds a ${{...}} that is not one of "
                    "Hydra's resolvers, and Hydra would read it as its own "
                    'interpolation: write a shell variable as $VAR (or "$VAR"'
                    "suffix), or ${oc.env:VAR} to have Hydra read it."
                )
        if "${" in line and "\\" in line:
            raise ValueError(
                f"setup line {line!r} holds both ${{...}} and a backslash, which "
                "Hydra's override writer doubles in such a line."
            )
    return list(setup)


#: The head of each interpolation in a line: a resolver's name before its
#: `:`, or whatever else opens it (a key, a shell expansion).
_INTERPOLATION_HEAD = re.compile(r"\$\{([^:}]*)")


def _has_resolver(name: str) -> bool:
    from omegaconf import OmegaConf

    return bool(name) and OmegaConf.has_resolver(name)


def _check(
    partition: str | None,
    minutes: float | None,
    qos: str | None,
    *,
    source: str | None = None,
) -> None:
    """Refuse a partition SLURM does not have, or a time limit past its
    MaxTime, as `scontrol` states them; judge nothing it does not state.
    ``source`` names the environment variable the partition came from."""
    if partition is None or "," in partition:
        return  # none (the cluster may route it), or several for SLURM to pick
    partitions = _partitions()
    if partitions is None:
        return
    names = {p["PartitionName"]: p for p in partitions}
    if partition not in names:
        origin = f" (from {source})" if source else ""
        raise ValueError(
            f"SLURM has no partition {partition!r}{origin}. "
            f"{_fitting(partitions, minutes)}{_SKIP}"
        )
    target = names[partition]
    limit = _minutes(target.get("MaxTime", ""))
    if qos is None and limit is not None and minutes is not None and minutes > limit:
        raise ValueError(
            f"A time limit of {minutes:g} minutes is past partition {partition!r}'s "
            f"(MaxTime={target['MaxTime']}). {_fitting(partitions, minutes)}{_SKIP}"
        )


_SKIP = "Pass check=False to submit without this check."


def _fitting(partitions: list[dict[str, str]], minutes: float | None) -> str:
    if minutes is None:
        return ""
    fit = [
        p["PartitionName"]
        for p in partitions
        if (limit := _minutes(p.get("MaxTime", ""))) is not None and minutes <= limit
    ]
    if not fit:
        return f"No partition here allows {minutes:g} minutes. "
    return f"Partitions whose time limit allows {minutes:g} minutes: {', '.join(fit)}. "


#: `scontrol`'s answer, once it has given one: asked once per process.
_ANSWER: list[dict[str, str]] | None = None


def _partitions() -> list[dict[str, str]] | None:
    """Every partition `scontrol` reports, hidden ones too, as its fields; None
    when there is no `scontrol`, it fails, it takes too long, or it reports
    nothing readable. A readable answer is kept for the process; no answer is
    asked again next time."""
    global _ANSWER
    if _ANSWER is not None:
        return _ANSWER
    exe = shutil.which("scontrol")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--all", "--oneliner", "show", "partition"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SCONTROL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    _ANSWER = _parse_partitions(out.stdout)
    return _ANSWER


def _parse_partitions(text: str) -> list[dict[str, str]] | None:
    """`scontrol --oneliner show partition` output: a record per line, of
    space-separated ``Key=value`` fields."""
    partitions = []
    for line in text.splitlines():
        fields = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        if fields.get("PartitionName"):
            partitions.append(fields)
    return partitions or None


#: SLURM's time formats: minutes, minutes:seconds, hours:minutes:seconds,
#: days-hours, days-hours:minutes, days-hours:minutes:seconds.
_DAYS = re.compile(r"(\d+)-(\d+)(?::(\d+))?(?::(\d+))?")
_CLOCK = re.compile(r"(\d+)(?::(\d+))?(?::(\d+))?")


def _minutes(text: str) -> float | None:
    """A SLURM time limit in minutes (``inf`` for UNLIMITED), or None when
    it is not one of SLURM's formats."""
    if text.upper() in ("UNLIMITED", "INFINITE"):
        return math.inf
    if m := _DAYS.fullmatch(text):
        days, hours, minutes, seconds = (int(g or 0) for g in m.groups())
        return days * 1440 + hours * 60 + minutes + seconds / 60
    if m := _CLOCK.fullmatch(text):
        parts = [int(g) for g in m.groups() if g is not None]
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return parts[0] + parts[1] / 60
        hours, minutes, seconds = parts
        return hours * 60 + minutes + seconds / 60
    return None
