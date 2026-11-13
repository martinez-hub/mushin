# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.slurm`: the resources of a sweep's SLURM jobs, one cell per job (#346).

Standard library only: `import mushin` loads this module, and the check below
runs only when the helper is called.
"""

from __future__ import annotations

import functools
import math
import os
import re
import shutil
import subprocess
import warnings
from typing import Any

#: How long the check waits for `scontrol` before leaving the request unjudged.
_SCONTROL_TIMEOUT_S = 10

#: Fields the helper sets from its own arguments; given again in ``extra``
#: they would override them unchecked. `cpus_per_gpu` and `gpus_per_task`
#: are SLURM's other way to ask for the same, which sbatch refuses beside
#: `--cpus-per-task` and `--gpus-per-node`.
_OWN_FIELDS = {
    "nodes": "a job of one task has one node",
    "tasks_per_node": "a job of one task",
    "cpus_per_task": "cpus=",
    "gpus_per_node": "gpus=",
    "timeout_min": "timeout_min=",
    "mem_gb": "mem_gb=",
    "partition": "partition=",
    "account": "account=",
    "qos": "qos=",
    "setup": "setup=",
    "cpus_per_gpu": "cpus=",
    "gpus_per_task": "gpus=",
}

#: sbatch options in ``additional_parameters`` (written last into the job
#: script) that would give a job several tasks, or that send it somewhere the
#: check does not look.
_SEVERAL_TASKS = ("nodes", "ntasks", "ntasks_per_node")
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

#: A shell parameter expansion, ``${NAME}`` or ``${NAME:-word}`` and the like,
#: which Hydra would read as its own interpolation (a key ``NAME``).
_SHELL_PARAMETER = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::?[-=?+][^}]*)?\}")


def slurm(
    *,
    gpus: int = 0,
    cpus: int = 1,
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
    cpus : int
        CPUs for the job's task (``--cpus-per-task``).
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
        ``$HOME``, not ``${HOME}``, which Hydra would read as its own
        interpolation (``"$HOME"suffix`` where text follows it).
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
    sbatch_options = extra.get("additional_parameters") or {}
    several = [key for key in _SEVERAL_TASKS if key in sbatch_options]
    if several:
        raise ValueError(
            f"additional_parameters {several!r} would give a job several tasks, "
            "each running the cell: mushin.slurm runs a cell as a job of one task. "
            "For DDP, use mushin.submitit_slurm_config."
        )
    gpus = _whole("gpus", gpus, 0)
    cpus = _whole("cpus", cpus, 1)
    timeout_min = _whole("timeout_min", timeout_min, 1)
    config: dict[str, Any] = {
        "nodes": 1,
        "tasks_per_node": 1,
        "cpus_per_task": cpus,
        "timeout_min": timeout_min,
    }
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
    effective = {
        key: os.environ.get(_ENV[key]) or value for key, value in named.items()
    }
    for key, value in named.items():
        if value is not None and effective[key] != value:
            warnings.warn(
                f"{_ENV[key]}={effective[key]!r} in this environment overrides "
                f"{key}={value!r}: sbatch reads its environment over the job "
                "script.",
                UserWarning,
                stacklevel=2,
            )
    elsewhere = [key for key in _ELSEWHERE if key in sbatch_options]
    if check and not elsewhere and not os.environ.get("SUBMITIT_EXECUTOR"):
        # Not inside a job: a module that builds its config at import is
        # imported by every job, which must not query the controller.
        _check(
            effective["partition"],
            _minutes(os.environ[_ENV["time"]])
            if os.environ.get(_ENV["time"])
            else timeout_min,
            effective["qos"],
        )
    return config


def _whole(name: str, value: Any, least: int) -> int:
    """``value`` as an int of at least ``least``; a fraction is refused, not
    truncated."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if value != int(value) or int(value) < least:
        raise ValueError(f"{name} must be a whole number >= {least}, got {value!r}")
    return int(value)


def _setup_lines(setup: Any) -> list[str]:
    if isinstance(setup, str) or not all(isinstance(line, str) for line in setup):
        raise ValueError(f"setup must be a list of shell lines, got {setup!r}")
    braced = [line for line in setup if _SHELL_PARAMETER.search(line)]
    if braced:
        # run() writes launcher_config as Hydra overrides, and OmegaConf takes
        # `${NAME}` for its own interpolation; no escape survives (`_quoted`
        # doubles a backslash before it).
        raise ValueError(
            f"setup lines {braced!r} hold a shell variable in braces, which Hydra "
            'would read as its own interpolation: write $VAR (or "$VAR"suffix).'
        )
    return list(setup)


def _check(partition: str | None, minutes: float | None, qos: str | None) -> None:
    """Refuse a partition SLURM does not have, or a time limit past its
    MaxTime, as `scontrol` states them; judge nothing it does not state."""
    if partition is None or "," in partition:
        return  # none (the cluster may route it), or several for SLURM to pick
    partitions = _partitions()
    if partitions is None:
        return
    names = {p["PartitionName"]: p for p in partitions}
    if partition not in names:
        raise ValueError(
            f"SLURM has no partition {partition!r}. {_fitting(partitions, minutes)}"
            f"{_SKIP}"
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


@functools.lru_cache(maxsize=1)
def _partitions() -> list[dict[str, str]] | None:
    """Every partition `scontrol` reports, hidden ones too, as its fields; None
    when there is no `scontrol`, it fails, it takes too long, or it reports
    nothing readable. Asked once per process."""
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
    return _parse_partitions(out.stdout)


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
