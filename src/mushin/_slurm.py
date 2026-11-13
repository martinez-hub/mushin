# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`mushin.slurm`: the resources of a sweep's SLURM jobs, one cell per job (#346).

Standard library only: `import mushin` loads this module, and the check below
runs only when the helper is called.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from typing import Any

#: How long the check waits for `scontrol` before leaving the request unjudged.
_SCONTROL_TIMEOUT_S = 10

#: `launcher_config` keys that would give a job several tasks: a job of one
#: cell runs one task, and DDP's several are `submitit_slurm_config`'s.
_SEVERAL_TASKS = ("nodes", "tasks_per_node", "ntasks_per_node")


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
    runs once (a job of several tasks runs the cell once per task). For
    multi-node DDP, where every GPU is a task, use
    ``mushin.submitit_slurm_config`` instead.

    Parameters
    ----------
    gpus : int
        GPUs per job (``--gpus-per-node``); 0, the default, requests none.
    cpus : int
        CPUs for the job's task (``--cpus-per-task``).
    mem_gb : int, optional
        Memory per job, in GB (``--mem``); unset leaves SLURM's default.
    timeout_min : int
        The job's time limit, in minutes (``--time``).
    partition, account, qos : str, optional
        Passed to SLURM as written; unset leaves SLURM's defaults (an
        ``SBATCH_PARTITION`` or ``SBATCH_ACCOUNT`` in the environment, then
        the cluster's).
    setup : list of str, optional
        Shell lines the job runs before the cell, such as ``module load cuda``
        or ``source .venv/bin/activate``, run as written. A line cannot hold
        ``${...}``, which Hydra reads as its own interpolation: write
        ``$HOME``, not ``${HOME}``.
    check : bool
        Check the request against the cluster when this is called: see below.
    **extra
        Other fields of hydra-submitit-launcher's SLURM config, passed through
        (``array_parallelism``, ``constraint``, ``additional_parameters``, ...).

    Returns
    -------
    dict
        The launcher config; nothing is submitted.

    Notes
    -----
    With ``check=True``, on a machine with ``scontrol`` (where ``run()`` will
    submit from), it refuses a request SLURM would reject or never start: a
    partition that does not exist, or a ``timeout_min`` past the partition's
    ``MaxTime`` (judged only without ``qos``, which can lift that limit). It
    names the partitions whose time limit fits, and never picks one for you.
    GPUs and accounts are not judged: ``scontrol`` lists a partition's GPUs
    only where the cluster accounts for them, and an account may run in a
    partition through its parent account. When ``scontrol`` is missing, slow
    or unreadable, and inside a job, nothing is judged.
    """
    for key in _SEVERAL_TASKS:
        if key in extra:
            raise ValueError(
                f"mushin.slurm runs each cell as a job of one task; {key}= would "
                "give a job several tasks, each running the cell. For multi-node "
                "DDP, where every GPU is a task, use mushin.submitit_slurm_config."
            )
    if int(gpus) < 0 or int(cpus) < 1 or int(timeout_min) < 1:
        raise ValueError(
            f"gpus must be >= 0, cpus and timeout_min >= 1; got gpus={gpus}, "
            f"cpus={cpus}, timeout_min={timeout_min}"
        )
    config: dict[str, Any] = {
        "nodes": 1,
        "tasks_per_node": 1,
        "cpus_per_task": int(cpus),
        "timeout_min": int(timeout_min),
    }
    if int(gpus) > 0:
        config["gpus_per_node"] = int(gpus)
    if mem_gb is not None:
        config["mem_gb"] = int(mem_gb)
    for key, value in (("partition", partition), ("account", account), ("qos", qos)):
        if value is not None:
            config[key] = value
    if setup is not None:
        braced = [line for line in setup if "${" in line]
        if braced:
            # run() writes launcher_config as Hydra overrides, and OmegaConf
            # takes `${...}` for its own interpolation; no escape survives
            # (`_quoted` doubles a backslash before it).
            raise ValueError(
                f"setup lines {braced!r} hold `${{`, which Hydra would read as "
                "its own interpolation rather than the shell's variable: write "
                "$VAR without braces."
            )
        config["setup"] = list(setup)
    config.update(extra)
    if check and not os.environ.get("SUBMITIT_EXECUTOR"):
        # Not inside a job: a module that builds its config at import is
        # imported by every job, which must not query the controller.
        _check(partition or os.environ.get("SBATCH_PARTITION"), int(timeout_min), qos)
    return config


def _check(partition: str | None, timeout_min: int, qos: str | None) -> None:
    """Refuse a partition SLURM does not have, or a time limit past its
    MaxTime, as `scontrol` states them; judge nothing it does not state."""
    partitions = _partitions()
    if partitions is None or (partition is not None and "," in partition):
        return  # no answer; or several partitions, for SLURM to choose among
    names = {p["PartitionName"]: p for p in partitions}
    if partition is None:
        defaults = [p for p in partitions if p.get("Default") == "YES"]
        if len(defaults) != 1:
            return
        target, what = (
            defaults[0],
            f"the default partition, {defaults[0]['PartitionName']!r}",
        )
    elif partition not in names:
        raise ValueError(
            f"SLURM has no partition {partition!r}. {_fitting(partitions, timeout_min)}"
            f" {_SKIP}"
        )
    else:
        target, what = names[partition], f"partition {partition!r}"
    limit = _minutes(target.get("MaxTime", ""))
    if qos is None and limit is not None and timeout_min > limit:
        raise ValueError(
            f"timeout_min={timeout_min} is past the time limit of {what} "
            f"(MaxTime={target['MaxTime']}). {_fitting(partitions, timeout_min)} {_SKIP}"
        )


_SKIP = "Pass check=False to submit without this check."


def _fitting(partitions: list[dict[str, str]], timeout_min: int) -> str:
    fit = [
        p["PartitionName"]
        for p in partitions
        if (limit := _minutes(p.get("MaxTime", ""))) is not None
        and timeout_min <= limit
    ]
    if not fit:
        return f"No partition here allows {timeout_min} minutes."
    return (
        f"Partitions whose time limit allows {timeout_min} minutes: {', '.join(fit)}."
    )


def _partitions() -> list[dict[str, str]] | None:
    """Every partition `scontrol` reports, hidden ones too, as its fields; None
    when there is no `scontrol`, it fails, it takes too long, or it reports
    nothing readable."""
    exe = shutil.which("scontrol")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--all", "--oneliner", "show", "partition"],
            capture_output=True,
            text=True,
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
