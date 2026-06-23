# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Run a method x seed training sweep via MultiRunMetricsWorkflow and recover
checkpoint paths deterministically."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Union

from hydra_zen import multirun  # import from source, not `mushin`, to avoid a

from mushin.workflows import (
    MultiRunMetricsWorkflow,  # circular import via mushin/__init__
)

TrainFn = Callable[[int], Union[str, "os.PathLike[str]"]]


def run_training_sweep(
    methods: dict[str, TrainFn],
    seeds: Sequence[int],
    ckpt_dir: str | os.PathLike[str],
    working_dir: str | None = None,
) -> dict[str, list[str]]:
    """Run ``methods[name](seed)`` for every (name, seed) via a Hydra sweep.

    Each call returns the path to a saved checkpoint; the job relocates it to
    ``ckpt_dir/{method}__seed{seed}.ckpt`` so paths are recoverable without
    relying on Hydra job ordering. Returns ``{method: [path_per_seed]}``.
    """
    ckpt_dir = Path(ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)

    def task(method, seed):
        src = methods[method](seed)
        dest = ckpt_dir / f"{method}__seed{seed}.ckpt"
        os.replace(src, dest)
        return {"checkpoint": str(dest)}

    sweep_cls = type(
        "_StudySweep", (MultiRunMetricsWorkflow,), {"task": staticmethod(task)}
    )
    wf = sweep_cls()
    wf.run(
        method=multirun(list(methods)), seed=multirun(seeds), working_dir=working_dir
    )

    return {m: [str(ckpt_dir / f"{m}__seed{s}.ckpt") for s in seeds] for m in methods}
