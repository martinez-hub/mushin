# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Run a method x seed training sweep via MultiRunMetricsWorkflow and recover
checkpoint paths deterministically."""

from __future__ import annotations

import os
import shutil
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
    on_error: str = "raise",
) -> dict[str, list[str]]:
    """Run ``methods[name](seed)`` for every (name, seed) via a Hydra sweep.

    Each call returns the path to a saved checkpoint; the job relocates it to
    ``ckpt_dir/m{method_index}__seed{seed}.ckpt`` so paths are recoverable
    without relying on Hydra job ordering. Returns ``{method: [path_per_seed]}``.

    The sweep is run over integer method *indices* rather than the method names
    themselves: a name that Hydra would parse as another scalar (e.g. ``"1"`` or
    ``"true"``) or that contains a comma would otherwise be reinterpreted or
    split as a Hydra override. Indices are unambiguous and are mapped back to
    names here.

    ``on_error`` is forwarded to the underlying ``MultiRunMetricsWorkflow.run``
    (``"raise"`` — default, abort on first failure; ``"nan"`` — fail-soft, keep
    going and record failures). Regardless of ``on_error``, this function never
    returns checkpoints for an incomplete sweep: if the workflow finishes with
    any recorded failures (``wf.is_complete`` is ``False``), it raises
    ``IncompleteSweepError`` instead, so ``Study.run`` can never proceed to
    evaluate/compare checkpoints from a sweep that did not fully complete.
    """
    ckpt_dir = Path(ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    names = list(methods)

    def task(method_index, seed):
        name = names[method_index]
        src = methods[name](seed)
        if src is None:
            raise ValueError(
                f"train_fn for method={name!r} seed={seed} returned no checkpoint path"
            )
        dest = ckpt_dir / f"m{method_index}__seed{seed}.ckpt"
        shutil.move(str(src), str(dest))
        return {"checkpoint": str(dest)}

    sweep_cls = type(
        "_StudySweep", (MultiRunMetricsWorkflow,), {"task": staticmethod(task)}
    )
    wf = sweep_cls()
    wf.run(
        method_index=multirun(list(range(len(names)))),
        seed=multirun(seeds),
        working_dir=working_dir,
        on_error=on_error,
    )

    if not wf.is_complete:
        # local import: keep `import mushin` from pulling in the benchmark
        # subsystem (see mushin/study/_load.py for the same pattern).
        from mushin.benchmark import IncompleteSweepError

        failed = [f["combo"] for f in wf.failures]
        raise IncompleteSweepError(
            f"{len(failed)} run(s) failed ({', '.join(map(str, failed))}); "
            "fix the cause and re-run with resume=True to complete the sweep "
            "before comparing."
        )

    return {
        names[i]: [str(ckpt_dir / f"m{i}__seed{s}.ckpt") for s in seeds]
        for i in range(len(names))
    }
