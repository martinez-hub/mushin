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
    resume: bool = False,
    capture_env: bool = False,
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

    ``resume`` and ``capture_env`` are forwarded to the workflow's ``run`` as
    well: ``resume=True`` (requires a stable ``working_dir``) re-executes only
    the failed/missing cells of a prior sweep, and ``capture_env=True`` writes a
    full dependency snapshot alongside the per-job provenance records. Because
    cells are swept by method *index*, resume additionally fingerprints the
    methods mapping itself (each name paired with its function's source hash):
    renaming, reordering, adding/removing a method, or editing any method's
    body invalidates every completed cell (with a warning) rather than silently
    reusing checkpoints trained by different code — coarse, but never stale.
    Like the core resume guard, the fingerprint covers each function's own
    source only, not helpers it calls or values it closes over.
    """
    ckpt_dir = Path(ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    names = list(methods)

    # One fingerprint over (name, identity) pairs IN ORDER. Passed to the
    # task as a config value so the per-cell config fingerprint — the resume
    # guard — covers the methods mapping, which the swept (method_index, seed)
    # combo alone cannot see.
    import functools
    import hashlib
    import warnings

    from mushin._resume import code_fingerprint

    def _method_identity(fn) -> str:
        """A string that changes when the method's code (or bound arguments)
        change. Falls back through source-less callables: a functools.partial
        keys on its wrapped function plus bound args; a callable object keys
        on its class source plus instance state. '?' means undetectable."""
        fp = code_fingerprint(fn)
        if fp is not None:
            return fp
        if isinstance(fn, functools.partial):
            kw = sorted((fn.keywords or {}).items())
            return f"partial:{_method_identity(fn.func)}:{fn.args!r}:{kw!r}"
        cls_fp = code_fingerprint(type(fn))
        if cls_fp is not None:
            try:
                state = repr(sorted(vars(fn).items()))
            except Exception:  # noqa: BLE001 - unorderable/unreprable state
                state = "?"
            return f"obj:{cls_fp}:{state}"
        return "?"

    _identities = {n: _method_identity(methods[n]) for n in names}
    _blind = sorted(n for n, ident in _identities.items() if "?" in ident)
    if _blind:
        warnings.warn(
            f"resume cannot detect code changes for method(s) "
            f"{', '.join(map(repr, _blind))}: their source is unreadable and no "
            "fallback identity applies. Re-run from a fresh working_dir after "
            "editing them.",
            UserWarning,
            stacklevel=2,
        )
    _fp_src = ",".join(f"{n}:{_identities[n]}" for n in names)
    methods_fp = hashlib.sha256(_fp_src.encode()).hexdigest()[:16]

    def task(method_index, seed, methods_fingerprint=""):
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
        methods_fingerprint=methods_fp,
        working_dir=working_dir,
        on_error=on_error,
        resume=resume,
        capture_env=capture_env,
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
