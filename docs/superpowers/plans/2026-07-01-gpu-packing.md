# GPU Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pin_gpu_round_robin(num_gpus)` — a tiny opt-in helper that pins each sweep job to one GPU (round-robin via `CUDA_VISIBLE_DEVICES`) — plus a GPU-packing guide.

**Architecture:** A single pure-ish function in a new `src/mushin/_packing.py` (arithmetic + one env-var write + guards), exported from `mushin`. It does only device-pinning; the user sets job concurrency via their launcher (`hydra.launcher.n_jobs`). Fully CI-testable with monkeypatching — no GPU, no cluster gate.

**Tech Stack:** Python 3.10+, hydra-core (`HydraConfig`), torch (lazy import for the CUDA-initialized check), pytest, mkdocs, uv.

**Reference (read once before starting):**
- Spec: `docs/superpowers/specs/2026-07-01-gpu-packing-design.md`
- `HydraConfig` API (verified present): `from hydra.core.hydra_config import HydraConfig`; `HydraConfig.initialized() -> bool`; `HydraConfig.get().hydra.job.num` (mushin already reads `hydra.job.num` at `src/mushin/workflows.py:421`).
- Exports: `src/mushin/__init__.py` — `from ._utils import load_experiment, load_from_checkpoint` (line 6) and the `__all__` list (starts line 32, contains `"load_experiment"`).
- mkdocs nav: `mkdocs.yml` `- Guides:` block; new entry goes after `- Workflows & sweeps: guides/workflows.md`.

**Conventions:**
- Source files start with the two-line MIT copyright header (see `src/mushin/_utils.py`).
- After edits: `uv run ruff check <paths>` + `uv run ruff format <paths>` (target py310 — modern idioms, `X | None`).
- Commit messages imperative; **no Claude attribution / no `Co-Authored-By`**.
- Tests via `uv run pytest`. Do NOT `git add -A` (untracked `.worktrees/` present) — stage named files.

---

### Task 1: The `pin_gpu_round_robin` helper + unit tests

**Files:**
- Create: `src/mushin/_packing.py`
- Test: `tests/test_packing.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_packing.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
import os

import pytest


def test_round_robin_mapping(monkeypatch):
    from mushin._packing import pin_gpu_round_robin

    for job_index, expected in [(0, "0"), (1, "1"), (4, "0"), (5, "1")]:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        gpu = pin_gpu_round_robin(num_gpus=4, job_index=job_index)
        assert gpu == int(expected)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == expected


def test_num_gpus_must_be_positive():
    from mushin._packing import pin_gpu_round_robin

    with pytest.raises(ValueError, match="num_gpus"):
        pin_gpu_round_robin(num_gpus=0)
    with pytest.raises(ValueError, match="num_gpus"):
        pin_gpu_round_robin(num_gpus=-1)


def test_job_index_defaults_to_hydra_job_num(monkeypatch):
    from types import SimpleNamespace

    import hydra.core.hydra_config as hc

    from mushin._packing import pin_gpu_round_robin

    cfg = SimpleNamespace(hydra=SimpleNamespace(job=SimpleNamespace(num=3)))

    class _FakeHydraConfig:
        @staticmethod
        def initialized():
            return True

        @staticmethod
        def get():
            return cfg

    # the helper does `from hydra.core.hydra_config import HydraConfig` at call
    # time, so patching the attribute on the module is seen by the function.
    monkeypatch.setattr(hc, "HydraConfig", _FakeHydraConfig)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    gpu = pin_gpu_round_robin(num_gpus=2)  # 3 % 2 == 1
    assert gpu == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_no_active_hydra_raises(monkeypatch):
    import hydra.core.hydra_config as hc

    from mushin._packing import pin_gpu_round_robin

    class _NotInit:
        @staticmethod
        def initialized():
            return False

    monkeypatch.setattr(hc, "HydraConfig", _NotInit)
    with pytest.raises(RuntimeError, match="job_index"):
        pin_gpu_round_robin(num_gpus=2)


def test_warns_when_cuda_already_initialized(monkeypatch):
    import torch

    from mushin._packing import pin_gpu_round_robin

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.warns(UserWarning, match="already initialized"):
        gpu = pin_gpu_round_robin(num_gpus=2, job_index=1)
    assert gpu == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"  # still set
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_packing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mushin._packing'`.

- [ ] **Step 3: Implement the helper**

Create `src/mushin/_packing.py`:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Opt-in GPU packing: co-locate small sweep jobs on shared devices."""

from __future__ import annotations

import os
import warnings


def pin_gpu_round_robin(num_gpus: int, job_index: int | None = None) -> int:
    """Pin this process to a single GPU (round-robin) via ``CUDA_VISIBLE_DEVICES``.

    Call this at the TOP of your task function, before any CUDA use (importing
    torch is fine; the first CUDA op is not). It maps a job to one physical GPU
    with ``job_index % num_gpus`` and returns the chosen index.

    Parameters
    ----------
    num_gpus : int
        Number of physical GPUs to spread jobs across (must be >= 1).
    job_index : int or None
        The job's index. Defaults to the current Hydra sweep index
        (``HydraConfig.get().hydra.job.num``); pass it explicitly outside Hydra.

    Notes
    -----
    This only maps a job to a device. Running ``num_gpus * jobs_per_gpu`` jobs
    concurrently (so ``jobs_per_gpu`` land on each GPU) is set by your launcher,
    e.g. ``hydra.launcher.n_jobs``. Placement does not change results, only
    scheduling.
    """
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1; got {num_gpus}")

    if job_index is None:
        from hydra.core.hydra_config import HydraConfig

        if not HydraConfig.initialized():
            raise RuntimeError(
                "pin_gpu_round_robin: no active Hydra job to read the job index "
                "from. Pass job_index=... explicitly, or call this inside a Hydra "
                "(multirun) task function."
            )
        job_index = int(HydraConfig.get().hydra.job.num)

    gpu = job_index % num_gpus

    try:
        import torch

        if torch.cuda.is_initialized():
            warnings.warn(
                "pin_gpu_round_robin: CUDA is already initialized, so setting "
                "CUDA_VISIBLE_DEVICES now has no effect on this process. Call it at "
                "the top of your task function, before any CUDA use.",
                UserWarning,
                stacklevel=2,
            )
    except ImportError:  # pragma: no cover
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return gpu
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_packing.py -v`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/_packing.py tests/test_packing.py
uv run ruff format src/mushin/_packing.py tests/test_packing.py
git add src/mushin/_packing.py tests/test_packing.py
git commit -m "feat: add pin_gpu_round_robin GPU-packing helper"
```

---

### Task 2: Export `pin_gpu_round_robin`

**Files:**
- Modify: `src/mushin/__init__.py`
- Test: `tests/test_packing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_packing.py`:

```python
def test_pin_gpu_round_robin_exported():
    import mushin
    from mushin import pin_gpu_round_robin

    assert "pin_gpu_round_robin" in mushin.__all__
    assert callable(pin_gpu_round_robin)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_packing.py::test_pin_gpu_round_robin_exported -v`
Expected: FAIL — `cannot import name 'pin_gpu_round_robin' from 'mushin'`.

- [ ] **Step 3: Implement the export**

In `src/mushin/__init__.py`:
- After the line `from ._utils import load_experiment, load_from_checkpoint`, add:
  `from ._packing import pin_gpu_round_robin`
- Add `"pin_gpu_round_robin",` to the `__all__` list (put it right after `"load_from_checkpoint",`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_packing.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check src/mushin/__init__.py tests/test_packing.py
uv run ruff format src/mushin/__init__.py tests/test_packing.py
git add src/mushin/__init__.py tests/test_packing.py
git commit -m "feat: export pin_gpu_round_robin from mushin"
```

---

### Task 3: Docs guide + nav + changelog

**Files:**
- Create: `docs/guides/packing.md`
- Modify: `mkdocs.yml`
- Create: `changes/+gpu-packing.added.md`

- [ ] **Step 1: Write the guide**

Create `docs/guides/packing.md`:

````markdown
# Packing small jobs onto GPUs

When you sweep (Hydra `--multirun`) over **small** models that each fit
comfortably on one GPU, the default is **one job per GPU** — a model using ~10%
of a device still occupies the whole thing, wasting most of the cluster. Packing
runs **several small jobs per GPU** to use each device's full potential.

mushin does not assign GPUs — placement is decided by your **launcher** and by
Lightning/torch inside each job. So packing is a launcher/environment recipe.

## joblib / basic launcher: `pin_gpu_round_robin`

Pin each job to one GPU round-robin, and run more jobs concurrently than you have
GPUs. `pin_gpu_round_robin` sets `CUDA_VISIBLE_DEVICES` from the Hydra job index —
call it at the **top** of your task function, before any CUDA use:

```python
from mushin import pin_gpu_round_robin

def task(cfg):
    pin_gpu_round_robin(num_gpus=4)   # this job -> GPU (job_num % 4)
    # ... build the Trainer(devices=1, accelerator="gpu") and train ...
```

Then run the sweep with the joblib launcher and enough concurrency to place
`jobs_per_gpu` jobs on each GPU (here 4 GPUs x 3 jobs/GPU = 12 concurrent):

```bash
python train.py --multirun \
    hydra/launcher=joblib hydra.launcher.n_jobs=12 \
    model=a,b,c,d,e,f,g,h,i,j,k,l
```

`pin_gpu_round_robin` only maps a job to a device; the concurrency
(`n_jobs = num_gpus * jobs_per_gpu`) is your launcher setting.

## Ray: true fractional-GPU sharing (recommended for heavier sharing)

`hydra-ray-launcher` supports fractional GPUs natively — no mushin code needed:

```bash
python train.py --multirun hydra/launcher=ray \
    hydra.launcher.ray.remote.num_gpus=0.25   # 4 jobs share one GPU
```

Ray schedules the fractions for you; use it when you want many jobs truly
time-slicing a device rather than each pinned to a whole one.

## MPS and MIG

- **NVIDIA MPS** (Multi-Process Service) improves compute overlap when several
  small processes share a GPU — start `nvidia-cuda-mps-control -d` before the
  sweep. Combine with the round-robin pinning above.
- **MIG** (A100/H100) partitions one physical GPU into isolated instances; assign
  jobs to slices via `CUDA_VISIBLE_DEVICES=MIG-<uuid>` (MIG instances appear as
  devices), which `pin_gpu_round_robin` does not compute for you — set it directly.

## Caveats

- **Memory / compute contention:** co-located jobs share the device's memory and
  compute. Tune `jobs_per_gpu` down if you hit OOM.
- **Single-GPU-per-job only:** packing is for sweeps where each job uses one GPU.
  It is mutually exclusive with a job that itself claims multiple GPUs
  (`HydraDDP`, FSDP).
- **Reproducibility:** packing changes only *scheduling*, not results — a packed
  sweep produces the same numbers as one-job-per-GPU.
````

- [ ] **Step 2: Wire the nav**

In `mkdocs.yml`, in the `- Guides:` list, add this line immediately after `- Workflows & sweeps: guides/workflows.md` (match the 6-space indentation of sibling entries):

```yaml
      - Packing small jobs: guides/packing.md
```

- [ ] **Step 3: Changelog fragment**

Create `changes/+gpu-packing.added.md`:

```markdown
`pin_gpu_round_robin(num_gpus)`: an opt-in helper to pack several small sweep jobs
onto each GPU. Called at the top of a Hydra task, it sets `CUDA_VISIBLE_DEVICES`
to `job_index % num_gpus` so jobs round-robin across devices; run
`num_gpus * jobs_per_gpu` jobs concurrently (via your launcher's `n_jobs`) to
co-locate them. New "Packing small jobs onto GPUs" guide covers the joblib recipe,
Ray fractional-GPU, and MPS/MIG.
```

- [ ] **Step 4: Verify docs build**

Run: `uv run --group docs mkdocs build --strict && rm -rf site`
Expected: builds with no warnings/errors.

- [ ] **Step 5: Commit**

```bash
git add docs/guides/packing.md mkdocs.yml changes/+gpu-packing.added.md
git commit -m "docs: GPU-packing guide + changelog"
```

---

### Task 4: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: all pass; no new failures vs. `main`.

- [ ] **Step 2: Lint, format, spelling**

```bash
uv run ruff check .
uv run ruff format --check .
uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes
```
Expected: all clean. (If codespell flags a real word in the new files, prefer rewording; only add to `ignore-words-list` for genuine abbreviations.)

- [ ] **Step 3: Import smoke**

Run: `uv run python -c "from mushin import pin_gpu_round_robin; print(pin_gpu_round_robin(num_gpus=3, job_index=7))"`
Expected: `1` (7 % 3), and no error (sets `CUDA_VISIBLE_DEVICES=1`).

- [ ] **Step 4: Docs build (strict)**

Run: `uv run --group docs mkdocs build --strict && rm -rf site`
Expected: clean.

- [ ] **Step 5: Final formatting commit (if needed)**

```bash
git add -A ':!.worktrees'
git commit -m "chore: formatting/lint pass for GPU packing" || echo "nothing to commit"
```

---

## Done criteria

- `from mushin import pin_gpu_round_robin` works; in `__all__`.
- `pin_gpu_round_robin(num_gpus, job_index)` sets `CUDA_VISIBLE_DEVICES = job_index % num_gpus` and returns it; defaults `job_index` to `hydra.job.num`; raises on `num_gpus<1` and no-Hydra; warns if CUDA already initialized. All unit-tested without a GPU.
- `docs/guides/packing.md` in the nav; changelog fragment present.
- Full suite green; ruff/format/codespell/mkdocs `--strict` clean.
- **No hardware gate** — this merges to `main` via the normal CI + review gate.
