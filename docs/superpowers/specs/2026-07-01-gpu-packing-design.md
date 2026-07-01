# GPU packing for small sweep jobs — Design

**Status:** Approved for planning
**Date:** 2026-07-01
**Issue:** #46. **Branch base:** `main`. Fully CI-testable — **no cluster gate**, merges to main normally.

## Context

Sweeping (Hydra `--multirun`) over **small** models spreads jobs one-per-GPU, leaving each device mostly idle (a model using ~10% of a GPU still occupies the whole device). mushin does not assign GPUs — it only forwards a launcher choice as a Hydra override (`hydra/launcher={launcher}`, `workflows.py:363`); device placement is decided by the launcher's concurrency + how it sets `CUDA_VISIBLE_DEVICES`, and by Lightning/torch inside each job. So packing is a launcher/environment behavior mushin currently passes through.

The two real packing routes:
- **Ray** (`hydra-ray-launcher`) supports fractional GPUs (`num_gpus=0.25` → 4 jobs truly share a device). Cleanest primitive, but a heavy dependency + Ray-cluster setup, and needs **no mushin code**.
- **basic/joblib launcher** (the common default) has no packing: you pin each job to a whole GPU round-robin (`CUDA_VISIBLE_DEVICES = job_index % num_gpus`) and run `num_gpus × jobs_per_gpu` jobs concurrently. The fiddly bit is setting `CUDA_VISIBLE_DEVICES` from the Hydra job index, before torch initializes CUDA.

## Goal

Let users pack N small jobs per GPU on the **basic/joblib** launcher via a tiny, opt-in helper that removes the `CUDA_VISIBLE_DEVICES`-round-robin boilerplate, plus a guide covering the joblib recipe, Ray (recommended for true sharing), and MPS/MIG.

## Non-Goals

- Assigning or managing devices automatically — the helper is opt-in and does one thing.
- Owning launcher concurrency config (`jobs_per_gpu` as a `Study` option) — concurrency stays the user's `hydra.launcher.n_jobs=…` override, which mushin already forwards.
- A Ray convenience — Ray does fractional GPUs natively; documented only.
- Multi-GPU-per-job packing — packing is for single-GPU-per-job sweeps, mutually exclusive with `HydraDDP`/FSDP jobs.

## Architecture

### `pin_gpu_round_robin` helper

New module `src/mushin/_packing.py`, exported from `mushin`:

```python
def pin_gpu_round_robin(num_gpus: int, job_index: int | None = None) -> int:
    """Pin this process to a single GPU (round-robin) by setting
    ``CUDA_VISIBLE_DEVICES``. Call at the TOP of your task function, before any
    CUDA use (importing torch is fine; the first CUDA op is not).

    ``job_index`` defaults to the current Hydra sweep index
    (``HydraConfig.get().hydra.job.num``); pass it explicitly outside Hydra.
    Returns the chosen GPU index. Concurrency (running ``num_gpus * jobs_per_gpu``
    jobs at once) is set by your launcher (e.g. ``hydra.launcher.n_jobs``), not
    here — this only maps a job to a device.
    """
```

Behavior:
- `gpu = job_index % num_gpus`; `os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)`; `return gpu`.
- `job_index` defaults to `HydraConfig.get().hydra.job.num` (mushin already reads this at `workflows.py:421`, so the access pattern is established).

Guards / error handling:
- `num_gpus < 1` → `ValueError("num_gpus must be >= 1")`.
- `job_index is None` and no active `HydraConfig` (`HydraConfig.initialized()` is False, or `.get()` raises) → `RuntimeError` telling the user to pass `job_index=` or run under Hydra.
- `torch.cuda.is_initialized()` already True → `warnings.warn(..., UserWarning)` that `CUDA_VISIBLE_DEVICES` won't take effect after CUDA init (still sets it and returns, for the non-CUDA-yet case). Import torch lazily inside the function so the module has no hard torch-at-import cost beyond what mushin already pays.

### Exports

`pin_gpu_round_robin` added to `src/mushin/__init__.py` imports + `__all__` (next to `load_experiment`).

## Data flow

Unchanged elsewhere. In a joblib sweep, each job process calls `pin_gpu_round_robin(num_gpus=N)` at the top of its task function → its `CUDA_VISIBLE_DEVICES` is set to one physical GPU → Lightning/torch then see a single device and train there. The user runs `hydra/launcher=joblib` with `hydra.launcher.n_jobs = N * jobs_per_gpu` so `jobs_per_gpu` jobs land on each GPU. Results are identical to a one-per-GPU run — only scheduling changes.

## Testing (hermetic, no GPU)

`tests/test_packing.py`:
- Round-robin mapping: for `num_gpus=4`, `job_index` in `{0,1,4,5}` → `CUDA_VISIBLE_DEVICES` in `{"0","1","0","1"}` and return value matches; use `monkeypatch.setenv`/`delenv` for isolation.
- `num_gpus=0` (and negative) → `ValueError`.
- `job_index=None` with `HydraConfig` monkeypatched so `initialized()` True and `get().hydra.job.num` returns e.g. 3 → sets/returns `3 % num_gpus`.
- `job_index=None` with no active Hydra (`HydraConfig.initialized()` False) → `RuntimeError`.
- Already-initialized CUDA: monkeypatch `torch.cuda.is_initialized` → True, assert `pytest.warns(UserWarning)` and that it still sets the env var.
- Export test: `from mushin import pin_gpu_round_robin`; in `mushin.__all__`.

No hardware gate — every path is exercised on CPU with monkeypatching.

## Docs

`docs/guides/packing.md` (+ `mkdocs.yml` nav):
- The problem (tiny jobs waste a whole GPU each).
- **joblib recipe**: `pin_gpu_round_robin(num_gpus=N)` at the task top; run `hydra/launcher=joblib` with `hydra.launcher.n_jobs=N*jobs_per_gpu`; a runnable snippet.
- **Ray** (recommended for true intra-GPU sharing): `hydra/launcher=ray` + `num_gpus=0.25`; no mushin code.
- **MPS / MIG** pointers (compute overlap / hardware partitioning) with one-line setup notes.
- Caveats: OOM/contention (tune `jobs_per_gpu`); single-GPU-per-job only (not with `HydraDDP`/FSDP); reproducibility unaffected (placement only).
- `changes/+gpu-packing.added.md` fragment.

## Build order

One spec, one plan: `pin_gpu_round_robin` + unit tests → export + export test → docs guide + nav + changelog → verification. Small, additive, no hardware gate — merges to main via the normal CI + review gate.
