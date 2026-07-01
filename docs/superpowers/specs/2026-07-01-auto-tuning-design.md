# Reproducibility-preserving auto-tuning — Design

**Status:** Approved for planning
**Date:** 2026-07-01
**Issue:** #47. **Branch base:** `main`. Hermetic (Tuner monkeypatched) — **no cluster gate**, merges to main normally.

## Context

PyTorch Lightning's `Tuner.scale_batch_size()` finds the largest *device* batch that fits in memory, and `Tuner.lr_find()` runs an LR range test. A mushin user can already call both directly, but the naive use conflicts with mushin's identity: batch size and learning rate are results-affecting hyperparameters, and a finder that maximizes the *device* batch to whatever fits the *current* GPU makes the same config produce different results on a 24 GB vs 80 GB card — breaking "same config → same result" and confounding sweep comparisons.

This feature adds two small, opt-in helpers that keep the convenience while protecting reproducibility, by **decoupling the device batch (memory/throughput) from the scientifically meaningful effective batch**, and by **recording found values to a sidecar pin file so re-runs reuse them deterministically ("found once, then pinned")** rather than re-discovering each run.

## Decisions (locked during brainstorming)

1. **API surface:** standalone opt-in helpers (like `pin_gpu_round_robin`), called inside the user's task/`train_fn` where they build the Trainer — not a declarative config flag. Reaches both the `zen(task)` path and the user-owned `Study` `TrainFn` path.
2. **Persistence:** a sidecar pin file (YAML via OmegaConf). If present, the helper reads the pinned value and **skips the search**. mushin does not patch Hydra's `.hydra/config.yaml` (Hydra owns it, writes it before the task runs, and DDP subprocesses read it early).
3. **Batch math:** **maximize the device batch, approximate the effective batch**, with bounded drift — when the max-that-fits already meets the per-device target, use exactly that with no accumulation; only when accumulation is *required* does integer rounding introduce a small drift, which is **recorded and warned on**.
4. **Scope:** batch size **and** learning rate, both record-and-pin.

## Architecture

New module `src/mushin/_tuning.py` (sibling to `src/mushin/_packing.py`), exporting `tune_batch_size` and `tune_learning_rate`, plus internal shared pin-file read/write and small result dataclasses. Both helpers are exported from `mushin` (`__init__.py` + `__all__`).

Each helper follows the same control flow:

```
if pin file exists and not retune:
    read pinned value → apply it → return (no search)
else:
    run Lightning Tuner (real steps) → derive values → write pin file → apply → return
```

### `tune_batch_size`

```python
def tune_batch_size(
    trainer: Trainer,
    module: LightningModule,
    datamodule: LightningDataModule | None = None,
    *,
    effective_batch_size: int,
    pin_path: str | Path | None = None,   # default: <trainer.log_dir>/mushin_batch_pin.yaml
    num_devices: int | None = None,       # default: derive from trainer
    safety_margin: float = 0.0,           # back off the found max by this fraction (OOM noise)
    batch_arg: str = "batch_size",        # attr on datamodule/module the tuner scales
    retune: bool = False,                 # force re-search even if a pin exists
    **scale_kwargs,                       # forwarded to Tuner.scale_batch_size
) -> BatchPin:
    ...
```

`BatchPin` (frozen dataclass): `device_batch: int`, `accumulate_grad_batches: int`, `effective_batch_size: int` (the **actual** realized value), `num_devices: int`, `drift: int` (actual − requested).

**Behavior:**
- `num_devices` := the argument, else derived from the trainer (`trainer.num_devices`), min 1.
- `per_device_total = effective_batch_size / num_devices`; if it does not divide evenly → `ValueError` naming both numbers and suggesting a divisible target.
- If a pin file exists and `retune` is False: read `device_batch` from it and **skip the search** (deterministic re-run). Otherwise:
  - `B_max = Tuner(trainer).scale_batch_size(module, datamodule=datamodule, **scale_kwargs)`.
  - Apply safety margin: `B = max(1, floor(B_max * (1 - safety_margin)))`.
- `device_batch = min(B, per_device_total)`.
  - If `B >= per_device_total`: `device_batch = per_device_total`, `accumulate_grad_batches = 1` → **exact** effective batch, no accumulation.
  - Else: `accumulate_grad_batches = max(1, round(per_device_total / device_batch))`.
- `actual_effective = device_batch * accumulate_grad_batches * num_devices`; `drift = actual_effective - effective_batch_size`. If `drift != 0`, `warnings.warn(...)` reporting requested vs actual.
- If a near-prime target forces `device_batch` far below `B` (e.g. `accumulate` becomes very large), `warn` suggesting a more composite `effective_batch_size`.
- **Apply:** set `datamodule.<batch_arg> = device_batch` (or `module.<batch_arg>` when the datamodule lacks it, mirroring the tuner's own attr lookup); set accumulation on the trainer (mechanism is a TDD'd implementation detail — see Risks); write the pin file with `device_batch`, requested `effective_batch_size`, and `num_devices` recorded at tune time.

### `tune_learning_rate`

```python
def tune_learning_rate(
    trainer: Trainer,
    module: LightningModule,
    datamodule: LightningDataModule | None = None,
    *,
    pin_path: str | Path | None = None,   # default: <trainer.log_dir>/mushin_lr_pin.yaml
    lr_attr: str = "lr",                  # attr on the module set to the found LR
    retune: bool = False,
    **lr_find_kwargs,                     # forwarded to Tuner.lr_find
) -> LRPin:
    ...
```

`LRPin` (frozen dataclass): `learning_rate: float`.

**Behavior:** learning rate is hardware-independent, so there is no device math — pinning simply makes the stochastic range test skip on re-runs and reuse the exact found value. If a pin file exists and not `retune`: read `learning_rate`, set `module.<lr_attr>`, return. Otherwise: `lr = Tuner(trainer).lr_find(module, datamodule=datamodule, **lr_find_kwargs).suggestion()`; if `suggestion()` is `None` → `RuntimeError` telling the user the range test found no clear suggestion (widen the range / add steps); else write the pin, set `module.<lr_attr> = lr`, return.

### Shared pin-file machinery

Internal `_read_pin(path) -> dict | None` and `_write_pin(path, mapping)` using OmegaConf (`OmegaConf.save` / `OmegaConf.load`), already a mushin dependency. `_read_pin` returns `None` when the file is absent. Pin files are small, human-readable YAML and safe to commit alongside a config.

## Data flow

A user calls the helper at the top of their task/`train_fn`, after constructing the Trainer/module/datamodule but before `trainer.fit(...)`. First run: the Tuner runs real steps, the found values are written to the sidecar and applied to the datamodule/trainer/module, and training proceeds. Subsequent runs: the sidecar is read, the search is skipped, the same values are applied — identical, deterministic behavior regardless of hardware (as long as the pinned device batch still fits). Placement/throughput changes with hardware; the pinned scientific quantity does not.

## Error handling

- `effective_batch_size` not divisible by `num_devices` → `ValueError`.
- `safety_margin` outside `[0, 1)` → `ValueError`.
- `effective_batch_size < 1` or `num_devices < 1` → `ValueError`.
- `lr_find` returns no suggestion → `RuntimeError`.
- Non-zero effective-batch drift, or a near-prime target forcing a tiny device batch → `warnings.warn(UserWarning)` (not fatal).

## Testing (hermetic, no GPU)

`tests/test_tuning.py`, mirroring `tests/test_packing.py`'s monkeypatch style — no hardware, every path on CPU:

- **Batch, exact path:** monkeypatch `Tuner.scale_batch_size` → returns `B_max >= per_device_total`; assert `device_batch == per_device_total`, `accumulate == 1`, `drift == 0`, datamodule attr set.
- **Batch, accumulation path:** `B_max < per_device_total`; assert `device_batch == B`, `accumulate == round(...)`, and the recorded `actual_effective`/`drift`.
- **Drift warning:** a `B` that does not divide `per_device_total` → `pytest.warns(UserWarning)` and recorded actual ≠ requested.
- **Safety margin:** margin backs off `B_max`; out-of-range margin → `ValueError`.
- **num_devices:** explicit override changes `per_device_total`; non-divisible → `ValueError`.
- **Pin round-trip:** first call writes the sidecar (monkeypatched search invoked once); second call reads it and the search is **not** called (assert via a call counter); `retune=True` forces the search again.
- **LR:** monkeypatch `Tuner.lr_find` to return an object whose `.suggestion()` is a float; assert `module.<lr_attr>` set and pin written; second call skips the search; `suggestion() is None` → `RuntimeError`.
- **Exports:** `from mushin import tune_batch_size, tune_learning_rate`; both in `mushin.__all__`.

The Tuner is always monkeypatched, so `scale_batch_size`/`lr_find` never run real steps in CI. The **real** Tuner-application path (does setting accumulation on a live Trainer take effect; does the datamodule attr propagate) is verified once in the Linux/Docker container during implementation (torch is capped at 2.2.2 on the Intel Mac), and captured as a documented, version-checked mechanism.

## Docs

`docs/guides/auto-tuning.md` (+ `mkdocs.yml` nav):
- Why the naive `scale_batch_size` breaks reproducibility; the decouple-and-pin model.
- `tune_batch_size` recipe: call it before `fit`, pin the effective batch, commit the sidecar; note the recorded actual effective batch and drift.
- `tune_learning_rate` recipe: record-and-pin the LR finder.
- Caveats: opt-in and explicit (runs real steps, mutates then resets trainer/model state); tune on a single device, pinned values apply at scale; a pinned device batch that no longer fits on smaller hardware is a deliberate re-tune (`retune=True`) decision; not on-by-default.
- `changes/+auto-tuning.added.md` fragment.

## Risks / verification points

1. **Applying `accumulate_grad_batches` to an already-constructed Trainer.** Lightning normally takes it at `Trainer(...)` construction. The plan will TDD the exact mechanism against the installed Lightning version — mutating the attribute, or injecting/replacing a `GradientAccumulationScheduler` callback — rather than assume the attribute is live.
2. **Torch/Lightning version sensitivity.** Tuner internals and the attr-application contract vary by version; verify in the Linux/Docker container, not only on the torch-2.2.2-capped Intel Mac.

## Non-goals

- A declarative `Trainer`/`Study` config flag — helpers only for now.
- Automatic DDP-tuner orchestration — documented recipe (tune on one device; pinned values apply at scale).
- Re-searching every run — the sidecar exists precisely to avoid that.
- Any coupling to `pin_gpu_round_robin` — packing and tuning are independent opt-in helpers.

## Build order

`_tuning.py` shared pin I/O + `BatchPin`/`LRPin` → `tune_batch_size` (math, search-or-pin, apply) + unit tests → `tune_learning_rate` + unit tests → exports + export test → docs guide + nav + changelog → container verification of the real apply path → final review. Small, additive, no hardware gate — merges to main via the normal CI + review gate.
