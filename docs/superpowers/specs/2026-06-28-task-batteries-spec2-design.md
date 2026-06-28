# More Task Batteries + `update_fn` Hook (Spec 2 of 2) — Design

**Status:** Approved for planning
**Date:** 2026-06-28
**Builds on:** Spec 1 (public Task API, merged in PR #51 / `d7b6250`).

## Context

Spec 1 made tasks first-class: a public `Task` dataclass, `register_task`/`get_task`/`list_tasks`, and `compare`/`Study` accepting a `Task` or a registered name. It deliberately deferred (a) new built-in batteries beyond classification/segmentation/detection and (b) the `update_fn` hook for metrics whose `.update()` signature is not `(preds, target)`.

This spec delivers both: four new built-in batteries — **regression, image_quality, audio, retrieval** — and the **`update_fn`** hook (designed against retrieval, the one domain the `(preds, target)` streaming contract cannot express).

torchmetrics spans ~14 domains; we are not bundling all of them. These four are the curated additions; the long tail remains reachable via a user `Task`.

## Goals

1. Add a per-`Task` `update_fn` hook so a battery can own its `.update()` dispatch.
2. Ship four new built-in batteries registered in `_TASKS`, reachable via `compare(task=...)`/`Study(task=...)`.
3. Put dependency-heavy metrics (LPIPS, PESQ, STOI) behind optional install extras, detection-style.
4. Zero behavior change to existing tasks; the default update path stays byte-for-byte identical.

## Non-Goals

- Distribution-level metrics (FID, KID, Inception Score) — they need accumulate-two-sets-then-compute, not per-sample streaming. Out of scope; documented as unsupported.
- Other torchmetrics domains (text, clustering, multimodal, nominal, pairwise, shape, video) — reachable via a custom `Task`, not bundled.
- Any change to the Spec 1 registry/resolution surface.

## Architecture

### 1. The `update_fn` hook

Add one optional field to `Task` (`src/mushin/benchmark/_tasks.py`), keeping it a frozen dataclass:

```python
update_fn: UpdateFn | None = None
```

with, in `src/mushin/benchmark/_inference.py`:

```python
UpdateFn = Callable[[dict[str, Metric], torch.Tensor, Optional[torch.Tensor], object], None]
# (battery, preds, probs, target) -> None ; owns all metric.update() calls for one batch
```

`evaluate(...)` gains an `update_fn: UpdateFn | None = None` parameter. The marked seam becomes:

```python
if update_fn is None:
    def update_fn(battery, preds, probs, target):
        for name, metric in battery.items():
            metric.update(probs if name in prob_metrics else preds, target)
...
for x, y in data:
    x = _to_device(x, device); y = _to_device(y, device)
    preds, probs = predict_fn(model, x)
    update_fn(battery, preds, probs, y)
```

When `update_fn is None`, behavior is identical to today (the default closure captures `prob_metrics`). `compare(...)` passes `spec.update_fn` through to `evaluate`. The `update_fn` is taken from the resolved `Task` regardless of a `metrics=` override (the task defines *how* to update; the override only changes *which* metrics). Documented caveat: overriding `metrics=` on the `retrieval` task requires retrieval-shaped metrics, since `_retrieval_update` calls `update(preds, target, indexes=...)`.

### 2. The four batteries

All live in `src/mushin/benchmark/_metrics.py` (battery factories) with default predict_fns in `src/mushin/benchmark/_predict.py`, and are registered in `_TASKS` (`src/mushin/benchmark/_tasks.py`). All four set `requires_num_classes=False` and `prob_metrics=frozenset()`. The battery factory signature stays `(num_classes=None, ignore_index=None)` for the uniform interface (args unused by these batteries).

**regression** — no new deps (core `torchmetrics.regression`):

```python
{
  "mse": MeanSquaredError(),
  "mae": MeanAbsoluteError(),
  "rmse": MeanSquaredError(squared=False),
  "r2": R2Score(),
  "pearson": PearsonCorrCoef(),
  "spearman": SpearmanCorrCoef(),
}
```
Default predict_fn: `(model(x), None)`. `target = y` is a continuous tensor.

**image_quality** — core `torchmetrics.image` (SSIM/PSNR/MS-SSIM) + LPIPS behind `[image]`:

```python
{
  "ssim": StructuralSimilarityIndexMeasure(),
  "psnr": PeakSignalNoiseRatio(),
  "ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(),
  "lpips": LearnedPerceptualImagePatchSimilarity(),   # requires [image]
}
```
Default predict_fn: `(model(x), None)` (the generated image). `target = y` is the reference image.

**audio** — core `torchmetrics.audio` (SI-SDR/SI-SNR) + PESQ/STOI behind `[audio]`:

```python
{
  "si_sdr": ScaleInvariantSignalDistortionRatio(),
  "si_snr": ScaleInvariantSignalNoiseRatio(),
  "pesq": PerceptualEvaluationSpeechQuality(fs=16000, mode="wb"),   # requires [audio]
  "stoi": ShortTimeObjectiveIntelligibility(fs=16000),             # requires [audio]
}
```
Default predict_fn: `(model(x), None)` (estimated waveform). `target = y` is the reference waveform. (PESQ/STOI need a sample rate; default to 16 kHz wideband — documented, overridable via a custom `Task`.)

**retrieval** — no new deps (core `torchmetrics.retrieval`); uses `update_fn`:

```python
{
  "retrieval_map": RetrievalMAP(),
  "ndcg": RetrievalNormalizedDCG(),
  "mrr": RetrievalMRR(),
  "precision": RetrievalPrecision(),
  "recall": RetrievalRecall(),
}
```
Default predict_fn: `(model(x), None)` (relevance scores). Batches yield `y = (relevance, indexes)`. `update_fn = _retrieval_update`:

```python
def _retrieval_update(battery, preds, probs, target):
    relevance, indexes = target
    for metric in battery.values():
        metric.update(preds, relevance, indexes=indexes)
```

Exact torchmetrics class names / constructor kwargs / floors are verified against the installed torchmetrics version during implementation; the names above are the design intent.

### 3. The "all-or-nothing battery" rule for optional extras

`image_quality` and `audio` include their optional metrics and **raise a clear `ImportError` naming the extra if it is not installed** — same all-or-nothing contract as `detection_battery` today (lazy import inside the factory, `try/except ImportError` → `pip install mushin-py[image]` / `[audio]`). A battery is a curated, complete set; a silently-partial battery is confusing. Users who want only the core metrics (e.g. SSIM alone) pass `metrics={...}` or build a custom `Task`. The core metrics themselves (SSIM/PSNR/MS-SSIM, SI-SDR/SI-SNR) need no extra, but the built-in battery as a whole does — this is intentional and documented.

### 4. Packaging & CI

`[project.optional-dependencies]` in `pyproject.toml` gains:

- `image` — the dependency `LearnedPerceptualImagePatchSimilarity` needs (torchvision + `lpips`; exact package set and floors verified at build, platform-gated where a C-extension is painful, mirroring the pycocotools Windows gate).
- `audio` — `pesq` and `pystoi` (both C/native; platform-gated as needed; the battery raises a clear error if used without them).

The `test` CI job installs `--extra image --extra audio` (joining `--extra detection`) and runs pytest with them, so the optional-dep metrics are exercised in CI. The `min-versions` job stays lean (no new extras), keeping honest floors for the core. New deps are isolated from the dev group exactly as detection's are.

## Data flow

Unchanged from Spec 1 except the update step: `compare` resolves the `Task`, builds the battery (or uses `metrics=`), and calls `evaluate(model, data, battery, predict_fn, prob_metrics, device, update_fn=spec.update_fn)`. `evaluate` streams batches and calls `update_fn(battery, preds, probs, y)` per batch (default closure or the task's). Aggregation/stats/`BenchmarkResult` are untouched.

## Error handling

- Missing extra: lazy import inside `image_quality`/`audio` factories; `except ImportError` re-raised as a clear "install mushin-py[image]/[audio]" message (detection precedent).
- Retrieval target shape: `_retrieval_update` unpacks `y = (relevance, indexes)`; a wrong shape raises a `ValueError`/`TypeError` from the unpack or from torchmetrics — documented in the guide so users supply the 2-tuple.
- Building a `Task(update_fn=...)` is unconstrained (any callable); contract documented.

## Testing

Hermetic, synthetic tensors; no real data, no GPU. Per battery:

- **regression / retrieval** (no extras): always run in CI and locally. Tiny tensors; for retrieval, a small `(scores, relevance, indexes)` batch over 2–3 queries, asserting the metrics land in `result.data` and the `update_fn` path works end-to-end via `compare(task="retrieval", ...)`.
- **image_quality / audio**: core-metric tests always run; the optional-dep metrics (LPIPS/PESQ/STOI) are guarded with `pytest.importorskip(...)` so they run only when the extra is installed (CI's `test` job has them; bare installs skip cleanly).
- **update_fn unit test**: a `Task` with a custom `update_fn` confirms `evaluate` routes through it, and that `update_fn=None` is byte-for-byte the old behavior (regression guard).
- **all-or-nothing error test**: constructing `image_quality`/`audio` without the extra raises the clear `ImportError` (simulated by monkeypatching the import, matching how detection's missing-extra path is tested).
- Public-surface test: `list_tasks()` now contains the four new names with descriptions; each reachable via `compare(task="<name>")` and `register_task` round-trips.

## Docs

- `docs/guides/custom.md` (or a new short guide): a per-domain table of what `predict_fn` returns and what `target` is, including retrieval's `(relevance, indexes)` 2-tuple and the `update_fn` extension point. Note the optional extras and the all-or-nothing battery rule.
- `changes/+task-batteries.added.md` towncrier fragment.
- Reference docs pick up new public symbols via mkdocstrings.

## Build order

One spec, one plan (~8–9 tasks): `update_fn` hook → regression → retrieval (+update_fn) → image_quality (+`[image]` extra) → audio (+`[audio]` extra) → register tasks → pyproject/CI extras → docs/changelog → full verification. The `update_fn` hook lands first (regression/image/audio don't need it; retrieval does), then the batteries, then packaging, then docs.
