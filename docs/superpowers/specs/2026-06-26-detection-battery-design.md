# Object-detection benchmark battery — design

- **Date:** 2026-06-26
- **Issue:** #44
- **Status:** approved (brainstorm complete; pending implementation plan)

## Motivation

mushin ships task-specific benchmark batteries that feed the shared
`(method × seed) → BenchmarkResult` significance spine: classification
(`compare_methods`), semantic segmentation (mIoU battery), and LLM systems
(`compare_llms`). Object detection is the obvious next per-task battery, and
`torchmetrics.detection` provides a family of ready bounding-box metrics. This
adds a `task="detection"` battery so users can compare trained detection models
across reproducible seeds and report Holm-corrected significance — the same
"is this difference real, or sampling noise?" guarantee, applied to detection.

## Goals

- `compare(models, data, task="detection")` and `Study` work for detection,
  mirroring the classification/segmentation entry points.
- Wrap the **whole bounding-box detection family** from `torchmetrics.detection`
  and surface **every** value each metric returns as an xarray data variable,
  each with its own per-seed significance.
- Hermetic CI tests proving the integration is numerically correct, plus an
  optional gated real-dataset end-to-end validation.

## Non-goals (YAGNI)

- Training detection models (this is the evaluation/comparison layer only).
- `PanopticQuality` / `ModifiedPanopticQuality` — panoptic-segmentation metrics
  with different (map-based) I/O; they belong in a future panoptic battery, not
  the bbox battery.
- A standalone "predictions-in" entry (`compare_detections(preds=...)`) — can be
  added later; the model-streaming path is the primary entry.
- Per-class AP (`class_metrics=True`) by default — expensive; left off.

## Decisions (settled in brainstorm)

1. **Entry point:** the existing model-streaming path —
   `compare(models, data, task="detection", ...)` + `Study`.
2. **Metric set:** all `torchmetrics.detection` **bbox** metrics, full outputs
   verbatim in the xarray (see table below). `PanopticQuality` excluded.
3. **Architecture:** generalize the shared `evaluate`/`compute_battery` path to be
   metric-shape-agnostic (a recursive `_to_device` + dict-valued-metric
   expansion) rather than forking a detection-specific streaming loop.
4. **Sentinel handling:** torchmetrics/COCO returns `-1.0` for buckets with no
   matching ground truth (e.g. `map_small` when no small objects). Normalize
   `-1.0` → `NaN` in mAP/mAR outputs so the significance machinery treats those
   buckets as *missing* (already handled by the Holm/NaN path) rather than a
   misleading real `-1` score. All keys are still exposed.

## Architecture & data flow

```
compare(models, data, task="detection")
  → get_task_spec("detection")
        → TaskSpec(detection_battery, default_detection_predict_fn, prob_metrics=∅)
  → evaluate(model, data, battery, predict_fn, prob_metrics):
        for x, y in data:                       # x: list[Tensor] images; y: list[dict] targets
            x = _to_device(x, dev); y = _to_device(y, dev)
            preds, _ = predict_fn(model, x)      # preds: list[dict] {boxes, scores, labels}
            for m in battery.values(): m.update(preds, y)
        raw = {name: m.compute() for name, m in battery.items()}   # values may be dict OR scalar
        → _expand(raw)        # dict values → one entry per key; scalars kept; -1 → NaN
  → to_dataset(per method/seed) → compare_methods → BenchmarkResult(.summary/.comparisons/.data)
```

Because every bbox metric shares the same `list[dict]` I/O, a **single streaming
pass updates all of them**; no metric needs probabilities (`prob_metrics` is
empty for detection).

## Components

### 1. `detection_battery(num_classes=None, ignore_index=None)` — `_metrics.py` (new)

Returns the full bbox family:

| battery key | class | data variables it contributes |
|---|---|---|
| `map` | `MeanAveragePrecision` | `map`, `map_50`, `map_75`, `map_small`, `map_medium`, `map_large`, `mar_1`, `mar_10`, `mar_100`, `mar_small`, `mar_medium`, `mar_large` |
| `iou` | `IntersectionOverUnion` | `iou` |
| `giou` | `GeneralizedIntersectionOverUnion` | `giou` |
| `ciou` | `CompleteIntersectionOverUnion` | `ciou` |
| `diou` | `DistanceIntersectionOverUnion` | `diou` |

- `box_format` defaults to `"xyxy"` (torchvision convention); exposed as a
  passthrough.
- `num_classes` / `ignore_index` are accepted for the uniform task-interface
  signature but unused (mAP infers classes from `labels`); documented, not warned.
- Metric classes are **imported lazily** inside this function so the dependency is
  only required when the detection task is actually used.

### 2. `default_detection_predict_fn(model, x)` — `_predict.py` (new)

Returns `(model(x), None)` — the torchvision detection convention (an eval-mode
detector maps `list[Tensor]` → `list[dict]` with `boxes`/`scores`/`labels`). Users
override `predict_fn` for non-torchvision detectors (DETR, YOLO, …).

### 3. `_to_device(obj, device)` — shared helper (new)

Recursively moves tensors to `device` through tensors, lists/tuples, and dicts;
anything else passes through. Replaces `evaluate`'s tensor-only
`x.to(device); y.to(device)` so the one streaming loop serves both tensor tasks
and detection's `list[dict]` shapes. Tensor tasks are unaffected (a bare tensor
still hits the `Tensor → .to(device)` branch).

### 4. Generic dict-valued-metric expansion — `_inference.py` / `_metrics.py`

When a battery metric's `compute()` returns a **dict**, emit one output per key
using the metric's **own key names** (`map`, `map_50`, `iou`, …, which are
distinct across this family); a **scalar** `compute()` is emitted under the
battery key (current behavior). During expansion, mAP/mAR values equal to the
COCO `-1.0` sentinel are converted to `NaN`. This mirrors the dict-expansion
already used in the LLM path, now applied to the benchmark path.

### 5. Registry entry — `_tasks.py`

```python
"detection": TaskSpec(detection_battery, default_detection_predict_fn, frozenset()),
```

## I/O contract (documented)

- **Targets** (`y`, per image): `dict` with `boxes` `[N,4]` (xyxy float), `labels`
  `[N]` int.
- **Predictions** (`predict_fn` output, per image): `dict` with `boxes` `[M,4]`,
  `scores` `[M]`, `labels` `[M]`.
- A batch is a `list` of these dicts (one per image); images `x` are a
  `list[Tensor]` (variable size allowed). mushin does not reshape — same "shape
  your data to the metric" stance as the LLM path.

## Optional dependency

- Add a `detection` extra: `mushin-py[detection]` → `torchmetrics[detection]`
  (+ `pycocotools` / `torchvision` as required by the installed torchmetrics).
- `detection_battery`'s lazy import raises a clear `ImportError` naming the extra
  if the detection metrics are unavailable.
- `torchmetrics >= 1.0` (mushin's existing floor) provides all five classes.

## Error handling & edge cases

- **Missing/odd shapes** surface torchmetrics' own errors (we don't reshape).
- **All-NaN data variable** (e.g. `map_small` when no targets carry small
  objects, after `-1 → NaN`): must not crash aggregation or stats. `to_dataset`
  builds the column; `compare_methods`/Holm already skip NaN as missing →
  reported non-significant with NaN p-value. Covered by a test.
- **Empty predictions** for an image (`M = 0`): valid; metrics handle it
  (contributes to lowered recall/precision).
- **Key collisions** across metrics: none in this family; if a future metric
  reused a key, `to_dataset`'s ragged-check would surface it.

## Testing

**Hermetic (CI, always):**
- *Reference equivalence* — feed known `preds`/`targets` through `detection_battery`
  via `evaluate` and assert each value equals calling the underlying torchmetrics
  metric directly on the same data (proves our wiring/`_to_device`/expansion don't
  corrupt numbers; torchmetrics is the oracle).
- *Perfect predictions* (preds == targets, score 1.0) → `map == 1.0`, `iou == 1.0`.
- *Disjoint predictions* → `map ≈ 0`, low IoU.
- *Sentinel* — a target set with no small objects → `map_small` is `NaN` (not
  `-1.0`) and does not break significance.
- *Dict expansion* — the expected data variables appear; *`_to_device`* moves a
  `list[dict[str, Tensor]]` correctly; *no regression* — classification/
  segmentation tests still pass (tensor path unchanged).
- *End-to-end* — `compare(task="detection")` over two tiny synthetic "models"
  across seeds → `BenchmarkResult` with the full set of detection data variables.

**Gated real-dataset (optional, not in CI):**
- Marked (e.g. `@pytest.mark.slow`/network) and skipped by default: pull a small
  real COCO sample + a pretrained torchvision detector, run
  `compare(task="detection")` end-to-end, and assert the resulting mAP lands in a
  plausible published range. Runnable manually to confirm real-world correctness.

## Files touched

- `src/mushin/benchmark/_metrics.py` — `detection_battery`, `-1 → NaN` + dict
  handling in the expansion path.
- `src/mushin/benchmark/_predict.py` — `default_detection_predict_fn`.
- `src/mushin/benchmark/_inference.py` — `_to_device`, dict-valued `compute()`
  handling in `evaluate`.
- `src/mushin/benchmark/_tasks.py` — registry entry.
- `pyproject.toml` — `detection` optional-dependency extra.
- `tests/test_benchmark/` — hermetic detection tests; gated real-dataset test.
- `docs/` — detection section in the benchmark/compare guide; changelog fragment.

## Open questions

None — all resolved in the brainstorm.
