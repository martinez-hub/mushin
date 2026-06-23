# Segmentation support for `compare` — Design

*Date: 2026-06-23*

## Goal

Extend `mushin.benchmark.compare` (and therefore `Study`) to semantic
segmentation via `task="segmentation"`, proving the `task=` seam the
classification version was built around. Along the way, refactor the shared
evaluation step to **stream** (update metrics per batch) so per-pixel
segmentation outputs don't blow up memory — a change that also lets
classification scale to large datasets.

## Design decisions

| Element | Decision |
| --- | --- |
| Battery | `miou` (mean IoU / Jaccard), `dice`, `pixel_acc`, `precision` (macro), `recall` (macro) — 5 scalars. |
| Eval loop | **Streaming, unified for all tasks.** Replace collect-then-compute with a per-batch `metric.update(...)` then `metric.compute()`. torchmetrics accumulates a confusion matrix → `O(C²)` memory. |
| Dispatch | A **task registry** maps each task to its battery factory, predict_fn, and the set of metric names that consume probabilities. |
| Void/ignore pixels | `compare` gains an optional `ignore_index: int | None = None`, passed to the segmentation battery (e.g. Pascal-VOC 255); ignored by classification. |
| Scope | **Semantic** segmentation only. |

## Owned vs. delegated (north-star check)

Unchanged from classification: metrics are delegated to torchmetrics, statistics
to scipy, training to the user. This change only adds a new battery + predict
step and makes the owned evaluation loop stream.

## Architecture / components

`src/mushin/benchmark/`:

- **`_tasks.py`** (new) — the task registry:
  ```python
  @dataclass(frozen=True)
  class TaskSpec:
      battery: Callable[..., dict[str, Metric]]
      predict_fn: PredictFn
      prob_metrics: frozenset[str]

  _TASKS = {
      "classification": TaskSpec(classification_battery,
                                 default_classification_predict_fn,
                                 frozenset({"auroc", "ece"})),
      "segmentation":   TaskSpec(segmentation_battery,
                                 default_segmentation_predict_fn,
                                 frozenset()),
  }
  ```
  An unknown `task` raises `NotImplementedError` listing the known tasks.

- **`_inference.py`** (refactor) — replace `run_inference` + the separate
  `compute_metrics` with a single streaming evaluator:
  ```python
  def evaluate(model, data, battery, predict_fn, prob_metrics, device=None
               ) -> dict[str, float]:
      # set device/eval/no_grad; reset every metric
      # for x, y in data:
      #     preds, probs = predict_fn(model, x.to(device))
      #     for name, m in battery.items():
      #         m.update((probs if name in prob_metrics else preds), y.to(device))
      # return {name: float(m.compute()) for name, m in battery.items()}
  ```
  Metrics are reset once before the loop. Classification results are identical to
  the old collect-then-compute path (metric accumulation is associative).

- **`_metrics.py`** — keep `classification_battery`; add
  `segmentation_battery(num_classes, ignore_index=None) -> dict[str, Metric]`
  using torchmetrics multiclass metrics, which treat per-pixel `(N, H, W)` preds
  and targets as samples:
  - `miou` — `MulticlassJaccardIndex(num_classes, ignore_index=...)`
  - `dice` — torchmetrics Dice/F1 over pixels (exact class pinned in the plan
    after API verification)
  - `pixel_acc` — `MulticlassAccuracy(num_classes, average="micro", ignore_index=...)`
  - `precision` — `MulticlassPrecision(num_classes, average="macro", ignore_index=...)`
  - `recall` — `MulticlassRecall(num_classes, average="macro", ignore_index=...)`

  The old module-level `_PROB_METRICS` is removed; that knowledge now lives in
  each `TaskSpec.prob_metrics`. (The `compute_metrics` helper is superseded by
  the streaming `evaluate`.)

- **`_predict.py`** — keep the classification predict_fn; add
  `default_segmentation_predict_fn(model, x)`: `logits = model(x)` of shape
  `(N, C, H, W)`; `probs = softmax(dim=1)`; `preds = argmax(dim=1)` of shape
  `(N, H, W)`; return `(preds, probs)`.

- **`compare.py`** — dispatch through the registry:
  - look up `spec = _TASKS[task]` (raise `NotImplementedError` if missing);
  - `battery = metrics if metrics is not None else spec.battery(num_classes,
    ignore_index=ignore_index)` — both factories take the same
    `(num_classes, *, ignore_index=None)` signature (`segmentation_battery` uses
    it; `classification_battery` accepts and ignores it, for a uniform call);
  - `predict_fn = predict_fn or spec.predict_fn`;
  - evaluate each model with the streaming `evaluate(..., spec.prob_metrics)`;
  - aggregation (`to_dataset`) and statistics (`compare_methods`) are unchanged.

  New keyword: `ignore_index: int | None = None`.

- **`Study`** — no change needed; it already forwards `task` to `compare`.

## Data flow

For each `(method, seed)` model: stream the dataloader → per-batch predict →
update the battery → compute 5 scalar metrics → one row in the
`(method × seed)` `xarray.Dataset` → `compare_methods` for significance. Memory
stays `O(C²)` regardless of image size or dataset length.

## Error handling

- Unknown `task` → `NotImplementedError` naming the supported tasks.
- `num_classes` required when `metrics` is not provided (unchanged).
- A predict_fn whose output shape doesn't match the target's → the torchmetrics
  `update` raises with a clear shape error; propagate it.

## Testing strategy

- **`segmentation_battery`**: a perfect synthetic mask (`preds == targets`,
  shape `(N, H, W)`) → `miou == dice == pixel_acc == 1.0`; a known partial
  confusion → a hand-checked mIoU.
- **`evaluate` (streaming)**: a tiny model returning fixed `(N, C, H, W)` logits
  over a 2-batch loader yielding `(x, mask)` → returns the expected metric dict;
  assert it matches a single-shot computation (associativity).
- **`compare(task="segmentation")`** end-to-end: two tiny seg models (one
  deliberately better) × small synthetic masks → `BenchmarkResult` with dims
  `(method, seed)`, `miou` present, the better model flagged.
- **Classification regression**: the entire existing `tests/test_benchmark`
  suite must still pass after the streaming refactor — this is the change's main
  risk and its main guard. Same for `tests/test_study`. The tests that exercised
  the removed `compute_metrics` / `_PROB_METRICS` (notably the metric-state-leak
  test in `test_metrics.py`) are migrated to the streaming `evaluate` (which
  resets metrics before the batch loop, preserving the no-state-carryover
  guarantee).
- **`ignore_index`**: a mask with an ignored label → that label is excluded from
  the metric (e.g. all-correct except ignored pixels → `pixel_acc == 1.0`).

## Non-goals

- **Instance / panoptic** segmentation (semantic only).
- **Detection** (its variable-length contract is a separate design cycle).
- **Per-class IoU vectors** — only scalar aggregates, to fit the existing
  dataset/stats pipeline.
- No change to aggregation, statistics, or `Study` orchestration.
