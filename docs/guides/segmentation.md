# Segmentation

`compare` and `Study` support semantic segmentation via `task="segmentation"`.

## Basic segmentation comparison

```python
from mushin.benchmark import compare

result = compare(
    methods={"fcn": [m0, m1, m2], "deeplab": [d0, d1, d2]},
    data=test_loader,
    task="segmentation",
    num_classes=21,
)
result.summary()
```

The default segmentation battery includes **mean IoU**, **Dice**, **pixel
accuracy**, and **macro precision/recall** (computed via torchmetrics).

## Ignoring void / boundary labels

Many segmentation datasets use a special label (e.g. 255 in PASCAL VOC) to
mark void or boundary pixels. Pass `ignore_index` to exclude these from all
metrics:

```python
result = compare(
    methods={"fcn": [m0, m1]},
    data=test_loader,
    task="segmentation",
    num_classes=21,
    ignore_index=255,
)
```

## Custom predict_fn for models that return dicts

Some models (e.g. `torchvision.models.segmentation`) return a dict instead of
a plain tensor. Use `predict_fn` to adapt the output:

```python
def seg_predict(model, x):
    logits = model(x)["out"]
    probs = logits.softmax(dim=1)
    return probs.argmax(dim=1), probs

compare(
    {"fcn": [m0, m1]},
    data=test_loader,
    task="segmentation",
    num_classes=21,
    ignore_index=255,
    predict_fn=seg_predict,
)
```

The `predict_fn` signature is `(model, batch_x) -> (predictions, probabilities)`,
where `predictions` is a `(N, H, W)` long tensor of class indices and
`probabilities` is a `(N, C, H, W)` float tensor of per-class probabilities.

!!! note "predict_fn must always return a 2-tuple"
    `predict_fn` always returns `(predictions, probabilities)` — the evaluation
    loop unpacks both. If you have no probabilities to provide, just return the
    predictions twice (`return preds, preds`); the duplicate is never used,
    because the segmentation battery has no probability-based metrics. (For
    `task="segmentation"`, `prob_metrics` is already empty, so you don't need to
    set it.)

## Using Study for segmentation

```python
from mushin import Study

study = Study(
    methods={"fcn": train_fcn, "deeplab": train_deeplab},
    load_fn=SegModel.load_from_checkpoint,
    seeds=[0, 1, 2],
    data=test_loader,
    task="segmentation",
    num_classes=21,
    ignore_index=255,
)
result = study.run()
```

## See also

- [Comparing methods guide](compare.md) — statistical tests and result reading
- [API Reference — benchmark](../reference/benchmark.md)
