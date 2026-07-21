# Segmentation

!!! note "Requires the `eval` extra"
    `compare` and `Study` are mushin's optional evaluation layer — install them
    with `pip install "mushin-py[eval]"`. Importing them without it raises a
    clear install hint. See [Installation](../install.md#optional-extras).

`compare` and `Study` support semantic segmentation via `task="segmentation"`.
Models receive `(N, C, H, W)` input tensors and must produce `(N, num_classes, H, W)`
logit tensors; the default `predict_fn` takes the argmax over classes and softmax
probabilities for you.

## Runnable example

The example below compares two tiny segmentation models on synthetic pixel masks:

```python
--8<-- "examples/segmentation_demo.py:run"
```

The default segmentation battery includes:

| Metric | Notes |
|---|---|
| `miou` | Mean Intersection over Union (macro-averaged) |
| `dice` | Macro-averaged Dice coefficient (= macro F1) |
| `pixel_acc` | Micro-averaged pixel accuracy |
| `precision` | Macro-averaged per-class precision |
| `recall` | Macro-averaged per-class recall |

All are confusion-matrix based and computed via torchmetrics, so streaming
evaluation uses O(C²) memory.

## Ignoring void / boundary labels

Many segmentation datasets use a special label (e.g. 255 in PASCAL VOC) to
mark void or boundary pixels. Pass `ignore_index` to exclude these from all
metrics:

```python
# fcn_models and deeplab_models are each a list of trained nn.Module (one per seed)
result = compare(
    methods={"fcn": fcn_models, "deeplab": deeplab_models},
    data=val_loader,
    task="segmentation",
    num_classes=21,
    ignore_index=255,
)
```

## Custom predict_fn for models that return dicts

Some models (e.g. `torchvision.models.segmentation`) return a dict instead of
a plain tensor. Use `predict_fn` to adapt the output:

```python
--8<-- "examples/segmentation_demo.py:dict_predict"
```

Pass it to `compare`:

```python
compare(
    {"fcn": fcn_models, "deeplab": deeplab_models},
    data=val_loader,
    task="segmentation",
    num_classes=21,
    ignore_index=255,
    predict_fn=torchvision_seg_predict,
)
```

The `predict_fn` signature is `(model, batch_x) -> (predictions, probabilities)`,
where `predictions` is a `(N, H, W)` long tensor of class indices and
`probabilities` is a `(N, C, H, W)` float tensor of per-class probabilities.

!!! note "predict_fn must always return a 2-tuple"
    `predict_fn` always returns `(predictions, probabilities)`. If you have no
    probabilities to provide, return the predictions twice
    (`return preds, preds`). For `task="segmentation"`, `prob_metrics` is
    already empty, so the duplicate is never used.

## Using Study for segmentation

```python
from mushin import Study

study = Study(
    methods={"fcn": train_fcn, "deeplab": train_deeplab},
    load_fn=lambda p: torch.load(p, weights_only=False),
    seeds=[0, 1, 2],
    data=val_loader,
    task="segmentation",
    num_classes=21,
    ignore_index=255,
)
result = study.run()
```

!!! tip "Pitfalls"
    - **Input shape:** Models must accept `(N, C, H, W)` and return
      `(N, num_classes, H, W)` logits. A 1×1 `Conv2d` is the minimal example.
    - **ignore_index:** Not supported by AUROC/ECE, but the segmentation
      battery has neither — `ignore_index` works correctly for all five
      segmentation metrics.
    - **Dict-output models:** Always wrap them with a `predict_fn`; passing
      a dict to the default `predict_fn` will raise an error.

## See also

- [Comparing methods guide](compare.md) — statistical tests and result reading
- [Custom metrics & predict_fn](custom.md) — override the metric battery
- [API Reference — benchmark](../reference/benchmark.md)
