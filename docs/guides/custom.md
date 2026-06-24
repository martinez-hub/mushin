# Custom metrics & predict_fn

mushin's metric batteries and prediction logic are fully replaceable. This guide
shows how to extend mushin with custom metrics and how to adapt models that
don't return plain tensors.

## Custom metrics dict

Pass a `metrics` dict to `compare` or `Study` to replace the default battery:

```python
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

compare(
    methods={"cnn": [m0, m1, m2]},
    data=test_loader,
    task="classification",     # still sets the default predict_fn
    metrics={
        "accuracy": MulticlassAccuracy(num_classes=10, average="micro"),
        "f1_macro": MulticlassF1Score(num_classes=10, average="macro"),
        "f1_weighted": MulticlassF1Score(num_classes=10, average="weighted"),
    },
    # num_classes is not required when metrics is provided
)
```

Each value must be a `torchmetrics.Metric` instance. The keys become the
data variable names in `result.data`.

### prob_metrics

Some metrics require class probabilities rather than hard predictions (e.g.
AUROC, ECE). mushin uses a `prob_metrics` frozenset to know which metrics to
feed probabilities to. The default is the task's built-in set; override it
when you add probability-based custom metrics:

```python
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy

compare(
    methods={"cnn": [m0, m1]},
    data=test_loader,
    task="classification",
    metrics={
        "accuracy": MulticlassAccuracy(num_classes=10, average="micro"),
        "auroc": MulticlassAUROC(num_classes=10),
    },
    prob_metrics=frozenset({"auroc"}),  # feed probabilities only to auroc
    num_classes=10,
)
```

## Custom predict_fn

The default `predict_fn` calls the model, takes the argmax over class logits,
and returns `(predictions, softmax_probabilities)`. Replace it when your model
returns something other than a plain `(N, C)` or `(N, C, H, W)` logit tensor.

### Adapting torchvision segmentation models

torchvision segmentation models return a dict `{"out": logits, ...}`. Here is
the adapter from the segmentation example:

```python
--8<-- "examples/segmentation_demo.py:dict_predict"
```

Pass it to `compare`:

```python
compare(
    {"fcn": [m0, m1]},
    data=test_loader,
    task="segmentation",
    num_classes=21,
    ignore_index=255,
    predict_fn=torchvision_seg_predict,
)
```

### predict_fn contract

The `predict_fn` signature is:

```python
def predict_fn(model: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
    ...
    return predictions, probabilities
```

- `predictions`: long tensor of class indices — `(N,)` for classification,
  `(N, H, W)` for segmentation.
- `probabilities`: float tensor of per-class probabilities — `(N, C)` for
  classification, `(N, C, H, W)` for segmentation.

If no probabilities are available, return predictions twice; the second element
is only consumed by metrics listed in `prob_metrics`.

!!! tip "Pitfalls"
    - **Always return a 2-tuple:** The evaluation loop always unpacks both
      elements. Returning only `predictions` will raise a `ValueError`.
    - **prob_metrics mismatch:** If you add a probability-based metric but
      forget to add its name to `prob_metrics`, it receives hard predictions
      and will likely error or silently produce wrong results.
    - **Metric state:** torchmetrics metrics are stateful; mushin calls
      `.reset()` before each model evaluation. Do not share metric instances
      across calls.

## See also

- [Comparing methods](compare.md)
- [Segmentation guide](segmentation.md) — `ignore_index` and dict-output models
- [API Reference — benchmark](../reference/benchmark.md)
