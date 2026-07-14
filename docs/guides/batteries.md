# Built-in batteries

mushin ships **seven** benchmark batteries as first-class, reusable
[tasks](custom.md#define-a-reusable-task): `classification`, `segmentation`,
`detection`, `regression`, `retrieval`, `image_quality`, and `audio`. Each is a
registered `Task` — a metric battery (built on
[torchmetrics](https://lightning.ai/docs/torchmetrics/)), a `predict_fn` that
extracts `(predictions, probabilities)` from a model, and the subset of metrics
that consume probabilities. List them at runtime:

```python
from mushin.benchmark import list_tasks

list_tasks()   # {name: description} for every registered task
```

Every battery flows through the same [`compare`](compare.md) / [`Study`](study.md)
machinery: the same pairwise significance tests, Holm correction, and the
resilient [`IncompleteSweepError`](resilience.md) resume path. Pass the task name
as `task="<name>"`; mushin runs each model over your `data` loader (which yields
`(x, y)` batches), calls `predict_fn(model, x)`, updates the battery's metrics
against `y`, and returns a `BenchmarkResult` with `.summary()`, `.comparisons`,
and `.data`. You are not limited to these seven — register your own with
`register_task` (see [Custom metrics & predict_fn](custom.md)).

Each section below gives an **illustrative real-model recipe** (bring your own
weights — not run here) and a **runnable toy** that is exactly the CI-tested code
from `examples/batteries.py`.

!!! note "The `predict_fn` contract"
    `predict_fn(model, x)` must return a `(preds, probs)` tuple. `probs` may be
    `None` when the battery has no probability metrics. When a real model's output
    does not already match the battery's expected format, override `predict_fn` to
    adapt it — the recipes below show exactly where.

## Classification

Multiclass classification: `accuracy`, `f1`, `precision`, `recall`, `auroc`,
`ece` (expected calibration error). **Requires `num_classes`.** The default
predict_fn reads `model(x)` as `(N, num_classes)` logits, softmaxes them into
`probs`, and argmaxes into `preds` — so a standard image classifier needs no
override.

### Real-model recipe

A fine-tuned **ViT** or **ResNet-50** returns class logits directly, so the
default predict_fn applies as-is:

```python
# Bring your own weights (not run here).
import torch
from mushin.benchmark import compare

# vit_seeds / resnet_seeds: each a list of fine-tuned nn.Module (one per seed),
# in eval mode, that map an (N, 3, H, W) image batch to (N, num_classes) logits.
result = compare(
    methods={"vit": vit_seeds, "resnet50": resnet_seeds},
    data=val_loader,          # yields (images, labels); labels are (N,) class ids
    task="classification",
    num_classes=1000,
    test="welch",
)
result.summary()   # accuracy / f1 / precision / recall / auroc / ece + significance
```

### Runnable toy

```python
--8<-- "examples/batteries.py:classification"
```

## Segmentation

Semantic segmentation over per-pixel class labels: `miou`, `dice` (macro Dice /
F1), `pixel_acc`, `precision`, `recall`. **Requires `num_classes`.** The default
predict_fn reads `model(x)` as `(N, C, H, W)` logits and argmaxes over the channel
dim to a `(N, H, W)` label map. Pass `ignore_index` to exclude void/boundary
pixels (see the [Segmentation guide](segmentation.md)).

### Real-model recipe

**SAM 3.1** (Segment Anything) emits object masks, not a per-pixel class map, so
you override `predict_fn` to fold its masks into a `(N, H, W)` label tensor whose
values are class ids in `[0, num_classes)`:

```python
# Bring your own weights (not run here).
import torch
from mushin.benchmark import compare


def sam_predict(model, x):
    """Adapt SAM-style output to the segmentation contract: (preds, probs).

    preds must be a (N, H, W) long tensor of class labels; probs may be None
    (the battery has no probability metrics). Fill in the mask -> class-label
    mapping for your prompts / label scheme.
    """
    n, _, h, w = x.shape
    outputs = model(x)  # e.g. per-image list of {"masks": (K, H, W) bool, "labels": (K,)}

    preds = x.new_zeros((n, h, w), dtype=torch.long)  # 0 = background
    for i, out in enumerate(outputs):
        for mask, label in zip(out["masks"], out["labels"]):
            preds[i][mask.bool()] = int(label)  # paint each mask with its class id
    return preds, None


result = compare(
    methods={"sam": sam_seeds},   # each a list of models (one per seed), eval mode
    data=val_loader,              # yields (images, masks); masks are (N, H, W) labels
    task="segmentation",
    num_classes=21,
    predict_fn=sam_predict,
    test="welch",
)
result.summary()   # miou / dice / pixel_acc / precision / recall + significance
```

### Runnable toy

```python
--8<-- "examples/batteries.py:segmentation"
```

## Detection

Object detection over the full `torchmetrics.detection` bounding-box family:
the 12 mAP/mAR scalars (`map`, `map_50`, `map_75`, `map_small|medium|large`,
`mar_1|10|100`, `mar_small|medium|large`) plus the IoU variants `iou`, `giou`,
`ciou`, `diou`. **No `num_classes`** (mAP infers classes from labels). Requires
the detection extra:

```
pip install "mushin-py[detection]"    # torchvision + pycocotools
```

Each batch is `(images, targets)` where `images` is a `list[Tensor]` and each
target is a `dict` with `boxes` (`[N, 4]`, xyxy) and `labels` (`[N]`). The default
predict_fn assumes a torchvision detector returning `list[dict]` with
`boxes`/`scores`/`labels`. A size bucket with no matching ground truth reports
`NaN` (COCO's `-1` sentinel), excluded from significance.

### Real-model recipe

**YOLO-World** does not follow the torchvision detector convention, so override
`predict_fn` to repackage its output into the torchmetrics format:

```python
# Bring your own weights (not run here).
from mushin.benchmark import compare


def yolo_world_predict(model, x):
    """Adapt YOLO-World output to torchmetrics detection format: (list_of_dicts, None).

    Return one dict per image with float tensors: boxes (M, 4) in xyxy pixels,
    scores (M,), labels (M,) as integer class ids.
    """
    results = model(x)  # e.g. an Ultralytics Results list, one entry per image
    preds = [
        {
            "boxes": r.boxes.xyxy,          # (M, 4), xyxy
            "scores": r.boxes.conf,         # (M,)
            "labels": r.boxes.cls.long(),   # (M,)
        }
        for r in results
    ]
    return preds, None  # no probabilities for detection metrics


result = compare(
    methods={"yolo_world": yolo_seeds},   # each a list of models (one per seed)
    data=coco_val_loader,                 # yields (images, targets) as above
    task="detection",
    predict_fn=yolo_world_predict,
    test="welch",
)
result.summary()   # map / map_50 / map_75 / mar_* / iou / giou / ciou / diou + significance
```

### Runnable toy

```python
--8<-- "examples/batteries.py:detection"
```

## Regression

Scalar regression: `mse`, `mae`, `rmse`, `r2`, `pearson`, `spearman`. **No
`num_classes`.** This is a single-target battery — predictions and targets are
continuous tensors of shape `(N,)` or `(N, 1)`. The passthrough predict_fn feeds
`model(x)` straight to the metrics against the target (no probabilities).

### Real-model recipe

An **aesthetic / image-quality scorer** (a model that regresses a scalar quality
score per input) needs no override — its scalar output is the prediction:

```python
# Bring your own weights (not run here).
from mushin.benchmark import compare

# scorer_seeds: each a list of models (one per seed) mapping an (N, ...) input
# batch to an (N,) scalar-score tensor.
result = compare(
    methods={"aesthetic_scorer": scorer_seeds},
    data=val_loader,          # yields (inputs, scores); scores are (N,) floats
    task="regression",
    test="welch",
)
result.summary()   # mse / mae / rmse / r2 / pearson / spearman + significance
```

### Runnable toy

```python
--8<-- "examples/batteries.py:regression"
```

## Retrieval

Information retrieval, scored per query: `retrieval_map`, `ndcg`, `mrr`,
`precision`, `recall`. **No `num_classes`.** The passthrough predict_fn feeds
`model(x)` (the per-document relevance scores) straight through, and a **grouped
update** scores each query separately: batches must yield `y = (relevance,
indexes)`, where `indexes` assigns every candidate to a query and `relevance` is
binary 0/1 (only `ndcg` accepts graded relevance).

### Real-model recipe

**CLIP** image↔text retrieval is conceptually a grouped ranking: for each query
(say, a text prompt) you score every candidate (image), then group by query id.
Keep the model producing a flat score vector and let the battery's grouped update
do the per-query aggregation — the exact data/grouping contract is what the
runnable toy below demonstrates:

```python
# Bring your own weights (not run here) — conceptual; see the toy for the exact
# (relevance, indexes) grouping contract.
from mushin.benchmark import compare

# Each batch yields (scores, (relevance, indexes)) where, for a flat list of
# (query, candidate) pairs:
#   scores    : (P,) CLIP similarity for each pair (model output, passthrough)
#   relevance : (P,) binary 0/1 — is this candidate relevant to its query?
#   indexes   : (P,) query id per pair, so metrics rank within each query
result = compare(
    methods={"clip": clip_seeds},   # each a list of models (one per seed)
    data=retrieval_loader,
    task="retrieval",
    test="welch",
)
result.summary()   # retrieval_map / ndcg / mrr / precision / recall + significance
```

### Runnable toy

```python
--8<-- "examples/batteries.py:retrieval"
```

## Image quality

Paired (full-reference) image quality: `ssim`, `psnr`, `ms_ssim`, `lpips`. **No
`num_classes`.** The passthrough predict_fn feeds `model(x)` (the generated /
restored image) to the metrics against the reference target. Images are
`(N, C, H, W)` in `[0, 1]`; `ms_ssim` needs `H, W > 160`. Requires the image
extra:

```
pip install "mushin-py[image]"    # torchvision + lpips
```

### Real-model recipe

A super-resolution / restoration model like **Real-ESRGAN** or **SwinIR** returns
the restored image directly, so the passthrough predict_fn applies as-is — just
make `data` yield `(low_quality_input, high_quality_reference)`:

```python
# Bring your own weights (not run here).
from mushin.benchmark import compare

# esrgan_seeds / swinir_seeds: each a list of models (one per seed) mapping a
# degraded (N, 3, H, W) image in [0, 1] to a restored (N, 3, H, W) image in [0, 1].
result = compare(
    methods={"real_esrgan": esrgan_seeds, "swinir": swinir_seeds},
    data=val_loader,          # yields (degraded, reference); both (N, 3, H, W) in [0, 1]
    task="image_quality",
    test="welch",
)
result.summary()   # ssim / psnr / ms_ssim / lpips + significance
```

### Runnable toy

```python
--8<-- "examples/batteries.py:image_quality"
```

## Audio

Speech / audio quality (estimated vs reference waveform): `si_sdr`, `si_snr`,
`stoi`. **No `num_classes`.** The passthrough predict_fn feeds `model(x)` (the
enhanced waveform) to the metrics against the clean reference. Waveforms are
`(N, T)`; STOI assumes a 16 kHz sample rate. Requires the audio extra:

```
pip install "mushin-py[audio]"    # pystoi
```

(PESQ is intentionally omitted — its only released package fails to import under
NumPy 2; add it via a custom task on a NumPy-1.x environment if you need it.)

### Real-model recipe

A source separator / speech enhancer like **Demucs** returns the enhanced
waveform directly, so the passthrough predict_fn applies as-is — make `data`
yield `(noisy_input, clean_reference)`:

```python
# Bring your own weights (not run here).
from mushin.benchmark import compare

# demucs_seeds: each a list of models (one per seed) mapping a noisy (N, T)
# waveform to an enhanced (N, T) waveform at 16 kHz.
result = compare(
    methods={"demucs": demucs_seeds},
    data=val_loader,          # yields (noisy, clean); both (N, T) at 16 kHz
    task="audio",
    test="welch",
)
result.summary()   # si_sdr / si_snr / stoi + significance
```

### Runnable toy

```python
--8<-- "examples/batteries.py:audio"
```

## See also

- [Comparing methods](compare.md) — the `compare` API, statistical tests, and reading the result
- [Custom metrics & predict_fn](custom.md) — override a battery or register your own task
- [Segmentation guide](segmentation.md) — `ignore_index` and segmentation specifics
- [Understanding the statistics](statistics.md) — tests, Holm correction, effect size
- [API Reference — benchmark](../reference/benchmark.md)
