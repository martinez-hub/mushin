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

## Walkthrough: comparing two classifiers

Before the per-battery reference, here is one end-to-end comparison run through
the whole machinery — code cell, then its real output, cell by cell. It answers
the question `compare` exists for: **is method A really better than method B, or
is the gap just noise?** All numbers below are the verbatim output of the toy in
`examples/batteries.py` (fully seeded, so you get the same values).

### 1. Build the data and two models

Two classifiers over a 4-class problem, evaluated across **8 seeds**. Each seed
memorizes the true labels then corrupts a fraction of its own predictions with a
per-seed RNG — so accuracy genuinely varies from seed to seed (real
within-method variance, not a deterministic constant). `strong` corrupts ~15% of
labels, `weak` ~40%:

```python
--8<-- "examples/batteries.py:walkthrough"
```

### 2. Run the comparison and read the summary

```python
result = run_walkthrough()          # returns a BenchmarkResult
print(result.summary().to_string(index=False))
```

```text
method    metric     mean   ci_low  ci_high significant_vs_ref
strong  accuracy 0.859375 0.824157 0.894593                   
strong        f1 0.858928 0.823638 0.894217                   
strong precision 0.860580 0.825959 0.895202                   
strong    recall 0.859348 0.824177 0.894519                   
strong     auroc 0.906214 0.882756 0.929672                   
strong       ece 0.140488 0.105270 0.175706                   
  weak  accuracy 0.597500 0.564284 0.630716                  *
  weak        f1 0.596200 0.562269 0.630131                  *
  weak precision 0.599170 0.566766 0.631575                  *
  weak    recall 0.598655 0.565313 0.631997                  *
  weak     auroc 0.732234 0.710031 0.754437                  *
  weak       ece 0.402363 0.369147 0.435579                  *
```

`strong` (the first method) is the reference. Each `weak` row carries a `*` in
`significant_vs_ref`: `weak` differs significantly from `strong` on **every**
metric. The `mean` ± the `[ci_low, ci_high]` 95% CI is the per-method effect
size you would report; the CIs for `strong` and `weak` accuracy (`0.824–0.895`
vs `0.564–0.631`) do not overlap.

### 3. The payoff — pairwise significance

`result.comparisons` is the raw pairwise table the `*` markers come from:

```python
print(result.comparisons.to_string(index=False))
```

```text
   metric method_a method_b  mean_diff  effect_size      p_value  p_corrected  significant
 accuracy   strong     weak   0.261875     6.395643 4.271657e-09 4.271657e-09         True
       f1   strong     weak   0.262728     6.345039 4.630911e-09 4.630911e-09         True
precision   strong     weak   0.261410     6.517591 3.383477e-09 3.383477e-09         True
   recall   strong     weak   0.260693     6.359941 4.561598e-09 4.561598e-09         True
    auroc   strong     weak   0.173980     6.368445 4.492403e-09 4.492403e-09         True
      ece   strong     weak  -0.261875    -6.395643 4.271657e-09 4.271657e-09         True
```

A real, non-NaN verdict: accuracy differs by `+0.262` (Welch's t-test
`p = 4.27e-09`, Cohen's `d ≈ 6.4`), `significant = True`. `p_corrected` is the
Holm-adjusted p-value across the six metric comparisons — still far below
`alpha = 0.05`, so the result survives multiple-comparison correction.

### 4. The underlying per-seed data

Every scalar above is aggregated from the `(method, seed)` grid in
`result.data` — one accuracy (etc.) per seed, which is exactly the spread the
test consumes:

```python
result.data
```

```text
<xarray.Dataset> Size: 880B
Dimensions:    (method: 2, seed: 8)
Coordinates:
  * method     (method) <U6 48B 'strong' 'weak'
  * seed       (seed) int64 64B 0 1 2 3 4 5 6 7
Data variables:
    accuracy   (method, seed) float64 128B 0.83 0.88 0.84 ... 0.575 0.6 0.585
    f1         (method, seed) float64 128B 0.8284 0.8785 0.842 ... 0.6013 0.5836
    precision  (method, seed) float64 128B 0.8283 0.8795 ... 0.6037 0.5863
    recall     (method, seed) float64 128B 0.8286 0.8785 0.843 ... 0.6 0.5886
    auroc      (method, seed) float64 128B 0.886 0.9193 0.8947 ... 0.7329 0.725
    ece        (method, seed) float64 128B 0.1699 0.1199 ... 0.3999 0.4149
```

### Interpretation

The summary reports what each method scores; the CIs quantify how confident that
estimate is (they shrink as you add seeds). The significance verdict answers the
*comparative* question: with `strong` accuracy per seed clustered near `0.86` and
`weak` near `0.60` — and the two bands separated by far more than their
seed-to-seed jitter — Welch's t-test returns `p = 4.27e-09`, and Holm keeps it
significant after correcting for the six simultaneous metric tests. The verdict
is trustworthy precisely **because** each method carries real variance across
seeds: had a method produced identical scores on every seed (a deterministic
model that ignores the seed), `compare` would warn and refuse to treat the
duplicated points as independent samples — the same guard the
[LLM path](llm.md#seeds-and-stochasticity) uses. This is also why the seed grid
matters for resilience: a long real sweep can lose individual `(method, seed)`
cells to crashes, and mushin resumes from a partial grid via
[`IncompleteSweepError`](resilience.md) rather than discarding the whole run.

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

**Output** (from running the toy):

```text
method    metric     mean   ci_low  ci_high significant_vs_ref
  good  accuracy 1.000000 1.000000 1.000000                   
  good        f1 1.000000 1.000000 1.000000                   
  good precision 1.000000 1.000000 1.000000                   
  good    recall 1.000000 1.000000 1.000000                   
  good     auroc 1.000000 1.000000 1.000000                   
  good       ece 0.000091 0.000091 0.000091                   
   bad  accuracy 0.276042 0.195243 0.356841                   
   bad        f1 0.268352 0.221906 0.314797                   
   bad precision 0.298513 0.252285 0.344741                   
   bad    recall 0.273085 0.206776 0.339394                   
   bad     auroc 0.479942 0.387758 0.572127                   
   bad       ece 0.212327 0.081076 0.343579                   
```

The `good` models memorize the labels (perfect, zero-variance scores); the `bad`
untrained baselines land near chance. Because `good` is deterministic across
seeds, `compare` emits a warning and leaves `significant_vs_ref` blank rather
than reporting a false positive — see the [walkthrough](#walkthrough-comparing-two-classifiers)
for a version with real seed variance.

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

**Output** (from running the toy):

```text
method    metric     mean   ci_low  ci_high significant_vs_ref
  good      miou 1.000000 1.000000 1.000000                   
  good      dice 1.000000 1.000000 1.000000                   
  good pixel_acc 1.000000 1.000000 1.000000                   
  good precision 1.000000 1.000000 1.000000                   
  good    recall 1.000000 1.000000 1.000000                   
   bad      miou 0.105903 0.105903 0.105903                   
   bad      dice 0.160738 0.160738 0.160738                   
   bad pixel_acc 0.317708 0.317708 0.317708                   
   bad precision 0.105903 0.105903 0.105903                   
   bad    recall 0.333333 0.333333 0.333333                   
```

The `good` models paint the exact ground-truth mask (perfect scores); the `bad`
models predict all-background, so `miou`/`precision` collapse while `pixel_acc`
(`0.318`) still reflects the background-heavy label distribution.

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

**Output** (from running the toy — the full 16-metric battery):

```text
method     metric      mean    ci_low   ci_high significant_vs_ref
  good        map  1.000000  1.000000  1.000000                   
  good     map_50  1.000000  1.000000  1.000000                   
  good     map_75  1.000000  1.000000  1.000000                   
  good  map_small  1.000000  1.000000  1.000000                   
  good map_medium       NaN       NaN       NaN                   
  good  map_large       NaN       NaN       NaN                   
  good      mar_1  1.000000  1.000000  1.000000                   
  good     mar_10  1.000000  1.000000  1.000000                   
  good    mar_100  1.000000  1.000000  1.000000                   
  good  mar_small  1.000000  1.000000  1.000000                   
  good mar_medium       NaN       NaN       NaN                   
  good  mar_large       NaN       NaN       NaN                   
  good        iou  1.000000  1.000000  1.000000                   
  good       giou  1.000000  1.000000  1.000000                   
  good       ciou  1.000000  1.000000  1.000000                   
  good       diou  1.000000  1.000000  1.000000                   
   bad        map  0.000000  0.000000  0.000000                   
   bad     map_50  0.000000  0.000000  0.000000                   
   bad     map_75  0.000000  0.000000  0.000000                   
   bad  map_small  0.000000  0.000000  0.000000                   
   bad map_medium       NaN       NaN       NaN                   
   bad  map_large       NaN       NaN       NaN                   
   bad      mar_1  0.000000  0.000000  0.000000                   
   bad     mar_10  0.000000  0.000000  0.000000                   
   bad    mar_100  0.000000  0.000000  0.000000                   
   bad  mar_small  0.000000  0.000000  0.000000                   
   bad mar_medium       NaN       NaN       NaN                   
   bad  mar_large       NaN       NaN       NaN                   
   bad        iou  0.000000  0.000000  0.000000                   
   bad       giou -0.944444 -0.944444 -0.944444                   
   bad       ciou -0.694444 -0.694444 -0.694444                   
   bad       diou -0.694444 -0.694444 -0.694444                   
```

The single ground-truth box is small, so the `medium`/`large` size buckets have
no matching target and report `NaN` (COCO's `-1` sentinel, excluded from
significance). The `good` detector places the box exactly (mAP `1.0`); the `bad`
detector's box misses entirely, driving mAP to `0` and the `giou`/`ciou`/`diou`
variants negative.

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

**Output** (from running the toy):

```text
method   metric      mean    ci_low   ci_high significant_vs_ref
  good      mse  0.000000  0.000000  0.000000                   
  good      mae  0.000000  0.000000  0.000000                   
  good     rmse  0.000000  0.000000  0.000000                   
  good       r2  1.000000  1.000000  1.000000                   
  good  pearson  1.000000  1.000000  1.000000                   
  good spearman  1.000000  1.000000  1.000000                   
   bad      mse  5.076191  5.076191  5.076191                   
   bad      mae  1.911885  1.911885  1.911885                   
   bad     rmse  2.253040  2.253040  2.253040                   
   bad       r2 -0.253489 -0.253489 -0.253489                   
   bad  pearson       NaN       NaN       NaN                   
   bad spearman  0.000000  0.000000  0.000000                   
```

The `good` model fits the affine relation exactly (zero error, `r2 = 1`); the
`bad` constant-`0` predictor has no variance, so `pearson` is undefined (`NaN`)
and `r2` goes negative (worse than predicting the mean).

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

**Output** (from running the toy):

```text
  method        metric     mean   ci_low  ci_high significant_vs_ref
identity retrieval_map 0.750000 0.750000 0.750000                   
identity          ndcg 0.815465 0.815465 0.815465                   
identity           mrr 0.750000 0.750000 0.750000                   
identity     precision 0.500000 0.500000 0.500000                   
identity        recall 1.000000 1.000000 1.000000                   
reversed retrieval_map 0.000000 0.000000 0.000000                   
reversed          ndcg 0.815465 0.815465 0.815465                   
reversed           mrr 0.000000 0.000000 0.000000                   
reversed     precision 0.000000 0.000000 0.000000                   
reversed        recall 0.000000 0.000000 0.000000                   
```

`identity` keeps the scores' ranking (query 0 scores AP `0.5`, query 1 AP `1.0`
→ `retrieval_map 0.75`); `reversed` inverts every ranking, sending the relevant
docs to the bottom (`retrieval_map`/`mrr` → `0`). `ndcg` is identical (`0.815`)
for both because the two queries are exact mirror images: reversing the ranking
swaps which query scores well and which scores poorly, so the two-query average
is unchanged — a reminder to read what each metric actually measures.

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

**Output** (from running the toy):

```text
method  metric      mean    ci_low   ci_high significant_vs_ref
     m    ssim  0.999384  0.999384  0.999384                   
     m    psnr 40.055252 40.055252 40.055252                   
     m ms_ssim  0.999481  0.999481  0.999481                   
     m   lpips  0.000724  0.000724  0.000724                   
```

The reconstruction is a near-copy of the reference (small added noise), so `ssim`
and `ms_ssim` sit just under `1`, `psnr` is high (`40 dB`), and the learned
`lpips` perceptual distance is near `0`. A single method just profiles the
battery; supply a second to get a significance verdict.

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

**Output** (from running the toy):

```text
method metric      mean    ci_low   ci_high significant_vs_ref
     m si_sdr 40.029053 40.029053 40.029053                   
     m si_snr 40.028419 40.028419 40.028419                   
     m   stoi  0.999894  0.999894  0.999894                   
```

The enhanced waveform is a near-copy of the clean reference, so `si_sdr`/`si_snr`
are high (`~40 dB`) and `stoi` (intelligibility) is essentially `1`. As with
image quality, a second method would turn this profile into a comparison.

## See also

- [Comparing methods](compare.md) — the `compare` API, statistical tests, and reading the result
- [Custom metrics & predict_fn](custom.md) — override a battery or register your own task
- [Segmentation guide](segmentation.md) — `ignore_index` and segmentation specifics
- [Understanding the statistics](statistics.md) — tests, Holm correction, effect size
- [API Reference — benchmark](../reference/benchmark.md)
