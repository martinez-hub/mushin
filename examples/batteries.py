# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Runnable toys for all seven built-in mushin benchmark batteries.

One ``run_<task>()`` function per battery: each builds tiny synthetic data and
tiny CPU-only torch models (no external weights), calls
``compare(task="<name>")``, and returns the resulting ``BenchmarkResult``. These
are the CI-tested toys embedded in ``docs/guides/batteries.md`` via mkdocs
snippet markers (``# --8<-- [start:<task>] ... [end:<task>]``).

The detection, image_quality, and audio batteries need optional extras
(``mushin-py[detection|image|audio]``); their metric imports happen inside
``compare`` (not at module import time), so this module always imports cleanly
and the tests ``pytest.importorskip`` the extra to skip when it is absent.

Run one:  python examples/batteries.py
Requires the eval extra (plus detection/image/audio for those batteries):  pip install "mushin-py[eval]"
"""

from __future__ import annotations


# --8<-- [start:walkthrough]
def run_walkthrough():
    """Flagship classification comparison of two models that vary across seeds.

    A realistic ``compare`` scenario: two classifiers whose accuracy genuinely
    varies seed to seed (each seed corrupts a fraction of the labels with its own
    RNG), separated by a wide enough accuracy gap that Welch's t-test is reliably
    significant. Eight seeds per method give the test real within-method variance
    to work with, so the reported p-value is non-trivial (not a deterministic
    zero-variance artifact). Fully seeded, tiny, CPU-only.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import compare

    num_classes = 4
    n = 200
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 8, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=32)

    class NoisyClassifier(torch.nn.Module):
        """Memorizes each row's true label, then corrupts a fraction of its
        predictions with a per-seed RNG so accuracy varies across seeds."""

        def __init__(self, error_rate: float, seed: int):
            super().__init__()
            gen = torch.Generator().manual_seed(seed)
            flip = torch.rand(n, generator=gen) < error_rate
            # map each flipped row to some *other* class (a genuine mistake)
            offset = 1 + torch.randint(0, num_classes - 1, (n,), generator=gen)
            wrong = (y + offset) % num_classes
            pred = torch.where(flip, wrong, y)
            self._map = {
                tuple(xi.tolist()): int(pi) for xi, pi in zip(x, pred, strict=True)
            }

        def forward(self, xb):
            idx = torch.tensor([self._map[tuple(r.tolist())] for r in xb])
            return torch.nn.functional.one_hot(idx, num_classes).float() * 10.0

    seeds = range(8)
    methods = {
        "strong": [NoisyClassifier(0.15, s) for s in seeds],  # ~85% accuracy
        "weak": [NoisyClassifier(0.40, 100 + s) for s in seeds],  # ~60% accuracy
    }
    return compare(
        methods,
        data=loader,
        task="classification",
        num_classes=num_classes,
        test="welch",
    )


# --8<-- [end:walkthrough]


# --8<-- [start:classification]
def run_classification():
    """Multiclass classification: accuracy/f1/precision/recall/auroc/ece.

    Requires ``num_classes``. The default predict_fn turns model logits into
    softmax probabilities and an argmax prediction.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import compare

    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 4, generator=g)
    y = torch.randint(0, 3, (64,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    class Perfect(torch.nn.Module):
        """Memorizes the label of each input row -> emits confident, correct logits."""

        def __init__(self):
            super().__init__()
            self._map = {
                tuple(xi.tolist()): int(yi) for xi, yi in zip(x, y, strict=True)
            }

        def forward(self, xb):
            idx = torch.tensor([self._map[tuple(r.tolist())] for r in xb])
            return torch.nn.functional.one_hot(idx, 3).float() * 10.0

    torch.manual_seed(0)  # make the untrained `bad` baselines reproducible
    methods = {
        "good": [Perfect() for _ in range(3)],
        "bad": [torch.nn.Linear(4, 3) for _ in range(3)],
    }
    return compare(
        methods, data=loader, task="classification", num_classes=3, test="welch"
    )


# --8<-- [end:classification]


# --8<-- [start:segmentation]
def run_segmentation():
    """Semantic segmentation: miou/dice/pixel_acc/precision/recall.

    Requires ``num_classes``. Models emit per-pixel ``(N, C, H, W)`` logits; the
    default predict_fn argmaxes over the channel dim to a ``(N, H, W)`` label map.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import compare

    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, 1, 8, 8, generator=g)
    masks = torch.randint(0, 3, (12, 8, 8), generator=g)
    loader = DataLoader(TensorDataset(x, masks), batch_size=4)

    class Perfect(nn.Module):
        """Memorizes each image's mask -> emits confident, correct per-pixel logits."""

        def __init__(self):
            super().__init__()
            self._m = {
                tuple(xi.flatten().tolist()): mi
                for xi, mi in zip(x, masks, strict=True)
            }

        def forward(self, xb):
            out = [
                nn.functional.one_hot(self._m[tuple(xi.flatten().tolist())], 3)
                .permute(2, 0, 1)
                .float()
                * 10.0
                for xi in xb
            ]
            return torch.stack(out)

    class Bad(nn.Module):
        def forward(self, xb):
            return torch.zeros(xb.shape[0], 3, 8, 8)

    methods = {
        "good": [Perfect() for _ in range(3)],
        "bad": [Bad() for _ in range(3)],
    }
    return compare(
        methods, data=loader, task="segmentation", num_classes=3, test="welch"
    )


# --8<-- [end:segmentation]


# --8<-- [start:detection]
def run_detection():
    """Object detection: mAP/mAR family + IoU variants (needs the detection extra).

    No ``num_classes``. Each batch is ``(images, targets)`` with targets a
    ``list[dict]`` of ``boxes``/``labels``; the default predict_fn expects the
    torchvision convention (model returns ``list[dict]`` of boxes/scores/labels).
    """
    import torch

    from mushin.benchmark import compare

    def box(x0, y0, x1, y1):
        return torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])

    class FixedDetector(torch.nn.Module):
        """Ignores the image and emits fixed torchvision-style predictions."""

        def __init__(self, preds):
            super().__init__()
            self._preds = preds

        def forward(self, x):
            return self._preds

    good = [
        {
            "boxes": box(0, 0, 10, 10),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    bad = [
        {
            "boxes": box(50, 50, 60, 60),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    tgts = [{"boxes": box(0, 0, 10, 10), "labels": torch.tensor([0])}]
    data = [([torch.zeros(3, 16, 16)], tgts)]  # a re-iterable list of one batch

    methods = {
        "good": [FixedDetector(good), FixedDetector(good)],
        "bad": [FixedDetector(bad), FixedDetector(bad)],
    }
    return compare(methods, data, task="detection", test="welch")


# --8<-- [end:detection]


# --8<-- [start:regression]
def run_regression():
    """Scalar regression: mse/mae/rmse/r2/pearson/spearman.

    No ``num_classes``. The passthrough predict_fn feeds ``model(x)`` (a scalar
    prediction of shape ``(N,)``) straight to the metrics against the target.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import compare

    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 1, generator=g)
    y = x[:, 0] * 2.0 + 1.0  # the true affine relation

    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    class Affine(torch.nn.Module):
        def __init__(self, w, b):
            super().__init__()
            self.w, self.b = w, b

        def forward(self, x):
            return x[:, 0] * self.w + self.b  # shape (N,)

    methods = {
        "good": [Affine(2.0, 1.0) for _ in range(3)],  # exact fit
        "bad": [Affine(0.0, 0.0) for _ in range(3)],  # constant 0
    }
    return compare(methods, data=loader, task="regression", test="welch")


# --8<-- [end:regression]


# --8<-- [start:retrieval]
def run_retrieval():
    """Information retrieval: retrieval_map/ndcg/mrr/precision/recall.

    No ``num_classes``. Batches yield ``y = (relevance, indexes)`` and the
    grouped update scores each query separately (metrics take
    ``(preds, relevance, indexes=...)``). The passthrough predict_fn feeds
    ``model(x)`` (the per-document scores) straight through.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    from mushin.benchmark import compare

    class RetrievalDS(Dataset):
        """Two queries, two docs each. Query grouping matters: query 0's relevant
        doc is ranked last (AP 0.5), query 1's is ranked first (AP 1.0)."""

        def __init__(self):
            self.scores = torch.tensor([0.9, 0.1, 0.5, 0.4])
            self.rel = torch.tensor([0, 1, 1, 0])  # binary relevance
            self.idx = torch.tensor([0, 0, 1, 1])  # query id per document

        def __len__(self):
            return 1  # a single batch

        def __getitem__(self, _i):
            return self.scores, (self.rel, self.idx)

    def collate(batch):  # one item; pass the tensors through unbatched
        return batch[0]

    loader = DataLoader(RetrievalDS(), batch_size=1, collate_fn=collate)

    class Reversed(torch.nn.Module):
        def forward(self, scores):
            return -scores  # inverts the ranking -> worse retrieval

    methods = {
        "identity": [torch.nn.Identity() for _ in range(3)],
        "reversed": [Reversed() for _ in range(3)],
    }
    return compare(methods, data=loader, task="retrieval", test="welch")


# --8<-- [end:retrieval]


# --8<-- [start:image_quality]
def run_image_quality():
    """Paired image quality: ssim/psnr/ms_ssim/lpips (needs the image extra).

    No ``num_classes``. The passthrough predict_fn feeds ``model(x)`` (the
    generated/restored image) to the metrics against the reference target. Images
    are ``(N, C, H, W)`` in ``[0, 1]``; ms_ssim needs H, W > 160 (5 scales).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import compare

    g = torch.Generator().manual_seed(0)
    ref = torch.rand(2, 3, 256, 256, generator=g)
    # a near (not exact) reconstruction keeps psnr finite (identical -> inf)
    gen = (ref + 0.01 * torch.randn(2, 3, 256, 256, generator=g)).clamp(0, 1)
    loader = DataLoader(TensorDataset(gen, ref), batch_size=2)

    class Recon(torch.nn.Module):
        def forward(self, x):
            return x  # returns the generated image; the target is the reference

    methods = {"m": [Recon() for _ in range(3)]}
    return compare(methods, data=loader, task="image_quality")


# --8<-- [end:image_quality]


# --8<-- [start:audio]
def run_audio():
    """Speech/audio quality: si_sdr/si_snr/stoi (needs the audio extra).

    No ``num_classes``. The passthrough predict_fn feeds ``model(x)`` (the
    enhanced waveform) to the metrics against the clean reference. Waveforms are
    ``(N, T)``; STOI assumes a 16 kHz sample rate.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import compare

    g = torch.Generator().manual_seed(0)
    ref = torch.randn(2, 16000, generator=g)  # ~1 s at 16 kHz
    est = ref + 0.01 * torch.randn(2, 16000, generator=g)  # near reconstruction
    loader = DataLoader(TensorDataset(est, ref), batch_size=2)

    class Enhancer(torch.nn.Module):
        def forward(self, x):
            return x  # returns the enhanced waveform; target is the clean reference

    methods = {"m": [Enhancer() for _ in range(3)]}
    return compare(methods, data=loader, task="audio")


# --8<-- [end:audio]


if __name__ == "__main__":
    import sys

    from mushin.benchmark import BenchmarkResult

    _RUNNERS = {
        "walkthrough": run_walkthrough,
        "classification": run_classification,
        "segmentation": run_segmentation,
        "detection": run_detection,
        "regression": run_regression,
        "retrieval": run_retrieval,
        "image_quality": run_image_quality,
        "audio": run_audio,
    }
    _failed = []
    for _name, _fn in _RUNNERS.items():
        try:
            _result = _fn()
            assert isinstance(_result, BenchmarkResult), "not a BenchmarkResult"
            _keys = list(_result.data.data_vars)
            assert _keys, "no metrics produced"
            print(f"  {_name}: OK  metrics={_keys}")
        except Exception as _exc:  # noqa: BLE001
            _failed.append(_name)
            print(f"  {_name}: FAIL  {_exc!r}")
    if _failed:
        print(f"\n{len(_failed)} battery example(s) failed: {_failed}")
        sys.exit(1)
    print(f"\nAll {len(_RUNNERS)} battery examples ran cleanly.")
