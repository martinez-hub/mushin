import torch

from mushin.benchmark._metrics import classification_battery


def test_battery_has_expected_metrics():
    battery = classification_battery(num_classes=3)
    assert set(battery) == {"accuracy", "f1", "precision", "recall", "auroc", "ece"}


def test_segmentation_battery_perfect_masks():
    from mushin.benchmark._metrics import compute_battery, segmentation_battery

    battery = segmentation_battery(num_classes=3)
    assert set(battery) == {"miou", "dice", "pixel_acc", "precision", "recall"}
    preds = torch.randint(0, 3, (2, 8, 8))
    out = compute_battery(battery, preds, preds, prob_metrics=frozenset())
    assert out["miou"] == 1.0
    assert out["dice"] == 1.0
    assert out["pixel_acc"] == 1.0


def test_segmentation_battery_ignore_index():
    from mushin.benchmark._metrics import compute_battery, segmentation_battery

    battery = segmentation_battery(num_classes=3, ignore_index=255)
    pred = torch.zeros(1, 4, 4, dtype=torch.long)
    tgt = torch.zeros(1, 4, 4, dtype=torch.long)
    tgt[0, 0, 0] = 255  # one void pixel, excluded
    out = compute_battery(battery, pred, tgt, prob_metrics=frozenset())
    assert out["pixel_acc"] == 1.0


def test_classification_battery_warns_on_ignore_index():
    import pytest

    from mushin.benchmark._metrics import classification_battery

    with pytest.warns(UserWarning, match="ignore_index"):
        classification_battery(3, ignore_index=0)


def test_compute_battery_expands_dict_metric():
    import torch
    from torchmetrics import Metric

    from mushin.benchmark._metrics import compute_battery

    class DictMetric(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")

        def update(self, preds, target):
            self.v = preds.float().mean()

        def compute(self):
            return {"x": self.v, "y": self.v + 1}

    out = compute_battery(
        {"m": DictMetric()},
        preds=torch.tensor([1.0]),
        targets=torch.tensor([1.0]),
        prob_metrics=frozenset(),
    )
    # the dict expands to one data var per key; no metric-agnostic sentinel here
    assert out == {"x": 1.0, "y": 2.0}


def test_detection_battery_contents_and_map_drops_metadata():
    pytest = __import__("pytest")
    # The detection metric classes are gated behind torchvision + pycocotools;
    # the `torchmetrics.detection` module itself imports without them, so skip on
    # the real deps (otherwise this would error instead of skip when absent).
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    import torch

    from mushin.benchmark._metrics import detection_battery

    battery = detection_battery()
    assert set(battery) == {"map", "iou", "giou", "ciou", "diou"}

    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    tgts = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}
    ]
    battery["map"].update(preds, tgts)
    keys = set(battery["map"].compute())
    # the three non-scalar bookkeeping keys are dropped
    assert {"classes", "map_per_class", "mar_100_per_class"}.isdisjoint(keys)
    # the 12 scalar AP/AR values remain
    assert {"map", "map_50", "map_75", "map_small", "mar_100"} <= keys


def test_detection_battery_clear_error_without_extra(monkeypatch):
    import builtins

    import pytest

    from mushin.benchmark import _metrics

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torchmetrics.detection":
            raise ImportError("no detection extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="mushin-py\\[detection\\]"):
        _metrics.detection_battery()


def test_regression_battery_end_to_end():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import BenchmarkResult, compare

    class _AffineModel(torch.nn.Module):
        def __init__(self, w, b):
            super().__init__()
            self.w, self.b = w, b

        def forward(self, x):
            return x[:, 0] * self.w + self.b  # shape (N,)

    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 1, generator=g)
    y = x[:, 0] * 2.0 + 1.0  # true relation
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    good = [_AffineModel(2.0, 1.0) for _ in range(3)]  # exact
    bad = [_AffineModel(0.0, 0.0) for _ in range(3)]  # constant 0

    result = compare(methods={"good": good, "bad": bad}, data=loader, task="regression")
    assert isinstance(result, BenchmarkResult)
    for name in ["mse", "mae", "rmse", "r2", "pearson", "spearman"]:
        assert name in result.data

    # The exact "good" model has known values: zero error, perfect correlation.
    # Assert them (not just key presence) so a mis-wired metric or a degenerate /
    # NaN result is caught.
    import pytest

    def good_val(name):
        return float(result.data[name].sel(method="good").values.ravel()[0])

    assert good_val("mse") == pytest.approx(0.0, abs=1e-5)
    assert good_val("mae") == pytest.approx(0.0, abs=1e-5)
    assert good_val("rmse") == pytest.approx(0.0, abs=1e-5)
    assert good_val("r2") == pytest.approx(1.0, abs=1e-4)
    assert good_val("pearson") == pytest.approx(1.0, abs=1e-4)
    assert good_val("spearman") == pytest.approx(1.0, abs=1e-4)


def test_retrieval_battery_end_to_end():
    import torch
    from torch.utils.data import DataLoader, Dataset

    from mushin.benchmark import BenchmarkResult, compare

    # Two queries, two docs each. y = (relevance, indexes); the model maps x -> a
    # score (Identity here). The data is chosen so that the per-query grouping
    # MATTERS: query 0's relevant doc has a LOW score (ranked last -> AP 0.5) while
    # query 1's relevant doc has a HIGH score (AP 1.0). Grouped MAP = 0.75; if the
    # update_fn dropped the `indexes=` grouping (merging both queries into one
    # ranking) the MAP would be 0.5 instead — so asserting 0.75 guards the hook.
    class _RetrievalDS(Dataset):
        def __init__(self):
            self.scores = torch.tensor([0.9, 0.1, 0.5, 0.4])
            self.rel = torch.tensor([0, 1, 1, 0])
            self.idx = torch.tensor([0, 0, 1, 1])

        def __len__(self):
            return 1  # single batch

        def __getitem__(self, _i):
            return self.scores, (self.rel, self.idx)

    def collate(batch):  # one item; pass tensors through unbatched
        return batch[0]

    loader = DataLoader(_RetrievalDS(), batch_size=1, collate_fn=collate)
    models = [torch.nn.Identity() for _ in range(3)]

    result = compare(methods={"m": models}, data=loader, task="retrieval")
    assert isinstance(result, BenchmarkResult)
    for name in ["retrieval_map", "ndcg", "mrr", "precision", "recall"]:
        assert name in result.data

    import pytest

    def val(name):
        return float(result.data[name].sel(method="m").values.ravel()[0])

    # Grouping-sensitive values: per-query MAP/MRR are mean(0.5, 1.0) = 0.75.
    assert val("retrieval_map") == pytest.approx(0.75, abs=1e-4)
    assert val("mrr") == pytest.approx(0.75, abs=1e-4)
    for name in ["retrieval_map", "ndcg", "mrr", "precision", "recall"]:
        assert 0.0 <= val(name) <= 1.0
