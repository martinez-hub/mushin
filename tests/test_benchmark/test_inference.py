def test_evaluate_with_dict_output_model_and_custom_predict_fn():
    # Real models (e.g. torchvision segmentation) return a dict {"out": logits},
    # not a tensor, so the default predict_fn can't be used. A custom predict_fn
    # extracting ["out"] must flow through evaluate end-to-end. This locks in the
    # segmentation-dogfood friction.
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import classification_battery

    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 4, generator=g)
    y = torch.randint(0, 3, (20,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    mapping = {tuple(xi.tolist()): int(yi) for xi, yi in zip(x, y)}

    class DictModel(torch.nn.Module):
        def forward(self, xb):
            idx = torch.tensor([mapping[tuple(r.tolist())] for r in xb])
            logits = torch.nn.functional.one_hot(idx, 3).float() * 10.0
            return {"out": logits}  # dict output, like torchvision seg models

    def predict(model, xb):
        logits = model(xb)["out"]
        probs = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs

    out = evaluate(
        DictModel(),
        loader,
        classification_battery(3),
        predict,
        prob_metrics=frozenset({"auroc", "ece"}),
    )
    assert out["accuracy"] == 1.0


def test_evaluate_segmentation_ignores_void_across_batches():
    # ignore_index must exclude void pixels through the *streaming* evaluate path
    # (the existing test only covers the one-shot compute_battery). Model is
    # correct on every non-void pixel; its arbitrary void prediction must not
    # count because the void label is excluded.
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import segmentation_battery
    from mushin.benchmark._predict import default_segmentation_predict_fn

    g = torch.Generator().manual_seed(0)
    N, C, H, W = 8, 3, 6, 6
    x = torch.randn(N, 1, H, W, generator=g)
    true = torch.randint(0, C, (N, H, W), generator=g)  # the real classes
    target = true.clone()
    target[:, 0, 0] = 255  # one void pixel per image
    loader = DataLoader(TensorDataset(x, target), batch_size=4)  # 2 batches
    mapping = {tuple(xi.flatten().tolist()): t for xi, t in zip(x, true)}

    class PerfectSeg(torch.nn.Module):
        def forward(self, xb):
            outs = []
            for xi in xb:
                t = mapping[tuple(xi.flatten().tolist())]  # predict the true class
                outs.append(
                    torch.nn.functional.one_hot(t, C).permute(2, 0, 1).float() * 10
                )
            return torch.stack(outs)

    res = evaluate(
        PerfectSeg(),
        loader,
        segmentation_battery(C, ignore_index=255),
        default_segmentation_predict_fn,
        prob_metrics=frozenset(),
    )
    assert res["pixel_acc"] == 1.0
    assert res["miou"] == 1.0


def test_evaluate_streams_and_matches_one_shot():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import classification_battery, compute_battery
    from mushin.benchmark._predict import default_classification_predict_fn

    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 4, generator=g)
    y = torch.randint(0, 3, (20,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = torch.nn.Linear(4, 3)

    battery = classification_battery(3)
    streamed = evaluate(
        model,
        loader,
        battery,
        default_classification_predict_fn,
        prob_metrics=frozenset({"auroc", "ece"}),
    )
    with torch.no_grad():
        preds, probs = default_classification_predict_fn(model, x)
    one_shot = compute_battery(
        classification_battery(3), preds, y, frozenset({"auroc", "ece"}), probs=probs
    )
    assert streamed.keys() == one_shot.keys()
    for k in streamed:
        assert abs(streamed[k] - one_shot[k]) < 1e-5


def test_evaluate_explicit_device_and_resets_between_calls():
    # evaluate(device=...) must run on the given device, and reusing the same
    # battery dict across calls must NOT leak metric state (evaluate resets each
    # metric before the batch loop).
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import classification_battery
    from mushin.benchmark._predict import default_classification_predict_fn

    def loader(seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(16, 4, generator=g)
        y = torch.randint(0, 3, (16,), generator=g)
        return DataLoader(TensorDataset(x, y), batch_size=8)

    model = torch.nn.Linear(4, 3)
    cpu = torch.device("cpu")
    pm = frozenset({"auroc", "ece"})

    battery = classification_battery(3)
    evaluate(
        model,
        loader(1),
        battery,
        default_classification_predict_fn,
        prob_metrics=pm,
        device=cpu,
    )  # first call dirties the metric state
    reused = evaluate(
        model,
        loader(2),
        battery,
        default_classification_predict_fn,
        prob_metrics=pm,
        device=cpu,
    )  # same battery, different data
    fresh = evaluate(
        model,
        loader(2),
        classification_battery(3),
        default_classification_predict_fn,
        prob_metrics=pm,
        device=cpu,
    )

    assert reused.keys() == fresh.keys()
    for k in reused:
        assert abs(reused[k] - fresh[k]) < 1e-6  # no carryover from the first call


def test_to_device_moves_tensors_in_nested_structures():
    import torch

    from mushin.benchmark._inference import _to_device

    dev = torch.device("cpu")
    obj = [
        {"boxes": torch.zeros(2, 4), "labels": torch.tensor([1, 2])},
        torch.ones(3),
    ]
    moved = _to_device(obj, dev)
    assert isinstance(moved, list)
    assert moved[0]["boxes"].device == dev and moved[0]["labels"].device == dev
    assert moved[1].device == dev
    # non-tensors pass through unchanged
    assert _to_device("a string", dev) == "a string"
    assert _to_device(7, dev) == 7


def test_to_device_handles_namedtuple():
    import collections

    import torch

    from mushin.benchmark._inference import _to_device

    Point = collections.namedtuple("Point", ["a", "b"])
    p = Point(torch.zeros(2), torch.ones(3))
    moved = _to_device(p, torch.device("cpu"))
    assert isinstance(moved, Point)  # type preserved, not collapsed to a plain tuple
    assert moved.a.device == torch.device("cpu")
    assert torch.equal(moved.b, torch.ones(3))


def test_expand_metric_value_scalar_dict_and_passthrough():
    import torch

    from mushin.benchmark._inference import expand_metric_value

    # scalar -> kept under the battery name
    assert expand_metric_value("acc", torch.tensor(0.5)) == {"acc": 0.5}
    # dict -> one entry per key (the metric's own key names)
    out = expand_metric_value(
        "map", {"map": torch.tensor(0.25), "map_50": torch.tensor(0.75)}
    )
    assert out == {"map": 0.25, "map_50": 0.75}
    # `expand_metric_value` does NOT special-case -1.0 — the COCO sentinel is
    # normalized to NaN upstream (inside the detection mAP battery), so a -1 from
    # any other metric (e.g. an IoU variant) passes through verbatim here.
    assert expand_metric_value("giou", {"giou": torch.tensor(-1.0)}) == {"giou": -1.0}


def test_evaluate_rejects_colliding_metric_names():
    """Two battery metrics producing the same data-variable name must raise, not
    silently overwrite (a scalar `score` vs another metric returning {"score": ...})."""
    import pytest
    import torch
    from torchmetrics import Metric

    from mushin.benchmark._inference import evaluate

    class Scalar(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")

        def update(self, preds, target):
            self.v = torch.tensor(1.0)

        def compute(self):
            return self.v

    class DictScore(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")

        def update(self, preds, target):
            self.v = torch.tensor(2.0)

        def compute(self):
            return {"score": self.v}

    data = [(torch.tensor([1.0]), torch.tensor([0]))]
    with pytest.raises(ValueError, match="colliding"):
        evaluate(
            torch.nn.Identity(),
            data,
            {"score": Scalar(), "other": DictScore()},
            predict_fn=lambda m, x: (m(x), None),
            prob_metrics=frozenset(),
        )


def test_expand_metric_value_rejects_non_scalar():
    import pytest
    import torch

    from mushin.benchmark._inference import expand_metric_value

    with pytest.raises(TypeError, match="non-scalar"):
        expand_metric_value("classes", {"classes": torch.tensor([0, 1, 2])})


def test_evaluate_expands_dict_metric_and_keeps_scalar():
    import torch
    from torchmetrics import Metric

    from mushin.benchmark._inference import evaluate

    class ScalarMetric(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")

        def update(self, preds, target):
            self.v = preds.float().mean()

        def compute(self):
            return self.v

    class DictMetric(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")

        def update(self, preds, target):
            self.v = preds.float().mean()

        def compute(self):
            return {"a": self.v, "b": self.v + 1}

    model = torch.nn.Identity()
    data = [(torch.tensor([1.0, 1.0]), torch.tensor([0, 0]))]  # one re-iterable batch

    out = evaluate(
        model,
        data,
        {"s": ScalarMetric(), "d": DictMetric()},
        predict_fn=lambda m, x: (m(x), None),
        prob_metrics=frozenset(),
    )
    assert out == {"s": 1.0, "a": 1.0, "b": 2.0}


def test_evaluate_uses_custom_update_fn():
    import torch
    from torchmetrics import MeanMetric

    from mushin.benchmark._inference import evaluate

    # data yields (x, y) where y is a (value, weight) tuple — a shape the default
    # (preds, target) loop could not handle; the custom update_fn unpacks it.
    data = [(torch.zeros(2, 1), (torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0])))]
    model = torch.nn.Identity()
    battery = {"m": MeanMetric()}

    def predict_fn(model, x):
        return torch.tensor([1.0, 3.0]), None

    calls = {"n": 0}

    def update_fn(battery, preds, probs, target):
        calls["n"] += 1
        value, _weight = target
        battery["m"].update(value)

    out = evaluate(model, data, battery, predict_fn, frozenset(), update_fn=update_fn)
    assert calls["n"] == 1
    assert out["m"] == 2.0  # mean of [1.0, 3.0]


def test_evaluate_default_update_fn_unchanged():
    import torch
    from torchmetrics.classification import MulticlassAccuracy

    from mushin.benchmark._inference import evaluate

    data = [(torch.zeros(3, 2), torch.tensor([0, 1, 2]))]
    model = torch.nn.Identity()
    battery = {"acc": MulticlassAccuracy(num_classes=3, average="micro")}

    def predict_fn(model, x):
        return torch.tensor([0, 1, 2]), None  # all correct

    out = evaluate(model, data, battery, predict_fn, frozenset())  # update_fn omitted
    assert out["acc"] == 1.0
