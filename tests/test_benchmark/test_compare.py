import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mushin.benchmark import BenchmarkResult, compare


def _loader(seed, n=64, d=4, num_classes=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


class _Perfect(torch.nn.Module):
    """Cheats: reads the target off a fixed mapping -> always correct."""

    def __init__(self, loader, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self._map = {
            tuple(x.tolist()): int(y) for xb, yb in loader for x, y in zip(xb, yb)
        }

    def forward(self, x):
        idx = torch.tensor([self._map[tuple(row.tolist())] for row in x])
        return torch.nn.functional.one_hot(idx, self.num_classes).float() * 10.0


class _AlmostPerfect(_Perfect):
    """Like `_Perfect` but deterministically mislabels its first ``n_wrong`` inputs,
    giving a high-but-non-constant per-model accuracy. A list of these with varying
    ``n_wrong`` has real across-seed variance, so significance is legitimately
    testable (a constant-across-seeds method has no sampling distribution and is
    masked)."""

    def __init__(self, loader, n_wrong, num_classes=3):
        super().__init__(loader, num_classes)
        self.n_wrong = n_wrong
        self._seen = 0

    def forward(self, x):
        out = []
        for row in x:
            y = self._map[tuple(row.tolist())]
            if self._seen < self.n_wrong:
                y = (y + 1) % self.num_classes  # deterministic error
            self._seen += 1
            out.append(y)
        idx = torch.tensor(out)
        return torch.nn.functional.one_hot(idx, self.num_classes).float() * 10.0


def test_compare_end_to_end():
    data = _loader(seed=0)
    good = [_Perfect(data) for _ in range(3)]
    bad = [torch.nn.Linear(4, 3) for _ in range(3)]

    result = compare(
        methods={"good": good, "bad": bad},
        data=data,
        task="classification",
        num_classes=3,
        test="welch",
    )

    assert isinstance(result, BenchmarkResult)
    assert set(result.data.dims) == {"method", "seed"}
    assert result.data.sizes == {"method": 2, "seed": 3}
    assert "accuracy" in result.data.data_vars
    assert float(result.data["accuracy"].sel({"method": "good"}).mean()) == 1.0
    assert len(result.summary()) == 2 * len(result.data.data_vars)


def test_compare_default_test_with_tied_methods():
    # two perfect methods => identical metrics => default wilcoxon must not crash
    data = _loader(seed=0)
    a = [_Perfect(data) for _ in range(3)]
    b = [_Perfect(data) for _ in range(3)]
    result = compare(methods={"a": a, "b": b}, data=data, num_classes=3)
    assert isinstance(result, BenchmarkResult)
    # all comparisons are between identical methods -> none significant
    assert not result.comparisons["significant"].any()


def test_compare_segmentation_end_to_end():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark import BenchmarkResult, compare

    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, 1, 8, 8, generator=g)
    masks = torch.randint(0, 3, (12, 8, 8), generator=g)
    loader = DataLoader(TensorDataset(x, masks), batch_size=4)

    class Perfect(torch.nn.Module):
        def __init__(self, masks):
            super().__init__()
            self._m = {tuple(xi.flatten().tolist()): mi for xi, mi in zip(x, masks)}

        def forward(self, xb):
            out = []
            for xi in xb:
                m = self._m[tuple(xi.flatten().tolist())]
                out.append(
                    torch.nn.functional.one_hot(m, 3).permute(2, 0, 1).float() * 10
                )
            return torch.stack(out)

    class Bad(torch.nn.Module):
        def forward(self, xb):
            return torch.zeros(xb.shape[0], 3, 8, 8)

    result = compare(
        methods={
            "good": [Perfect(masks) for _ in range(3)],
            "bad": [Bad() for _ in range(3)],
        },
        data=loader,
        task="segmentation",
        num_classes=3,
        test="welch",
    )
    assert isinstance(result, BenchmarkResult)
    assert set(result.data.dims) == {"method", "seed"}
    assert "miou" in result.data.data_vars
    assert float(result.data["miou"].sel({"method": "good"}).mean()) == 1.0


def test_compare_rejects_unknown_task():
    import pytest

    from mushin.benchmark import compare

    with pytest.raises(NotImplementedError, match="not supported"):
        compare(methods={"a": []}, data=[], task="bogus_task", num_classes=2)


def test_compare_detection_does_not_demand_num_classes(monkeypatch):
    import torch
    from torchmetrics import Metric

    from mushin.benchmark import _tasks, compare

    class Const(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("v", default=torch.tensor(0.0), dist_reduce_fx="sum")

        def update(self, preds, target):
            self.v = torch.tensor(0.5)

        def compute(self):
            return self.v

    def fake_battery(num_classes=None, ignore_index=None):
        return {"score": Const()}

    monkeypatch.setitem(
        _tasks._TASKS,
        "detection",
        _tasks.TaskSpec(
            fake_battery,
            lambda m, x: (m(x), None),
            frozenset(),
            requires_num_classes=False,
        ),
    )

    class M(torch.nn.Module):
        def forward(self, x):
            return x

    data = [(torch.tensor([1.0]), torch.tensor([1.0]))]
    result = compare({"a": [M(), M()], "b": [M(), M()]}, data, task="detection")
    assert "score" in result.data.data_vars


def test_compare_rejects_one_shot_iterator():
    import pytest
    import torch

    from mushin.benchmark import compare

    one_shot = iter([(torch.randn(4, 4), torch.randint(0, 3, (4,)))])
    with pytest.raises(TypeError, match="re-iterable"):
        compare(
            methods={"a": [torch.nn.Linear(4, 3)]},
            data=one_shot,
            task="classification",
            num_classes=3,
        )


def test_compare_single_seed_warns_underpowered_and_no_significance():
    # n=1: the underpowered warning (from the default wilcoxon) must propagate
    # through the public compare() API, and nothing may be flagged significant.
    # (The stats engine's n=1 NaN handling is covered separately in test_stats.)
    data = _loader(seed=0)
    with pytest.warns(UserWarning, match="cannot reach alpha"):
        result = compare(
            methods={"a": [_Perfect(data)], "b": [torch.nn.Linear(4, 3)]},
            data=data,
            task="classification",
            num_classes=3,
        )  # default test == wilcoxon
    assert result.data.sizes["seed"] == 1
    assert "accuracy" in result.data.data_vars
    assert not result.comparisons["significant"].any()


def test_compare_flags_significant_difference_end_to_end():
    # positive significance through the public compare(): clearly-better models
    # across several seeds -> the accuracy comparison is flagged significant with
    # the correct sign. (Only ever asserted on synthetic dicts at the
    # compare_methods level before.)
    data = _loader(seed=0)
    torch.manual_seed(0)  # make the bad models' random init deterministic
    result = compare(
        methods={
            # high but non-constant accuracy across seeds, so significance is a real
            # test (a zero-variance method would be masked, not significant).
            "good": [_AlmostPerfect(data, n_wrong=k) for k in (0, 1, 2, 1, 2, 3)],
            "bad": [torch.nn.Linear(4, 3) for _ in range(6)],
        },
        data=data,
        task="classification",
        num_classes=3,
        test="welch",  # wilcoxon is underpowered at small n; welch can reach alpha
    )
    row = result.comparisons[result.comparisons["metric"] == "accuracy"].iloc[0]
    assert row["significant"]
    # mean_diff is method_a - method_b; identify the better method sign-robustly.
    better = row["method_a"] if row["mean_diff"] > 0 else row["method_b"]
    assert better == "good"


def test_compare_accepts_task_object():
    from torchmetrics.classification import MulticlassAccuracy

    from mushin.benchmark import compare
    from mushin.benchmark._tasks import Task

    data = _loader(seed=0)
    good = [_Perfect(data) for _ in range(3)]
    bad = [torch.nn.Linear(4, 3) for _ in range(3)]

    task = Task(
        battery=lambda num_classes, ignore_index=None: {
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro")
        },
        predict_fn=lambda model, x: (model(x).argmax(dim=-1), model(x).softmax(dim=-1)),
        description="acc-only classification",
    )
    result = compare(
        methods={"good": good, "bad": bad},
        data=data,
        task=task,
        num_classes=3,
    )
    assert isinstance(result, BenchmarkResult)
    assert "accuracy" in result.data


def test_compare_accepts_registered_task_name():
    from torchmetrics.classification import MulticlassAccuracy

    from mushin.benchmark import compare
    from mushin.benchmark._tasks import Task, register_task

    register_task(
        "acc_only",
        Task(
            battery=lambda num_classes, ignore_index=None: {
                "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro")
            },
            predict_fn=lambda model, x: (
                model(x).argmax(dim=-1),
                model(x).softmax(dim=-1),
            ),
        ),
        overwrite=True,
    )
    data = _loader(seed=1)
    models = [_Perfect(data) for _ in range(3)]
    result = compare(methods={"m": models}, data=data, task="acc_only", num_classes=3)
    assert "accuracy" in result.data
