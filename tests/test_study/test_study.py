import torch
from torch.utils.data import DataLoader, TensorDataset

from mushin import Study
from mushin.benchmark import BenchmarkResult


def _loader(n=40, d=4, num_classes=3):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


def _save(tmp_path, names, n_seeds=2, d=4, num_classes=3):
    ckpts = {}
    for name in names:
        paths = []
        for s in range(n_seeds):
            torch.manual_seed(hash((name, s)) % 1000)
            p = tmp_path / f"{name}_{s}.pt"
            torch.save(torch.nn.Linear(d, num_classes), p)
            paths.append(str(p))
        ckpts[name] = paths
    return ckpts


def test_from_checkpoints_runs_and_compares(tmp_path):
    ckpts = _save(tmp_path, ["m1", "m2"])
    study = Study.from_checkpoints(
        checkpoints=ckpts,
        load_fn=lambda p: torch.load(p, weights_only=False),
        data=_loader(),
        num_classes=3,
        test="welch",
    )
    result = study.run()
    assert isinstance(result, BenchmarkResult)
    assert result.data.sizes == {"method": 2, "seed": 2}
    assert study.checkpoints == ckpts
    assert study.working_dir is None


def test_from_checkpoints_rejects_empty():
    import pytest

    with pytest.raises(ValueError, match="must not be empty"):
        Study.from_checkpoints({}, load_fn=lambda p: p, data=None, num_classes=2)


def test_full_motion_trains_then_compares(tmp_path):
    # train_fn just instantiates and saves a tiny model (no real training needed
    # to exercise the plumbing) and returns its checkpoint path.
    def make_train(name):
        def train(seed):
            torch.manual_seed(hash((name, seed)) % 1000)
            p = tmp_path / f"{name}_seed{seed}_raw.pt"
            torch.save(torch.nn.Linear(4, 3), p)
            return str(p)

        return train

    # Use non-arange seeds (5, 9, 17) so the seed coordinate bug is detectable:
    # with seeds=[0,1,2] the bug is hidden because arange(3) coincides with real seeds.
    explicit_seeds = [5, 9, 17]
    study = Study(
        methods={"m1": make_train("m1"), "m2": make_train("m2")},
        load_fn=lambda p: torch.load(p, weights_only=False),
        seeds=explicit_seeds,
        data=_loader(),
        num_classes=3,
        test="welch",
        working_dir=str(tmp_path / "run"),
    )
    result = study.run()

    assert isinstance(result, BenchmarkResult)
    assert result.data.sizes == {"method": 2, "seed": 3}
    assert set(study.checkpoints) == {"m1", "m2"}
    assert all(len(v) == 3 for v in study.checkpoints.values())
    # full-motion run records the resolved working directory (not left as the input)
    assert study.working_dir == str((tmp_path / "run").resolve())
    # seed coordinate must carry the real seed values, not arange(n_seeds)
    assert result.data.coords["seed"].values.tolist() == explicit_seeds


def test_study_on_error_nan_raises_incomplete_sweep(tmp_path):
    """A training sweep requested with on_error="nan" whose train_fn fails for one
    seed must surface as IncompleteSweepError from Study.run (the gate lives in
    run_training_sweep), never a silently-compared partial result."""
    import pytest

    from mushin.benchmark import IncompleteSweepError

    def make_train(name):
        def train(seed):
            if name == "m1" and seed == 9:
                raise RuntimeError("boom")
            torch.manual_seed(hash((name, seed)) % 1000)
            p = tmp_path / f"{name}_seed{seed}_raw.pt"
            torch.save(torch.nn.Linear(4, 3), p)
            return str(p)

        return train

    study = Study(
        methods={"m1": make_train("m1"), "m2": make_train("m2")},
        load_fn=lambda p: torch.load(p, weights_only=False),
        seeds=[5, 9],
        data=_loader(),
        num_classes=3,
        test="welch",
        working_dir=str(tmp_path / "run"),
        on_error="nan",
    )
    with pytest.raises(IncompleteSweepError):
        study.run()


def test_study_default_on_error_raise_propagates(tmp_path):
    """Default on_error="raise" is unchanged: the train_fn's own exception
    propagates (not swallowed into an IncompleteSweepError)."""
    import pytest

    def make_train(name):
        def train(seed):
            if name == "m1" and seed == 9:
                raise RuntimeError("boom")
            torch.manual_seed(hash((name, seed)) % 1000)
            p = tmp_path / f"{name}_seed{seed}_raw.pt"
            torch.save(torch.nn.Linear(4, 3), p)
            return str(p)

        return train

    study = Study(
        methods={"m1": make_train("m1"), "m2": make_train("m2")},
        load_fn=lambda p: torch.load(p, weights_only=False),
        seeds=[5, 9],
        data=_loader(),
        num_classes=3,
        test="welch",
        working_dir=str(tmp_path / "run"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        study.run()


class _PerfectSegmenter(torch.nn.Module):
    """Returns logits that perfectly reproduce the provided mask (clamped to num_classes)."""

    def __init__(self, x_ref, masks_ref):
        super().__init__()
        self._m = {tuple(xi.flatten().tolist()): mi for xi, mi in zip(x_ref, masks_ref)}

    def forward(self, xb):
        out = []
        for xi in xb:
            m = self._m[tuple(xi.flatten().tolist())].clamp(max=2)
            out.append(torch.nn.functional.one_hot(m, 3).permute(2, 0, 1).float() * 10)
        return torch.stack(out)


def test_study_forwards_ignore_index_for_segmentation(tmp_path):
    x = torch.randn(8, 1, 8, 8)
    masks = torch.randint(0, 3, (8, 8, 8))
    masks[:, 0, 0] = 255  # void pixels at (0,0)
    loader = DataLoader(TensorDataset(x, masks), batch_size=4)

    ckpts = {}
    for name in ("a", "b"):
        paths = []
        for s in range(2):
            p = tmp_path / f"{name}_{s}.pt"
            torch.save(_PerfectSegmenter(x, masks), p)
            paths.append(str(p))
        ckpts[name] = paths

    study = Study.from_checkpoints(
        checkpoints=ckpts,
        load_fn=lambda p: torch.load(p, weights_only=False),
        data=loader,
        task="segmentation",
        num_classes=3,
        test="welch",
        ignore_index=255,
    )
    result = study.run()
    assert float(result.data["pixel_acc"].mean()) == 1.0


def test_study_from_checkpoints_accepts_task_object(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torchmetrics.classification import MulticlassAccuracy

    from mushin import Study
    from mushin.benchmark._tasks import Task

    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 4, generator=g)
    y = torch.randint(0, 3, (32,), generator=g)
    data = DataLoader(TensorDataset(x, y), batch_size=16)

    def load_fn(_path):
        return torch.nn.Linear(4, 3)

    task = Task(
        battery=lambda num_classes, ignore_index=None: {
            "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro")
        },
        predict_fn=lambda model, x: (
            model(x).argmax(dim=-1),
            model(x).softmax(dim=-1),
        ),
    )
    study = Study.from_checkpoints(
        checkpoints={"m": ["a.ckpt", "b.ckpt", "c.ckpt"]},
        load_fn=load_fn,
        data=data,
        num_classes=3,
        task=task,
    )
    result = study.run()
    assert "accuracy" in result.data


def test_from_checkpoints_classless_task_no_num_classes(tmp_path):
    # Classless tasks (requires_num_classes=False) must run through Study WITHOUT a
    # num_classes argument — it now defaults to None and is forwarded to compare().
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin import Study

    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 1, generator=g)
    y = x[:, 0] * 2.0 + 1.0
    data = DataLoader(TensorDataset(x, y), batch_size=8)

    class _Affine(torch.nn.Module):
        def forward(self, t):
            return t[:, 0] * 2.0 + 1.0

    study = Study.from_checkpoints(
        checkpoints={"m": ["a", "b", "c"]},
        load_fn=lambda _p: _Affine(),
        data=data,
        task="regression",  # no num_classes passed
    )
    result = study.run()
    assert "mse" in result.data
