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

    study = Study(
        methods={"m1": make_train("m1"), "m2": make_train("m2")},
        load_fn=lambda p: torch.load(p, weights_only=False),
        seeds=[0, 1, 2],
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
    from pathlib import Path

    assert study.working_dir == str((tmp_path / "run").resolve())
