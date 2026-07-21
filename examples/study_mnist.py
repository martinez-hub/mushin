"""Train two small classifiers with Study on MNIST and compare them.

Run it (downloads MNIST):  python examples/study_mnist.py
Requires the eval extra + torchvision:  pip install "mushin-py[eval]" torchvision

The reusable `run()` core is exercised by the test suite on tiny synthetic data,
so CI never downloads MNIST.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from mushin import Study
from mushin.benchmark import BenchmarkResult


# --8<-- [start:train_fn]
def _make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 10),
    )


def _make_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def _train_and_save(model: nn.Module, loader: DataLoader, path: Path) -> str:
    """Train model for one epoch and save checkpoint; return its path."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for x, y in loader:
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    torch.save(model, path)
    return str(path)


def make_train_fn(name: str, model_factory, loader: DataLoader, ckpt_dir: Path):
    """Return a train_fn(seed) -> checkpoint_path for Study."""

    def train_fn(seed: int) -> str:
        torch.manual_seed(seed)
        model = model_factory()
        path = ckpt_dir / f"{name}_seed{seed}.pt"
        return _train_and_save(model, loader, path)

    return train_fn


# --8<-- [end:train_fn]


# --8<-- [start:run]
def run(
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    seeds=(0, 1, 2),
    working_dir: str | os.PathLike[str],
) -> BenchmarkResult:
    """Train CNN and MLP across seeds on ``train_loader``, then compare them on
    the held-out ``test_loader``."""
    # Resolve to an absolute path: each train_fn runs inside Hydra's per-job
    # working directory, so a relative checkpoint dir would not point at the
    # directory we create here.
    work = Path(working_dir).resolve()
    ckpt_dir = work / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "cnn": make_train_fn("cnn", _make_cnn, train_loader, ckpt_dir),
        "mlp": make_train_fn("mlp", _make_mlp, train_loader, ckpt_dir),
    }

    study = Study(
        methods=methods,
        load_fn=lambda p: torch.load(p, weights_only=False),
        seeds=list(seeds),
        data=test_loader,
        num_classes=10,
        test="welch",
        working_dir=str(work),
    )
    return study.run()


# --8<-- [end:run]


def _load_mnist(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    tf = transforms.ToTensor()
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
    )


if __name__ == "__main__":
    train_loader, test_loader = _load_mnist()
    result = run(
        train_loader, test_loader, seeds=(0, 1, 2), working_dir="./study_output"
    )
    print(result.summary().to_string(index=False))
