"""Compare two small classifiers on MNIST with statistical significance.

Run it (downloads MNIST):  python examples/compare_classifiers.py

The reusable `run()` core is exercised by the test suite on tiny synthetic data,
so CI never downloads MNIST.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from mushin.benchmark import BenchmarkResult, compare


# --8<-- [start:models]
def small_cnn() -> nn.Module:
    """A tiny convolutional classifier for 1x28x28 images."""
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 10),
    )


def mlp() -> nn.Module:
    """A tiny fully-connected classifier."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
# --8<-- [end:models]


def _train(model: nn.Module, loader: DataLoader, epochs: int = 1) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
    return model.eval()


# --8<-- [start:run]
def run(
    train_loader: DataLoader, test_loader: DataLoader, *, seeds=(0, 1, 2)
) -> BenchmarkResult:
    """Train one CNN and one MLP per seed, then compare them with statistics."""
    methods: dict[str, list[nn.Module]] = {"cnn": [], "mlp": []}
    for seed in seeds:
        torch.manual_seed(seed)
        methods["cnn"].append(_train(small_cnn(), train_loader))
        methods["mlp"].append(_train(mlp(), train_loader))

    return compare(
        methods,
        data=test_loader,
        task="classification",
        num_classes=10,
        test="welch",
    )
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
    result = run(train_loader, test_loader)
    print(result.summary().to_string(index=False))
