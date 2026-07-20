"""Synthetic segmentation demo: compare two tiny segmentation models.

No heavy data needed — uses synthetic (N, C, H, W) inputs and (N, H, W) masks.

Run it:  python examples/segmentation_demo.py
Requires the eval extra:  pip install "mushin-py[eval]"

The reusable `run()` core is exercised by the test suite on tiny synthetic data.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from mushin.benchmark import BenchmarkResult, compare


def _tiny_seg_model(in_channels: int, num_classes: int) -> nn.Module:
    """A minimal (N, C_in, H, W) -> (N, num_classes, H, W) segmentation head."""
    return nn.Conv2d(in_channels, num_classes, kernel_size=1)


def _train(model: nn.Module, loader: DataLoader, epochs: int = 1) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            # y: (N, H, W) long tensor of class indices
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
    return model.eval()


# --8<-- [start:run]
def run(
    loader: DataLoader,
    *,
    in_channels: int = 3,
    num_classes: int = 4,
    seeds=(0, 1, 2),
) -> BenchmarkResult:
    """Train two tiny segmentation models per seed, then compare."""
    methods: dict[str, list[nn.Module]] = {"model_a": [], "model_b": []}
    for seed in seeds:
        torch.manual_seed(seed)
        methods["model_a"].append(
            _train(_tiny_seg_model(in_channels, num_classes), loader)
        )
        torch.manual_seed(seed + 100)
        methods["model_b"].append(
            _train(_tiny_seg_model(in_channels, num_classes), loader)
        )

    return compare(
        methods,
        data=loader,
        task="segmentation",
        num_classes=num_classes,
        test="welch",
    )


# --8<-- [end:run]


# --8<-- [start:dict_predict]
def torchvision_seg_predict(model, x):
    """Adapt a torchvision segmentation model (returns {"out": logits})."""
    logits = model(x)["out"]
    probs = logits.softmax(dim=1)
    return probs.argmax(dim=1), probs


# --8<-- [end:dict_predict]


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(42)
    N, C, H, W, num_classes = 16, 3, 16, 16, 4
    x = torch.randn(N, C, H, W, generator=g)
    masks = torch.randint(0, num_classes, (N, H, W), generator=g)
    loader = DataLoader(TensorDataset(x, masks), batch_size=8)

    result = run(loader, in_channels=C, num_classes=num_classes, seeds=(0, 1, 2))
    print(result.summary().to_string(index=False))
