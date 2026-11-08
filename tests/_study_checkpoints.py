# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""A small evaluation loader and saved linear models, shared by the Study
tests."""

import zlib

import torch
from torch.utils.data import DataLoader, TensorDataset


def loader(n=40, d=4, num_classes=3):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


def save_models(tmp_path, names, n_seeds=2, d=4, num_classes=3):
    """``{name: [checkpoint path per seed]}`` of freshly saved linear models."""
    checkpoints = {}
    for name in names:
        paths = []
        for s in range(n_seeds):
            # crc32, not hash(): a str's hash is salted per process, so the
            # weights would differ on every run.
            torch.manual_seed(zlib.crc32(f"{name}:{s}".encode()) % 1000)
            p = tmp_path / f"{name}_{s}.pt"
            torch.save(torch.nn.Linear(d, num_classes), p)
            paths.append(str(p))
        checkpoints[name] = paths
    return checkpoints
