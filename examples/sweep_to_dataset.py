"""Flagship example: run a sweep, get a labeled xarray dataset back.

Trains a tiny logistic-regression classifier on a fixed synthetic 2-class
dataset across a grid of learning rates and seeds, records validation accuracy,
and returns the results as an ``xarray.Dataset`` with dims ``(lr, seed)``.

Run as a script to also print the dataset and save a plot::

    python examples/sweep_to_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch as tr

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

LEARNING_RATES = [0.01, 0.1, 1.0]
SEEDS = [0, 1, 2]


def _make_data(seed: int, n: int = 256) -> tuple[tr.Tensor, tr.Tensor]:
    g = tr.Generator().manual_seed(seed)
    x0 = tr.randn(n, 2, generator=g) + tr.tensor([2.0, 2.0])
    x1 = tr.randn(n, 2, generator=g) + tr.tensor([-2.0, -2.0])
    x = tr.cat([x0, x1])
    y = tr.cat([tr.zeros(n), tr.ones(n)])
    return x, y


class LRSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr: float, seed: int) -> dict:
        tr.manual_seed(seed)
        x, y = _make_data(seed)
        model = tr.nn.Linear(2, 1)
        opt = tr.optim.SGD(model.parameters(), lr=lr)
        for _ in range(100):
            opt.zero_grad()
            logits = model(x).squeeze(1)
            loss = tr.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
        with tr.no_grad():
            preds = (model(x).squeeze(1) > 0).float()
            acc = (preds == y).float().mean().item()
        # returning the dict is what populates the dataset; saving is optional
        result = dict(accuracy=acc)
        tr.save(result, "metrics.pt")
        return result


def build_dataset(working_dir: Optional[Path] = None):
    """Run the learning-rate x seed sweep and return an ``xarray.Dataset``."""
    wf = LRSweep()
    wf.run(
        lr=multirun(LEARNING_RATES),
        seed=multirun(SEEDS),
        working_dir=str(working_dir) if working_dir is not None else None,
    )
    return wf.to_xarray()


if __name__ == "__main__":
    ds = build_dataset()
    print(ds)
