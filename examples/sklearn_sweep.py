"""scikit-learn sweep -> labeled xarray dataset (mushin is framework-agnostic).

`MultiRunMetricsWorkflow` never inspects your model — it sweeps configurations
and collects the ``dict`` your ``task`` returns. So you can train a
**scikit-learn** model inside ``task`` (no torch, no Lightning in *your* code)
and still get results back as a labeled ``xarray.Dataset``, exactly as with a
torch model. mushin needs no scikit-learn integration for this — the sweep
layer is neutral. (mushin itself still imports torch for provenance if it is
installed; only your task body is framework-free.)

This sweeps ``LogisticRegression``'s inverse-regularization strength ``C`` across
seeds on a fixed synthetic 2-class dataset, recording held-out accuracy, and
returns an ``xarray.Dataset`` with dims ``(C, seed)``.

Run as a script to also print the mean accuracy per ``C``::

    python examples/sklearn_sweep.py
"""

from __future__ import annotations

from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

C_VALUES = [0.01, 0.1, 1.0, 10.0]
SEEDS = [0, 1, 2]


def _split(seed: int):
    x, y = make_classification(
        n_samples=400, n_features=20, n_informative=5, random_state=seed
    )
    return train_test_split(x, y, test_size=0.25, random_state=seed)


# --8<-- [start:workflow]
class LogRegCSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(C: float, seed: int) -> dict:
        x_train, x_test, y_train, y_test = _split(seed)
        model = LogisticRegression(C=C, max_iter=1000, random_state=seed)
        model.fit(x_train, y_train)
        # The returned dict populates the dataset — no torch, no file to save.
        return dict(accuracy=float(model.score(x_test, y_test)))


def build_dataset(working_dir: Path | None = None):
    """Run the ``C`` x ``seed`` scikit-learn sweep and return an ``xarray.Dataset``."""
    wf = LogRegCSweep()
    wf.run(
        C=multirun(C_VALUES),
        seed=multirun(SEEDS),
        working_dir=str(working_dir) if working_dir is not None else None,
    )
    return wf.to_xarray()


# --8<-- [end:workflow]


def main() -> None:
    ds = build_dataset()
    print(ds)

    mean_acc = ds["accuracy"].mean("seed")
    print("\nmean held-out accuracy by C:")
    print(mean_acc)


if __name__ == "__main__":
    main()
