# Flagship Example + Quickstart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the flagship "run a sweep, get a dataset" example — runnable, tested, and wired into a README Quickstart — and reset the version to 0.1.0, so the positioning's lead message is provable in ~10 lines.

**Architecture:** Add a self-contained `examples/sweep_to_dataset.py` that trains a tiny logistic-regression classifier on synthetic data across a learning-rate × seed grid using `mushin.workflows.MultiRunMetricsWorkflow`, then returns the aggregated results as an `xarray.Dataset`. A test executes the example and asserts the dataset's shape. The README Quickstart presents the same example as the product pitch.

**Tech Stack:** Python, mushin (MultiRunMetricsWorkflow + `multirun`), PyTorch (tiny CPU model), xarray, matplotlib (plot, script-only), pytest.

**Governance note:** `main` is branch-protected (PR + green CI required). Do all work on a branch (e.g. `flagship-example`) and open a PR; the "Commit" steps create commits on that branch. Do not add AI authorship/attribution to commits.

---

## Background the implementer must know

- `MultiRunMetricsWorkflow` is subclassed; you define a `@staticmethod task(...)`
  that returns a `dict[str, number | sequence]`. The sweep is launched by
  `wf.run(param=multirun([...]), ...)`, where each `param` becomes a sweep
  dimension.
- After `run()`, `wf.metrics` is built from **each task's return value** (not from
  a file). `wf.to_xarray()` returns an `xarray.Dataset` whose dims/coords are the
  swept parameters and whose data-variables are the returned metric keys.
- `run()` accepts a keyword `working_dir: str | None`; sweep params are passed as
  extra keywords. Example: `wf.run(lr=multirun([...]), seed=multirun([...]))`.
- Tests can run a workflow inside a temp dir using the existing `cleandir`
  fixture (defined in `tests/conftest.py`), which `chdir`s into `tmp_path`.

## File Structure

- Create: `examples/sweep_to_dataset.py` — the flagship example (importable
  `build_dataset()` + script `main()`).
- Create: `tests/test_examples.py` — executes the example, asserts the dataset.
- Modify: `README.md` — add a "Quickstart" section after the badges/intro.
- Modify: `pyproject.toml` — `version = "0.4.0"` → `"0.1.0"`.
- Modify: `CHANGELOG.md` — re-version the unreleased `0.4.0` entry to `0.1.0`.

---

### Task 1: Flagship example returns a labeled dataset

**Files:**
- Create: `examples/sweep_to_dataset.py`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_examples.py
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))


@pytest.mark.usefixtures("cleandir")
def test_build_dataset_returns_labeled_grid():
    import sweep_to_dataset as ex

    ds = ex.build_dataset()

    # dims are the swept parameters; data var is the returned metric
    assert set(ds.dims) == {"lr", "seed"}
    assert ds.sizes == {"lr": len(ex.LEARNING_RATES), "seed": len(ex.SEEDS)}
    assert "accuracy" in ds.data_vars
    # accuracy is a probability in [0, 1]
    assert float(ds["accuracy"].min()) >= 0.0
    assert float(ds["accuracy"].max()) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_examples.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sweep_to_dataset'`.

- [ ] **Step 3: Write the example module**

```python
# examples/sweep_to_dataset.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_examples.py -q`
Expected: PASS (1 passed). The sweep launches 9 Hydra jobs in a temp dir.

- [ ] **Step 5: Commit**

```bash
git add examples/sweep_to_dataset.py tests/test_examples.py
git commit -m "Add flagship sweep-to-dataset example with test"
```

---

### Task 2: Script entry point prints the dataset and saves a plot

**Files:**
- Modify: `examples/sweep_to_dataset.py`
- Test: `tests/test_examples.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_examples.py`:

```python
@pytest.mark.usefixtures("cleandir")
def test_main_writes_plot():
    import matplotlib

    matplotlib.use("Agg")  # headless backend for CI
    import sweep_to_dataset as ex

    ex.main()
    assert Path("sweep_accuracy.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_examples.py::test_main_writes_plot -q`
Expected: FAIL with `AttributeError: module 'sweep_to_dataset' has no attribute 'main'`.

- [ ] **Step 3: Add `main()` and the script guard**

Append to `examples/sweep_to_dataset.py`:

```python
def main() -> None:
    ds = build_dataset()
    print(ds)

    # mean accuracy across seeds, as a function of learning rate
    mean_acc = ds["accuracy"].mean("seed")
    print("\nmean accuracy by learning rate:")
    print(mean_acc)

    import matplotlib.pyplot as plt

    mean_acc.plot.line(x="lr", marker="o")
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("mean accuracy")
    plt.savefig("sweep_accuracy.png", dpi=120, bbox_inches="tight")
    print("\nsaved plot to sweep_accuracy.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_examples.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the example as a real script (smoke check)**

Run: `cd /tmp && uv run --project "$OLDPWD" python "$OLDPWD/examples/sweep_to_dataset.py"`
Expected: prints an `<xarray.Dataset>` with dims `(lr: 3, seed: 3)` and a
`accuracy` data variable, then "saved plot to sweep_accuracy.png".

- [ ] **Step 6: Commit**

```bash
git add examples/sweep_to_dataset.py tests/test_examples.py
git commit -m "Add runnable script entry point with plot to example"
```

---

### Task 3: README Quickstart built around the example

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the Quickstart section**

Insert the following markdown immediately **before** the existing
`## What it provides` section in `README.md`. (The outer fence below is `~~~`
only so this plan renders; in the README the content keeps its ` ``` ` fences.)

~~~markdown
## Quickstart: run a sweep, get a dataset

Define your experiment as a function, sweep over parameters, and get the results
back as a labeled `xarray.Dataset` — not rows in a dashboard you have to export.

```python
import torch as tr
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

class LRSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr: float, seed: int) -> dict:
        tr.manual_seed(seed)
        # ... train a model with this lr/seed ...
        return dict(accuracy=acc)  # whatever you return becomes a data variable

wf = LRSweep()
wf.run(lr=multirun([0.01, 0.1, 1.0]), seed=multirun([0, 1, 2]))  # 9 runs

ds = wf.to_xarray()
# <xarray.Dataset> Dimensions: (lr: 3, seed: 3)
#   Data variables: accuracy (lr, seed)

ds["accuracy"].mean("seed")   # average over seeds, per learning rate
```

The full runnable version is in [`examples/sweep_to_dataset.py`](examples/sweep_to_dataset.py):

```bash
uv run python examples/sweep_to_dataset.py
```
~~~

- [ ] **Step 2: Verify the fenced code blocks are balanced**

Run: `uv run python -c "print('balanced' if open('README.md').read().count(chr(96)*3) % 2 == 0 else 'UNBALANCED')"`
Expected: prints `balanced`.

- [ ] **Step 3: Spell-check the README**

Run: `uv run codespell README.md`
Expected: no output (clean).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Add Quickstart section to README"
```

---

### Task 4: Reset version to 0.1.0

**Files:**
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump the version in pyproject.toml**

In `pyproject.toml`, change:

```toml
version = "0.4.0"
```

to:

```toml
version = "0.1.0"
```

- [ ] **Step 2: Verify uv reports the new version**

Run: `uv version --short`
Expected: prints `0.1.0`.

- [ ] **Step 3: Re-version the changelog entry**

In `CHANGELOG.md`, change the heading `## [0.4.0] - 2026-06-22` to
`## [0.1.0] - 2026-06-22`, and update the two link references at the bottom:

```markdown
[Unreleased]: https://github.com/martinez-hub/mushin/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/martinez-hub/mushin/releases/tag/v0.1.0
```

- [ ] **Step 4: Run the full gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run codespell src tests examples README.md CHANGELOG.md && uv run pytest tests/ --hypothesis-profile fast -p no:cacheprovider -q`
Expected: ruff clean, format clean, codespell clean, all tests pass (incl. the 2 new example tests).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Reset version to 0.1.0 for fresh fork lineage"
```

- [ ] **Step 6: Open the PR**

```bash
git push -u origin flagship-example
gh pr create --base main --title "Flagship example, Quickstart, and 0.1.0" \
  --body "Adds the runnable, tested sweep-to-dataset example, a README Quickstart built around it, and resets the version to 0.1.0."
```

Then wait for CI to pass and squash-merge (linear history required).

---

## Out of scope (non-engineering proof points — owner-driven)

These are part of "earn the position" but are not code tasks:

- **Dogfood:** run a real research workflow of your own through mushin once.
- **hydra-zen outreach:** contact the maintainers re: coordinate-vs-compete.
- **PyPI publish:** register the pending publisher, then tag `v0.1.0` and create
  the GitHub Release (the publish workflow does the rest).

## Self-review notes

- Spec coverage: proof points #1 (killer example → Tasks 1-2), #4 (docs/quickstart
  → Task 3), #5 (version reset → Task 4). Proof points #2, #3, #6 are owner-driven
  and listed as out of scope above.
- The example's `task` returns the metric dict (populates `to_xarray`) and also
  saves it (demonstrates the reload path); both behaviors match the workflow API.
- Types/names consistent across tasks: `build_dataset`, `main`, `LEARNING_RATES`,
  `SEEDS`, data variable `accuracy`, dims `lr`/`seed`.
