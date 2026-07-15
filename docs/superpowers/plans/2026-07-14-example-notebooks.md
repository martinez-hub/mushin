# EQUINE-style Example Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship six runnable Jupyter notebooks — narrative + real code + committed outputs + inline plots — rendered in the docs site and CI-executed so they can never rot.

**Architecture:** Real `.ipynb` files under `docs/notebooks/`, each authored from a jupytext "percent" `.py` source that is executed to populate outputs, then metadata-stripped with `nbstripout --keep-output`. `mkdocs-jupyter` renders the committed outputs at docs-build time (`execute: false`, so the docs build needs no torch). A new `notebooks` CI job runs `pytest --nbmake docs/notebooks/` to *execute* every notebook and fail on any cell error.

**Tech Stack:** Python, torch, mushin, matplotlib (`viz` extra), jupytext + nbconvert + ipykernel + nbstripout + nbmake (dev group), mkdocs-jupyter (docs group), mkdocs-material.

**Spec:** `docs/superpowers/specs/2026-07-14-example-notebooks-design.md`

---

## Conventions used by every notebook task

Each notebook is authored the same way. The plan gives the **full percent-format source**; the build/verify/commit steps are identical, so they are written out once here and referenced (with exact commands) in each task.

**Authoring → committed notebook (run from repo root):**

```bash
# 1. Write the percent source to the scratchpad (throwaway; NOT committed)
#    (the task provides the exact file content)
# 2. Convert + execute into the committed .ipynb
uv run jupytext --to notebook --execute \
  --output docs/notebooks/<NN_name>.ipynb <scratch>/<NN_name>.py
# 3. Strip nondeterministic metadata but KEEP the outputs (for display)
uv run nbstripout --keep-output docs/notebooks/<NN_name>.ipynb
```

`<scratch>` is the session scratchpad directory. Only the `.ipynb` is committed.

**Rules baked into every notebook source:**
- Fully seeded (`torch.manual_seed` / `Generator().manual_seed`) so outputs are stable.
- Any `working_dir` uses `tempfile.mkdtemp()` — never a repo-relative path — so no run artifacts land in the tree.
- Tiny compute only (≤ a few hundred synthetic points, ≤ 100 optimizer steps) so nbmake stays fast.
- Every plotting cell ends with `plt.show()` so the inline backend embeds a PNG output.
- First cell after the title is a markdown "what this builds"; last cell is a markdown "See also" linking the relevant guide(s).

---

## Task 1: Tooling & dependencies

**Files:**
- Modify: `pyproject.toml` (`[dependency-groups]` `docs` + `dev`)
- Modify: `.pre-commit-config.yaml`
- Modify: `.gitignore`
- Create: `docs/notebooks/.gitkeep` (placeholder so the dir exists before the first notebook)

- [ ] **Step 1: Add render + authoring/execution deps**

In `pyproject.toml`, add `mkdocs-jupyter` to the `docs` group:

```toml
docs = [
    "mkdocs-material >= 9.5, < 10",
    "mkdocstrings[python]>=0.26,<2",
    # Renders committed .ipynb notebooks under docs/notebooks/ into the site.
    # execute:false in mkdocs.yml — the build shows the committed outputs and
    # never runs torch. Notebooks are executed separately by the `notebooks` CI
    # job (nbmake) to prevent rot.
    "mkdocs-jupyter >= 0.24",
]
```

Add the notebook toolchain to the `dev` group (append inside the existing `dev = [ ... ]` list, before the closing `]`):

```toml
    # Example-notebook toolchain (docs/notebooks/). jupytext converts the
    # percent-format .py sources into .ipynb and executes them; nbstripout keeps
    # outputs but strips nondeterministic metadata; nbmake (a pytest plugin)
    # re-executes every notebook in CI so they can't rot. nbconvert+ipykernel
    # provide the execution kernel.
    "jupytext >= 1.16",
    "nbconvert >= 7.0",
    "ipykernel >= 6.0",
    "nbstripout >= 0.7",
    "nbmake >= 1.5",
```

- [ ] **Step 2: Add the nbstripout pre-commit hook**

In `.pre-commit-config.yaml`, append a new repo block (keeps outputs, strips metadata, so committed notebooks stay clean):

```yaml
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
        args: ["--keep-output"]
```

- [ ] **Step 3: Ignore notebook checkpoint dirs**

Append to `.gitignore`:

```gitignore

# Jupyter
.ipynb_checkpoints/
```

- [ ] **Step 4: Create the notebooks directory**

Create `docs/notebooks/.gitkeep` (empty file).

- [ ] **Step 5: Sync and verify the tools resolve and import**

Run:
```bash
uv sync --group dev --group docs
# Register a `python3` kernelspec in this env so jupytext --execute / nbconvert /
# nbmake can find a kernel (a fresh uv venv registers none). Idempotent.
uv run python -m ipykernel install --user --name python3 --display-name "Python 3"
uv run jupytext --version
uv run nbstripout --version
uv run python -c "import nbconvert, ipykernel, nbmake; print('ok')"
uv run python -c "import mkdocs_jupyter; print('mkdocs-jupyter ok')"
```
Expected: each prints a version / `ok` with exit code 0. The kernel install prints
`Installed kernelspec python3 in …`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .pre-commit-config.yaml .gitignore docs/notebooks/.gitkeep
git commit -m "build: add notebook toolchain (mkdocs-jupyter, jupytext, nbmake, nbstripout)"
```

---

## Task 2: mkdocs-jupyter plugin + nav + Notebook 01 (sweep → dataset)

This task also proves the whole render pipeline end-to-end on the first notebook.

**Files:**
- Modify: `mkdocs.yml` (plugins + nav)
- Create: `docs/notebooks/01_sweep_to_dataset.ipynb` (built from scratch source)

- [ ] **Step 1: Register the plugin**

In `mkdocs.yml`, add `mkdocs-jupyter` to the `plugins:` list (after `search`, before `mkdocstrings`):

```yaml
plugins:
  - search
  - mkdocs-jupyter:
      execute: false
      include_source: true
  - mkdocstrings:
```

- [ ] **Step 2: Add the "Example notebooks" nav section**

In `mkdocs.yml`, insert a new top-level nav entry **between** `- Examples: examples.md` and `- API Reference:`:

```yaml
  - Example notebooks:
      - Sweeps → datasets: notebooks/01_sweep_to_dataset.ipynb
```

(Later tasks append their notebook to this section.)

- [ ] **Step 3: Write the notebook source**

Write to `<scratch>/01_sweep_to_dataset.py`:

```python
# %% [markdown]
# # Sweeps → labeled datasets
#
# The core mushin move: define a `task(**config)`, sweep it over a grid with
# `multirun`, and get the results back as a **labeled `xarray.Dataset`** whose
# dimensions are exactly the parameters you swept. No manual bookkeeping of runs.
#
# Here we train a tiny logistic-regression classifier on a fixed synthetic
# 2-class problem across a grid of **learning rates × seeds**, record training
# accuracy, and read the result as a dataset with dims `(lr, seed)`.

# %%
from __future__ import annotations

import tempfile

import torch as tr

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

# %% [markdown]
# A fixed, seeded synthetic dataset — two Gaussian blobs, one per class.

# %%
POINTS_PER_CLASS = 256


def make_data(seed: int, n: int = POINTS_PER_CLASS):
    g = tr.Generator().manual_seed(seed)
    x0 = tr.randn(n, 2, generator=g) + tr.tensor([2.0, 2.0])
    x1 = tr.randn(n, 2, generator=g) + tr.tensor([-2.0, -2.0])
    x = tr.cat([x0, x1])
    y = tr.cat([tr.zeros(n), tr.ones(n)])
    return x, y

# %% [markdown]
# The workflow: subclass `MultiRunMetricsWorkflow` and implement `task`. Whatever
# `dict` it returns becomes variables in the dataset.

# %%
class LRSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr: float, seed: int) -> dict:
        tr.manual_seed(seed)
        x, y = make_data(seed)
        model = tr.nn.Linear(2, 1)
        opt = tr.optim.SGD(model.parameters(), lr=lr)
        for _ in range(100):
            opt.zero_grad()
            logits = model(x).squeeze(1)
            tr.nn.functional.binary_cross_entropy_with_logits(logits, y).backward()
            opt.step()
        with tr.no_grad():
            acc = ((model(x).squeeze(1) > 0).float() == y).float().mean().item()
        return dict(accuracy=acc)

# %% [markdown]
# Run the `lr × seed` sweep with `multirun(...)` and read it back as a dataset.

# %%
wf = LRSweep()
wf.run(
    lr=multirun([0.01, 0.1, 1.0]),
    seed=multirun([0, 1, 2]),
    working_dir=tempfile.mkdtemp(),
)
ds = wf.to_xarray()
ds

# %% [markdown]
# The dataset is fully labeled — reduce over `seed` to get mean accuracy per
# learning rate.

# %%
mean_acc = ds["accuracy"].mean("seed")
mean_acc

# %% [markdown]
# And the payoff: a metric-vs-parameter curve with the per-seed spread.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3.2))
for s in ds["seed"].values:
    ax.plot(ds["lr"].values, ds["accuracy"].sel(seed=s).values, "o-",
            alpha=0.35, color="tab:blue")
ax.plot(ds["lr"].values, mean_acc.values, "ks-", lw=2, label="mean")
ax.set_xscale("log")
ax.set_xlabel("learning rate")
ax.set_ylabel("train accuracy")
ax.set_title("Accuracy vs learning rate (per-seed + mean)")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **See also**
#
# - [Workflows & sweeps guide](../guides/workflows.md) — the full sweep API.
# - [Notebook 06](06_sklearn_framework_agnostic.ipynb) — the same flow with a
#   scikit-learn model (no torch): the sweep layer is framework-agnostic.
```

- [ ] **Step 4: Build the committed notebook**

Run:
```bash
uv run jupytext --to notebook --execute \
  --output docs/notebooks/01_sweep_to_dataset.ipynb <scratch>/01_sweep_to_dataset.py
uv run nbstripout --keep-output docs/notebooks/01_sweep_to_dataset.ipynb
```
Expected: exit 0; the `.ipynb` now contains code cells with outputs and one `image/png` output for the plot.

- [ ] **Step 5: Verify nbmake executes it cleanly**

Run:
```bash
uv run --extra viz pytest --nbmake docs/notebooks/01_sweep_to_dataset.ipynb -p no:cacheprovider
```
Expected: `1 passed`.

- [ ] **Step 6: Verify the docs build renders it (strict)**

Run:
```bash
uv run --group docs mkdocs build --strict
```
Expected: exit 0, no warnings. (Confirms the plugin is wired and the nav entry resolves.)

- [ ] **Step 7: Commit**

```bash
git add mkdocs.yml docs/notebooks/01_sweep_to_dataset.ipynb
git commit -m "docs: add example notebook 01 (sweep to dataset) + mkdocs-jupyter"
```

---

## Task 3: Notebook 02 (compare + batteries)

**Files:**
- Modify: `mkdocs.yml` (append nav entry)
- Create: `docs/notebooks/02_compare_and_batteries.ipynb`

- [ ] **Step 1: Append the nav entry**

Under `- Example notebooks:` in `mkdocs.yml`, add after the notebook-01 line:

```yaml
      - Compare & batteries: notebooks/02_compare_and_batteries.ipynb
```

- [ ] **Step 2: Write the notebook source**

Write to `<scratch>/02_compare_and_batteries.py`:

```python
# %% [markdown]
# # Comparing methods, with statistics + built-in batteries
#
# `compare(...)` trains-free evaluation: give it a few trained models per method
# (one per seed), a data loader, and a task name; it computes the task's metrics
# per seed and runs a significance test between methods. The result is a
# `BenchmarkResult` with a tidy `.summary()` and pairwise `.comparisons`.

# %%
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mushin.benchmark import compare

# %% [markdown]
# Tiny synthetic 3-class image data (so this runs in seconds, no downloads).

# %%
NUM_CLASSES = 3


def make_loader(n: int = 256, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    # class-correlated signal so a model can actually learn something
    x = torch.randn(n, 1, 8, 8, generator=g) + y.view(n, 1, 1, 1)
    return DataLoader(TensorDataset(x, y), batch_size=64)


train_loader = make_loader(seed=0)
test_loader = make_loader(n=128, seed=99)

# %% [markdown]
# Two tiny classifiers — a conv net and an MLP.

# %%
def small_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        nn.Linear(8 * 4 * 4, NUM_CLASSES),
    )


def mlp() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(), nn.Linear(8 * 8, 32), nn.ReLU(), nn.Linear(32, NUM_CLASSES)
    )


def train(model: nn.Module) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(3):
        for x, y in train_loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
    return model.eval()

# %% [markdown]
# Train one of each per seed, then `compare` them on the held-out loader.

# %%
methods = {"cnn": [], "mlp": []}
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    methods["cnn"].append(train(small_cnn()))
    methods["mlp"].append(train(mlp()))

result = compare(
    methods, data=test_loader, task="classification",
    num_classes=NUM_CLASSES, test="welch",
)
result.summary()

# %% [markdown]
# The pairwise comparison, with effect size and (corrected) p-value.

# %%
result.comparisons

# %% [markdown]
# Per-method mean accuracy with 95% confidence intervals, straight from
# `.summary()`.

# %%
import matplotlib.pyplot as plt

acc = result.summary().query("metric == 'accuracy'")
fig, ax = plt.subplots(figsize=(4.2, 3.2))
lo = (acc["mean"] - acc["ci_low"]).values
hi = (acc["ci_high"] - acc["mean"]).values
ax.bar(acc["method"], acc["mean"], yerr=[lo, hi], capsize=6,
       color=["tab:blue", "tab:orange"])
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1)
ax.set_title("Accuracy by method (95% CI)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## A second battery: regression
#
# The task name is the only thing that changes. Regression needs no
# `num_classes`; the passthrough predict_fn feeds `model(x)` straight to the
# metrics.

# %%
gg = torch.Generator().manual_seed(0)
xr = torch.randn(64, 1, generator=gg)
yr = xr[:, 0] * 2.0 + 1.0
reg_loader = DataLoader(TensorDataset(xr, yr), batch_size=32)


class Affine(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w, self.b = w, b

    def forward(self, x):
        return x[:, 0] * self.w + self.b


reg_methods = {
    "good": [Affine(2.0, 1.0) for _ in range(3)],
    "bad": [Affine(0.0, 0.0) for _ in range(3)],
}
compare(reg_methods, data=reg_loader, task="regression", test="welch").summary()

# %% [markdown]
# **See also**
#
# - [Comparing methods guide](../guides/compare.md) — every knob on `compare`.
# - [Built-in batteries guide](../guides/batteries.md) — all seven tasks with
#   real-model recipes (SAM, YOLO-World, CLIP, …).
```

- [ ] **Step 3: Build the committed notebook**

```bash
uv run jupytext --to notebook --execute \
  --output docs/notebooks/02_compare_and_batteries.ipynb <scratch>/02_compare_and_batteries.py
uv run nbstripout --keep-output docs/notebooks/02_compare_and_batteries.ipynb
```

- [ ] **Step 4: Verify nbmake + strict docs build**

```bash
uv run --extra viz pytest --nbmake docs/notebooks/02_compare_and_batteries.ipynb -p no:cacheprovider
uv run --group docs mkdocs build --strict
```
Expected: `1 passed`; docs build exit 0.

- [ ] **Step 5: Commit**

```bash
git add mkdocs.yml docs/notebooks/02_compare_and_batteries.ipynb
git commit -m "docs: add example notebook 02 (compare + batteries)"
```

---

## Task 4: Notebook 03 (Study)

**Files:**
- Modify: `mkdocs.yml` (append nav entry)
- Create: `docs/notebooks/03_study.ipynb`

- [ ] **Step 1: Append the nav entry**

```yaml
      - Studies: notebooks/03_study.ipynb
```

- [ ] **Step 2: Write the notebook source**

Write to `<scratch>/03_study.py`:

```python
# %% [markdown]
# # Studies: train + compare in one call
#
# `compare` assumes you already have trained models. `Study` closes the loop: you
# give it a `train_fn(seed) -> checkpoint_path` per method and the seeds, and it
# runs the whole multi-seed training sweep **and** the comparison in a single
# call, returning the same `BenchmarkResult`.

# %%
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mushin import Study

# %% [markdown]
# Tiny synthetic 3-class data again (no downloads).

# %%
NUM_CLASSES = 3


def make_loader(n: int = 256, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    x = torch.randn(n, 1, 8, 8, generator=g) + y.view(n, 1, 1, 1)
    return DataLoader(TensorDataset(x, y), batch_size=64)


train_loader = make_loader(seed=0)
test_loader = make_loader(n=128, seed=99)

# %% [markdown]
# A `train_fn` per method: train, save a checkpoint, return its path.

# %%
def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        nn.Linear(8 * 4 * 4, NUM_CLASSES),
    )


def make_mlp():
    return nn.Sequential(
        nn.Flatten(), nn.Linear(8 * 8, 32), nn.ReLU(), nn.Linear(32, NUM_CLASSES)
    )


ckpt_dir = Path(tempfile.mkdtemp())


def make_train_fn(name, factory):
    def train_fn(seed: int) -> str:
        torch.manual_seed(seed)
        model = factory()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(3):
            for x, y in train_loader:
                opt.zero_grad()
                nn.functional.cross_entropy(model(x), y).backward()
                opt.step()
        path = ckpt_dir / f"{name}_seed{seed}.pt"
        torch.save(model.eval(), path)
        return str(path)

    return train_fn

# %% [markdown]
# Hand the methods, loader, and seeds to `Study` — it does the rest.

# %%
study = Study(
    methods={
        "cnn": make_train_fn("cnn", make_cnn),
        "mlp": make_train_fn("mlp", make_mlp),
    },
    load_fn=lambda p: torch.load(p, weights_only=False),
    seeds=[0, 1, 2],
    data=test_loader,
    num_classes=NUM_CLASSES,
    test="welch",
    working_dir=tempfile.mkdtemp(),
)
result = study.run()
result.summary()

# %% [markdown]
# Same `BenchmarkResult` as `compare` — plot the per-method accuracy with CIs.

# %%
import matplotlib.pyplot as plt

acc = result.summary().query("metric == 'accuracy'")
fig, ax = plt.subplots(figsize=(4.2, 3.2))
lo = (acc["mean"] - acc["ci_low"]).values
hi = (acc["ci_high"] - acc["mean"]).values
ax.bar(acc["method"], acc["mean"], yerr=[lo, hi], capsize=6,
       color=["tab:blue", "tab:orange"])
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1)
ax.set_title("Study: accuracy by method (95% CI)")
fig.tight_layout()
plt.show()

# %% [markdown]
# **See also**
#
# - [Studies guide](../guides/study.md) — resilience options, provenance, and
#   wiring `Study` to real training loops.
# - [Notebook 02](02_compare_and_batteries.ipynb) — the `compare`-only flow.
```

- [ ] **Step 3: Build, verify, commit**

```bash
uv run jupytext --to notebook --execute \
  --output docs/notebooks/03_study.ipynb <scratch>/03_study.py
uv run nbstripout --keep-output docs/notebooks/03_study.ipynb
uv run --extra viz pytest --nbmake docs/notebooks/03_study.ipynb -p no:cacheprovider
uv run --group docs mkdocs build --strict
git add mkdocs.yml docs/notebooks/03_study.ipynb
git commit -m "docs: add example notebook 03 (Study)"
```
Expected: `1 passed`; docs build exit 0.

---

## Task 5: Notebook 04 (resilient & resumable sweeps)

**Files:**
- Modify: `mkdocs.yml` (append nav entry)
- Create: `docs/notebooks/04_resilience.ipynb`

- [ ] **Step 1: Append the nav entry**

```yaml
      - Resilient sweeps: notebooks/04_resilience.ipynb
```

- [ ] **Step 2: Write the notebook source**

Write to `<scratch>/04_resilience.py`. The failing cell is gated on an absolute
sentinel path so the *same* notebook can fail, then "fix", then resume:

```python
# %% [markdown]
# # Resilient & resumable sweeps
#
# A long `method × seed` sweep should not throw away hours of good runs because
# one cell hit a transient failure — and it must never let you compute statistics
# on a grid with holes. mushin gives you **fail-soft** runs (`on_error="nan"`),
# a **manifest**, and **resume**. This notebook walks the full loop end to end.

# %%
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from mushin import multirun
from mushin.benchmark import IncompleteSweepError, compare_methods
from mushin.workflows import MultiRunMetricsWorkflow

# %% [markdown]
# One stable working directory for the whole notebook, plus a sentinel file that
# stands in for "we fixed the root cause". The `mlp / seed=3` cell fails until the
# sentinel exists.

# %%
WORK = Path(tempfile.mkdtemp())
SENTINEL = WORK / "cause_fixed.flag"


class FlakySweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(method: str, seed: int) -> dict:
        if method == "mlp" and seed == 3 and not SENTINEL.exists():
            raise RuntimeError("simulated transient failure on mlp/seed=3")
        torch.manual_seed(seed)
        base = 0.90 if method == "cnn" else 0.82
        return dict(accuracy=base + 0.01 * torch.rand(1).item())

# %% [markdown]
# ## 1. Fail-soft: `on_error="nan"`
#
# The bad cell becomes `NaN`, a warning is emitted, and every other cell finishes.

# %%
wf = FlakySweep()
wf.run(
    method=multirun(["cnn", "mlp"]),
    seed=multirun([0, 1, 2, 3, 4]),
    working_dir=str(WORK),
    on_error="nan",
)
print("is_complete:", wf.is_complete)
wf.failures

# %% [markdown]
# The grid has exactly one hole:

# %%
ds_before = wf.to_xarray()
ds_before["accuracy"]

# %% [markdown]
# Every requested cell's status is recorded in the sweep manifest — this is what
# makes a resume possible.

# %%
manifest = json.loads((WORK / "mushin_sweep_manifest.json").read_text())
[(c["combo"], c["status"]) for c in manifest["cells"]]

# %% [markdown]
# ## 2. Statistics refuse an incomplete sweep
#
# The missing cell is *missing data*, not a measurement, so `compare_methods`
# raises rather than quietly averaging over the hole.

# %%
try:
    compare_methods(ds_before)
except IncompleteSweepError as e:
    print("IncompleteSweepError:", e)

# %% [markdown]
# ## 3. Resume: fill only the failed cell
#
# "Fix the cause" (touch the sentinel), then re-run against the **same**
# `working_dir` with `resume=True`. Completed cells are reused from disk; only the
# failed cell re-executes.

# %%
SENTINEL.write_text("fixed")

wf2 = FlakySweep()
wf2.run(
    method=multirun(["cnn", "mlp"]),
    seed=multirun([0, 1, 2, 3, 4]),
    working_dir=str(WORK),
    resume=True,
)
print("is_complete:", wf2.is_complete)
ds_after = wf2.to_xarray()
ds_after["accuracy"]

# %% [markdown]
# ## 4. The grid, before and after
#
# The single NaN cell (left) is filled in on resume (right).

# %%
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
for ax, ds, title in [
    (axes[0], ds_before, "after fail-soft run"),
    (axes[1], ds_after, "after resume"),
]:
    arr = ds["accuracy"].transpose("method", "seed").values
    im = ax.imshow(arr, vmin=0.8, vmax=0.95, cmap="viridis", aspect="auto")
    ax.set_xticks(range(ds.sizes["seed"]))
    ax.set_xticklabels(ds["seed"].values)
    ax.set_yticks(range(ds.sizes["method"]))
    ax.set_yticklabels(ds["method"].values)
    ax.set_xlabel("seed")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt = "NaN" if np.isnan(arr[i, j]) else f"{arr[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="w", fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Now the comparison runs

# %%
compare_methods(ds_after).summary()

# %% [markdown]
# **See also**
#
# - [Resilient & resumable sweeps guide](../guides/resilience.md) — provenance,
#   `capture_env`, and the same options threaded through `Study`.
```

- [ ] **Step 3: Build, verify, commit**

```bash
uv run jupytext --to notebook --execute \
  --output docs/notebooks/04_resilience.ipynb <scratch>/04_resilience.py
uv run nbstripout --keep-output docs/notebooks/04_resilience.ipynb
uv run --extra viz pytest --nbmake docs/notebooks/04_resilience.ipynb -p no:cacheprovider
uv run --group docs mkdocs build --strict
git add mkdocs.yml docs/notebooks/04_resilience.ipynb
git commit -m "docs: add example notebook 04 (resilient & resumable sweeps)"
```
Expected: `1 passed` (the intentional `UserWarning`/`RuntimeError` are captured as
cell output, not errors); docs build exit 0.

> **Note for the implementer:** the manifest key names (`cells`, `combo`,
> `status`) and the sidecar filename (`mushin_sweep_manifest.json`) are taken from
> the resilience guide. Before building, confirm them against the actual file:
> after the first `wf.run(..., on_error="nan")`, print
> `json.loads((WORK / "mushin_sweep_manifest.json").read_text())` and adjust the
> manifest cell to the real schema if it differs. The rest of the notebook does
> not depend on the manifest shape.

---

## Task 6: Notebook 05 (LLM evaluation)

**Files:**
- Modify: `mkdocs.yml` (append nav entry)
- Create: `docs/notebooks/05_llm_eval.ipynb`

- [ ] **Step 1: Append the nav entry**

```yaml
      - LLM evaluation: notebooks/05_llm_eval.ipynb
```

- [ ] **Step 2: Write the notebook source**

Write to `<scratch>/05_llm_eval.py`:

```python
# %% [markdown]
# # Evaluating LLM systems, with statistics
#
# `compare_llms` brings mushin's seed-based significance testing to LLM/system
# evaluation. You give it `systems` (callables `(inputs, seed) -> outputs`), a
# `data` eval set, and a `metric`; it runs each system across seeds and reports
# which differences are statistically real. Here both "systems" are fakes (no
# network, no keys) whose per-seed noise stands in for `temperature > 0`.

# %%
from __future__ import annotations

import random

from mushin.llm import compare_llms

# %% [markdown]
# A tiny even/odd classification eval set.

# %%
data = [{"input": i, "reference": "even" if i % 2 == 0 else "odd"} for i in range(20)]


def exact_match(output: str, reference: str) -> float:
    return float(output.strip() == reference.strip())

# %% [markdown]
# Two fake systems: a strong one that occasionally slips, and a biased one that
# leans "even". Each wires the trial `seed` to its randomness, so the per-seed
# scores form a real sampling distribution.

# %%
def strong(inputs, seed):
    rng = random.Random(seed)
    out = []
    for i in inputs:
        label = "even" if i % 2 == 0 else "odd"
        if rng.random() < 0.15:  # occasional slip
            label = "odd" if label == "even" else "even"
        out.append(label)
    return out


def biased(inputs, seed):
    rng = random.Random(1000 + seed)
    return ["even" if rng.random() < 0.85 else "odd" for _ in inputs]

# %% [markdown]
# Compare them across five seeds with Welch's t-test.

# %%
result = compare_llms(
    systems={"strong": strong, "biased": biased},
    data=data,
    metric=exact_match,
    seeds=range(5),
    test="welch",
)
result.summary()

# %% [markdown]
# The per-seed scores live in `result.data` (dims `method × seed`) — plot the
# distribution each system produces across seeds.

# %%
import matplotlib.pyplot as plt

scores = result.data["score"]
methods = [str(m) for m in scores["method"].values]
fig, ax = plt.subplots(figsize=(4.2, 3.2))
for k, m in enumerate(methods):
    ys = scores.sel(method=m).values
    ax.scatter([k] * len(ys), ys, alpha=0.7, zorder=3)
    ax.hlines(ys.mean(), k - 0.2, k + 0.2, color="k", lw=2, zorder=4)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods)
ax.set_ylabel("exact-match score")
ax.set_ylim(0, 1)
ax.set_title("Per-seed scores by system (bar = mean)")
fig.tight_layout()
plt.show()

# %% [markdown]
# **See also**
#
# - [LLM evaluation guide](../guides/llm.md) — real systems, hydra-zen configs,
#   and the deterministic-system caveat.
```

- [ ] **Step 3: Build, verify, commit**

```bash
uv run jupytext --to notebook --execute \
  --output docs/notebooks/05_llm_eval.ipynb <scratch>/05_llm_eval.py
uv run nbstripout --keep-output docs/notebooks/05_llm_eval.ipynb
uv run --extra viz pytest --nbmake docs/notebooks/05_llm_eval.ipynb -p no:cacheprovider
uv run --group docs mkdocs build --strict
git add mkdocs.yml docs/notebooks/05_llm_eval.ipynb
git commit -m "docs: add example notebook 05 (LLM evaluation)"
```
Expected: `1 passed`; docs build exit 0.

---

## Task 7: Notebook 06 (scikit-learn / framework-agnostic)

**Files:**
- Modify: `mkdocs.yml` (append nav entry)
- Create: `docs/notebooks/06_sklearn_framework_agnostic.ipynb`

- [ ] **Step 1: Append the nav entry**

```yaml
      - scikit-learn (framework-agnostic): notebooks/06_sklearn_framework_agnostic.ipynb
```

- [ ] **Step 2: Write the notebook source**

Write to `<scratch>/06_sklearn_framework_agnostic.py`:

```python
# %% [markdown]
# # Framework-agnostic sweeps (scikit-learn, no torch)
#
# `MultiRunMetricsWorkflow` never inspects your model — it sweeps configurations
# and collects the `dict` your `task` returns. So you can train a **scikit-learn**
# model inside `task` (no torch, no Lightning) and still get a labeled
# `xarray.Dataset` back, exactly as with a torch model. mushin needs no
# scikit-learn integration for this; the sweep layer is neutral.

# %%
from __future__ import annotations

import tempfile

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

# %% [markdown]
# A fixed, seeded synthetic classification problem, split per seed.

# %%
def split(seed: int):
    x, y = make_classification(
        n_samples=400, n_features=20, n_informative=5, random_state=seed
    )
    return train_test_split(x, y, test_size=0.25, random_state=seed)

# %% [markdown]
# Sweep the inverse-regularization strength `C` across seeds; the returned dict
# populates the dataset — no torch, no checkpoint to save.

# %%
class LogRegCSweep(MultiRunMetricsWorkflow):
    @staticmethod
    def task(C: float, seed: int) -> dict:
        x_train, x_test, y_train, y_test = split(seed)
        model = LogisticRegression(C=C, max_iter=1000, random_state=seed)
        model.fit(x_train, y_train)
        return dict(accuracy=float(model.score(x_test, y_test)))


wf = LogRegCSweep()
wf.run(
    C=multirun([0.01, 0.1, 1.0, 10.0]),
    seed=multirun([0, 1, 2]),
    working_dir=tempfile.mkdtemp(),
)
ds = wf.to_xarray()
ds

# %% [markdown]
# Same labeled dataset as the torch sweep in notebook 01 — plot accuracy vs `C`.

# %%
import matplotlib.pyplot as plt

mean_acc = ds["accuracy"].mean("seed")
fig, ax = plt.subplots(figsize=(5, 3.2))
for s in ds["seed"].values:
    ax.plot(ds["C"].values, ds["accuracy"].sel(seed=s).values, "o-",
            alpha=0.35, color="tab:green")
ax.plot(ds["C"].values, mean_acc.values, "ks-", lw=2, label="mean")
ax.set_xscale("log")
ax.set_xlabel("C (inverse regularization strength)")
ax.set_ylabel("held-out accuracy")
ax.set_title("scikit-learn LogisticRegression: accuracy vs C")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **See also**
#
# - [Notebook 01](01_sweep_to_dataset.ipynb) — the identical flow with a torch
#   model.
# - [Core concepts](../concepts.md) — why the sweep layer is framework-agnostic.
```

- [ ] **Step 3: Build, verify, commit**

```bash
uv run jupytext --to notebook --execute \
  --output docs/notebooks/06_sklearn_framework_agnostic.ipynb <scratch>/06_sklearn_framework_agnostic.py
uv run nbstripout --keep-output docs/notebooks/06_sklearn_framework_agnostic.ipynb
uv run --extra viz pytest --nbmake docs/notebooks/06_sklearn_framework_agnostic.ipynb -p no:cacheprovider
uv run --group docs mkdocs build --strict
git add mkdocs.yml docs/notebooks/06_sklearn_framework_agnostic.ipynb
git commit -m "docs: add example notebook 06 (scikit-learn framework-agnostic)"
```
Expected: `1 passed` (scikit-learn is in the dev group, synced by `uv run`); docs
build exit 0.

---

## Task 8: CI job, guide cross-links, changelog

**Files:**
- Modify: `.github/workflows/ci.yml` (new `notebooks` job)
- Modify: `docs/guides/resilience.md`, `docs/guides/llm.md`, `docs/guides/workflows.md` (cross-links)
- Create: `changes/+docs-example-notebooks.misc.md`

- [ ] **Step 1: Add the `notebooks` CI job**

In `.github/workflows/ci.yml`, add a new job (place it after `batteries-clean-install`, before `changelog`). Match the existing jobs' 2-space indentation:

```yaml
  notebooks:
    # Execute every docs/notebooks/*.ipynb with nbmake so the example notebooks
    # can never rot. nbmake checks that cells RUN (not output equality), so
    # numerical drift never makes this flaky; committed outputs are for display.
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
      - name: Register python3 kernel
        run: uv run python -m ipykernel install --user --name python3 --display-name "Python 3"
      - name: Execute example notebooks
        run: uv run --extra viz pytest --nbmake docs/notebooks/ -p no:cacheprovider
```

- [ ] **Step 2: Verify the job command locally (all six notebooks)**

Run:
```bash
uv run --extra viz pytest --nbmake docs/notebooks/ -p no:cacheprovider
```
Expected: `6 passed`.

- [ ] **Step 3: Add guide → notebook cross-links**

Add a short pointer near the top of each guide (under its first paragraph). Use
the exact insertion text below.

In `docs/guides/resilience.md`, after the opening paragraph (the one ending
"…**resume**, and **provenance**."), add:

```markdown

> **Prefer to follow along?** [Notebook 04 — Resilient sweeps](../notebooks/04_resilience.ipynb)
> runs this whole fail-soft → resume loop end to end with live output.
```

In `docs/guides/llm.md`, after its first paragraph, add:

```markdown

> **Prefer to follow along?** [Notebook 05 — LLM evaluation](../notebooks/05_llm_eval.ipynb)
> runs a full `compare_llms` example with outputs and a per-seed score plot.
```

In `docs/guides/workflows.md`, after its first paragraph, add:

```markdown

> **Prefer to follow along?** [Notebook 01 — Sweeps → datasets](../notebooks/01_sweep_to_dataset.ipynb)
> builds a sweep end to end and plots the result.
```

- [ ] **Step 4: Verify strict docs build with the new cross-links**

```bash
uv run --group docs mkdocs build --strict
```
Expected: exit 0, no warnings (confirms all three relative notebook links resolve).

- [ ] **Step 5: Add the changelog fragment**

Create `changes/+docs-example-notebooks.misc.md` with content:

```markdown
Added six runnable EQUINE-style example notebooks (sweeps, compare + batteries, Study, resilient sweeps, LLM evaluation, scikit-learn) under an "Example notebooks" section of the docs, executed in CI via nbmake so they stay current.
```

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml docs/guides/resilience.md docs/guides/llm.md docs/guides/workflows.md changes/+docs-example-notebooks.misc.md
git commit -m "ci+docs: nbmake job for example notebooks + guide cross-links"
```

---

## Task 9: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full notebook execution**

```bash
uv run --extra viz pytest --nbmake docs/notebooks/ -p no:cacheprovider
```
Expected: `6 passed`.

- [ ] **Step 2: Strict docs build**

```bash
uv run --group docs mkdocs build --strict
```
Expected: exit 0, no warnings.

- [ ] **Step 3: Confirm committed notebooks are metadata-stripped**

```bash
uv run nbstripout --keep-output docs/notebooks/*.ipynb
git diff --exit-code -- docs/notebooks/
```
Expected: no diff (notebooks were already stripped during authoring). If there is
a diff, `git add` + amend the relevant notebook commit.

- [ ] **Step 4: Confirm no stray run artifacts were committed**

```bash
git status --porcelain
git ls-files docs/notebooks/ | grep -vE '\.(ipynb|gitkeep)$' || echo "clean: only notebooks tracked"
```
Expected: clean working tree; only `.ipynb` (+ `.gitkeep`) tracked under
`docs/notebooks/`.

- [ ] **Step 5: Push and open the PR** (only when the user asks — see repo policy on PRs)

The branch is `example-notebooks`. When instructed, push and open a PR whose body
summarizes the six notebooks and the new `notebooks` CI check. Do **not** add any
Claude attribution (repo policy; a `commit-msg` hook strips trailers).
```

---

## Notes for the executor

- **Rendering caveat:** the very first strict docs build (Task 2, Step 6) is the real test of the mkdocs-jupyter ↔ mkdocs-material integration. If it warns about the `.ipynb` nav path, confirm the `mkdocs-jupyter` plugin block is *before* `mkdocstrings` and that `include_source: true` is set; if a notebook emits a broken relative link, fix the link in the notebook source and rebuild.
- **Dark/light rendering** is only visible on the deployed site; the strict build cannot catch it. After merge + Pages deploy, spot-check one notebook in both themes and open a follow-up only if a plot/code block is unreadable.
- **Determinism:** notebooks are seeded, but nbmake asserts *execution*, not output equality — never add output-diffing to the nbmake job.
