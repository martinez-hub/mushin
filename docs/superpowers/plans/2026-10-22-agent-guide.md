# Agent Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a tested usage guide for AI coding agents that set up experiments with mushin, reachable from the installed package (`python -m mushin guide`), `help(mushin)`, the docs site, `llms.txt` and the README.

**Architecture:** One ASCII Markdown file, `src/mushin/agent_guide.md`, shipped as package data, is the single source. A small `python -m mushin` CLI prints it, a docs page includes it with a snippet include, and `llms.txt` and the README link to it. `tests/test_agent_guide.py` executes its code, checks its `run()` keywords and links, and exercises the CLI; `tests/test_docs_claims.py` pins its seed floors. `__init__.py` gains a docstring and lists only names that resolve, so `help(mushin)` works on a core install.

**Tech Stack:** Python 3.11+, argparse, `importlib.resources`, pytest, `ast`, mkdocs-material with pymdownx.snippets, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-15-agent-guide-design.md` (decisions D1 to D8; D9 shipped as #254).

---

## Conventions for every task

- Work in the `feat/agent-guide` worktree (`/Users/josuemartinez/code/mushin-agentdocs`), rebased on `main`.
- Run Python with the repo's venv: `uv run ...` from the worktree root. If the worktree has no `.venv`, `uv sync --all-extras --group docs` first. Put scratch output (sweeps, builds) outside the repo, e.g. under `/Volumes/ExtremeSSD/scratch/`; the internal disk is nearly full.
- Commits: conventional prefix, no AI attribution (no `Co-Authored-By`, no "Generated with" footer).
- Docs prose avoids em dashes. `src/mushin/agent_guide.md` is ASCII only.
- After each task: `uv run ruff check . && uv run ruff format --check .` passes.

## File structure

| File | Responsibility |
|---|---|
| `src/mushin/_extras.py` (new) | The `eval` extra's module list and `eval_available()`, the one source both the import guards and `__dir__` use. |
| `src/mushin/benchmark/__init__.py`, `src/mushin/llm/__init__.py` | Their install-hint guards import `EVAL_MODULES` instead of hard-coding names. |
| `src/mushin/__init__.py` | Module docstring with the guide pointer; `__all__` and `__dir__` list only names that resolve. |
| `src/mushin/agent_guide.md` (new) | The guide (D8). |
| `src/mushin/__main__.py` (new) | `python -m mushin guide [--path]`. |
| `pyproject.toml` | Ship `agent_guide.md` as package data. |
| `tests/_docs_site.py` (new) | mkdocs nav and URL helpers, moved out of `tests/test_llms_txt.py` so `tests/test_agent_guide.py` shares them. |
| `tests/test_agent_guide.py` (new) | D7 items 1 to 5. |
| `tests/test_docs_claims.py` | D7 item 6: the Holm seed floors. |
| `tests/test_lazy_imports.py` | `dir(mushin)` and `__all__` behavior on a full install. |
| `docs/for-agents.md` (new), `mkdocs.yml` | The docs page (D4) and its nav entry. |
| `docs/llms.txt`, `tests/test_llms_txt.py` | The guide pointer and nav entry (D5); `raw.githubusercontent.com` becomes an allowed host. |
| `README.md` | "Using mushin with an AI agent" (D6). |
| `AGENTS.md` | Contributors learn the guide is tested and where it lives. |
| `.github/workflows/ci.yml` | Docs job runs the guide tests; `core-only` and `batteries-clean-install` check the shipped guide and `help(mushin)`. |
| `changes/+agent-guide.added.md` (new) | The changelog fragment, renamed to the PR number later. |

---

### Task 0: Baseline

- [ ] **Step 1: Confirm the branch and a green baseline**

Run:
```bash
cd /Users/josuemartinez/code/mushin-agentdocs
git fetch -q origin && git rebase origin/main
git log --oneline -1 origin/main
uv run pytest tests/test_lazy_imports.py tests/test_llms_txt.py tests/test_docs_claims.py -q -p no:cacheprovider
```
Expected: rebase succeeds; tests pass.

---

### Task 1: One definition of the `eval` extra's modules

**Files:**
- Create: `src/mushin/_extras.py`
- Modify: `src/mushin/benchmark/__init__.py:5-16`, `src/mushin/llm/__init__.py:3-11`
- Test: `tests/test_lazy_imports.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_lazy_imports.py`)

```python
def test_eval_available_reflects_the_eval_modules(monkeypatch):
    import importlib.util

    from mushin import _extras

    assert _extras.EVAL_MODULES == ("scipy", "torchmetrics")
    assert _extras.eval_available() is True  # the dev environment has the extra

    real_find_spec = importlib.util.find_spec

    def without_scipy(name, *args, **kwargs):
        return None if name == "scipy" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_scipy)
    assert _extras.eval_available() is False


def test_eval_available_does_not_import_the_eval_modules():
    """dir(mushin) calls eval_available, and IPython's tab completion calls
    dir(): it must not import scipy or torchmetrics (and so torch)."""
    code = (
        "import sys; from mushin._extras import eval_available; eval_available(); "
        "print('torchmetrics' in sys.modules, 'scipy' in sys.modules)"
    )
    assert _fresh_import_probe(code) == "False False"
```

`_fresh_import_probe` is the subprocess helper already at the top of `tests/test_lazy_imports.py`.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_lazy_imports.py -k eval_available -q -p no:cacheprovider`
Expected: FAIL, `ModuleNotFoundError: No module named 'mushin._extras'`.

- [ ] **Step 3: Create `src/mushin/_extras.py`**

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The optional `eval` extra: the modules it brings, and whether they are here."""

from __future__ import annotations

import importlib.util

# The third-party modules the evaluation layer imports at module level. The
# install-hint guards in `mushin.benchmark` and `mushin.llm` import these, and
# `eval_available` looks them up, so the two cannot drift apart.
EVAL_MODULES = ("scipy", "torchmetrics")


def eval_available() -> bool:
    """Whether the `eval` extra's modules are installed, without importing them.

    `dir(mushin)` calls this, and IPython's tab completion calls `dir()`, so it
    must stay cheap: importing torchmetrics would import torch.
    """
    try:
        return all(importlib.util.find_spec(name) is not None for name in EVAL_MODULES)
    except (ImportError, ValueError):  # a broken or half-imported module
        return False
```

- [ ] **Step 4: Point the two guards at it**

In `src/mushin/benchmark/__init__.py`, replace

```python
try:
    import scipy as _scipy  # noqa: F401
    import torchmetrics as _torchmetrics  # noqa: F401
except ImportError as _exc:  # pragma: no cover - exercised by the core-only CI job
```

with

```python
import importlib as _importlib

from .._extras import EVAL_MODULES as _EVAL_MODULES

try:
    for _module in _EVAL_MODULES:
        _importlib.import_module(_module)
except ImportError as _exc:  # pragma: no cover - exercised by the core-only CI job
```

keeping the `raise ImportError(...) from _exc` body unchanged. In `src/mushin/llm/__init__.py`, make the same replacement of its `try: import torchmetrics ...` line, keeping its own message, and update the comment above it to say "a raw scipy/torchmetrics ImportError".

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_lazy_imports.py tests/test_benchmark -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mushin/_extras.py src/mushin/benchmark/__init__.py src/mushin/llm/__init__.py tests/test_lazy_imports.py
git commit -m "refactor: define the eval extra's modules once, in mushin._extras"
```

---

### Task 2: `help(mushin)` describes the package and lists only names that resolve (D3)

**Files:**
- Modify: `src/mushin/__init__.py` (top of file; `__dir__`; after `__all__`)
- Test: `tests/test_lazy_imports.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_lazy_imports.py`)

```python
def test_module_docstring_points_agents_to_the_guide():
    import mushin

    assert mushin.__doc__ and "python -m mushin guide" in mushin.__doc__


def test_dir_lists_no_deprecated_aliases():
    """help(mushin) touches every name dir() lists; a deprecated alias then
    warns on every render (and raises under -W error)."""
    import warnings

    import mushin

    assert "BaseWorkflow" not in dir(mushin)
    assert "RobustnessCurve" not in dir(mushin)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mushin.BaseWorkflow  # still importable  # noqa: B018
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_all_lists_the_eval_names_when_the_extra_is_installed():
    import mushin

    assert {"compare", "llm", "classification_battery"} <= set(mushin.__all__)
    assert sorted(mushin.__all__) == dir(mushin)
```

The core-install side (no scipy) cannot be simulated here: in the full dev environment torchmetrics imports `scipy.signal` once the audio extras are present, so blocking scipy also breaks Lightning. The `core-only` CI job checks it on a real core install (Task 8).

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_lazy_imports.py -k "docstring or deprecated_aliases or eval_names" -q -p no:cacheprovider`
Expected: FAIL (no docstring; `BaseWorkflow` in `dir(mushin)`).

- [ ] **Step 3: Add the module docstring**

In `src/mushin/__init__.py`, directly after the three header comment lines and before `import importlib`, add:

```python
"""mushin: boilerplate-free, reproducible machine-learning experiment sweeps.

Sweep a Python function over a grid of parameters and seeds, and get the
results back as a labeled ``xarray.Dataset``, with per-run provenance and paired
statistics across seeds.

Main entry points:

- ``@mushin.sweep``: decorate a function, then ``.run(...)`` it over a grid
  built with ``mushin.multirun([...])``.
- ``mushin.MultiRunMetricsWorkflow``: the class API, for custom result
  collection or Lightning training.
- ``mushin.benchmark.compare_methods``: paired, corrected comparisons of methods
  across seeds (needs the ``eval`` extra).
- ``mushin.Study`` and ``mushin.compare``: train-then-compare, and metric
  batteries for torch models (need the ``eval`` extra).

AI coding agents: run ``python -m mushin guide`` for a usage guide matching this
version.
"""
```

- [ ] **Step 4: Filter `__all__` and `__dir__`**

Add `from ._extras import eval_available` to the eager imports at the top (it imports nothing heavy). Replace

```python
def __dir__():
    return sorted(set(__all__) | set(_DEPRECATED))
```

with

```python
def __dir__():
    # Only names that resolve: `__all__` (filtered below on a core install). The
    # deprecated aliases stay importable but are not listed, since help(mushin)
    # touches every listed name and each alias warns on access.
    return sorted(__all__)
```

and directly after the `__all__ = [...]` literal add:

```python
# On a core install (no `eval` extra) the evaluation-layer names raise on
# access, which breaks help(mushin), pydoc, `from mushin import *` and
# inspect.getmembers. Advertise them only when they resolve; accessing one still
# raises the install hint.
_EVAL_NAMES = _LAZY_BENCHMARK | {"llm"}
if not eval_available():
    __all__ = [name for name in __all__ if name not in _EVAL_NAMES]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_lazy_imports.py -q -p no:cacheprovider && uv run ruff check src/mushin/__init__.py`
Expected: PASS, and ruff clean. If ruff flags the `__all__` reassignment, add the rule's `noqa` code to that line.

- [ ] **Step 6: Commit**

```bash
git add src/mushin/__init__.py tests/test_lazy_imports.py
git commit -m "feat: help(mushin) describes the package and renders on a core install"
```

---

### Task 3: The guide, shipped, and `python -m mushin guide` (D1, D2, D8)

**Files:**
- Create: `src/mushin/agent_guide.md`, `src/mushin/__main__.py`, `tests/test_agent_guide.py`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data]`)

- [ ] **Step 1: Write the failing tests** (create `tests/test_agent_guide.py`)

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Keep the agent guide (`src/mushin/agent_guide.md`) honest.

Agents copy the guide's code and follow its links, and the guide ships inside
the package, so a stale keyword, a broken example or a dead link misleads every
agent that reads it. CI runs this file in the docs job too, because the `test`
job skips the docs-only PRs most likely to break it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib import resources
from pathlib import Path

GUIDE = resources.files("mushin") / "agent_guide.md"


def _text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_the_guide_ships_as_ascii_markdown():
    text = _text()
    assert text.isascii(), sorted({c for c in text if not c.isascii()})
    first, second = text.splitlines()[:2]
    # docs/for-agents.md includes the guide from line 3, below its own title.
    assert first.startswith("# ") and second == ""


def test_python_m_mushin_guide(capsys):
    from mushin.__main__ import _version, main

    assert main(["guide"]) == 0
    assert capsys.readouterr().out == f"<!-- mushin {_version()} -->\n" + _text()

    assert main(["guide", "--path"]) == 0
    assert Path(capsys.readouterr().out.strip()).is_file()

    assert main([]) == 0
    assert "guide" in capsys.readouterr().out


def test_the_guide_prints_on_a_cp1252_pipe():
    """A Windows pipe writes in the ANSI code page; the guide must survive it."""
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    env.pop("PYTHONUTF8", None)
    result = subprocess.run(
        [sys.executable, "-m", "mushin", "guide"],
        capture_output=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-2000:]
    assert result.stdout.decode("cp1252").startswith("<!-- mushin ")
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_agent_guide.py -q -p no:cacheprovider`
Expected: FAIL (`FileNotFoundError` for the guide; no `mushin.__main__`).

- [ ] **Step 3: Create `src/mushin/agent_guide.md`** with exactly this content (it was run and checked while writing this plan: ASCII, 261 lines, its three python blocks run in order as one script, ruff-formatted, codespell-clean):

````markdown
# Using mushin: a guide for AI coding agents

This guide is for a coding agent setting up an experiment for a scientist or
engineer. The copy that `python -m mushin guide` prints matches the installed
mushin. For exact signatures, use `help(obj)`.

## What mushin is

mushin runs a Python function over a grid of parameters and seeds, and returns
the results as a labeled `xarray.Dataset` (or a pandas table), with each run's
provenance and paired statistics across seeds. The function can use any
framework: scikit-learn, XGBoost, plain NumPy or PyTorch. torch and
pytorch-lightning are installed with mushin either way; they also power the
optional evaluation layer (metric batteries, `compare`, `Study`).

## Install

The PyPI name is `mushin-py`, and the import is `import mushin`. The PyPI
package named `mushin` is not this project.

| Install | Adds |
|---|---|
| `pip install mushin-py` | sweeps to datasets, provenance, resume |
| `pip install "mushin-py[eval]"` | `compare_methods` (needs scipy), `compare`, metric batteries, `mushin.llm`, `Study` |
| `pip install "mushin-py[viz]"` | matplotlib, for plots |
| `pip install "mushin-py[netcdf]"` | saving datasets as `.nc` files |
| `pip install "mushin-py[mcp]"` | `mushin-mcp`, a read-only MCP server over finished runs |
| `pip install "mushin-py[detection]"` (or `image`, `audio`) | more metric batteries |
| `pip install hydra-joblib-launcher` | `launcher="joblib"`, to run cells in parallel (a Hydra plugin, not an extra) |

Python 3.11 to 3.14 is supported. Intel macOS is not.

## Choosing an entry point

| Job | Use |
|---|---|
| Run a function over a grid of parameters and seeds | `@mushin.sweep` |
| Collect results yourself, or train with Lightning | `mushin.MultiRunMetricsWorkflow` |
| Compare methods across seeds, with significance | `mushin.benchmark.compare_methods` |
| Score torch models on a metric battery | `mushin.compare` |
| Train several methods, then compare them | `mushin.Study` |
| Compare LLM systems | `mushin.llm` |
| Inspect finished runs | `mushin.show`, `mushin.best`, `mushin.diff`, or the MCP server |
| Run cells in parallel on one machine | `launcher="joblib"` |
| Run several jobs on each GPU | the GPU packing guide (see Going further) |
| Run on a cluster | the multi-node guide (see Going further) |

## A template experiment

Write the experiment as one script in the scientist's project. Leave their code
where it is: the sweep task is a new module-level function that calls it and
returns a dict of numbers. Do not put `@mushin.sweep` on their function. It
replaces the function with a sweep handle, which cannot be called, so their
other call sites break.

```python
import random

import mushin
from mushin.benchmark import compare_methods


# The scientist's code. Import theirs; this stub stands in for it.
def train_and_evaluate(method: str, lr: float, seed: int) -> float:
    rng = random.Random(seed)
    base = {"baseline": 0.70, "new": 0.75}[method]
    return base - abs(lr - 0.01) + rng.gauss(0, 0.005)


# The sweep task: a new module-level function that calls their code and
# returns a dict of numbers. Never decorate their function itself.
@mushin.sweep
def experiment(method: str, lr: float, seed: int):
    return {"score": float(train_and_evaluate(method, lr, seed))}


GRID = dict(
    method=mushin.multirun(["baseline", "new"]),
    lr=mushin.multirun([0.001, 0.01]),
    seed=mushin.multirun(list(range(6))),  # 6 seeds: the floor for 2 methods
)
RUN_DIR = "runs/compare-v1"  # under a gitignored runs/; one per sweep

if __name__ == "__main__":
    plan = experiment.run(**GRID, working_dir=RUN_DIR, dry_run=True)
    assert plan["num_cells"] == 24, plan

    ds = experiment.run(**GRID, working_dir=RUN_DIR, on_error="nan")
    print(experiment.workflow.to_dataframe().head())

    # compare_methods needs exactly the dims (method, seed): select lr away.
    table = compare_methods(ds.sel(lr=0.01))
    print(table)  # mean_diff = method_a - method_b
    print("results in", experiment.workflow.working_dir)
```

What it does:

- `dry_run=True` prints the grid and returns
  `{"num_cells", "axes", "fixed", "working_dir"}` without running anything.
  Check `num_cells` before every expensive run.
- Each `mushin.multirun([...])` argument becomes a dimension of `ds`, here
  `(method, lr, seed)`. A plain value, such as `lr=0.01`, is fixed for every
  cell.
- `experiment.workflow` is the sweep's `MultiRunMetricsWorkflow`: its
  `failures`, `skipped`, `working_dir` and `to_dataframe()`.
- `compare_methods` runs a paired test (Wilcoxon by default) for each pair of
  methods across seeds, and corrects the p-values for the number of pairs
  (Holm). Its table has `mean_diff` and `effect_size`, the raw and corrected
  p-values, and `significant`. `mean_diff` is `method_a - method_b`, so a
  negative value means `method_b` scored higher.

## Rules that prevent wrong results and wasted compute

- **Sweep a value with `mushin.multirun([...])`.** A bare list raises:

```python
try:
    experiment.run(method=["baseline", "new"], lr=0.01, seed=0, dry_run=True)
except TypeError as e:
    print(e)  # ... wrap them: `method=mushin.multirun(['baseline', 'new'])` ...
```

  To pass a list as one value instead, use `mushin.hydra_list([...])`.
- **Return a dict of numbers.** The values are converted to JSON into each
  cell's `mushin_metrics.json`.
- **Each cell runs in its own directory**, so a relative path in the task
  resolves there, not in the project. Build input paths from
  `mushin.original_cwd()`, for example
  `mushin.original_cwd() / "data" / "train.csv"`.
- **Check the size first.** Run with `dry_run=True` before anything expensive.
  `confirm_above=N`, or the `MUSHIN_MAX_CELLS` environment variable, refuses a
  grid of more than `N` cells. `max_total_seconds=` stops starting cells once a
  time budget runs out, but only with the default launcher: under
  `launcher="joblib"` it has no effect, so bound such a run another way.
- **Failures.** With `on_error="nan"`, the sweep goes on when a cell raises:
  the cell becomes NaN, `experiment.workflow.failures` lists it, and its
  traceback is in `mushin_error.txt` in the cell's directory. `compare_methods`
  then refuses the incomplete grid with `IncompleteSweepError`, unless you pass
  `allow_incomplete=True`. Fix the cause, then rerun the same script with
  `resume=True`.
- **Explore with `sample=N`.** It runs N cells of the grid and records the
  rest as skipped; a later run with `resume=True` and no `sample` fills them
  in.
- **Resume.** `resume=True` with the same `working_dir` reruns the cells that
  are incomplete or whose config or task function changed. It does not see an
  edit to anything the task calls: the scientist's code, a helper, a constant
  or an installed package. After such an edit, use a new `working_dir`.
  `cache_dir=` has the same blind spot.
- **`compare_methods` needs exactly two dims, `method` and `seed`,** and string
  `method` values. Select or average away other dims first (`ds.sel(lr=0.01)`,
  `ds.mean("lr")`), and rename yours if needed
  (`ds.rename({"model": "method"})`). A method whose score is the same for
  every seed gets NaN for its pairs, with a warning.
- **Seeds.** With `k` methods there are `m = k * (k - 1) / 2` pairs, and with
  Wilcoxon a pair can only be significant at 0.05 if
  `m * 2 ** (1 - n) < 0.05` for `n` seeds. So use at least 6 seeds for 2
  methods, 7 for 3 and 8 for 4. These are floors, not enough power: a real but
  small difference needs more seeds.
- **Commit before the final runs.** Each cell's `mushin_provenance.json`
  records the git commit and whether the tree was dirty, and untracked files
  count. Add `runs/` and any logs to `.gitignore`, commit, and check that
  `git status --porcelain` prints nothing before the final run.
- **`best()` is per cell.** `mushin.best(root, "score")` returns the single
  best cell, which is one seed. To pick a configuration, average over seeds
  first: `ds["score"].mean("seed")`.
- **On Windows, set `PYTHONUTF8=1`** (or run `python -X utf8`) when mushin's
  output is captured: `mushin.diff` and the dry-run preview print characters
  that a Windows console code page may lack.

## Running long sweeps from an agent

An agent's shell commands time out, and a sweep can take hours. So:

1. Put the sweep in a script, like the template, with a fixed `working_dir`
   under the gitignored `runs/`.
2. Start it in the background, with its log under `runs/` too:

   ```bash
   mkdir -p runs
   nohup python experiment.py > runs/compare-v1.log 2>&1 &
   ```

3. Poll it:

   ```bash
   python -c "import mushin; mushin.show('runs/compare-v1')"
   ```

   `show` prints one row per cell that has started, with its status
   (`running`, `completed`, `failed` or `skipped`) and metrics.
4. Use the default launcher for a sweep you run this way. If the process died
   with cells still `running`, check that nothing from that run is still alive
   before you relaunch the same script with `resume=True`: with
   `launcher="joblib"`, the worker processes outlive the main one and keep
   computing cells.

## Where results go, and how to reload them

A sweep writes to its `working_dir`, or by default to `multirun/<date>/<time>/`
under the directory it was launched from:

```text
runs/compare-v1/
  multirun.yaml                the sweep's overrides
  mushin_sweep_manifest.json   every cell's status, written when the run ends
  0/, 1/, ...                  one directory per cell, holding:
    mushin_cell_status.json    completed, failed, skipped or running
    mushin_metrics.json        the dict the task returned (completed cells)
    mushin_error.txt           the traceback (failed cells, on_error="nan")
    mushin_provenance.json     git commit, package versions, hardware
    mushin_workflow.log, .hydra/
```

Reload a finished sweep in a new process:

```python
from mushin.workflows import MultiRunMetricsWorkflow

wf = MultiRunMetricsWorkflow().load_from_dir(RUN_DIR, "mushin_metrics.json")
assert wf.is_complete, wf.failures + wf.skipped  # every cell has a result
ds = wf.to_xarray()  # the same (method, lr, seed) dataset as above
```

A reloaded sweep with failed, skipped or unfinished cells is not complete, and
`compare_methods` refuses it. In mushin 0.15.1 and earlier, a reload lost that
record: there, check that `mushin.show(root)` lists a `completed` row for every
cell of the grid before comparing.

`mushin.show(root)`, `mushin.best(root, metric)` and `mushin.diff(root_a,
root_b)` read a sweep directory directly; `show` prints its table and also
returns it. To let an MCP client such as Claude Code inspect the runs, install
`"mushin-py[mcp]"` and see the MCP guide.

## Before reporting to the scientist

- [ ] Every cell of the grid completed: `wf.is_complete` is true, or
  `mushin.show(root)` lists one `completed` row per cell (the `num_cells` of
  the dry run). If not, report the failed or missing cells.
- [ ] There are enough seeds for the number of methods compared (see Rules).
- [ ] The report gives corrected p-values (`p_corrected`) and effect sizes
  (`mean_diff`, `effect_size`), with the direction stated, not raw means alone.
- [ ] The provenance is clean: `"dirty": false` in `mushin_provenance.json`.
- [ ] The report names the sweep's directory, so the run can be found and
  reloaded.

## Going further

- [Understanding the statistics](https://martinez-hub.github.io/mushin/guides/statistics/)
- [Resilient and resumable sweeps](https://martinez-hub.github.io/mushin/guides/resilience/)
- [From exploration to a paper](https://martinez-hub.github.io/mushin/guides/exploration-to-paper/)
- [Workflows, launchers, and using W&B or TensorBoard alongside](https://martinez-hub.github.io/mushin/guides/workflows/)
- [Comparing methods](https://martinez-hub.github.io/mushin/guides/compare/) and [the metric batteries](https://martinez-hub.github.io/mushin/guides/batteries/)
- [Studies](https://martinez-hub.github.io/mushin/guides/study/)
- [LLM evaluation](https://martinez-hub.github.io/mushin/guides/llm/)
- [Packing small jobs onto GPUs](https://martinez-hub.github.io/mushin/guides/packing/)
- [Multi-node](https://martinez-hub.github.io/mushin/guides/multinode/) and [sharded](https://martinez-hub.github.io/mushin/guides/sharding/) training
- [Analyzing runs from Claude Code (MCP)](https://martinez-hub.github.io/mushin/guides/mcp/)

For exact signatures: `help(mushin.sweep)` and
`help(mushin.workflows.BaseWorkflow.run)`.
````

- [ ] **Step 4: Create `src/mushin/__main__.py`**

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""``python -m mushin``: print the usage guide for AI coding agents."""

from __future__ import annotations

import argparse
import sys
from importlib import metadata, resources

GUIDE = "agent_guide.md"


def _version() -> str:
    try:
        return metadata.version("mushin-py")
    except metadata.PackageNotFoundError:  # importable but not installed
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mushin",
        description="mushin's command line. `guide` prints the usage guide for "
        "AI coding agents that matches this version.",
    )
    commands = parser.add_subparsers(dest="command")
    guide = commands.add_parser(
        "guide", help="print the usage guide for AI coding agents"
    )
    guide.add_argument(
        "--path", action="store_true", help="print the guide's file path instead"
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    path = resources.files("mushin") / GUIDE
    if args.path:
        print(path)
        return 0
    # An HTML comment: valid Markdown, invisible when rendered.
    sys.stdout.write(f"<!-- mushin {_version()} -->\n")
    sys.stdout.write(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Ship the guide as package data**

In `pyproject.toml`, change

```toml
[tool.setuptools.package-data]
# PEP 561: ship the py.typed marker so type checkers see our annotations.
mushin = ["py.typed"]
```

to

```toml
[tool.setuptools.package-data]
# PEP 561: ship the py.typed marker so type checkers see our annotations, and
# the agent guide that `python -m mushin guide` prints.
mushin = ["py.typed", "agent_guide.md"]
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_agent_guide.py -q -p no:cacheprovider`
Expected: PASS (3 tests).

- [ ] **Step 7: Check that the wheel and sdist carry the guide**

Run:
```bash
rm -rf /Volumes/ExtremeSSD/scratch/dist && uv build -q -o /Volumes/ExtremeSSD/scratch/dist
unzip -l /Volumes/ExtremeSSD/scratch/dist/*.whl | grep agent_guide.md
tar tzf /Volumes/ExtremeSSD/scratch/dist/*.tar.gz | grep agent_guide.md
```
Expected: one line from each command. If the sdist lacks it, add `include src/mushin/agent_guide.md` to `MANIFEST.in` (create it if absent) and rebuild.

- [ ] **Step 8: Commit**

```bash
git add src/mushin/agent_guide.md src/mushin/__main__.py pyproject.toml tests/test_agent_guide.py
git commit -m "feat: ship a usage guide for AI coding agents, printed by python -m mushin guide"
```

---

### Task 4: Share the mkdocs nav helpers

**Files:**
- Create: `tests/_docs_site.py`
- Modify: `tests/test_llms_txt.py`

`tests/` is not a package; pytest's default import mode puts a test file's directory on `sys.path`, so `import _docs_site` resolves for every test under `tests/`.

- [ ] **Step 1: Create `tests/_docs_site.py`** by moving these definitions out of `tests/test_llms_txt.py` unchanged: `ROOT`, `_MkdocsLoader`, `_construct_tagged`, the `add_multi_constructor` call, `_mkdocs`, `_site`, `_nav_pages`, `_site_url`, `_nav_urls`. The new file starts:

```python
# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The docs site's nav and URLs, read from mkdocs.yml, for the tests that check
links into the site (`test_llms_txt.py`, `test_agent_guide.py`)."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
```

followed by the moved definitions, in their current order.

- [ ] **Step 2: Import them back in `tests/test_llms_txt.py`**

Delete the moved definitions there and add, with the other imports:

```python
from _docs_site import (
    ROOT,
    _MkdocsLoader,
    _mkdocs,
    _nav_pages,
    _nav_urls,
    _site,
    _site_url,
)
```

Remove any import that ruff then reports unused (`os`, probably). Keep `EXTERNAL_HOSTS`, `_pyproject`, `_llms_txt`, `_llms_txt_links` and `_llms_txt_urls` in `test_llms_txt.py`.

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_llms_txt.py -q -p no:cacheprovider && uv run ruff check tests && uv run ruff format --check tests`
Expected: PASS, the same number of tests as before.

- [ ] **Step 4: Commit**

```bash
git add tests/_docs_site.py tests/test_llms_txt.py
git commit -m "test: share the mkdocs nav helpers in tests/_docs_site.py"
```

---

### Task 5: The guide's code runs, its keywords exist, its links resolve (D7 items 2 to 4)

**Files:**
- Modify: `tests/test_agent_guide.py`

- [ ] **Step 1: Add the tests** (append to `tests/test_agent_guide.py`; add `import ast`, `import inspect` and `import pytest` to its imports, and `from _docs_site import ROOT, _nav_urls, _site`)

```python
RAW_PREFIX = "https://raw.githubusercontent.com/martinez-hub/mushin/main/"
# The files that point agents on mushin < 0.16 at the raw guide.
RAW_LINK_FILES = ["docs/for-agents.md", "README.md", "docs/llms.txt"]
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _python_blocks() -> list[str]:
    return re.findall(r"^```python\n(.*?)^```$", _text(), re.DOTALL | re.MULTILINE)


def test_every_python_example_is_tagged_python():
    """Only ```python blocks run below; another Python tag would dodge that."""
    other = re.findall(r"^\s*```(py|python3|pycon)\b", _text(), re.MULTILINE)
    assert not other, other


def test_the_guide_code_runs(tmp_path):
    """Every python block, in order, as one script, the way an agent runs it.
    A real script, so `inspect.getsource` works for mushin's fingerprints, and
    Hydra's global state stays out of this process."""
    pytest.importorskip("scipy")  # the template calls compare_methods
    blocks = _python_blocks()
    assert blocks, "the guide has no python example"
    (tmp_path / "guide_example.py").write_text("\n".join(blocks), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "guide_example.py"],
        cwd=tmp_path,
        # As the guide tells agents to run mushin on Windows.
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]


def _run_parameters() -> set[str]:
    from mushin.workflows import BaseWorkflow, MultiRunMetricsWorkflow

    names: set[str] = set()
    for fn in (MultiRunMetricsWorkflow.run, BaseWorkflow.run):
        for p in inspect.signature(fn).parameters.values():
            if p.kind is not p.VAR_KEYWORD and p.name != "self":
                names.add(p.name)
    return names


def _is_mushin_sweep(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "sweep"
        and isinstance(node.value, ast.Name)
        and node.value.id == "mushin"
    )


def _parameters(fn: ast.FunctionDef) -> set[str]:
    return {a.arg for a in [*fn.args.args, *fn.args.kwonlyargs]}


def test_the_guide_run_keywords_exist():
    """Each keyword of a mushin `run()` call in the guide is a parameter of the
    task or of `run()`. `run()` accepts arbitrary task overrides, so a renamed
    safety keyword (`dry_run=`, `working_dir=`) would otherwise pass silently.
    `Sweep.run` is `(self, **kwargs)`, so its signature proves nothing."""
    tree = ast.parse("\n".join(_python_blocks()))
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    handles: dict[str, set[str]] = {}  # name -> its task's parameters
    dicts: dict[str, list[str]] = {}  # name bound to a dict literal or dict(...)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            _is_mushin_sweep(d) for d in node.decorator_list
        ):
            handles[node.name] = _parameters(node)
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        target, value = node.targets[0].id, node.value
        if isinstance(value, ast.Call) and _is_mushin_sweep(value.func):
            first = value.args[0] if value.args else None
            fn = functions.get(first.id) if isinstance(first, ast.Name) else None
            handles[target] = _parameters(fn) if fn else set()
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "dict"
            and not value.args
        ):
            dicts[target] = [k.arg for k in value.keywords if k.arg]
        elif isinstance(value, ast.Dict) and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in value.keys
        ):
            dicts[target] = [k.value for k in value.keys]

    allowed_run = _run_parameters()
    bad, checked = [], 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in handles
        ):
            continue  # not a mushin run(), e.g. subprocess.run
        checked += 1
        keywords: list[str] = []
        for kw in node.keywords:
            if kw.arg is not None:
                keywords.append(kw.arg)
            elif isinstance(kw.value, ast.Name) and kw.value.id in dicts:
                keywords += dicts[kw.value.id]
            else:
                bad.append(f"line {node.lineno}: `**{ast.unparse(kw.value)}`")
        allowed = allowed_run | handles[node.func.value.id]
        bad += [
            f"line {node.lineno}: `{k}=` is not a run() parameter"
            for k in keywords
            if k not in allowed
        ]
    assert checked, "found no mushin run() call in the guide"
    assert not bad, bad


def test_the_guide_links_are_nav_pages():
    links = [url.split("#", 1)[0] for url in LINK.findall(_text())]
    site_links = [url for url in links if url.startswith(_site())]
    # If site_url moves, every old link stops matching it: fail, not pass.
    assert site_links, f"the guide has no link into the docs site {_site()}"
    stale = sorted(set(site_links) - _nav_urls())
    assert not stale, f"the guide links to pages not in mkdocs.yml's nav: {stale}"
    others = [url for url in links if not url.startswith(_site())]
    assert not others, f"the guide links outside the docs site: {others}"


def test_raw_guide_links_name_a_file_in_the_repo():
    """The raw-GitHub URL is the only fallback D2 gives agents on mushin < 0.16."""
    found = []
    for rel in RAW_LINK_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for url in re.findall(re.escape(RAW_PREFIX) + r"[^\s)>`\"']+", text):
            found.append(url)
            assert (ROOT / url.removeprefix(RAW_PREFIX)).is_file(), f"{rel}: {url}"
    assert found, f"none of {RAW_LINK_FILES} links the raw guide"
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_agent_guide.py -q -p no:cacheprovider`
Expected: every test passes except `test_raw_guide_links_name_a_file_in_the_repo`, which fails with "none of [...] links the raw guide" until Task 7 adds the links.

- [ ] **Step 3: Mutation check** (do not commit the mutation)

Temporarily change `dry_run=True` to `dryrun=True` in the guide's template and run `uv run pytest tests/test_agent_guide.py -k keywords -q -p no:cacheprovider`. Expected: FAIL naming `dryrun=`. Then also change a guide link to `https://martinez-hub.github.io/mushin/guides/nope/` and run `-k links`: expected FAIL. Revert both with `git checkout src/mushin/agent_guide.md`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_agent_guide.py
git commit -m "test: run the agent guide's code and check its run() keywords and links"
```

---

### Task 6: Pin the guide's seed floors (D7 item 6)

**Files:**
- Modify: `tests/test_docs_claims.py`

- [ ] **Step 1: Add the tests** (append; add `import numpy as np`, `import xarray as xr` and `from mushin.benchmark import compare_methods` to its imports)

```python
AGENT_GUIDE = (
    Path(__file__).resolve().parent.parent / "src" / "mushin" / "agent_guide.md"
)


def _holm_floor(n_methods: int, alpha: float = ALPHA) -> int:
    """Seeds at which compare_methods first calls every pair significant on data
    that separates the methods perfectly. Each method's scores vary across seeds
    (a method constant across seeds gets NaN), and each pair's differences are
    distinct and nonzero (tied differences make the Wilcoxon p approximate)."""
    for n in range(2, 32):
        scores = np.array([(np.arange(n) + 1.0) * (i + 1) for i in range(n_methods)])
        ds = xr.Dataset(
            {"score": (("method", "seed"), scores)},
            coords={
                "method": [f"m{i}" for i in range(n_methods)],
                "seed": np.arange(n),
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the underpowered warning, below the floor
            if compare_methods(ds, alpha=alpha)["significant"].all():
                return n
    raise AssertionError(f"{n_methods} methods never clear alpha={alpha}")


@pytest.mark.parametrize("n_methods, floor", [(2, 6), (3, 7), (4, 8)])
def test_holm_seed_floor(n_methods: int, floor: int):
    """With m pairs, Holm needs m * 2 ** (1 - n) < alpha."""
    assert _holm_floor(n_methods) == floor


def test_agent_guide_quotes_the_holm_seed_floors():
    text = AGENT_GUIDE.read_text(encoding="utf-8")
    match = re.search(
        r"at least (\d+) seeds for 2\s+methods, (\d+) for 3 and (\d+) for 4", text
    )
    assert match, "could not find the seed floors in the agent guide"
    assert tuple(map(int, match.groups())) == tuple(_holm_floor(k) for k in (2, 3, 4))
```

- [ ] **Step 2: Run them**

Run: `uv run --extra eval pytest tests/test_docs_claims.py -q -p no:cacheprovider`
Expected: PASS. (These floors were checked while writing this plan: 6, 7 and 8 seeds, with `p_corrected` 0.031, 0.047 and 0.047 at the floor.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_docs_claims.py
git commit -m "test: pin the agent guide's Holm seed floors to compare_methods"
```

---

### Task 7: The docs page, `llms.txt`, README and AGENTS.md (D4, D5, D6)

**Files:**
- Create: `docs/for-agents.md`
- Modify: `mkdocs.yml` (nav), `docs/llms.txt`, `tests/test_llms_txt.py` (`EXTERNAL_HOSTS`), `README.md`, `AGENTS.md`

- [ ] **Step 1: Create `docs/for-agents.md`**

````markdown
# For AI agents

mushin ships a usage guide for AI coding agents (Claude Code, Codex, Cursor and
the like) that set up experiments for scientists and engineers. It covers
which API to use for which job, a tested template, the pitfalls that give wrong
results, how to run long sweeps from an agent, and what to check before
reporting. This page shows the guide from the main branch; from mushin 0.16,
`python -m mushin guide` prints the copy that matches the installed version.

## Point your agent at it

Add this to your project's `AGENTS.md` or `CLAUDE.md`:

```text
This project uses mushin. Install it with `pip install mushin-py` (the PyPI
package named `mushin` is a different project). Before writing or changing an
experiment, run `python -m mushin guide` and follow it. On mushin older than
0.16, read
https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md
instead.
```

To let an agent inspect finished runs, see
[Analyzing from Claude Code](guides/mcp.md): `mushin-mcp` is a read-only MCP
server over a directory of runs.

## The guide

--8<-- "src/mushin/agent_guide.md:3"
````

The include starts at line 3, below the guide's own title (Task 3's test pins that line 1 is the title and line 2 is blank). The line has no fence around it, which #253's guard allows.

- [ ] **Step 2: Add the page to the nav** in `mkdocs.yml`, right after Quickstart:

```yaml
  - Quickstart: quickstart.md
  - For AI agents: for-agents.md
```

- [ ] **Step 3: Point `llms.txt` at the guide**

In `docs/llms.txt`, add this as the last bullet of the opening facts list (after the "framework-agnostic" bullet):

```markdown
- To set up, run and report an experiment, read the [agent guide](https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md) first. On mushin 0.16 or later, `python -m mushin guide` prints the copy for the installed version.
```

and, in its "Docs" list, right after the Quickstart entry:

```markdown
- [For AI agents](https://martinez-hub.github.io/mushin/for-agents/): the usage guide for coding agents, and a block to paste into a project's `AGENTS.md` or `CLAUDE.md`
```

In `tests/test_llms_txt.py`, change `EXTERNAL_HOSTS = {"github.com"}` to `EXTERNAL_HOSTS = {"github.com", "raw.githubusercontent.com"}`.

- [ ] **Step 4: Add the README section**, right before `## What else it does`:

```markdown
## Using mushin with an AI agent

mushin ships a guide for AI coding agents that set up experiments: which API to
use, a tested template, the pitfalls that give wrong results, and what to check
before reporting. From mushin 0.16, `python -m mushin guide` prints the copy
for the installed version. The
[For AI agents](https://martinez-hub.github.io/mushin/for-agents/) page has a
block to paste into your project's `AGENTS.md` or `CLAUDE.md`, and
[`llms.txt`](https://martinez-hub.github.io/mushin/llms.txt) indexes the docs.
On older versions, point the agent at the
[raw guide](https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md).
```

- [ ] **Step 5: Tell contributors** in `AGENTS.md`

Add a Layout row after the `src/mushin/__init__.py` row:

```markdown
| `src/mushin/agent_guide.md`, `__main__.py` | The usage guide for agents that *use* mushin, printed by `python -m mushin guide` and shown at `docs/for-agents.md`. |
```

Add an invariant after the `docs/llms.txt` one:

```markdown
- **The agent guide is tested.** `src/mushin/agent_guide.md` is ASCII, its
  python blocks run in order as one script, the keywords of its `run()` calls
  exist and its links are nav pages (`tests/test_agent_guide.py`); its seed
  floors are pinned in `tests/test_docs_claims.py`. When you change behavior
  the guide describes, change the guide.
```

In "Using mushin from an agent", change the first sentence to:

```markdown
If you are writing code that *uses* mushin rather than changing it, run
`python -m mushin guide` (mushin 0.16 or later), or read the docs index at
<https://martinez-hub.github.io/mushin/llms.txt>.
```

- [ ] **Step 6: Run the docs tests and the strict build**

Run:
```bash
uv run pytest tests/test_agent_guide.py tests/test_llms_txt.py tests/test_docs_snippets.py -q -p no:cacheprovider
rm -rf /Volumes/ExtremeSSD/scratch/site && uv run --group docs mkdocs build --strict -d /Volumes/ExtremeSSD/scratch/site
grep -c "compare_methods(ds.sel(lr=0.01))" /Volumes/ExtremeSSD/scratch/site/for-agents/index.html
ls /Volumes/ExtremeSSD/scratch/site/llms.txt
uv run codespell src tests examples README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md AGENTS.md CLAUDE.md docs/llms.txt pyproject.toml changes
```
Expected: all tests pass (now including `test_raw_guide_links_name_a_file_in_the_repo`); the build succeeds; the grep prints at least `1`; `llms.txt` exists; codespell is silent.

- [ ] **Step 7: Commit**

```bash
git add docs/for-agents.md mkdocs.yml docs/llms.txt tests/test_llms_txt.py README.md AGENTS.md
git commit -m "docs: a For AI agents page, and pointers from llms.txt, the README and AGENTS.md"
```

---

### Task 8: CI checks the shipped guide and `help(mushin)` (D7 "In CI")

**Files:**
- Modify: `.github/workflows/ci.yml` (`docs`, `core-only`, `batteries-clean-install` jobs)

- [ ] **Step 1: Docs job.** In the "Docs-pinning tests" step, add the guide's tests:

```yaml
      - name: Docs-pinning tests
        run: uv run --extra eval pytest tests/test_llms_txt.py tests/test_docs_claims.py tests/test_agent_guide.py -p no:cacheprovider
```

and add "the agent guide's code must run and its keywords and links resolve" to the comment above it.

- [ ] **Step 2: core-only job.** At the end of its smoke-test heredoc, before `print("core-only smoke test passed")`, add:

```python
          # 4. help(mushin) renders on a core install and points to the agent
          #    guide; star-import and getmembers work without the eval extra.
          import inspect
          import pydoc

          assert "python -m mushin guide" in pydoc.render_doc(mushin)
          assert "compare" not in mushin.__all__
          inspect.getmembers(mushin)
          exec("from mushin import *", {})
```

and after that step add one that checks the installed (non-editable) package ships the guide:

```yaml
      - name: The core install ships the agent guide
        run: test -s "$(/tmp/core/bin/python -m mushin guide --path)"
```

- [ ] **Step 3: batteries-clean-install job.** After its last step add:

```yaml
      - name: The built wheel ships the agent guide
        run: test -s "$(/tmp/clean-batteries/bin/python -m mushin guide --path)"
```

- [ ] **Step 4: Check the workflow parses**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo ok`
Expected: `ok`. The real check is the PR's CI run.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: check the agent guide in the docs job, and that installs ship it"
```

---

### Task 9: Changelog fragment and full verification

**Files:**
- Create: `changes/+agent-guide.added.md`

- [ ] **Step 1: Write the fragment**

```markdown
mushin now ships a usage guide for AI coding agents that set up experiments: which API to use for which job, a tested template, the pitfalls that give wrong results, how to run long sweeps from an agent, and what to check before reporting. `python -m mushin guide` prints it (`--path` gives its file), the new [For AI agents](https://martinez-hub.github.io/mushin/for-agents/) page shows it with a block to paste into a project's `AGENTS.md` or `CLAUDE.md`, and `llms.txt` and the README point to it. `help(mushin)` now describes the package and points to the guide, and on a core install it renders, as do `from mushin import *` and `inspect.getmembers(mushin)`. `dir(mushin)` no longer lists the deprecated `BaseWorkflow` and `RobustnessCurve` aliases, which still import with a warning.
```

- [ ] **Step 2: Full verification**

Run:
```bash
make check
uv run --group docs mkdocs build --strict -d /Volumes/ExtremeSSD/scratch/site
```
Expected: lint, format, spell and the full test suite pass; the strict build passes.

- [ ] **Step 3: Mutation checks from the spec's Verification** (revert each after)

1. Rename `dry_run` to `dryrun` in `MultiRunMetricsWorkflow.run` and `BaseWorkflow.run` (signature only): `uv run pytest tests/test_agent_guide.py -k keywords -q` fails.
2. Change `compare_methods`' default `test="wilcoxon"` to `"welch"`: `uv run --extra eval pytest tests/test_docs_claims.py -k holm -q` fails.

Revert with `git checkout src/`.

- [ ] **Step 4: Commit**

```bash
git add changes/+agent-guide.added.md
git commit -m "docs: changelog fragment for the agent guide"
```

---

### Task 10: Review, PR and merge

- [ ] **Step 1: Reviews before the PR.** Run `/code-review feat/agent-guide`, one adversarial pass with a runtime-experiment lens (an agent following the guide end to end on a fresh project), and `/security-review` on the branch. Reproduce each finding before acting on it; stop at nothing merge-blocking.
- [ ] **Step 2: Open the PR**, then rename the fragment to `changes/<PR>.added.md` and push.
- [ ] **Step 3: Codex and CI.** Wait for Codex's review of the head commit and green CI; fix findings that hold up.
- [ ] **Step 4: Squash-merge** with a written message (no AI attribution), then delete the branch and worktree.
