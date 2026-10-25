# Using mushin: a guide for AI coding agents

This guide is for a coding agent setting up an experiment for a scientist or
engineer. The copy that `python -m mushin guide` prints matches the installed
mushin, whose version is its first line. A copy read from the web describes the
newest mushin; most of it applies from 0.10 on. For exact signatures, use
`help(obj)`.

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
| `pip install hydra-ray-launcher` | the Ray launcher, to pack several jobs onto each GPU (a Hydra plugin, not an extra) |

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
| Run several jobs on each GPU | Ray; see the GPU packing guide (Going further). `pin_gpu_round_robin` needs a fresh process per job, which joblib stops giving once cells outnumber `n_jobs` |
| Run on a cluster | the multi-node guide (see Going further) |

## A template experiment

Write the experiment as one script in the scientist's project. Leave their code
where it is: the sweep task is a new module-level function that calls it and
returns a dict of numbers. Do not put `@mushin.sweep` on their function. It
replaces the function with a sweep handle, which cannot be called, so their
other call sites break. The template's statistics need the `eval` extra:
`pip install "mushin-py[eval]"`.

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
- `experiment.workflow` is the `MultiRunMetricsWorkflow` of the latest `run()`
  call, a dry run included: read its `failures`, `skipped`, `working_dir` and
  `to_dataframe()` right after the run you care about.
- `compare_methods` runs a paired test (Wilcoxon by default) for each pair of
  methods across seeds, and corrects the p-values for the number of pairs
  (Holm). Its table has `mean_diff` and `effect_size`, the raw and corrected
  p-values, and `significant`. `mean_diff` is `method_a - method_b`, so a
  negative value means `method_b` scored higher.

## Rules that prevent wrong results and wasted compute

Sweep a value with `mushin.multirun([...])`. A bare list raises:

```python
try:
    experiment.run(method=["baseline", "new"], lr=0.01, seed=0, dry_run=True)
except TypeError as e:
    print(e)  # ... wrap them: `method=mushin.multirun(['baseline', 'new'])` ...
```

To pass a list as one value instead, use `mushin.hydra_list([...])`. The other
rules:

- **Return a dict of numbers.** The values are converted to JSON into each
  cell's `mushin_metrics.json`.
- **Each cell runs in its own directory**, so a relative path in the task
  resolves there, not in the project. Build input paths from
  `mushin.original_cwd()`, for example
  `mushin.original_cwd() / "data" / "train.csv"`.
- **Check the size first.** Run with `dry_run=True` before anything expensive.
  `confirm_above=N`, or the `MUSHIN_MAX_CELLS` environment variable, refuses a
  grid of more than `N` cells. `max_total_seconds=T` stops starting cells `T`
  seconds after the first cell that computes, and records the rest as skipped.
  A later `resume=True` runs them, with no time limit unless it passes
  `max_total_seconds` again, which starts a new clock. Under
  `launcher="joblib"` every worker honors the same deadline, so `T` bounds the
  elapsed time, not the compute summed over the workers, and each worker can
  finish a cell it started just before the deadline: the sweep can run past
  `T` by up to one cell per worker.
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
  are incomplete or whose config, task function or project code changed: the
  helpers, classes and constants the task (and `pre_task`) uses, followed
  through the project's own modules as the running process has loaded them
  (a notebook's functions, a script's main block and a registry dict of
  classes included), and the classes the config names as `_target_` (by
  their module's file). A comment or docstring in a helper is not an edit
  (one in the task function's own body is), and a value computed on import
  (`os.cpu_count()`) is compared by its source, so it reads alike on every
  node. It does not see an installed package,
  data files, or code reached only through some other string. After such a
  change, use a new
  `working_dir`. `cache_dir=` keys on the same fingerprints. Hydra still
  lists every cell when a resume launches; the completed ones return their
  saved metrics without running.
- **`compare_methods` needs exactly two dims, `method` and `seed`,** and string
  `method` values. Select or average away other dims on the Dataset first:
  `ds.sel(lr=0.01)`, or `ds.mean("lr", keep_attrs=True)` (older xarray drops
  attrs in a reduction unless asked). The record of failed cells lives in
  `ds.attrs`, and `compare_methods` checks it there: `ds["score"]`, and
  anything built from it, has lost it, so an incomplete sweep would be
  compared without a word. Check `wf.is_complete` before comparing. Rename
  yours if needed (`ds.rename({"model": "method"})`). To pick one method, use
  `ds.sel({"method": "new"})`: `ds.sel(method="new")` does not select, since
  `method` is also a keyword of `sel`. A method whose score is the same for
  every seed gets NaN for its pairs, with a warning.
- **Seeds.** With `k` methods there are `m = k * (k - 1) / 2` pairs, and with
  Wilcoxon a pair can only be significant at 0.05 if
  `m * 2 ** (1 - n) < 0.05` for `n` seeds. So use at least 6 seeds for 2
  methods, 7 for 3 and 8 for 4; `compare_methods` warns when the seed count
  cannot reach significance for the pairs it corrects over. These are floors,
  not enough power: a real but small difference needs more seeds.
- **Make each seed an independent repeat.** The statistics treat seeds as
  independent. If each seed only re-splits one small dataset into train and
  validation, the runs share most of their data and the p-values come out too
  small. Prefer a fixed held-out test set, with the seed varying only the
  model's own randomness (initialization, data order, subsampling). Otherwise
  say so in the report: `compare_methods` has no correction for overlapping
  splits.
- **Choose on some seeds, test on others.** Pick settings (a learning rate, a
  regularization strength) on exploration seeds, then run the comparison on
  fresh seeds: choosing and testing on the same seeds overstates the winner.
  If a knob means something different for each method, sweep a level
  (`level=mushin.multirun(["low", "high"])`) and map it to each method's value
  inside the task. Then select each method at its own level on the Dataset,
  for example `ds.sel(level=xr.DataArray(["high", "low"], dims="method"))`
  (in the order of `ds["method"]`), which keeps the record of failed cells.
- **Commit before the final runs.** Each cell's `mushin_provenance.json`
  records the git commit and whether the tree was dirty, and untracked files
  count. Add `runs/`, `__pycache__/` and any logs to `.gitignore`, commit,
  and check that `git status --porcelain` prints nothing before the final run.
- **Record the environment of the final run** with `capture_env=True`: it
  writes the full package list to `mushin_env.txt` in the sweep directory.
  Each cell's provenance lists only mushin's own stack (torch, numpy,
  Lightning, Hydra), not, for example, scikit-learn.
- **`best()` is per cell unless you pass `over=`.** `mushin.best(root, "score")` returns the single
  best cell, which is one seed. To pick a configuration, average over seeds:
  `mushin.best(root, "score", over="seed")` returns the configuration with
  the best mean, counting only configurations with a result at every seed
  another has (older versions: `ds["score"].mean("seed")`).
- **On Windows, set `PYTHONUTF8=1`** (or run `python -X utf8`) when output is
  captured: a pipe then writes in the ANSI code page (cp1252 on Western
  systems), and printing a character it lacks fails. mushin's own console
  output does not fail there (except `python -m mushin guide --path`, which
  exits with an error rather than print a path the code page cannot show),
  but the rest of a run (your code, the libraries it calls, their logs)
  still can, so set it anyway.

## Running long sweeps from an agent

An agent's shell commands time out, and a sweep can take hours. Put the sweep
in a script, like the template, with a fixed `working_dir` under the gitignored
`runs/`, and start it in the background, with its log under `runs/` too:

```bash
mkdir -p runs
nohup python experiment.py > runs/compare-v1.log 2>&1 &
```

Then poll it:

```bash
python -c "import mushin; mushin.show('runs/compare-v1')"
```

`show` prints one row per cell that has started, with its status (`running`,
`completed`, `failed` or `skipped`) and metrics, and returns the rows. A cell
that has not started has no row yet. This counts the finished cells; the
sweep is done when the count reaches the dry run's `num_cells`, or when the
process has exited and `mushin_sweep_manifest.json` exists:

```bash
python -c "import mushin; r = mushin.show('runs/compare-v1'); print(sum(x['status'] == 'completed' for x in r.rows))"
```

Use the default launcher for a sweep you run this way. If the process died
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
    mushin_provenance.json     git commit, package versions, hardware if torch ran (cells that ran)
    mushin_workflow.log, .hydra/
```

Reload a finished sweep, for example in a new process:

```python
from mushin.workflows import MultiRunMetricsWorkflow

wf = MultiRunMetricsWorkflow().load_from_dir("runs/compare-v1", "mushin_metrics.json")
assert wf.is_complete, wf.failures + wf.skipped  # every cell has a result
ds = wf.to_xarray()  # the same (method, lr, seed) dataset as above
```

A reloaded sweep with failed, skipped or unfinished cells is not complete, and
`compare_methods` refuses it. In mushin 0.15.1 and earlier (the first line of
`python -m mushin guide` is the installed version), a reload lost that record,
and `is_complete` was always true for a reloaded sweep. There, check that
`mushin.show(root)` lists a `completed` row for every cell of the grid before
comparing.

`mushin.show(root)`, `mushin.best(root, metric)` and `mushin.diff(root_a,
root_b)` read a sweep directory directly; `show` prints its table and also
returns it. To let an MCP client such as Claude Code inspect the runs, install
`"mushin-py[mcp]"` and see the MCP guide.

## Before reporting to the scientist

Check each of these:

- Every cell of the grid completed: `mushin.show(root)` lists one `completed`
  row per cell, as many as the dry run's `num_cells`. (From the mushin that
  ships this guide, `wf.is_complete` on a reloaded sweep says the same.) If
  not, report the failed or missing cells.
- There are enough seeds for the number of methods compared (see Rules).
- The report gives corrected p-values (`p_corrected`) and effect sizes
  (`mean_diff`, `effect_size`), with the direction stated, not raw means alone.
- The provenance is clean: `"dirty": false` in `mushin_provenance.json`, and
  the final run used `capture_env=True`.
- The seeds were independent repeats, or the report says they were not; and
  the settings compared were chosen on other seeds.
- The report names the sweep's directory, so the run can be found and
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

For exact signatures: `help(mushin.MultiRunMetricsWorkflow.run)` (every
keyword `run()` takes; `help(mushin.workflows.BaseWorkflow.run)` adds detail on
the launcher ones) and `help(compare_methods)`.
