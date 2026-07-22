# From exploration to a paper-ready sweep

Most projects have two phases with very different needs. **Exploration** is fast
and messy: you try a coarse grid, look at a few cells, change the model, try
again — you want speed and cheap iteration, and you do not care that the results
are throwaway. **The paper run** is the opposite: one clean, complete grid with
enough seeds to make a claim, captured well enough that a reviewer can reproduce
it.

mushin does not have an "explore mode" or a `promote()` button, and that is
deliberate. The two phases are the *same sweep primitive* run with different
options, and the transition between them is a convention, not a feature. This
guide is that convention — the safe path from a scratch grid to a claim you can
defend.

> **The one trap to avoid:** do not grow your exploration directory into your
> paper directory. Start the paper run **fresh**. The rest of this guide explains
> why, and what to do instead.

## Phase 1 — explore fast

In exploration you want to cover a lot of configurations cheaply and throw most
of them away. Four options make that cheap:

- **`sample=K`** runs a random `K`-cell subset of the grid instead of all of it,
  so you can probe a wide space for the price of a handful of runs. The subset is
  deterministic and resume-safe.
- **`cache_dir=`** is a content-addressed cache shared across directories: a cell
  you already computed — same config, same task code — is reused instead of
  recomputed, even in a brand-new `working_dir`. Refining a coarse grid into a
  finer one only pays for the *new* cells.
- **`dry_run=True`** prints the grid (cell count and each axis) without launching
  anything — catch a `range(100)` you meant to be `range(10)` before it costs you.
- **`confirm_above=N`** (or the `MUSHIN_MAX_CELLS` environment variable) refuses a
  grid larger than `N` cells, so an accidental combinatorial blow-up fails fast.
- **`max_total_seconds=T`** time-boxes a run: once the budget is spent the
  remaining cells are skipped (NaN) rather than the sweep running to the end, and
  a later `resume=True` finishes them when you have more time.

A typical exploration loop:

```python
from mushin import multirun, show, best, diff
from mushin.workflows import MultiRunMetricsWorkflow

wf = MyWorkflow()

# Preview first — how big is this really?
wf.run(lr=multirun([1e-4, 3e-4, 1e-3, 3e-3]), seed=multirun(range(5)), dry_run=True)

# Probe a coarse grid cheaply: 6 random cells, reusing anything already cached.
wf.run(
    lr=multirun([1e-4, 3e-4, 1e-3, 3e-3]),
    wd=multirun([0.0, 1e-4, 1e-2]),
    seed=multirun(range(5)),
    sample=6,
    cache_dir="~/.cache/mushin",
    working_dir="runs/explore-01",
)
```

Then look at what happened without loading the full `xarray` dataset — these
history tools read the sweep directory's sidecars directly:

```python
show("runs/explore-01")              # a status/metrics table, one row per cell
best("runs/explore-01", "accuracy")  # the winning cell + its config + its dir
diff("runs/explore-01", "runs/explore-00")  # what moved since the last idea
```

Iterate: change the model, narrow the grid, re-run. With `cache_dir=` set, cells
you have already computed stay free across every iteration.

### A word on exploration statistics

If you compute statistics during exploration — say `compare_methods` on a
`sample=`d sweep — mushin will **refuse**, because a subsampled or budget-limited
grid has holes and stats over holes are misleading. That refusal is the point.
When you knowingly want a rough read anyway, pass `allow_incomplete=True`; it
computes over the completed cells only and warns you that the result may be
under-powered. Treat any such number as a *hint that a configuration is worth a
real run*, never as a result.

## Phase 2 — the paper run

When a configuration looks promising, run it properly. This is a **new sweep in a
new directory** — not a resume of the exploration directory, not a `sample=`, not
a cache reuse of exploration cells.

```python
wf = MyWorkflow()
wf.run(
    method=multirun(["baseline", "ours"]),
    seed=multirun(range(10)),        # enough seeds to make a claim
    working_dir="runs/paper/main-result",
    capture_env=True,                # snapshot the exact environment
    on_error="nan",                  # fail-soft, then resume the stragglers
)
```

What changes from exploration:

- **Full grid, no `sample=`.** A claim needs the complete method × seed grid, not
  a subset.
- **More seeds.** Exploration might use 3 seeds; a claim usually needs more —
  mushin will warn you if your test cannot even reach significance at the seed
  count you chose (see [Understanding the statistics](statistics.md)).
- **`capture_env=True`.** This writes a full dependency snapshot next to the
  sweep so a reviewer can rebuild your environment. mushin already records
  per-cell provenance (git commit, package versions, config) on every run; this
  adds the pinned dependency list.
- **A stable, named `working_dir`** you will keep and cite, e.g.
  `runs/paper/main-result`.

Then verify the claim on the *complete* grid — here the incompleteness guard is
your friend, not an obstacle:

```python
from mushin.benchmark import compare_methods

ds = wf.to_xarray()
compare_methods(ds)   # runs only because the grid is complete; refuses if not
```

See [Comparing methods](compare.md) for the full statistical workflow.

## Why re-run instead of "promoting" exploration cells

It is tempting to keep the good cells from exploration and just add seeds by
resuming. Don't — and mushin actively stops you from doing it by accident.

- **Code eras must not mix.** Between exploration runs you edit the task,
  helpers, and config. If you resumed an exploration directory, some cells would
  carry results from an *older* version of your code and some from the current
  one, silently combined into one dataset. mushin's resume guards on a
  fingerprint of both the resolved config **and** the task source, so an edited
  task re-runs its cells rather than reusing a stale result — but the only way to
  be sure *every* cell in your paper grid came from one code era is to run that
  grid fresh. (See [Resilient & resumable sweeps](resilience.md) for how the
  fingerprint guard works.)
- **Winner's curse.** The configuration that looked best across a noisy
  exploration is, on average, an over-estimate — you selected it *because* it got
  a lucky draw. The only cure is a fresh evaluation on new seeds. Reusing the
  exploration cells that won bakes the optimism straight into your headline
  number.

So the handoff is a re-run, by design. Exploration tells you *which*
configuration to run for the paper; the paper run is what you actually report.

Two mushin features make the re-run cheap and honest at the same time:

- `cache_dir=` reuse is safe here because it is keyed on config **and** code — a
  cell only comes from the cache if it is byte-for-byte the same computation. It
  never mixes code eras. (If you changed anything since exploration, those cells
  simply recompute.)
- Because the paper run is a normal sweep, everything else in mushin applies to
  it: [fail-soft + resume](resilience.md) for the flaky cells, `show`/`best` to
  watch it, and the provenance record it writes automatically.

## The reproducibility handoff

When the paper run is done, the sweep directory *is* the artifact a reviewer
needs. To hand it off:

- **Per-cell provenance** — every cell's `mushin_provenance.json` records the git
  commit, dirty flag, package versions, and resolved config. `wf.provenance`
  reads a representative one.
- **Environment snapshot** — `capture_env=True` wrote `mushin_env.txt` (a pinned
  dependency list) at the sweep root.
- **A flat results table** — `mushin.export.table("runs/paper/main-result",
  path="results.csv")` writes a dependency-free CSV of every cell (params,
  status, metrics) for an appendix or a `pandas` reader.
- **The manifest** — `mushin_sweep_manifest.json` records that every requested
  cell completed, which is what lets `compare_methods` confirm the grid is whole.
- **Notes and tags** — annotate the run with `run(..., notes="...", tags=[...])`
  to record *why* it exists. They persist in the manifest, travel on the dataset
  (`ds.attrs["mushin_notes"]` / `mushin_tags`), survive a resume, and read back as
  `wf.notes` / `wf.tags`.

## Checklist

Before you call a sweep paper-ready:

- [ ] Ran as a **fresh** `working_dir`, not a resume/`sample=` of exploration.
- [ ] **Full grid**, no `sample=`.
- [ ] Enough **seeds** that the test can reach significance (no underpowered
      warning).
- [ ] `capture_env=True` — the environment is snapshotted.
- [ ] `wf.is_complete` is `True` and `compare_methods` runs without an
      `IncompleteSweepError`.
- [ ] The headline configuration was **evaluated fresh**, not carried over from
      the exploration that selected it.

## Related

- [Resilient & resumable sweeps](resilience.md) — fail-soft, resume, and the
  fingerprint guards referenced above.
- [Comparing methods](compare.md) and
  [Understanding the statistics](statistics.md) — turning the finished grid into
  a defensible claim.
