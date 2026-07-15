# Design: EQUINE-style example notebooks

**Date:** 2026-07-14
**Status:** Approved (brainstorming) — pending implementation plan
**Branch:** `example-notebooks`

## Problem

mushin's docs are guides (how-to/reference) plus markdown examples with embedded
outputs (the batteries guide). What's missing — modeled on
[EQUINE's example notebooks](https://mit-ll-responsible-ai.github.io/equine/example_notebooks/toy_example_GP/)
— is **runnable tutorial notebooks**: narrative prose interleaved with code cells,
their real outputs, and **inline plots**, building a flow end-to-end. These read as
"sit down and follow along," which guides don't, and they *show* results (tables,
significance, figures) instead of describing them.

## Goals

- A set of **real Jupyter notebooks** rendered in the docs site (EQUINE-style
  `In[]/Out[]` + inline matplotlib plots), covering mushin's main flows.
- Notebooks are **CI-executed** so they can never rot — matching mushin's
  tested-examples ethos.
- "Done better" than the reference: each notebook has a clear narrative, plots that
  carry the story (CIs, significance, metric-vs-parameter curves), and mushin's
  distinctive angles (reproducibility, statistics, resilience).

## Non-goals

- Not replacing the guides — notebooks **complement** them (guides = reference,
  notebooks = walkthroughs), exactly as EQUINE keeps both.
- Not executing notebooks at docs-build time (keeps the docs job fast and dep-light;
  see Testing).

## Design

### Toolchain

- **Authoring/render:** real `.ipynb` files under `docs/notebooks/`, rendered by
  **`mkdocs-jupyter`** (`execute: false` — render the committed outputs, so the docs
  build stays fast and needs no torch). Add `mkdocs-jupyter` to the `docs`
  dependency group; register the plugin in `mkdocs.yml`.
- **Anti-rot:** a CI job runs **`nbmake`** (`pytest --nbmake docs/notebooks/`) which
  *executes* every notebook and fails on any cell error. nbmake checks *runs*, not
  output equality, so numerical drift never causes flaky failures; committed outputs
  are for display only. Add `nbmake` + `jupyter`/`ipykernel` to the `dev` group.
- **Plots:** matplotlib (already the `viz` extra + dev group). Notebooks that plot
  set a non-interactive backend and produce figures inline.
- **Reproducibility:** every notebook is fully seeded (torch/numpy) so a reader's
  re-run matches the committed narrative, and `nbmake` re-execution is deterministic.

### Notebook set (6)

Under `docs/notebooks/`, numbered for nav order:

1. **`01_sweep_to_dataset.ipynb`** — the flagship. Define a `task(lr, seed)`, run a
   `multirun` sweep, get the labeled `xarray.Dataset`; **plot** mean metric vs the
   swept parameter with per-seed error bars. Establishes the core mental model.
2. **`02_compare_and_batteries.ipynb`** — train a few tiny models per seed, run
   `compare(task="classification")`, read `.summary()` / `.comparisons`; **plot** a
   bar chart of per-method means with CIs and significance markers. Then show a
   second battery (e.g. `regression` or `segmentation`) to convey the task API.
   Cross-link the Built-in batteries guide.
3. **`03_study.ipynb`** — `Study`: a multi-seed training sweep routed straight into
   `compare`, in one call; show the `BenchmarkResult` and a comparison plot.
4. **`04_resilience.ipynb`** — mushin's distinctive story: run a sweep where one
   cell fails with `on_error="nan"`; show the NaN cell + `wf.failures` + the
   `mushin_sweep_manifest.json`; show `compare`/stats raising `IncompleteSweepError`;
   fix and `resume=True`; show only the failed cell re-running and the grid filling
   in place; then a clean compare. **Plot** the grid before/after (NaN → filled).
5. **`05_llm_eval.ipynb`** — `llm.compare_llms` with two toy (deterministic-vs-seed-
   varying) systems and a simple metric; show the significance table and the
   deterministic-system warning; **plot** per-system score distributions over seeds.
6. **`06_sklearn_framework_agnostic.ipynb`** — the sweep layer is framework-agnostic:
   a scikit-learn `LogisticRegression` sweep (no torch) → labeled dataset; **plot**
   accuracy vs the swept `C`. Ties to the framework note in Core concepts.

Each notebook: a title + intro markdown cell stating what it builds, then
alternating markdown/code cells with committed outputs, at least one inline plot,
and a closing "next steps / see also" cell linking the relevant guide(s).

### Docs integration

- New top-level nav section **"Example notebooks"** (after "Examples", before "API
  Reference") listing the 6 notebooks.
- The `mkdocs-jupyter` plugin renders them; verify `mkdocs build --strict` stays
  clean (notebooks must not emit broken cross-references).
- Add short pointers from the most relevant guides to their notebook (e.g. the
  resilience guide → notebook 04).

### Testing / CI

- New CI job **`notebooks`** (ubuntu, with `--extra detection --extra image --extra
  audio` if any battery notebook needs them; at minimum `viz`): `uv run pytest
  --nbmake docs/notebooks/ -p no:cacheprovider`. Fast-ish (tiny models), executes
  all 6.
- Whether to make it a *required* status check: yes, once green — same rationale as
  `batteries-clean-install` (docs that claim to run must run). Decide at merge time.

## Risks / open questions

- **Notebook JSON diffs are noisy.** Committed `.ipynb` with outputs produce large,
  hard-to-review diffs. Mitigation: keep outputs small (tiny models, short tables),
  and strip nondeterministic metadata (execution counts/timestamps) via an
  `nbstripout`-style cleanup of *metadata only* (keep outputs) before commit — or
  accept the diff. The plan will pick one.
- **mkdocs-jupyter + Material theme** rendering quirks (code/plot styling in light
  and dark). Verify visually via the deployed site (or a local `mkdocs serve`
  screenshot) after the first notebook.
- **nbmake execution cost.** Six notebooks with torch import + tiny training; keep
  each notebook's compute trivial so the CI job stays well under a couple minutes.
- **matplotlib as a notebook dep.** Notebooks import matplotlib; the `notebooks` CI
  job and any local run need the `viz` extra (or dev group, which has it).

## Rollout

Additive: new `docs/notebooks/` + a nav section + a CI job + dev/docs deps. No
package/runtime change. Ships incrementally is possible (notebook-by-notebook) but
the plan will build all 6 in one branch to establish the pattern consistently.
This is a docs feature; no version bump required on its own (folds into the next
release's changelog via a fragment).
