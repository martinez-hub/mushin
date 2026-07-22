# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- towncrier release notes start -->

## [0.10.0] - 2026-07-22

### Added

- A pre-launch cell-count gate guards against an accidentally huge grid.
  `run(..., confirm_above=N)` refuses to launch a sweep with more than `N` cells,
  raising a `ValueError` that reports the count and how to proceed (preview with
  `dry_run=True`, or raise the ceiling). The `MUSHIN_MAX_CELLS` environment
  variable supplies a default ceiling when a call doesn't set one — so a shared
  cluster or CI profile can cap every sweep — and an explicit `confirm_above=`
  wins over it. A malformed `MUSHIN_MAX_CELLS` is ignored, and `dry_run=True`
  bypasses the gate so an over-limit sweep can still be previewed.
- `mushin.best(root, metric)` returns the completed cell that optimizes a metric
  across a sweep — `mode="max"` (default) or `mode="min"`. Like `mushin.show` it
  reads the per-cell sidecars directly (offline, no Hydra/xarray). The returned
  `BestResult` carries the winning `combo` (swept params), the optimized `value`,
  the full `metrics` dict, the cell `status`, and its job `dir` (for locating
  checkpoints/artifacts). Failed/running/skipped cells and non-finite metric
  values are ignored; an unknown metric raises a `ValueError` that lists the
  available scalar metrics.
- `mushin.diff(a, b)` compares two sweep directories. It aligns cells by their
  swept-parameter combination and, for each shared cell, reports the delta
  `b - a` of every metric that is a finite scalar in both — printed as a `Δmetric`
  table. Cells present in only one sweep are listed (`only_in_a` / `only_in_b`),
  and the two runs' environment provenance is diffed field-by-field
  (git/packages/python/accelerator), excluding volatile fields like the per-cell
  timestamp. Like `show`/`best` it reads the per-cell sidecars directly (offline,
  no Hydra/xarray); the returned `DiffResult` exposes `.rows`, `.only_in_a`,
  `.only_in_b`, `.provenance`, and `.table`.
- `mushin.export.table(root)` exports a sweep as CSV — one row per cell with its
  swept parameters, `status`, and metrics. It reads the per-cell sidecars directly
  (offline, no Hydra/xarray) and writes full-precision values so pandas parses
  numbers numerically. Returns the CSV string, or writes to `path=` and returns the
  `Path`; `metrics=` restricts the metric columns. A durable, spreadsheet-friendly
  substrate for researchers who prefer pandas to xarray.
- `mushin.show(root)` prints (and returns) a status/metrics table for a sweep
  directory. It reads each cell's status and metrics sidecars directly — pure
  JSON, so it needs neither Hydra nor xarray and works mid-sweep — making it handy
  for watching a live sweep or eyeballing a finished one before the full
  `to_xarray` load. Each row carries the cell's swept parameters, its status
  (`completed`/`running`/`failed`/`skipped`/…), and its metric values; `metrics=`
  restricts the metric columns and `sort=` orders the rows. The returned
  `ShowResult` exposes `.rows` (one dict per cell) and `.table`.
- `run()` now validates up front that every required task (and `pre_task`)
  parameter is satisfied — by the base config, an override, or the raw
  `overrides=[...]` list — and raises a clear `ValueError` naming the missing
  parameter(s) before launching. Previously a genuinely missing parameter surfaced
  as an opaque per-job Hydra `ConfigAttributeError` after the sweep had already
  started, once per cell. A parameter supplied only via an override is still valid
  (the common decorator case, where the base config is empty), so this adds no
  false positives.
- `run(..., cache_dir=...)` adds a content-addressed cache of completed cells that
  is shared across `working_dir`s. A cell whose resolved config AND task source
  match a previously-computed cell — keyed on the same fingerprints as `resume` —
  reuses that result instead of recomputing, so a cell computed in one sweep is
  free in another (a coarse grid refined into a finer one, a re-run in a fresh
  directory). Newly-computed cells are stored in the cache automatically. It
  complements `resume` (which reuses only within a single `working_dir`); a changed
  non-swept config value or an edited task body is a cache miss. Cache writes are
  best-effort and never fail the cell.
- `run(..., dry_run=True)` (and `@mushin.sweep` `.run(dry_run=True)`) previews a
  sweep instead of launching it: it prints the cell count and each swept axis with
  its values — so a range typo shows up as an unexpectedly wide axis before any
  compute is spent — and returns a summary dict (`num_cells`, `axes`, `fixed`,
  `working_dir`) with no jobs run.
- `run(..., max_total_seconds=T)` adds a graceful wall-clock budget. Once the
  budget is exhausted the remaining grid cells are skipped — recorded `'skipped'`
  in the manifest and per-cell status, NaN in the dataset, and surfaced in
  `self.skipped` (and a `mushin_skipped` dataset attr) — instead of the sweep
  running to the end. The clock starts at the first computed cell, so at least one
  cell always runs and resume cache hits don't consume the budget; a cell already
  running is never interrupted. Because a skipped cell is not `completed`, a later
  `resume=True` with more time finishes exactly the skipped cells. The budget is
  measured per launcher process, so it's best paired with the default sequential
  launcher.
- `run(..., notes="...", tags=[...])` annotates a sweep with free-form lineage. The
  note and tags are recorded in the sweep manifest, exposed on the workflow as
  `wf.notes` / `wf.tags` (and preserved when a sweep is reloaded with
  `load_from_dir`), and carried on the dataset as the `mushin_notes` /
  `mushin_tags` attrs — so "why did I run this?" travels with the results. `tags`
  must be a list of strings and `notes` a string. A resume that does not re-pass
  `notes`/`tags` preserves the original run's lineage rather than wiping it.
- `run(..., sample=K)` runs only a random `K`-cell subset of the grid — the rest
  are skipped (NaN, and listed in `self.skipped`) — for fast exploration of a large
  grid without paying for every cell. The subset is chosen deterministically from
  the Hydra job indices (seeded by `sample_seed`, default 0), so it is reproducible
  and identical across a resume; resuming *without* `sample` fills in the remaining
  cells. `sample >= n_cells` runs everything. Because selection is by job index it
  is launcher- and axis-type-agnostic. Note: the full grid is still composed by
  Hydra (only the sampled cells run), and a sampled sweep is intentionally
  incomplete (`is_complete` is False).

### Fixed

- A batch of defensive-handling fixes: corrupt-but-valid-JSON cell-status sidecars now degrade (resume treats a null/`"3"` attempt or a wrong-shape record as absent) instead of crashing; `load_from_dir` raises a real `FileNotFoundError` for a missing config (not a `-O`-stripped assert); the resume contextvar no longer leaks when a status write fails; `RobustnessCurve.plot` no longer leaks a figure when plotting raises; a malformed tuning pin reaches its friendly "delete it or retune" guard; `mushin.lightning`/`benchmark`/`mcp`/`testing` are reachable as lazy submodules after `import mushin`; `compare`/`compare_llms` reject an unknown `correction=` before any evaluation runs; the MCP `get_failures`/`get_provenance` tools tolerate wrong-type sidecar JSON; and a `value_check` message typo ("None orof") is fixed.
- A typo'd sweep-axis name is now caught before launch instead of silently
  producing a wrong dataset. Passing `run(lrate=multirun(...))` when the task
  parameter is `lr` previously ran to completion and added a phantom `lrate`
  dimension of constant values; it now raises a `ValueError` that names the likely
  intended parameter. Only near-misses of a real target are rejected — deliberate
  overrides beyond the task's own parameters (values consumed by `pre_task`,
  config groups, interpolations) and tasks declaring `**kwargs` are unaffected.
- Fixed `trainer.validate()` under `HydraDDP`/`HydraFSDP`: the child ranks were told to run `fit` (train) while rank 0 validated — a collective mismatch that hangs or crashes. The launcher now maps Lightning's `_validate_impl` to a `pl_validating` flag and `_pl_main` dispatches `trainer.validate` accordingly, alongside the existing test/predict paths.
- Fixed a cluster of silent-wrong-results bugs in the override → grid pipeline: a fixed (non-`multirun`) string containing `,` or `=` is no longer re-split into an accidental sweep and `"true"`/`"1"` stay strings; a duplicated sweep-axis value is rejected instead of silently collapsing two cells into one; `combo_key` is now injective (delimiter-containing values can't collide with a different combination); a dotted sweep (`model.width`) on a config field that already exists no longer crashes; and a batching sweeper's multi-batch job list is flattened instead of failing after every job ran.
- Fixed five defects found by an adversarial review of the exploration features:

  - **`cache_dir`** is now resolved to an absolute path (`expanduser().resolve()`)
    before launching. A relative or `~`-prefixed `cache_dir` was previously
    dereferenced inside each per-cell job dir (Hydra chdirs into it), so
    cross-cell/cross-directory reuse silently never hit.
  - **`resume=True` with `sample=`** is now rejected with a clear error. Combining
    them overwrote already-completed cells (skipped by job index) to `skipped`/NaN,
    silently discarding prior results; resume WITHOUT `sample` fills the rest.
  - **The cell count** (used by `sample=`, `confirm_above=`/`MUSHIN_MAX_CELLS`, and
    `dry_run`) now counts sweep axes supplied via the raw `overrides=[...]` list,
    not only `param=multirun(...)` kwargs — so those no longer undercount the grid.
  - **`compare_methods(allow_incomplete=True)`** now computes each comparison over
    only the seeds completed for both methods (dropping NaN cells), fulfilling its
    documented promise instead of feeding NaN into the tests and returning all-NaN
    statistics.
  - **`mushin.show(sort=...)`** places NaN metric values last deterministically
    (they no longer break the sort's total order) and raises a clear error when
    `sort` names a column that isn't present.
- Fixed metric-value coercion into the JSON sidecar: a task returning a nested-dict metric containing numpy/torch values (or a `datetime`/`Path`) no longer crashes the sidecar write *after* the task succeeded (which left the cell stuck and aborted the sweep); non-finite metrics (NaN/±Inf) now serialize as valid JSON (`null`) instead of the non-strict `NaN`/`Infinity` literals that broke strict parsers, and read back as NaN so metric columns stay float dtype on resume; a non-scalar battery metric (numpy array/list, not just a torch tensor) now gives the crafted "must return scalar" error.
- Provenance no longer leaks secrets: `mushin_provenance.json` records the config with `resolve=False` (so a `${oc.env:SECRET}` interpolation is kept as an unresolved reference instead of baking the resolved secret into the file), and values under secret-named keys (`api_key`, `token`, `password`, ...) or shaped like provider tokens (`sk-...`, `hf_...`) are redacted. This also fixes the MCP `get_provenance(include_config=True)` tool, which re-served the on-disk record.
- Reloading a sweep offline with `load_from_dir(dir, "mushin_metrics.json")` no
  longer crashes. The default `metric_load_fn` now sniffs the file — it reads the
  JSON metrics sidecar (written by any task that returns a dict, including
  `@mushin.sweep`/decorator sweeps) with `json` and falls back to `torch.load`
  only for torch pickle/`.pt` files. Previously it was hard-wired to `torch.load`,
  which raised `UnpicklingError` on the JSON sidecar unless you overrode the loader.
- Resume now guards against stale-code reuse: editing a task body and re-running with `resume=True` (same config, same swept params) re-runs the affected cells — with a clear warning — instead of silently returning the previous run’s cached metrics. A per-cell hash of the task source is stored alongside the config fingerprint; a completed cell is reused only when both still match. Legacy sweeps (no recorded code hash) resume as before.
- Resuming a sweep after changing the grid shape (adding or removing an axis) no longer silently reuses the wrong cells: previously, with legacy (pre-fingerprint) sidecars, adding an axis projected every new cell onto the old parameters, reused a stale cell across every new axis value, dropped the new dimension entirely, and reported the sweep complete. The resume now keys reuse on the current swept parameters, so a shape change re-runs the full grid, with a one-time warning.
- Resuming into a working_dir whose numeric job dirs get reused for different cells no longer leaves stale metrics behind: on a cache hit the reused cell now writes its metrics and status into the current job dir, so the dir stays consistent with the config Hydra wrote there. Previously a reused cell (e.g. a=3) kept the leftover metrics of whatever cell last used that dir (e.g. a=1), so the manifest and an offline `load_from_dir` mis-keyed the value.
- `compare_methods` now refuses a sweep that has **skipped/never-run** cells, not
  just failed ones. Previously it keyed only on `mushin_failures`, so a `sample=`
  subset or a `max_total_seconds`-limited sweep — both of which leave NaN cells —
  would compute statistics silently over a partial grid. It now also keys on the
  `mushin_skipped` completeness signal and raises `IncompleteSweepError`. A new
  `allow_incomplete=True` argument bypasses the guard (with a warning) to compute
  stats over only the completed cells, for exploratory analysis of a partial
  sweep. As before, the guard keys on the completeness signals (attrs), never on
  raw NaN values, so a metric that is legitimately NaN for other reasons does not
  trigger it.

### Misc

- Documentation fixes: `load_from_checkpoint` docstring corrected (its three params default to None, not "state_dict"/"model"; no `load_module` param; returns the passed `nn.Module`, not a `LightningModule`); `run()`/wrapper docstrings reference `hydra_zen.zen` (not the nonexistent `mushin.zen`); `capture_env`/`resume`/`load_metrics`/`workflow_overrides` docstrings match actual behavior; the Python support matrix reads 3.10–3.13 (3.14 is not yet a required CI leg); `study.md`/`concepts.md` list all seven task batteries; and the `correction=` option is documented in the compare/LLM/index guides.
- Documented the sweep-analysis API. `mushin.show`, `mushin.best`, `mushin.diff`,
  and `mushin.export.table` now have an API reference page (`reference/analysis`),
  and the exploration-to-paper guide covers `max_total_seconds`, `notes=`, and
  `tags=`. The workflows guide also gains a "Using mushin alongside a
  hyperparameter search" section documenting the Optuna/Ax/Nevergrad → mushin
  two-phase pattern (searcher finds configs; mushin runs the reproducible final
  grid), with no new dependency.
- Fixed several tests that did not test what they claimed: the LLM "clear winner flagged significant" test now uses seed-varying systems and asserts the comparison is actually flagged significant (its old deterministic systems made the comparison masked, so it asserted nothing about significance); the Holm "is monotone" test now pins the exact step-down values and the monotonicity property; `test_on_error_raise` asserts the specific `RuntimeError("boom")` rather than any exception; a conditional-pass tuning test now visibly `pytest.skip`s instead of silently returning; a workflow test no longer scatters `multirun/` into the repo root; and a config-group test cleans up its `ConfigStore` entry.
- New guide, "From exploration to a paper-ready sweep", documenting the convention
  for moving from fast throwaway exploration (`sample=`, `cache_dir=`, `dry_run`,
  `show`/`best`/`diff`) to a clean, complete, reproducible paper run — including why
  the paper grid must be run fresh (code-era mixing, winner's curse) rather than
  grown from the exploration directory.
- Repo hygiene & example fixes: the crashing MNIST examples (`compare_classifiers`, `study_mnist`) now name torchvision in their install hint; `batteries.py` skips (not fails, exit 0) batteries whose optional extra is missing and drops the false "Run one"; two tracked stray `mushin_seed*.json` files are removed and gitignored; the `omegaconf`/`typing-extensions` dependency floors are raised to versions that actually resolve; `mushin-mcp` on a core install gives an actionable "install [mcp]" message instead of a bare `ModuleNotFoundError`; and `make spell` now covers `examples` like CI.
- The publish workflow pins pypa/gh-action-pypi-publish to its release tag (a bare commit SHA cannot resolve the Docker-based action's container image, which broke the v0.9.0 publish).


## [0.9.0] - 2026-07-20

### Added

- Provenance records the accelerator identity (CUDA/cuDNN/device name), `seed_everything_per_rank` persists the effective seed to a per-rank JSON in the run dir, and a resume no longer overwrites the environment snapshot recorded for earlier cells.
- Study and evaluate_checkpoints now accept predict_fn/metrics/prob_metrics/correction and forward them to compare (custom-output models and custom batteries work through Study), and checkpoints load lazily one at a time instead of holding every method x seed model in RAM.
- Sweep axes got first-class support for nested (dotted) params and Hydra config groups (the dataset coordinate is the chosen option name), `range(...)` overrides expand to their values, and unsupported continuous/adaptive sweeps (`interval(...)`) raise a clear error instead of crashing or yielding an all-NaN grid.
- The MCP server gained `get_failures` (failed cells + tracebacks from the manifest) and `get_provenance` (git/package/accelerator records) tools, and now summarizes long coordinate axes and metric curves instead of flooding the agent's context with raw arrays.
- The wheel now ships a PEP 561 `py.typed` marker, so mypy/pyright/IDEs see mushin's type annotations.
- `MultiRunMetricsWorkflow.to_dataframe()` (and `experiment.workflow.to_dataframe()`) returns the sweep as a tidy pandas DataFrame in one call; README and quickstart show the pandas path alongside the dataset.
- `compare`/`compare_methods`/`compare_llms` accept `correction=` -- `holm` (default), `bonferroni`, `fdr_bh` (Benjamini-Hochberg), or `none`; the statistics guide documents the per-metric family scope.

### Changed

- `import mushin` is ~5x faster (~0.13s from ~0.61s): torch now loads on first use, not at import time.

### Fixed

- HydraDDP/HydraFSDP now track their child rank processes like Lightning's base launcher -- children are reaped if rank 0 dies (no more orphaned GPU processes), signal forwarding works, and per-rank thread pools are capped; checkpoint/experiment loading errors are real exceptions with actionable messages instead of bare asserts.
- HydraDDP/HydraFSDP rank startup is staggered by a deterministic, tunable delay (`MUSHIN_DDP_LAUNCH_DELAY`, default 1s) instead of a hard-coded random 1-5s per rank, and the re-launch command builds a Windows-safe `hydra.run.dir` override.
- Paired tests now report the paired effect size (Cohen's d_z) instead of the pooled-variance d; `mushin_failures` is stored as a JSON attr so failed-sweep datasets survive a netCDF round-trip; a custom `metrics=` battery no longer inherits the task's probability routing silently (pass `prob_metrics=` explicitly, unknown names now raise).
- Resilient sweeps got tougher: a corrupt (mid-write-killed) sweep manifest no longer aborts resume, atomic sidecar writes use unique temp names so concurrent writers cannot clobber each other, and `on_error="nan"` now writes the full traceback to `mushin_error.txt` in the failed cell's directory.
- Resume now verifies a config fingerprint before reusing a completed cell -- changing a non-swept value and resuming re-runs the affected cells (with a warning) instead of silently mixing two configurations into one dataset; resuming also no longer ships the full manifest to every worker.
- The MCP server now surfaces workflow-sweep metrics from `mushin_metrics.json` (previously only MetricsCallback `.pt` files were read), and a single corrupt per-run `config.yaml` no longer makes the whole experiment unqueryable.
- `parse_score` no longer misreads an incidental leading integer ("0 issues found...") as a 0.0 score -- a bare integer with trailing text now raises; the LLM guide documents cache-invalidation rules and when paired tests are valid.

### Misc

- CI now tests macOS (Apple Silicon + the Intel dependency branch), executes notebooks with the netCDF engine they recommend, and the publish workflow smoke-tests the built wheel and SHA-pins the OIDC publish action; the Python 3.14 classifier is withheld until its CI leg is required to pass.
- Docs: eval-extra callouts everywhere they were missing (examples index, notebooks 02/03/05, custom/segmentation guides, example docstrings), a "Versioning & scope" statement in the README, a tracker (W&B/TensorBoard) how-to in the workflows guide, `multirun`/`hydra_list` reference entries, and the decorator quickstart now points at the decorator example. `MultiRunMetricsWorkflow.run` is fully documented and `plot` no longer leaks figures.
- Notebook 07 demonstrates xarray group-by (split-apply-combine), notebook 05 finally shows its plot on the docs site, and a netCDF round-trip test pins string-coordinate datasets.


## [0.8.0] - 2026-07-20

### Changed

- The evaluation layer (`compare`, the metric batteries, LLM evaluation, and `Study`) now requires the optional `eval` extra: `pip install mushin-py[eval]`. This keeps the core sweep→dataset install lean — `torchmetrics` and `scipy` are no longer base dependencies. Accessing these features without the extra raises a clear install hint. The `detection`/`image`/`audio` battery extras now imply `eval`.

## [0.7.0] - 2026-07-18

### Added

- Multi-node DDP support: `submitit_slurm_config` (derives `tasks_per_node` from
  `gpus_per_node`) and `seed_everything_per_rank` helpers, a fail-fast check that the
  launched world size matches `num_nodes x devices`, `MetricsCallback` now writes only
  on global rank 0, and `_teardown` clears only mushin-set env vars (leaving
  scheduler-owned vars alone under SLURM/torchrun). See the new multi-node guide.
- `HydraFSDP`: a Fully-Sharded Data Parallel strategy that works under Hydra
  `--multirun`. Like `HydraDDP`, it reattaches ranks via the job's saved
  `config.yaml` instead of re-executing with `sys.argv` (which a sweep would run as
  the wrong job), so FSDP sharded training composes with Hydra sweeps. Exported from
  `mushin`; see the new "Sharded training under Hydra" guide.
- `pin_gpu_round_robin(num_gpus)`: an opt-in helper to pack several small sweep jobs
  onto each GPU. Called at the top of a Hydra task, it sets `CUDA_VISIBLE_DEVICES`
  to `job_index % num_gpus` so jobs round-robin across devices; run
  `num_gpus * jobs_per_gpu` jobs concurrently (via your launcher's `n_jobs`) to
  co-locate them. New "Packing small jobs onto GPUs" guide covers the joblib recipe,
  Ray fractional-GPU, and MPS/MIG.

### Fixed

- `HydraDDP`/`HydraFSDP` docs now show the working launcher-provided-ranks pattern; an imperative `@mushin.sweep` task with `strategy=HydraDDP(), devices=N` on the local launcher silently trains on one GPU. (#95)
- A bare `list`/`tuple` passed as a sweep argument (the common slip of forgetting `multirun`) now raises an actionable error — ``lr=mushin.multirun([...])`` to sweep, or `mushin.hydra_list([...])` to pass the list as a single value — instead of a generic type-list `TypeError`.

### Misc

- Added a runnable `examples/parallel_sweep.py` showing how to submit a sweep out-of-process (`run(..., launcher="joblib")`), wired into the workflows guide and examples index — the missing runnable counterpart to 0.6.0's out-of-process launcher support.
- Added an internal cluster-gated validation runbook (`docs/superpowers/cluster-validation-runbook.md`): a self-contained, per-PR checklist (Phase 1 single-node multi-GPU, Phase 2 multi-node/SLURM) that anyone with HPC access can run to validate HydraDDP, GPU-packing (#59), HydraFSDP (#58), multi-node DDP (#50), submitit dispatch (#86), and resume hard-kill/preemption durability (#83).
- Docs: README and the workflows guide now lead with the `@mushin.sweep` decorator; `mushin.sweep`, out-of-process launchers, and hard-kill-durable resume are covered in the README feature list + API reference; and a new "Analyzing your results" example notebook shows the common `xarray.Dataset` recipes (reduce over seeds, pick the best config, slice, tabulate, plot, save/reload).
- Docs: the workflows guide now explains the two axes of sweep parallelism — a Hydra `launcher=` (distributes the sweep across cells/nodes) vs. `HydraDDP` (multi-GPU training within one cell) — and how they compose (e.g. submitit + HydraDDP).


## [0.6.0] - 2026-07-15

### Added

- New `@mushin.sweep` decorator: turn a plain `task`-style function into a runnable sweep with no subclassing — `experiment.run(lr=multirun([...]), seed=multirun([...]))` returns the labeled `xarray.Dataset` directly. Drop to `experiment.workflow` (the last-run `MultiRunMetricsWorkflow`) or `experiment.workflow_cls` for power features; `mushin_resume` and all `run()` resilience options carry through. The `MultiRunMetricsWorkflow` subclass form is unchanged.
- Resumable sweeps now survive a hard process kill (OOM, SLURM preemption): each cell records its completion durably from inside its own job, so `resume=True` never recomputes finished cells. A task may also declare a `mushin_resume` parameter to receive a `ResumeContext` (the cell's directory, `is_resume`, and the last checkpoint) and resume its own training mid-run.
- Sweeps can now use out-of-process Hydra launchers (e.g. `hydra-joblib-launcher`, submitit): per-cell dispatch is stdlib-picklable, so `run(..., launcher="joblib")` parallelizes across worker processes. Previously any process-backed launcher failed with a `PicklingError`. Resilience, resume, and provenance semantics are unchanged in-process and preserved out-of-process.

### Changed

- Provenance capture no longer spawns git subprocesses per sweep cell. The sweep-constant part of the record (git state + package versions — three `git` subprocesses via `_git()`) is now captured **once** per `run()` and reused for every cell, instead of being recomputed per cell. An N-cell sweep now spawns 3 git subprocesses instead of 3N (e.g. ~30s less git overhead on a 1000-cell sweep). Each cell's `mushin_provenance.json` is byte-for-byte equivalent (only `timestamp`/`config` vary per cell, as before).
- `import mushin` is now ~65% faster (~1.7s → ~0.6s on a cold import): the Lightning integration (`HydraDDP`, `MetricsCallback`) and its heavy `pytorch_lightning` dependency (which also transitively pulled in matplotlib and scipy) now load lazily on first attribute access instead of at import time. The sweep → `xarray` core no longer pays for Lightning it never uses. All public names resolve unchanged; `_tuning`/`Study` still work (they import pytorch_lightning only inside functions).

### Fixed

- Sweeps no longer emit Hydra's "Future Hydra versions will no longer change working directory at job runtime" deprecation warning: `MultiRunMetricsWorkflow.run` now sets `hydra.job.chdir=True` explicitly (the behavior the workflow already relies on).

### Misc

- Added six runnable EQUINE-style example notebooks (sweeps, compare + batteries, Study, resilient sweeps, LLM evaluation, scikit-learn) under an "Example notebooks" section of the docs, executed in CI via nbmake so they stay current.
- CI: a batteries-clean-install job builds the wheel, installs it (non-editable) with the detection/image/audio extras into a fresh env, and runs all 7 battery examples against it — catching packaging / optional-extra issues the editable test job would miss.
- Documentation: a Built-in batteries guide covering all 7 benchmark batteries (classification, segmentation, detection, regression, retrieval, image_quality, audio) with real-model recipes and CI-tested runnable toys.
- Documentation: add a top-level Examples page indexing all runnable example scripts (sweep_to_dataset, sklearn_sweep, compare_classifiers, study_mnist, segmentation_demo, compare_llms_demo, batteries), each with a one-line description and a link, plus a pointer to the batteries guide.
- Documentation: add sweep resilience + provenance (0.5.0) to the README "What it provides" list.
- Documentation: the Built-in batteries guide now shows real captured outputs under every battery, plus a flagship notebook-style walkthrough (comparing two classifiers) with the actual summary table, significant p-values, and interpretation.


## [0.5.0] - 2026-07-14

### Added

- Sweep resilience and provenance: `run(on_error="nan")` records failed grid cells as
  NaN and keeps going (default stays `"raise"`); `run(working_dir=..., resume=True)`
  re-runs only the failed/missing cells and fills them in place; `compare`/`Study`
  refuse statistics on an incomplete sweep (`IncompleteSweepError`) until you resume;
  every run writes per-job provenance (`mushin_provenance.json`: git SHA, versions,
  config) with an opt-in `capture_env=True` full dependency snapshot. `Study` accepts
  `on_error`/`resume`/`capture_env` for training sweeps.


## [0.4.1] - 2026-07-13

### Added

- Added a runnable scikit-learn sweep example (`examples/sklearn_sweep.py`) demonstrating that the framework-agnostic `MultiRunMetricsWorkflow` wraps non-Lightning models (here `LogisticRegression`) and still returns a labeled `xarray.Dataset`. `scikit-learn` is a dev-only dependency for the example and its test; the package itself has no scikit-learn dependency.

### Misc

- Documentation: add a "Frameworks: Lightning-first, sweep layer agnostic" section to Core concepts, clarifying that the sweep->dataset workflow wraps any framework (scikit-learn, XGBoost, ...) while compare/auto-tuning/HydraDDP are PyTorch/Lightning-specific.
- Documentation: add a Changelog page to the docs site that embeds CHANGELOG.md (self-maintaining, no duplication).
- Documentation: add an API reference page for the auto-tuning helpers (`tune_batch_size`, `tune_learning_rate`), which had a usage guide but no auto-generated API docs.
- Documentation: announce the 0.4.0 release with a "What's new" highlights callout on the docs home page.
- Documentation: refresh the README for 0.4.0 — add auto-tuning and the task API/batteries to the feature list, note the framework-agnostic sweep layer (scikit-learn example), and fix the stale `viz` extra description that referenced the now-deprecated `RobustnessCurve`.


## [0.4.0] - 2026-07-13

### Added

- Four new built-in task batteries — `regression`, `image_quality`, `audio`, and
  `retrieval` — plus a per-`Task` `update_fn` hook for metrics whose update step is
  not `(preds, target)` (used by `retrieval`). LPIPS and STOI sit behind the
  optional `[image]` and `[audio]` extras. Each battery is exported from `mushin`.
- Overhauled the documentation: runnable, tested example scripts (MNIST) that the guides embed verbatim, deeper guides with annotated output, and new Tutorial, Core concepts, Custom metrics/predict_fn, and Statistics pages.
- Public task API: `Task` dataclass plus `register_task`, `get_task`, and
  `list_tasks` make evaluation tasks first-class and reusable. `compare(...)` and
  `Study(...)` now accept either a `Task` object or a registered task name, and the
  built-in batteries (`classification_battery`, `segmentation_battery`,
  `detection_battery`) are exported from `mushin`.
- `compare(task="detection")` — compare trained object detectors across seeds over
  the full `torchmetrics.detection` bounding-box family (mean-average-precision plus
  the IoU/GIoU/CIoU/DIoU variants), reporting every scalar metric with Holm-corrected
  significance. Needs the optional `mushin-py[detection]` extra.
- `mushin.llm.compare_llms` — compare LLM systems (callables or hydra-zen configs) across reproducible stochastic seeds with a metric (a plain scorer, a `torchmetrics` text metric, or the new `llm_judge` helper), reporting Holm-corrected statistical significance. Includes an on-disk output cache. Provider-agnostic: you bring the systems, data, and judge model.
- `tune_batch_size` / `tune_learning_rate`: opt-in, reproducibility-preserving
  auto-tuning. Lightning's batch/LR finder runs once, the result is pinned to a
  sidecar YAML, and later runs reuse it. `tune_batch_size` pins a hardware-
  independent effective batch, choosing the largest device batch that both fits and
  divides the per-device target exactly, so the effective batch is identical on any
  GPU count with no drift.

### Changed

- Modernized the codebase to Python 3.10+ idioms now that 3.9 is no longer
  supported: `ruff` `target-version` is `py310`, and the pyupgrade auto-fixes
  (`Optional[X]`/`Union[X, Y]` -> `X | None`/`X | Y`) plus explicit `zip(..., strict=True)`
  have been applied. No behavior change.
- `import mushin` is now lightweight: the `benchmark` and `llm` subsystems load on
  first use instead of at import time, so a bare import no longer pulls the
  battery/eval machinery. Every existing top-level name still resolves. The default
  Hydra config/job name is now `mushin_workflow` (was `rai_workflow`), and the new
  `mushin.original_cwd()` helper anchors relative paths in `task()` against the
  launch directory rather than Hydra's per-job output directory.

### Fixed

- Fixed the docs example scripts surfaced in review: the Study example now trains and evaluates on separate MNIST splits and resolves its checkpoint directory to an absolute path (so it works under Hydra's per-job chdir); the custom-metrics guide no longer implies `Study` accepts `metrics`/`predict_fn` (those are `compare`-only).
- Hardening from a repo-wide adversarial audit: `load_experiment` now loads DDP/nested-layout configs (was silently `None`) and labels each job's own `working_dir`; the benchmark `compare()` path masks zero within-group-variance comparisons (no more false-positive significance — now consistent with `compare_llms`, both via a single `compare_methods`); `MetricsCallback` keeps every metric series aligned to the epoch axis (NaN-padding missing metrics, reserving the `epoch` key); `Study` labels the seed coordinate with the real seed values and relocates checkpoints across filesystems (`shutil.move`); multirun overrides are built with Hydra's `choice(...)` syntax so comma-bearing and single-element values are no longer mis-split; and several legibility/robustness fixes (`to_dataset` empty-method error, `_to_device` namedtuple support, `'='`-bearing override parsing, `load_from_dir` cache reset, dropped the never-working dict-override path).

### Removed

- Dropped support for Python 3.9 (end-of-life October 2025); `mushin` now requires
  Python >= 3.10. This refreshes the dependency lockfile to patched versions of
  pillow, urllib3, aiohttp, filelock, requests, pytest, and pytorch-lightning,
  clearing the Dependabot security alerts anchored on the old Python-3.9 dependency
  branch. The `scipy` (>= 1.13) and `matplotlib` (>= 3.9) floors are raised to their
  first NumPy-2-compatible releases, and the `mcp` extra no longer needs a Python
  version gate.

### Deprecated

- `BaseWorkflow` and `RobustnessCurve` are deprecated at the top level and will be
  removed in a future release. Import them from `mushin.workflows` instead;
  accessing them as `mushin.BaseWorkflow` / `mushin.RobustnessCurve` now emits a
  `DeprecationWarning`.

### Misc

- Added a `@claude` mention bot (GitHub Actions): mention `@claude` in any issue or PR comment to have Claude answer questions or make changes. PR *reviews* stay with the Codex connector.


## [0.3.0] - 2026-06-24

### Added

- Optional read-only MCP server (`mushin-mcp`, `pip install "mushin-py[mcp]"`, Python >= 3.10) that lets Claude Code and other MCP clients list experiments, summarize swept parameters and metrics, read configs, and inspect saved datasets — with no training or sweep launching. (#32)
- Added a documentation website (MkDocs Material) with how-to guides and an auto-generated API reference, deployed to GitHub Pages.
- The test suite now runs on Windows in CI (windows-latest) across Python 3.9-3.14, alongside Linux.

### Changed

- Raised minimum dependency floors to a tested minimum and added a `min-versions` CI job that runs the suite against the lowest declared versions, so the floors stay honest. The old floors did not actually work: several were incompatible with NumPy 2 (the non-Intel floor), so `torch` (>= 2.4), `pandas` (>= 2.2.2), `xarray` (>= 2024.6), and `netCDF4` (>= 1.7.0) were raised to their first NumPy-2-compatible releases. `pytorch-lightning` was raised to >= 2.4 (the pre-2.0 1.5 API does not run this package) and `hydra-zen` to >= 0.10 (for `multirun`). Intel-macOS keeps torch 2.2.x / NumPy 1.x.


## [0.2.1] - 2026-06-23

### Added

- Adopted towncrier news fragments with a CI gate — each PR now adds a `changes/` fragment, assembled into the changelog at release.

### Fixed

- Point PyPI project links at the mushin repo (Homepage/Repository/Issues/Changelog) instead of only the upstream toolbox.
- Synced `uv.lock`'s recorded project version to 0.2.0.

## [0.2.0] - 2026-06-23

### Added
- `Study` — orchestrate a multi-seed training sweep (via Hydra/Lightning) and
  route the trained models into `compare` in one call;
  `Study.from_checkpoints(...)` for eval-only comparison of existing checkpoints.
- Segmentation support in `compare` (`task="segmentation"`): mean IoU, Dice,
  pixel accuracy, and macro precision/recall, with `ignore_index` for void
  labels; plumbed through `Study`.
- Streaming evaluation — metrics update per batch (O(C²) memory for the
  segmentation battery); one unified eval loop for all tasks.
- A task registry so new task types self-describe their battery and predict step.
- `compare` gains a `prob_metrics` override and rejects one-shot `data` iterators.

### Changed
- Package author/maintainer set to Josue Martinez-Martinez (previously the
  original MIT-LL authors; their attribution remains in LICENSE and the README).
- README: documented `compare` and `Study`; clarified install
  (`pip install mushin-py` → `import mushin`); absolute logo URLs so images
  render on PyPI; transparent dark-mode logo.

### Fixed
- Significance: `compare` warns when the chosen test cannot reach `alpha` at the
  given seed count (e.g. Wilcoxon with ≤5 seeds).
- `cohens_d` returns signed infinity (not `0.0`) when within-group variance is
  zero but the means differ.
- Several review-driven fixes: Hydra-scalar method names in `Study` sweeps,
  NaN-safe Holm correction, ragged-results validation, and an empty-checkpoints
  guard.

## [0.1.0] - 2026-06-22

First release of `mushin` as a standalone package — a fork of the
`rai_toolbox.mushin` workflow layer from MIT Lincoln Laboratory's
(unmaintained) responsible-ai-toolbox.

### Added
- Standalone packaging: `pyproject.toml`, MIT license (original MIT-LL copyright
  retained), README, and a vendored `value_check` so the package no longer
  depends on `rai_toolbox`.
- uv-based development workflow with a committed `uv.lock`.
- Ruff (lint + format) and codespell, with pre-commit hooks.
- `Makefile` developer shortcuts (`make check`, `make test-py`, ...).
- GitHub Actions CI across Python 3.9–3.14.
- Community health files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue forms,
  PR template, Dependabot, CODEOWNERS.

### Changed
- Modernized type hints (PEP 585 generics, `collections.abc` imports).
- Declared support for Python 3.9–3.14; NumPy >= 2 except on Intel-macOS, which
  is capped at NumPy 1.x / torch 2.2.x (no newer wheels on that platform).

### Fixed
- Compatibility with PyTorch 2.6+, which changed `torch.load`'s default
  `weights_only` to `True`: pass `weights_only=False` when loading trusted,
  self-produced metric and checkpoint files.
- `test_overrides_roundtrip`: exclude Hydra-reserved tokens (`null`/`none`/
  `nan`/`inf`) from the generated-string strategy.
- Updated deprecated `xarray.Dataset.dims` to `.sizes` in tests.

[Unreleased]: https://github.com/martinez-hub/mushin/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/martinez-hub/mushin/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/martinez-hub/mushin/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/martinez-hub/mushin/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/martinez-hub/mushin/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/martinez-hub/mushin/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/martinez-hub/mushin/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/martinez-hub/mushin/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/martinez-hub/mushin/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/martinez-hub/mushin/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/martinez-hub/mushin/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/martinez-hub/mushin/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/martinez-hub/mushin/releases/tag/v0.1.0
