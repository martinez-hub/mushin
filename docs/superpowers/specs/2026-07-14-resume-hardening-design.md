# Design: Resume hardening — survive process kills & resume mid-training

**Date:** 2026-07-14
**Status:** Approved (brainstorming) — pending implementation plan
**Component:** `MultiRunMetricsWorkflow` sweep resilience (`src/mushin/workflows.py`, `src/mushin/_sweep_io.py`)

## Problem

mushin's sweep resilience today is **cell-level and in-process**: each cell writes a
metrics sidecar when its `task` returns; the sweep manifest (per-combo status) is
written **once at the end** in `jobs_post_process`; `resume=True` skips cells the
manifest marks `completed` and re-runs the rest **from scratch**. Two real failure
modes are not covered:

1. **Out-of-process kill.** A hard kill — OOM-killer, SLURM preemption/time-limit,
   node death, SIGKILL — takes down the whole sweep process. Because the manifest
   is only written at the end, a mid-sweep kill loses it; on resume, cells that
   *did* finish get recomputed (their metrics sidecars survive on disk, but resume
   keys off manifest status, not sidecar presence).
2. **Intra-cell resume.** A single long-running cell killed mid-training (e.g. a
   model at epoch 50/100) restarts from epoch 0 on resume, discarding progress.
   mushin gives the task no stable directory to keep a checkpoint in, no signal
   that it is resuming, and no "partial/running" state distinct from "failed".

## Goals

- A resumed sweep **never recomputes a cell that already completed**, even after a
  hard process kill (durable, incremental progress).
- A re-executed cell can **resume its own training from a checkpoint** — mushin
  guarantees a stable per-cell directory and hands the task a resume signal; the
  task/Lightning owns the checkpoint format.
- Fully **backward compatible**: existing tasks (which declare only their swept
  params) run exactly as today.

## Non-goals

- mushin does **not** own the training loop, checkpoint format, or checkpoint
  cadence ("enable, don't own"). It does not call `save_state`/`load_state` hooks.
- Not Lightning-specific — the mechanism is framework-agnostic (plain-torch,
  Lightning, sklearn tasks all work).
- Not a distributed/multi-node coordination layer. Each cell is one job; mushin
  coordinates *cells*, not ranks within a cell.

## Decisions (from brainstorming)

- **Scope:** both gaps — durable-across-kill *and* intra-cell checkpoint resume.
- **Contract:** *enable, don't own* — mushin provides a stable per-combo dir + a
  resume signal; the task loads its own checkpoint (e.g. `ckpt_path="last"`).
- **Signal delivery:** an **introspected optional `mushin_resume` kwarg** — injected
  only if the task's signature declares it; tasks without it are called unchanged.
- **Directory layout:** **combo-named directories** (user-visible, a strict
  improvement over positional numeric dirs).
- **`last_ckpt`:** **best-effort** discovery (newest `*.ckpt` / `last.ckpt` in the
  cell dir), with a documented convention.

## Architecture (5 components)

### 1. Stable per-combo directories

Replace Hydra's positional multirun subdir (`working_dir/0,1,2…`, from the default
`hydra.sweep.subdir=${hydra.job.num}`) with a combo-keyed one:

```
hydra.sweep.subdir=${hydra.job.override_dirname}
```

so cell `(method=cnn, seed=3)` always lands in
`working_dir/method=cnn,seed=3/` on every run. `override_dirname` is derived from the
job's task overrides — which, for a mushin sweep, are exactly the swept params —
so it is deterministic and stable across runs.

- This override is appended in `run()` alongside `hydra.sweep.dir`, **only when
  `working_dir` is set** (a stable location is required for any of this to work;
  without `working_dir` behavior is unchanged). A caller who supplies their own
  `hydra.sweep.subdir` override wins (same guard pattern as the `hydra.job.chdir`
  fix).
- `to_xarray()` / `jobs_post_process` discover per-cell dirs from the returned
  `jobs` objects, so the dir *naming* does not affect dataset assembly.
- **Alignment note:** the plan must confirm `override_dirname`'s string form maps
  1:1 onto `combo_key(combo)` (the existing key used by `Manifest`), or add a small
  translation so the two agree. Hydra offers `hydra.job.config.override_dirname`
  knobs (kv-sep, item-sep, exclude keys) if sanitization is needed.

### 2. Durable per-cell status sidecar

Introduce a per-cell status file, `mushin_cell_status.json`, written **from inside
the job** into that cell's stable dir. Transitions:

- **`running`** — written by the pre-task wrapper *before* `task` executes (records
  the combo, the stable dir, and an incremented `attempt`).
- **`completed`** — written after `task` returns and its metrics sidecar is
  persisted.
- **`failed`** — written when the task raises under `on_error="nan"` (carries the
  exception string).

Because each cell writes only its own file, independently and via atomic
write (temp + rename), a process-wide kill cannot lose the progress of cells that
already finished, and there is no shared-file write contention under parallel
launchers. The aggregate `Manifest` becomes a **read view**: `Manifest.load`
(and `is_complete` / resume) reconstruct status by scanning the per-cell status
sidecars under `working_dir/*/`. The end-of-sweep manifest write is retained as a
convenience snapshot but is no longer the source of truth for completion.

This fixes **Gap 1**: a hard kill leaves durable per-cell `completed` markers, so
resume skips them.

### 3. `ResumeContext` + introspected injection

A small immutable dataclass (new, in `_sweep_io.py` or a new `_resume.py`):

```python
@dataclass(frozen=True)
class ResumeContext:
    dir: Path            # this cell's stable working directory (exists)
    is_resume: bool      # True if a prior attempt left artifacts in `dir`
    last_ckpt: Path | None  # newest *.ckpt / last.ckpt in `dir`, best-effort
    attempt: int         # 1 on first run, incremented each re-execution
```

Injection happens in mushin's own task wrapper layer (around
`_instrument_task(task_fn_wrapper(self.task))`), **not** through hydra-zen:

- The wrapper inspects the *underlying* task's signature (`inspect.signature`). If it
  declares a parameter named `mushin_resume`, the wrapper computes a `ResumeContext`
  for the current cell (from its stable dir + prior status sidecar + checkpoint
  scan) and passes it. Otherwise the task is called exactly as today.
- **Critical integration constraint:** `mushin_resume` must be invisible to
  hydra-zen's `zen` config population — it is not a swept param and has no config
  entry. The wrapper must present zen a call surface that excludes `mushin_resume`
  (e.g. bind swept params from config, then inject `mushin_resume` when invoking the
  real task), so `zen` never tries to resolve it. The plan must include a test that
  a task declaring `mushin_resume` does not cause a Hydra/zen config error.

This fixes **Gap 2**: a re-executed cell gets a stable dir and a resume signal.

### 4. Resume semantics

`resume=True` (which still requires `working_dir`) processes each requested cell:

- status **`completed`** → short-circuit, returning the cell's cached metrics from
  its stable dir (as today, via `_resume_short_circuit` / `read_metrics_sidecar`).
- status **`running` / `failed` / absent** → re-execute, injecting a `ResumeContext`
  with `is_resume = (prior artifacts exist in dir)` and `last_ckpt = newest ckpt or
  None`, and `attempt = prior_attempt + 1`.

`is_resume` is true whenever the stable dir already holds a prior `running`/`failed`
status sidecar or any checkpoint — so it also lights up on a *fresh* (non-`resume`)
re-run into an existing `working_dir`, which is the natural HPC requeue path (same
command, same dir).

### 5. `last_ckpt` discovery (best-effort convention)

`last_ckpt` = the most-recently-modified file matching `*.ckpt` directly in the cell
dir, preferring an exact `last.ckpt` if present. Documented convention: **a task
that wants intra-cell resume should write its checkpoint into `mushin_resume.dir`**
(Lightning: set `default_root_dir=mushin_resume.dir` and use
`ModelCheckpoint(save_last=True)` → `last.ckpt`). If no checkpoint is found,
`last_ckpt is None` and the task starts fresh — never an error.

## On-disk layout (after)

```
working_dir/
  method=cnn,seed=0/
    mushin_cell_status.json     # {"status": "completed", "attempt": 1, "dir": "...", ...}
    mushin_metrics.json         # cached metrics (existing)
    mushin_provenance.json      # existing
    last.ckpt                   # written by the task, if it checkpoints
    .hydra/ …                   # existing hydra job output
  method=mlp,seed=3/
    mushin_cell_status.json     # {"status": "running", ...}  ← killed mid-training
    last.ckpt                   # epoch-50 checkpoint the task left behind
  mushin_sweep_manifest.json    # end-of-sweep snapshot (read view; not source of truth)
```

## Backward compatibility

- Tasks that don't declare `mushin_resume` are unaffected (introspection gate).
- Sweeps without `working_dir` are unaffected (stable dirs / status sidecars are
  only engaged with a stable location).
- The `resume=True` public API is unchanged; it just becomes robust to kills.
- **Directory naming changes** for any `working_dir` sweep (numeric → combo-named).
  This is the one visible behavior change; it is a strict improvement (stable,
  human-readable) but any downstream code that hard-codes `working_dir/0` paths
  would need updating. Called out in the changelog as a behavior change.

## Testing strategy

Every failure mode is unit-testable **without a cluster**:

- **Durable-across-kill:** run a sweep in a subprocess, `SIGKILL` it after N cells
  complete (detect via status sidecars appearing), then re-run with `resume=True`
  in-process; assert the N completed cells are *not* recomputed (call counter) and
  the grid completes.
- **Intra-cell resume signal:** a task that records the `ResumeContext` it received;
  first run fails at a "checkpoint" it wrote to `mushin_resume.dir`; resume asserts
  `is_resume is True`, `last_ckpt` points at that file, `attempt == 2`.
- **Introspection gate:** a task with `mushin_resume` runs without a zen/Hydra
  config error; a task without it is called unchanged.
- **Stable dirs:** two runs of the same grid write to identical combo-named dirs;
  `to_xarray()` assembles correctly regardless of naming.
- **No-op guards:** no `working_dir` → no status sidecars, numeric-free behavior
  unchanged; caller-supplied `hydra.sweep.subdir` wins.

## Validation caveat

Unit tests simulate kills faithfully, but **true SLURM preemption/requeue can only
be validated on real HPC hardware**. This feature therefore lands CI-green and
unit-verified, then joins the **cluster-gated** set (with #50/#58/#59) for hardware
sign-off before being called production-ready. Do not merge to a release as
"HPC-validated" without that sign-off.

## Risks / open questions

- **`override_dirname` ↔ `combo_key` alignment.** The single most important
  implementation detail; the plan must verify the mapping empirically and add a
  translation if the string forms differ (separators, escaping, long names).
- **`override_dirname` length / illegal chars** for large grids or string values
  with commas/slashes. Mitigation: Hydra's override_dirname sanitization + the
  `exclude_keys` knob; if still unsafe, fall back to a hashed combo slug (keep it
  deterministic).
- **Parallel launchers.** Per-cell sidecars avoid shared-file contention, but the
  aggregate manifest snapshot and `to_xarray` assembly must tolerate a partially
  populated tree (some cells `running`/absent). MVP targets serial + submitit
  (one task per job); joblib parallelism should work via the same per-cell files
  but is a secondary test target.
- **`is_resume` on an unrelated dir collision.** If a user reuses a `working_dir`
  across *different* grids, a combo dir could carry stale artifacts. Mitigation:
  the status sidecar records the combo; a mismatch (or a `mushin_provenance` config
  mismatch) means "not a valid resume of this cell" → treat as fresh. The plan
  decides how strict to be.

## Rollout

Additive and backward-compatible except the directory-naming change. Ships as one
feature branch: `_sweep_io.py` (ResumeContext, per-cell status sidecar read/write,
Manifest-as-read-view), `workflows.py` (stable-subdir override, wrapper injection,
resume-semantics update), tests, docs (resilience guide + a note in notebook 04),
changelog. No dependency changes. Version bump: minor (new public `ResumeContext`
surface + behavior change), decided at release time.
