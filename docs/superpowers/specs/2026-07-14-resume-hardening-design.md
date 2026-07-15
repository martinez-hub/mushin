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
- **Directory layout:** **keep Hydra's positional numeric dirs** (`working_dir/0,1,2…`)
  — no user-visible naming change. Correctness under grid changes is preserved by
  a combo-match guard (below), not by dir naming.
- **`last_ckpt`:** **best-effort** discovery (newest `*.ckpt` / `last.ckpt` in the
  cell dir), with a documented convention.

## Architecture (5 components)

### 1. Directories: keep Hydra's positional numeric dirs (no naming change)

Cells keep landing in `working_dir/0,1,2…` (Hydra's default
`hydra.sweep.subdir=${hydra.job.num}`). No new subdir override is added. Why this is
sufficient:

- **Completion-skip (Gap 1) is dir-name-independent.** The durable status sidecar
  (component 2) records each cell's *combo*; `Manifest.from_cell_status` keys cells
  by `combo_key(combo)` and stores whatever dir the cell ran in. So resume maps
  combo→recorded-dir correctly no matter how dirs are named or numbered.
- **Intra-cell checkpoint reuse (Gap 2) relies on stable launch order.** For a
  resume of the *same* sweep (identical multirun spec — the SLURM requeue /
  re-run-same-command case), Hydra expands the grid deterministically, so each
  combo gets the same `job.num` → the same numeric dir → its prior checkpoint is
  right there. This is the dominant resume scenario.
- **Grid-change safety via the combo-match guard.** If the user *narrows or reorders*
  the grid on resume, numeric dir `0` can be reused by a different combo. To avoid a
  cell loading another cell's checkpoint, `build_resume_context` only treats a dir
  as resumable when the **prior status sidecar's `combo` equals the current cell's
  combo**; on a mismatch it reports `is_resume=False, last_ckpt=None` and the cell
  starts fresh. Safe in all cases (worst case: a checkpoint isn't reused, never a
  wrong one loaded).

This keeps the on-disk layout exactly as today — no backward-incompatible change.

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
  for the current cell (from its dir + prior status sidecar + checkpoint scan,
  **gated by the combo-match guard** — a prior sidecar whose recorded combo differs
  from the current cell's is ignored, so a reused numeric dir never yields another
  cell's checkpoint) and passes it. Otherwise the task is called exactly as today.
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

Numeric dirs are unchanged from today; only the `mushin_cell_status.json` sidecar
(and any checkpoint the task writes) are new:

```
working_dir/
  0/
    mushin_cell_status.json     # {"status": "completed", "attempt": 1, "combo": {...}}
    mushin_metrics.json         # cached metrics (existing)
    mushin_provenance.json      # existing
    last.ckpt                   # written by the task, if it checkpoints
    .hydra/ …                   # existing hydra job output
  1/
    mushin_cell_status.json     # {"status": "running", "combo": {...}}  ← killed mid-training
    last.ckpt                   # epoch-50 checkpoint the task left behind
  mushin_sweep_manifest.json    # end-of-sweep snapshot (read view; not source of truth)
```

## Backward compatibility

- Tasks that don't declare `mushin_resume` are unaffected (introspection gate).
- Sweeps without `working_dir` are unaffected (stable dirs / status sidecars are
  only engaged with a stable location).
- The `resume=True` public API is unchanged; it just becomes robust to kills.
- **No directory-naming change** — dirs stay positional/numeric (`working_dir/0,1,2…`),
  so downstream code that references those paths is unaffected. The only new
  on-disk artifact is the per-cell `mushin_cell_status.json` sidecar.
- **Resuming a pre-upgrade sweep still works.** A sweep dir created by the shipped
  mushin has the end-of-run `mushin_sweep_manifest.json` but no per-cell status
  sidecars. `Manifest.from_cell_status` therefore *seeds from the legacy manifest*
  and overlays any per-cell sidecars (sidecars authoritative), so upgrading and
  then `resume=True` on an old dir still skips its completed cells rather than
  recomputing the whole sweep. Covered by a dedicated backward-compat test.

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
- **Combo-match guard:** a cell whose stable dir holds a prior status sidecar for a
  *different* combo gets `is_resume=False, last_ckpt=None` (never loads the wrong
  checkpoint).
- **Same-grid checkpoint reuse:** a resume of the same grid re-executes a failed
  cell in the same numeric dir and its `ResumeContext.last_ckpt` points at the
  checkpoint the prior attempt wrote.

## Validation caveat

Unit tests simulate kills faithfully, but **true SLURM preemption/requeue can only
be validated on real HPC hardware**. This feature therefore lands CI-green and
unit-verified, then joins the **cluster-gated** set (with #50/#58/#59) for hardware
sign-off before being called production-ready. Do not merge to a release as
"HPC-validated" without that sign-off.

## Risks / open questions

- **Numeric-dir reuse across grid changes** (the reason for the combo-match guard).
  With positional dirs, narrowing/reordering the grid on resume can point a cell at
  a dir a *different* cell used. The guard (component 3: `build_resume_context`
  ignores a prior sidecar whose recorded combo ≠ the current combo) makes this safe
  — a cell never loads another cell's checkpoint; worst case it starts fresh. This
  is now an essential correctness invariant, and must have a dedicated test.
- **`zen` must not see `mushin_resume`.** The signature-strip wrapper is the load-
  bearing seam; a test must assert a task declaring `mushin_resume` runs without a
  Hydra/zen config error.
- **Parallel launchers.** Per-cell sidecars avoid shared-file contention, but the
  aggregate manifest snapshot and `to_xarray` assembly must tolerate a partially
  populated tree (some cells `running`/absent). MVP targets serial + submitit
  (one task per job); joblib parallelism should work via the same per-cell files
  but is a secondary test target.

## Rollout

Additive and fully backward-compatible (no directory-naming change). Ships as one
feature branch: new `_resume.py` (ResumeContext, per-cell status sidecar read/write,
checkpoint discovery, contextvar), `_sweep_io.py` (Manifest-as-read-view via
`from_cell_status`), `workflows.py` (durable status writes, combo-match-guarded
ResumeContext injection, resume-semantics update), tests, docs (resilience guide +
a note in notebook 04), changelog. No dependency changes. Version bump: minor (new
public `ResumeContext` surface), decided at release time.
