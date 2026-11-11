# SLURM job lifecycle (#345)

## Problem

`run(launcher="submitit_slurm")` hands every cell to hydra-submitit-launcher,
whose `launch()` ends with `[j.results()[0] for j in jobs]`. The driver blocks
until every job ends, and:

- the first job that ends without a result (time limit, out-of-memory kill,
  failed node, exit outside Python) raises `UncompletedJobError` from that
  comprehension, even under `on_error="nan"`. The later jobs keep running,
  unobserved; no manifest is written; the dead cell's status stays `running`;
- a resume or `sample=` still submits every cell, because each job decides
  reuse for itself;
- nothing records job ids or logs, and the driver must stay alive for the
  whole sweep, which is why users wrap mushin in their own `sbatch` script.

## Split

1. **Dead jobs (this PR).** mushin's own submitit launchers; a job that ends
   without a result is a failed cell.
2. **Keep a result a dead job already wrote.** A job that dies after its cell
   completed (a DDP teardown hang killed at the time limit) keeps the cell's
   result: the runner records the job id in the status sidecar, and a dead
   job whose cell says `completed` with its own id is a completed cell.
   Until then the manifest says failed and the sidecar says completed for
   such a cell; a later `resume=True` reuses it.
3. **Skip reusable cells before submission** (resume, `cache_dir`, `sample=`)
   in mushin's launcher, keeping Hydra's job numbers.
4. **Submit and return** (`wait=False`) with a submission record (cell to job
   id and log, `on_error`, notes and tags, the budget id) and a collect step;
   `show` lists queued cells.
5. **A guard against two drivers on one sweep, and `mushin.cancel(dir)`.**

#348 (`cells_per_job`) builds on the same launcher.

## Decisions (PR 1)

- **D1. Ship launcher subclasses in `src/hydra_plugins/mushin_submitit/`**, in
  the mushin-py wheel. Hydra instantiates only classes its scanner found under
  `hydra_plugins`. Rejected: patching the plugin's class or `Job.results`
  during `run()` (a process-wide patch of a third-party class), and catching
  the error in `run()` (raised at the first dead job, with the handles of the
  jobs still running lost).
- **D2. The module imports nothing heavy and never fails to import.** Hydra
  imports every `hydra_plugins` module in every Hydra application and reports
  an `ImportError` as a warning (an error under `-W error`). Without the
  plugin it defines nothing; the core-only CI job checks that the scan is
  silent.
- **D3. `run()` routes the plugin's names to mushin's**
  (`mushin_submitit_slurm`, `mushin_submitit_local`), for both `launcher=` and
  a raw `hydra/launcher=` override, only when mushin's launchers are importable;
  otherwise Hydra reports the name the user wrote, with the install hint. The
  config is the plugin's dataclass with only `_target_` changed, so
  `launcher_config` keys are unchanged. Rejected: overwriting the plugin's own
  `submitit_slurm` node, since which store wins depends on scan order. The
  launcher choice is not part of any hash a resume compares, so a sweep the
  plugin's launcher ran resumes here.
- **D4. `launch()` is `_submit()` then `_wait()`.** `_submit` is the plugin
  1.2.0 body up to `map_array`, copied; a test pins a digest of the plugin's
  `launch` and compares the copy line by line. `_wait` waits on every job in
  order; `UncompletedJobError` and `FailedJobError` become a COMPLETED
  `JobReturn` carrying `_DeadJob` (FAILED would make Hydra's sweeper re-raise
  whatever `on_error` says), composed in the driver as the job would have
  been, with the global `HydraConfig` restored. Only a task function marked
  `_mushin_collects_dead_jobs` (mushin's runner) gets this; any other sees the
  plugin's error. The launcher holds no new state: it is pickled into every
  job and into `checkpoint()`.
- **D5. `jobs_post_process` records each dead cell before its main loop**,
  whatever `on_error` is (that loop raises at the first failure under
  `"raise"`). A cell whose status is `running`, or that has none (the job died
  before the runner, so the dir may not exist), becomes `failed`, loses any
  metrics file a prior attempt left, and gets a `mushin_error.txt` with
  submitit's message and its log tail. A terminal status is the runner's own
  record and is left alone (see split item 2). The manifest and the failures
  list keep one line: the job id, its state and its log path.
- **D6. `on_error="raise"` now raises after every job has ended**, as it
  already did for a task that raised.

## Needs a cluster

Runbook T9: what `results()` raises on a time limit, an out-of-memory kill
and `scancel`; requeue through `checkpoint()` with the subclass; whether
`hydra.job.id` matches the job id submitit reports for an array task
(`<array>_<task>`), which item 2 relies on.
