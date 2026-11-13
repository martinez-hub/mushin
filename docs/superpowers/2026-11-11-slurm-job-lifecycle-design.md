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
   such a cell; a later `resume=True` reuses it (checked with the basic
   launcher: the task did not rerun and the manifest went back to
   completed). Under `"nan"`, then, a job that finished can come back NaN
   where before PR 1 the sweep aborted. Cases:
   - the result arrives after submitit's 15 to 30 s wait (a slow shared
     filesystem), so `results()` raises for a job SLURM calls COMPLETED;
   - the driver cannot read the result;
   - in a job of several tasks, a rank other than 0 dies after rank 0 has
     returned: submitit's `results()` raises the first failed task's error,
     so the plugin's `results()[0]` drops rank 0's result (`job.task(0)`
     would keep it).
   The same test tells a terminal status an earlier sweep or attempt left
   from this job's own, which PR 1 also leaves alone: the job died before the
   runner, so a reused dir keeps its old status and `mushin_error.txt` (on a
   resume, a failed cell rerun whose job dies before the runner keeps the
   previous attempt's error file). A stale `running` record or `.hydra/`
   snapshot from an earlier sweep in a reused dir is likewise kept, with its
   attempt and hashes; PR 1 records only this sweep's combo over it.
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
  order; a job that ended without a result becomes a `JobReturn` composed in
  the driver as the job would have been (a test pins the part of `run_job`
  this mirrors), with the global `HydraConfig` restored, and with the status
  and value the runner gives it (D6, D6b). Only a task function marked
  `_mushin_collects_dead_jobs` (mushin's runner) gets this; any other sees the
  plugin's error. The launcher holds no new state: it is pickled into every
  job and into `checkpoint()`.
- **D5. The launcher records each dead cell as it collects it**, through the
  runner's `record_dead_cell`, whatever `on_error` is. A cell whose status is
  `running`, or that has none (the job died before the runner), becomes
  `failed` and loses any metrics file a prior attempt left. A cell whose job
  died before Hydra made its dir also gets the `.hydra/` snapshot `run_job`
  would have written, so a reload finds it as failed rather than never run.
  As for a task that raised, `mushin_error.txt` (submitit's message, with the
  job's whole error stream; past 22,000 characters mushin keeps the first
  2,000 and the last 20,000) is written only under `"nan"`; under `"raise"`
  the raised error, bounded the same way, carries it. A terminal status is
  left alone (split item 2). The manifest and the failures list keep one
  line shaped like a repr: the job id, its state, its log path (for a job of
  several tasks, their folder) and what ended it: "gave no result", or, for
  a `FailedJobError` (the job pickled an error instead of a result: it could
  not import the task, or could not save its result), "raised an error
  instead of returning". The line does not name that error: reading it from
  the traceback's text named the wrong exception for some tracebacks (a
  lowercase `struct.error`, a note after the error line, an exception
  group), and the traceback is in `mushin_error.txt` under `"nan"` and in
  the raised error under `"raise"`. The bounded copy has no traceback, since
  the original's frames hold the whole message.
- **D6. A dead job under `on_error="raise"` is a FAILED return**, as
  `run_job` returns a task that raised, so the sweeper raises it after that
  job's batch (with the default sweeper, after every job). Under `"nan"` it
  is a COMPLETED return carrying `_DeadJob`. Every job of a batch is waited
  on either way.
- **D6b. A dead job for a cell `sample=` left out is skipped**, as the
  runner would have skipped it; `record_dead_cell` returns the status and
  value for every case, so the launcher holds no mushin semantics.
- **D7. An error that cannot become a failed cell** (a job submitit cannot
  say has ended, a cell the driver cannot compose) is raised once every job
  of the batch has been waited on. Any other error from `results()` for a job
  that has ended (an unreadable result) is a dead cell.
- **D8. Routing covers `launcher=` and a raw `hydra/launcher=` override.** A
  launcher chosen in a config's defaults list runs the plugin's own launcher:
  reading the composed `hydra.launcher._target_` would cost a compose on
  every sweep. The digest pin hashes the code lines of the plugin's
  `launch()` (comments and blank lines dropped; `ast.dump` differs between
  Python versions).

## Decisions (PR 2)

- **E1. A record is the job's own when it carries this run's `run_id`.** The
  runner gets a `uuid4` per `run()`, pickled into every job, and every status
  write records it with Hydra's `job.id`. Not the job id alone: a local job's
  id is a process id, which can recur in a reused dir, and whether SLURM's
  array id reads the same in the job and the driver is runbook T9.
- **E2. A dead job keeps its own terminal record.** Completed: the driver
  returns the cell's metrics file (as a resume returns it; None when the task
  returned None), under either `on_error`. Skipped: skipped. Failed: the
  runner's record and error file stay, and the cell fails as before.
- **E3. Otherwise the dead job's end replaces the record**, with this job's
  `.hydra/` snapshot (unless the record is this run's, whose snapshot Hydra
  wrote), a new attempt count (one more than a record of the same combo, as
  `build_resume_context` counts), no fingerprints from another run, and no
  error file from an earlier attempt.
- **E4. Except a completed result of the same cell** that this run's config
  and task code do not contradict (the same combo; a fingerprint the record
  lacks cannot differ): a resume reuses it, so a job that died before its
  runner leaves it, and the cell fails for this run only, as in PR 1.
- **E5. One read of the record, no retry.** On a slow filesystem the driver
  can still see an older record of a cell that completed and record the cell
  failed over it, as PR 1 did; T9 checks how often.

## Needs a cluster

A probe on `submitit_local` (a cell sleeping past `timeout_min=1`) shows a
time limit taking the runner's path, not this one: submitit's handler
catches the timeout signal, tries `checkpoint()` (which pickles mushin's
launcher fine) and, once out of retries, raises `UncompletedJobError("Job
not requeued because: timed-out too many times.")` inside the task, so the
cell fails with a traceback, as it already did before this PR. The dead-job
path is for kills Python cannot act on: out of memory, a failed node,
`scancel`, or a time limit's final kill while the task sits in C code.

Runbook T9: what `results()` raises on an out-of-memory kill, a failed node
and `scancel`, and on a time limit hit while the task is in C code; requeue
through `checkpoint()` with the subclass; a two-task job whose rank 1 dies
after rank 0 returned, and any COMPLETED job that comes back as "gave no
result" (item 2's cases); whether
`hydra.job.id` matches the job id submitit reports for an array task
(`<array>_<task>`), which item 2 relies on.
