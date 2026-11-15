# Running a sweep on SLURM

mushin can submit a sweep's cells to a SLURM cluster itself: no `sbatch` script
to write. Each cell runs as one SLURM job (a task of a job array, when several
are submitted together), or, for a sweep of many short cells, each job runs
several cells in turn ([Many small cells](#many-small-cells)). Every cell runs
in its own directory under the sweep's `working_dir`, and `run()` gathers the results into
the same `xarray.Dataset` a local sweep gives, or, with `wait=False`, returns
at once and leaves that to `mushin.collect`. This guide covers ordinary
sweeps, where each cell runs in one process. For a cell trained with DDP across
several GPUs or nodes, see the [multi-node guide](multinode.md).

## Install

```bash
pip install "mushin-py[slurm]"
```

The `slurm` extra adds hydra-submitit-launcher, which mushin drives through
`launcher="submitit_slurm"`.

## Describe each cell's job

`mushin.slurm` builds the launcher config: one task per job, with the GPUs,
CPUs, memory and time limit each cell needs.

```python
import mushin


@mushin.sweep
def experiment(lr, seed):
    ...  # train and evaluate one cell; return a dict of numbers
    return dict(accuracy=0.0)


ds = experiment.run(
    lr=mushin.multirun([0.01, 0.1]),
    seed=mushin.multirun([0, 1, 2]),
    working_dir="/scratch/me/lr-sweep",
    launcher="submitit_slurm",
    launcher_config=mushin.slurm(
        gpus=1,
        cpus=8,
        mem_gb=32,
        timeout_min=120,
        partition="gpu",
        account="my-lab",
        setup=["module load cuda/12.4"],
    ),
    on_error="nan",
)
```

- `timeout_min` is each job's time limit; a cell that runs past it is
  stopped.
- `setup` lines run in each job before the cell, for module loads or
  environment variables. They cannot choose the Python: each job runs the
  driver's own interpreter, by its path (see below), so start the driver from
  the environment the jobs should use. Write a shell variable as `$HOME`, not
  `${HOME}`: mushin passes the config through Hydra, which reads `${...}` as
  its own.
- Other fields of the plugin's SLURM config, such as `constraint`, `gres`,
  `array_parallelism` or `additional_parameters`, pass through unchecked. A
  field `mushin.slurm` writes itself (`nodes`, `tasks_per_node`, and the one of
  each argument you gave) is refused there.

When `scontrol` is available, which is where you submit from, `mushin.slurm`
checks the request before anything is queued, against the partition you name
(or `SBATCH_PARTITION` names). It refuses one that does not exist, or a time
limit past its `MaxTime`, and names the partitions whose limit fits. It does
not judge a job that names no partition (the cluster may route it), nor the
time limit under a QoS (`qos=` or `SBATCH_QOS`), which can lift it;
`check=False` skips the check. It judges what `sbatch` will use: an
`SBATCH_PARTITION`, `SBATCH_ACCOUNT`, `SBATCH_TIMELIMIT` or `SBATCH_QOS` set in
your shell overrides the job script, and mushin warns when one overrides an
argument you gave. It never chooses a partition for you.

## Submit, and come back later

By default `run()` submits the jobs and then waits for every one of them to
end before it returns the dataset. With `wait=False` it submits them and
returns at once, so you can log out of the login node while they queue and
run:

```python
import mushin

summary = experiment.run(
    lr=mushin.multirun([1e-4, 3e-4, 1e-3]),
    seed=mushin.multirun([0, 1, 2]),
    working_dir="/scratch/me/lr-sweep",
    launcher="submitit_slurm",
    launcher_config=mushin.slurm(gpus=1, timeout_min=240, partition="gpu"),
    wait=False,
)
# {"working_dir": "/scratch/me/lr-sweep", "run_id": "...", "submitted": 9,
#  "job_ids": ["4242"]}
```

`summary["job_ids"]` lists one id per job array, the ids `squeue` shows and
`sbatch --dependency=afterany:<id>` takes. Later, from any Python session on a
machine that sees the filesystem and can run `sacct` (a login node), once the
jobs have ended:

```python
ds = mushin.collect("/scratch/me/lr-sweep")
```

`collect` finishes the run as a waiting `run()` would have: a cell whose job
ended without a result is recorded failed, the sweep's manifest is written,
and the dataset is returned. It needs no import of the script that ran the
sweep. Some details:

- `wait=False` needs `launcher="submitit_slurm"` (or `"submitit_local"`), a
  `working_dir`, and Hydra's basic sweeper; `run()` raises a `ValueError`
  otherwise, before it writes anything. The workflow that ran it raises from
  `to_xarray()` until it is collected, and its `collect()` does what
  `mushin.collect` does.
- While any job of the run is queued, running, or preempted (SLURM may
  requeue it), `collect` raises a `RuntimeError` with the count of each state
  and writes nothing: collect again later. It reads each job's state from
  `sacct`. A job `sacct` does not report reads `UNKNOWN`, which also refuses:
  `sacct` may be missing here, may not list the job yet, or may not see the
  job's cluster.
- `collect` loads the jobs' submission and result files under
  `<working_dir>/.submitit/` and imports your task's code, as the jobs did:
  keep that folder until you have collected, run `collect` where the code
  and the types your task returns import as in the jobs, and only on a
  `working_dir` you trust. A result that does not load stops `collect`, with
  nothing written.
- Collect a run in the dir it was submitted in: the run's launch file names
  that dir, so `collect` refuses a copy or a move of it. A `submitit_folder`
  outside the `working_dir` is fine. With a relative `working_dir`, collect
  from the directory `run()` was called in.
- `on_error` works as in a waited run. Under `"raise"`, `collect` records
  every cell and raises the first failed cell's own error, with no manifest;
  a second `collect` then reloads the sweep, the failed cell NaN. Under
  `"nan"` it returns the dataset with failed cells NaN, and warns.
- A job that died gives no result file. `collect` waits once, up to 15
  seconds, for the result files of all such jobs (a shared filesystem can
  show a file late), then records their cells.
- With `max_total_seconds`, the budget runs from the first computed cell, so
  time a job spends queued after that counts, as with a waiting driver.
- Every batch of a `hydra.sweeper.max_batch_size` sweep is submitted at once,
  each its own job array; `array_parallelism` still caps how many run. A Hydra
  `on_multirun_end` callback runs at submission, not at collect.
- Collecting a run collected already, or run with the default `wait=True`,
  only reloads it (`load_from_dir`), and writes nothing.
- A `launcher="submitit_local"` job's id is a process id on the machine that
  submitted it, so collect such a run on that host.
- `wait` is `run()`'s own parameter. A task, `pre_task` or config field of
  that name takes its value as an override, `overrides=["++wait=..."]`;
  `run(wait=...)` for such a sweep raises a `ValueError` saying so.

## Or keep the driver running

With the default `wait=True`, the process that calls `run()` (the driver) does
little work, but it has to stay alive for the whole sweep. Either:

- run the script inside `tmux` or `screen` on a login node, so it survives your
  logout (not a login node that stops long-running processes: check your
  cluster's policy); or
- submit the driver itself as a small CPU job that lasts as long as the sweep,
  for example `sbatch --time=2-00:00:00 --cpus-per-task=1 --wrap "python sweep.py"`.
  SLURM passes a job's environment on to the jobs it submits, which can upset
  the cells' requests (memory, GPUs), so call `run()` inside submitit's
  `clean_env()` (submitit 1.4.6 or later), which hides those variables while it
  submits:

  ```python
  import submitit

  with submitit.helpers.clean_env():
      ds = experiment.run(...)
  ```

  It hides every `SBATCH_*` variable too, so under it a partition, account,
  QoS or time limit has to be passed to `mushin.slurm` itself, called inside
  the block as above.

If the driver stops anyway, the jobs keep running. Wait until they have ended
(`squeue --me` lists none of them), or cancel them (below), then run the same
script again with `resume=True`: it reuses the finished cells and runs only
the rest. A resume started while the earlier run's jobs are still queued or
running would submit those cells again, and two jobs would then write the
same cell, so `run()` refuses it (see
[Resume, and cancel](#resume-and-cancel)).

## What the jobs need

- **A shared filesystem.** The `working_dir`, the Python environment and your
  code must be at the same paths on the compute nodes. submitit starts each job
  with the driver's Python executable, and the job reads its task from files
  under `working_dir`.
- **Your code, at the same path.** Each job imports the modules your task
  uses, as the driver does.

Each cell runs in its own directory, `<working_dir>/<number>/`, and its
`mushin_cell_status.json` names the job that ran it (`job_id`). Its `submitit`
entry also names that job's log folder (`logs`), the node it ran on (`host`)
and, on SLURM, its `partition`, once the job's code has started; a job that
died before that names only its id and logs. A cell a resume reused keeps the entry of the
job that computed it. That job's logs are under
`<working_dir>/.submitit/<job id>/`: `<job id>_0_log.out` and
`<job id>_0_log.err`. A submission of several cells is one job array, whose
tasks' ids are `<array id>_<index>`, the index counting the cells that
submission sent (a resume sends only some, so it is not the cell's number); a
submission of one cell is a plain job.

## Watch a running sweep

```bash
squeue --me
```

lists the sweep's queued and running jobs. `mushin.show("/scratch/me/lr-sweep")`
prints a row for each cell the sweep has recorded or submitted, with its
status, from any machine that sees the filesystem: `queued` (its job has not
started), `running`, `completed`, `failed` or `skipped`. During a resume, a
cell sent to a job again is `queued` in place of the earlier run's status
until that job starts.
`mushin.show("/scratch/me/lr-sweep", jobs=True)` also lists each cell's job id
(`job`), the node it ran on (`host`) and the folder of its logs (`logs`); a
queued cell has only its job id. With `cells_per_job`, the cells a running job
has not reached yet are `queued`, with that job's id.

The driver names the jobs of each submission in a file under
`<working_dir>/mushin_jobs/`, which `show` reads, and removes the file once
it has waited on those jobs (with `wait=False`, `collect` removes it). Some
cases follow from that:

- A driver killed before its jobs end (Ctrl-C, a lost login session) leaves
  its file, and so does a driver whose wait raised before the scheduler could
  say every job had ended. A cell whose job never started then stays
  `queued` until a later run of the sweep, such as a `resume=True`, records
  the cell or removes the file, or until `mushin.cancel` (below) removes it.
- A job that dies before its code starts can show `queued` until the driver
  has waited on every job of its submission, or, for a run submitted with
  `wait=False`, until it is collected.
- A sweep whose `hydra.sweep.subdir` puts each cell more than one directory
  below `working_dir` lists no cell as `queued`.
- A login node may see a shared filesystem's files late, so a cell that has
  just started can show `queued` for a few seconds.

`array_parallelism` (256 by default) caps how many of the sweep's jobs run at
once; the rest wait in the queue.

## When a job dies

A cell that raises is a failed cell, as in a local sweep. A job that is
stopped at its time limit, or ends without returning (killed out of memory, by
`scancel`, or on a failed node the cluster does not requeue), fails its cell
too. A job SLURM requeues (after a preemption, or a node failure on clusters
that requeue then) runs its cell again instead, from the start or from its
checkpoint. With `on_error="nan"`, the
sweep goes on, the cell is NaN in the dataset, and `mushin_error.txt` in the
cell's directory holds the traceback, or, for a job that ended without
returning, what ended it and where its log is. A reload of the sweep
(`load_from_dir`) can then report the cell failed too, as the run did. The
[resilience guide](resilience.md) covers failed cells and checkpoints.

A job of several cells (`cells_per_job`, below) goes on to its next cell
after one raises. When the job itself ends without returning, the cells it
finished keep their results when their records hold them (a metrics file the
record names, or a task that returned None); the cell it was running and
those it had not reached fail, and a resume runs them again. Each of those
cells reports the job's error, not one of its own. At the time limit, a job
that is not requeued (the plugin's default for SLURM, `max_num_timeout=0`)
fails the cell it was running and goes on to the next ones until SLURM ends
it; a job that is requeued skips the cells it finished and runs the
interrupted one again.

## Resume, and cancel

Run the same sweep again with `resume=True` and the same `working_dir`.
Completed cells are reused from their records, and only the others are
submitted: a resume that fills 3 cells of 1,000 queues 3 jobs. `sample=` and
`cache_dir=` work the same way, submitting only the cells they leave to
compute. The driver still submits a cell, for its job to decide, whenever it
cannot judge the cell as the job would. Among others: a cell whose config
reads the environment (`${oc.env:...}`), a cell whose job imports modules the
driver cannot (a `PYTHONPATH` set in `setup`, say), and every cell of a sweep
with Hydra callbacks, since the job's environment, imports and callbacks are
not the driver's. With `cells_per_job`, a resume groups only the cells it
submits, so its jobs can be fewer than the first run's.

`run()` refuses a `working_dir` whose earlier jobs may still be queued or
running, whichever launcher it uses: two drivers on one sweep would each
submit the cells they find unfinished, and two jobs would then write one
cell. It reads the submissions listed under `<working_dir>/mushin_jobs/`
(below). It asks `scontrol` about each SLURM job, as `mushin.cancel` does,
even one whose cell has recorded its end: a job preempted after that may be
requeued. Where `scontrol` is not found, a job whose cell has recorded its end
is taken as over. For `launcher="submitit_local"` on the machine that
submitted it, it looks for the job's process. Then:

- A job SLURM lists as not ended (pending, running, or preempted, since SLURM
  may requeue a preempted job), or a local job whose process still runs, makes
  `run()` raise a `ValueError` that names the submission's file. Wait for those jobs to end:
  a driver still waiting on them finishes its run, and a `resume=True` then
  reuses the cells they completed. Or cancel them: `mushin.cancel` (below) for
  SLURM jobs; stop the processes of local ones.
- When `scontrol` is found but fails, takes too long or answers in a form
  mushin cannot read, `run()` raises a `RuntimeError`, as `cancel` cancels
  nothing then. So does a submission sent to another cluster, whose jobs
  plain `scontrol` does not see; `mushin.cancel` names the command that
  cancels those, and removes their file.
- Where the jobs cannot be checked at all (no `scontrol` on the machine, or
  `submitit_local` jobs of another machine), `run()` warns and goes ahead.
  It does the same when SLURM has given a job's id to another job.
- A run submitted with `wait=False` and not yet collected is refused while
  its jobs may run, and its message says to collect it. Where its jobs cannot
  be checked, `run()` raises a `RuntimeError` instead of going ahead: a run
  that went ahead would replace the sweep's `multirun.yaml`, and `collect`
  could no longer find the earlier run. Once its jobs have ended, a
  `resume=True` may go ahead in place of `collect`: it reuses the cells they
  completed, and the earlier run can no longer be collected.

Once you know a submission's jobs are over, or are not this sweep's, deleting
the file the message names lets the run go ahead. A run that goes ahead
removes the files of the submissions it found over, once its own cells have
run. The check can miss jobs: it reads only the files under
`<working_dir>/mushin_jobs/`, and a sweep whose directory is set by a
`hydra.sweep.dir=` override, not `working_dir`, is not checked.

To cancel a sweep, pass its `working_dir` to `mushin.cancel`, on a machine
where `scontrol` and `scancel` work (a login node), with the driver running or
not:

```python
import mushin

mushin.cancel("/scratch/me/lr-sweep", dry_run=True)  # the job ids it would cancel
mushin.cancel("/scratch/me/lr-sweep")  # cancels them, and returns their ids
```

It reads the submissions listed under `<working_dir>/mushin_jobs/`, asks
`scontrol` about each one's job array, and runs `scancel` once, for the arrays
still queued or running. A driver still running records the cancelled cells as
failed, and a later `resume=True` runs them again. With `cells_per_job`,
cancelling a running job also cancels the cells it had not reached. Once a submission's jobs are
cancelled or over, `cancel` removes its file, so `show` no longer lists its
cells as `queued`; a run submitted with `wait=False` is then over, and
`collect` only reloads what its cells recorded. Some cases follow:

- SLURM reuses job ids. A job is cancelled only when its script or logs lie
  under the sweep's submitit folder (`<working_dir>/.submitit/` by default);
  a job id SLURM has given another job is left alone, with a warning. The
  check is only as strict as that folder: a `submitit_folder` several sweeps
  share cannot tell their jobs apart, and one at or just below the
  filesystem's root (`/%j`, `/scratch/%j`) is not trusted at all, so its
  jobs are not cancelled.
- When `scontrol` takes too long, fails, or answers in a form `cancel` cannot
  read, it cancels nothing and raises, naming the `scancel` command to run by
  hand.
- Jobs sent to another cluster (`additional_parameters={"clusters": ...}`, or
  `SBATCH_CLUSTERS`) are not checked: `cancel` warns with the
  `scancel --clusters=...` command to run, and removes their file.
- Jobs of `launcher="submitit_local"` are processes, not SLURM jobs: `cancel`
  skips them with a warning.

By hand, `scancel <job id>` for each submission `squeue --me` lists cancels the
same jobs (a job array has one id for all its tasks). Each submission's folders
under `<working_dir>/.submitit/` begin with its id, and a resume adds a
submission of its own.

## Many small cells

By default each cell is its own job, so a sweep of many short cells spends
much of its time in the queue, and a large grid can hit the cluster's limits
on a job array's size (`MaxArraySize`) or on how many jobs you may have
queued. `cells_per_job` gives each job several cells, which it runs one after
another:

```python
launcher_config = mushin.slurm(gpus=1, timeout_min=120, cells_per_job=20)
```

The cells a sweep submits are grouped in order, up to 20 to a job (the last
job may get fewer), so 1,000 cells make 50 jobs. They are grouped within each
of the sweeper's batches, so a `hydra.sweeper.max_batch_size` can give more,
smaller jobs. Each cell keeps its own
directory, record, metrics file and `.hydra/` snapshot, and `run()` returns
the same dataset.

- `timeout_min` is the whole job's time limit, for all of its cells: allow
  for `cells_per_job` of them.
- The cells of a job share one Python process, as a reused joblib worker's
  do. What one cell leaves behind is there for the next: module-level state,
  the random state of a library the task does not seed, GPU memory a task
  still references, a `torch.distributed` process group left initialized,
  and environment variables the task sets (`hydra.job.env_set` is restored
  after each cell). `pin_gpu_round_robin` raises in a later cell once an
  earlier cell of the job has initialized CUDA. The cells of a job also share
  its submitit log.
- A job of several tasks (`tasks_per_node` or `nodes` above 1, as for DDP)
  cannot run several cells: the sweep is refused before anything is
  submitted. A task count set another way (`additional_parameters` with
  `ntasks`) is not seen: do not combine it with `cells_per_job`.
- `array_parallelism` caps how many jobs run at once, not cells.
- Choose `cells_per_job` so the number of jobs fits the cluster's
  `MaxArraySize` and your queue limit.

Other ways to do better:

- give each cell more of the sweep's work: fewer, longer cells; or
- run the sweep inside one allocation: request the GPUs once, in one `sbatch`
  job, then sweep with `launcher="joblib"` and `pin_gpu_round_robin`, as the
  [packing guide](packing.md#a-sweep-inside-one-slurm-allocation) shows. That
  suits a sweep of at most `n_jobs` cells, since each cell needs a fresh
  process to pin its GPU; for more, the same guide's Ray launcher shares GPUs
  without that limit.
