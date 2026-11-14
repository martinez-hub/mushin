# Running a sweep on SLURM

mushin can submit a sweep's cells to a SLURM cluster itself: no `sbatch` script
to write. Each cell runs as one SLURM job (a task of a job array, when several
are submitted together), in its own directory under the sweep's `working_dir`, and `run()` gathers the results into
the same `xarray.Dataset` a local sweep gives. This guide covers ordinary
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

## Submit, and keep the driver running

`run()` submits the jobs and then waits for every one of them to end before it
returns the dataset. The process that calls `run()` (the driver) does little
work, but it has to stay alive for the whole sweep. Either:

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
the rest. Do not resume while the earlier run's jobs are still queued or
running: the resume submits those cells again, and two jobs then write the
same cell.

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
queued cell has only its job id.

The driver names the jobs of each submission in a file under
`<working_dir>/mushin_jobs/`, which `show` reads, and removes the file once
it has waited on those jobs. Some cases follow from that:

- A driver killed before its jobs end (Ctrl-C, a lost login session) leaves
  its file, and so does a driver whose wait raised before the scheduler could
  say every job had ended. A cell whose job never started then stays
  `queued` until a later run of the sweep, such as a `resume=True`, records
  the cell.
- A job that dies before its code starts can show `queued` until the driver
  has waited on every job of its submission.
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
returning, what ended it and where its log is. The
[resilience guide](resilience.md) covers failed cells and checkpoints.

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
not the driver's.

To cancel a sweep, cancel its jobs: `scancel <job id>` for each submission
`squeue --me` lists (a job array has one id for all its tasks). Each
submission's folders under `<working_dir>/.submitit/` begin with its id, and a
resume adds a submission of its own. A driver still running records the
cancelled cells as failed, and a later `resume=True` runs them again.

## Many small cells

Each cell is its own job, so a sweep of many short cells spends much of its
time in the queue, and a large grid can hit the cluster's limits on a job
array's size (`MaxArraySize`) or on how many jobs you may have queued. Ways to
do better:

- give each job more of the sweep's work: fewer, longer cells; or
- run the sweep inside one allocation: request the GPUs once, in one `sbatch`
  job, then sweep with `launcher="joblib"` and `pin_gpu_round_robin`, as the
  [packing guide](packing.md#a-sweep-inside-one-slurm-allocation) shows. That
  suits a sweep of at most `n_jobs` cells, since each cell needs a fresh
  process to pin its GPU; for more, the same guide's Ray launcher shares GPUs
  without that limit.
