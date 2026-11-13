# Running a sweep on SLURM

mushin can submit a sweep's cells to a SLURM cluster itself: no `sbatch` script
to write. Each cell runs as one SLURM job (one task of a job array), in its own
directory under the sweep's `working_dir`, and `run()` gathers the results into
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
        setup=["module load cuda/12.4", "source $HOME/envs/ml/bin/activate"],
    ),
    on_error="nan",
)
```

- `timeout_min` is each job's time limit; a cell that runs past it is
  stopped.
- `setup` lines run in each job before the cell, for module loads or a
  virtual environment. Write a shell variable as `$HOME`, not `${HOME}`: mushin
  passes the config through Hydra, which reads `${...}` as its own.
- Other fields of the plugin's SLURM config, such as `constraint`, `qos`,
  `array_parallelism` or `additional_parameters`, pass through as given.

When `scontrol` is available, which is where you submit from, `mushin.slurm`
checks the request before anything is queued. It refuses a partition that
does not exist, or a time limit past the partition's, and names the partitions
whose limit fits; `check=False` skips the check. It judges what `sbatch` will
use: an `SBATCH_PARTITION`, `SBATCH_TIMELIMIT` or `SBATCH_QOS` set in your
shell overrides the job script, and mushin warns when one overrides an argument
you gave. It does not choose a partition for you.

## Submit, and keep the driver running

`run()` submits the jobs and then waits for every one of them to end before it
returns the dataset. The process that calls `run()` (the driver) does little
work, but it has to stay alive for the whole sweep. On a login node that stops
long-running processes, either:

- run the script inside `tmux` or `screen`, so it survives your logout; or
- submit the driver itself as a small CPU job that lasts as long as the sweep,
  for example `sbatch --time=2-00:00:00 --cpus-per-task=1 --wrap "python sweep.py"`.
  The driver submits the cells' jobs from inside it.

If the driver stops anyway, the jobs keep running. Run the same script again
with `resume=True` (below) once they are done: it collects the finished cells
and runs only the rest.

## What the jobs need

- **A shared filesystem.** The `working_dir`, the Python environment and your
  code must be at the same paths on the compute nodes. submitit starts each job
  with the driver's Python executable, and the job reads its task from files
  under `working_dir`.
- **Your code, at the same path.** Each job imports the modules your task
  uses, as the driver does.

Each cell runs in its own directory, `<working_dir>/<number>/`. The job logs go
under `<working_dir>/.submitit/<job id>/`, where an array task's job id is
`<array id>_<index>`: `<job id>_0_log.out` and `<job id>_0_log.err`.

## Watch a running sweep

```bash
squeue --me
```

lists the sweep's array tasks. `mushin.show("/scratch/me/lr-sweep")` prints each
cell that has started with its status (`running`, `completed`, `failed`,
`skipped`), from any machine that sees the filesystem.

`array_parallelism` (256 by default) caps how many of the sweep's jobs run at
once; the rest wait in the queue.

## When a job dies

A cell that raises is a failed cell, as in a local sweep. A job that is
stopped at its time limit, or ends without returning (killed out of memory, by
`scancel`, on a failed node), fails its cell too. With `on_error="nan"`, the
sweep goes on, the cell is NaN in the dataset, and `mushin_error.txt` in the
cell's directory holds the traceback, or, for a job that ended without
returning, what ended it and where its log is. The
[resilience guide](resilience.md) covers failed cells and checkpoints.

## Resume, and cancel

Run the same sweep again with `resume=True` and the same `working_dir`.
Completed cells are reused from their records, and only the others are
submitted: a resume that fills 3 cells of 1,000 queues 3 jobs. `sample=` and
`cache_dir=` work the same way, submitting only the cells they leave to
compute. A cell whose config reads the environment (`${oc.env:...}`), and every
cell of a sweep with Hydra callbacks, is still submitted, and its job decides,
since the job's environment and callbacks are not the driver's.

To cancel a sweep, cancel its job array: `scancel <array id>`. The array id is
in `squeue --me`, and it begins each folder name under
`<working_dir>/.submitit/`. A driver still running records the cancelled
cells as failed, and a later `resume=True` runs them again.

## Many small cells

Each cell is its own job, so a sweep of many short cells spends much of its
time in the queue. Two ways to do better:

- give each job more of the sweep's work: fewer, longer cells; or
- run the sweep inside one allocation: request the GPUs once, in one `sbatch`
  job, then sweep with `launcher="joblib"` and `pin_gpu_round_robin`, as the
  [packing guide](packing.md#a-sweep-inside-one-slurm-allocation) shows.
