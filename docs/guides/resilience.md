# Resilient & resumable sweeps

A long method x seed sweep is only as reliable as its flakiest cell. One OOM, a
corrupt data shard, or a transient cluster hiccup should not throw away the hours
already spent on the runs that *did* succeed, and it must never let you quietly
compute statistics on a half-finished grid. mushin gives you three tools for
this: **fail-soft** runs, **resume**, and **provenance**.

> **Prefer to follow along?** [Notebook 04: Resilient sweeps](../notebooks/04_resilience.ipynb)
> runs this whole fail-soft → resume loop end to end with live output.

## Fail-soft: `on_error="nan"`

By default a failing job aborts the whole sweep and re-raises the exception
(`on_error="raise"`). Pass `on_error="nan"` instead and mushin keeps going: the
failing cell is recorded, its grid position becomes `NaN`, a `UserWarning` is
emitted, and every other cell completes normally.

```python
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

wf = MyWorkflow()
wf.run(
    method=multirun(["cnn", "mlp"]),
    seed=multirun([0, 1, 2, 3, 4]),
    working_dir="runs/experiment",
    on_error="nan",
)
```

After the run you can inspect what happened:

```python
wf.is_complete  # False — at least one cell failed
wf.failures  # [{"combo": "method=mlp,seed=3", "exception": "...", "working_dir": "..."}]
ds = wf.to_xarray()
ds["accuracy"].sel({"method": "mlp", "seed": 3})  # nan
ds.attrs["mushin_failures"]  # JSON list: ["method=mlp,seed=3"]
```

To debug a failed cell, look inside its `working_dir`: mushin writes the full
stack trace to `mushin_error.txt` there (the one-line `exception` above is just
the repr), alongside Hydra's own `.hydra/` config snapshot and job log for that
cell.

The failed cells are also written to a **sweep manifest**,
`<working_dir>/mushin_sweep_manifest.json`, which records every requested cell's
status (`completed` / `failed`) and the job directory it ran in. This manifest is
what makes a resume possible.

## Statistics refuse an incomplete sweep

You cannot compare methods on a grid that has holes in it: the missing cells are
*missing data*, not measurements. Both `compare` (`compare_methods`) and `Study`
detect the completeness signal and refuse:

```python
from mushin.benchmark import compare_methods, IncompleteSweepError

try:
    compare_methods(ds)
except IncompleteSweepError as e:
    print(e)  # "1 run(s) failed (method=mlp,seed=3); fix the cause and re-run
    #  with resume=True to complete the sweep before comparing."
```

This is keyed purely on `ds.attrs["mushin_failures"]`: a plain user dataset, or
a metric that is legitimately `NaN` for some other reason, is unaffected. Only a
sweep that actually recorded failures triggers the guard.

A sweep reloaded from disk carries the same signal.
`MultiRunMetricsWorkflow().load_from_dir(working_dir, "mushin_metrics.json")`
records every cell of the sweep's grid that has no metrics: as failed if its
status says so, otherwise as skipped (not run), which covers a cell still
`running` and one a killed sweep never started. Its `to_xarray()` then has the
same attrs, `is_complete` is `False`, and `compare_methods` refuses it too.
Give each sweep its own `working_dir` (resuming it is fine): job dirs that an
earlier sweep left in a reused one can stand in for cells the new sweep has not
run yet.

## Resume: fill only the failed cells

Fix the underlying cause, then re-run against the **same `working_dir`** with
`resume=True`. mushin reads the prior manifest and short-circuits every cell that
already `completed` (reusing its cached metrics from disk); only the failed and
missing cells actually re-execute:

```python
wf = MyWorkflow()
wf.run(
    method=multirun(["cnn", "mlp"]),
    seed=multirun([0, 1, 2, 3, 4]),
    working_dir="runs/experiment",  # same directory as before
    resume=True,
)

wf.is_complete  # True — every cell now present
compare_methods(wf.to_xarray())  # no longer raises
```

`resume=True` requires `working_dir` to be set (there is nothing to resume from
without a stable location). The completed cells are not recomputed, so a resume
of a mostly-successful sweep is cheap.

That holds when the cause was outside your code: a full disk, a lost node, a
flaky download. If the fix is an edit to the task or to a project file its
module imports (a helper, a constant), the completed cells were computed by
the old code, so the resume re-runs them as well, with a warning naming the
file. An edit anywhere in such a file counts; a comment, a docstring or
formatting does not. In the driver script, only what the task's definition
can reach by name counts (its helpers and constants), not the `run(...)`
call, other top-level calls, `print` calls or what only they read, so
extending the grid and resuming reuses the completed cells. The task
function's own source is compared as text, so a comment inside its body is an
edit. If you edit a module and resume in the same Python process without
reloading it, the old code still runs: reload it (or restart) first. The
script rule follows names, so a statement that changes an object in place
(`CFG["lr"] = 0.1`), a function that rebinds a global when called, a
definition registered inside a block, or a registry filled by
`__init_subclass__` can hide an edit there; helpers kept in an imported module
are compared as a whole file.

In a notebook, what counts is what the task's code looks up in the kernel when
the run starts: each helper and class defined in a cell (its definition, and
the defaults and class attributes it captured) and each plain constant (a
number, a string, a tuple of them) by its value. Re-run the edited cell, then
resume, and the cells re-run with a warning naming it (`helper (notebook)`). A
mutable value (a dict config, a list), an object's state, a name the task
rebinds and a value named like a secret are not compared, nor is code whose
text the kernel does not keep (a `%%time` cell, `exec`, a `%run` file). A value
that differs per session, such as a timestamp or a seed drawn in a cell,
re-runs the cells after a restart; pass it to the task as a parameter instead.

### The full loop

```
run(on_error="nan")  ──►  inspect wf.failures  ──►  fix the cause
        ▲                                                │
        │                                                ▼
   compare / Study  ◄── run(working_dir=…, resume=True) ◄┘
   (once is_complete)
```

## Surviving a hard kill & resuming mid-training

`resume=True` is durable across a **hard process kill** (OOM, SLURM preemption,
node death), not just handled Python exceptions. Each cell records its status
(`running` → `completed`/`failed`) from inside its own job, so a mid-sweep kill
never loses the cells that already finished; resuming re-runs only the unfinished
ones.

A long-running cell can also resume its **own** training. Declare a
`mushin_resume` parameter on your `task`; mushin injects a `ResumeContext`:

```python
from mushin.workflows import MultiRunMetricsWorkflow


class Train(MultiRunMetricsWorkflow):
    @staticmethod
    def task(lr, seed, mushin_resume=None):
        # mushin_resume.dir       -> this cell's directory (write your checkpoint here)
        # mushin_resume.is_resume -> True when a prior attempt of THIS cell left artifacts
        # mushin_resume.last_ckpt -> newest checkpoint in dir, or None
        ckpt = mushin_resume.last_ckpt if mushin_resume else None
        trainer.fit(
            model, ckpt_path=ckpt
        )  # Lightning: default_root_dir=mushin_resume.dir
        return dict(accuracy=...)
```

Write your checkpoint into `mushin_resume.dir` (Lightning: set
`default_root_dir=mushin_resume.dir` with `ModelCheckpoint(save_last=True)`).
Re-running the **same** sweep (same grid) reuses each cell's directory, so a
resumed cell finds its own checkpoint; a cell is never handed a checkpoint from a
different cell. Tasks that don't declare `mushin_resume` are unaffected.

## Provenance

Every run (fail-soft or not) writes a per-job provenance record,
`mushin_provenance.json`, into each job directory *before* the task executes, so
even a failing cell leaves its lineage behind. It captures the git SHA, key
package versions, the resolved config and, once the task has returned, the
accelerator as the cell's process saw it (CUDA and cuDNN build, and the device
once the task initialized CUDA or when MPS is available). It is `null` when
that process never imported torch. With torch loaded, `cuda` and `cudnn` are
the wheel's build (`null` on a CPU-only build) and `device` is `null` until the
task initializes CUDA; records from before 0.17 named a present GPU even for a
task that stayed on the CPU, so `diff` against such a sweep reports a device
change once. A cell killed during its task keeps the view its process had
before the task:

```python
wf.provenance  # dict: {"git": {"sha": ...}, "packages": {...}, "config": {...}, ...}
ds.attrs[
    "provenance"
]  # the same record, JSON-encoded, so a saved dataset carries its lineage
```

For a fuller record, pass `capture_env=True` to write a complete dependency
snapshot (`mushin_env.txt`, via `uv export`/`pip freeze`, falling back to an
`importlib.metadata` dump) alongside the sweep:

```python
wf.run(..., working_dir="runs/experiment", capture_env=True)
```

## Using it from `Study`

`Study` threads the same options through to its training sweep, so you get
fail-soft, resumable *training* runs with the same guarantees: an incomplete
training sweep raises `IncompleteSweepError` from `Study.run` rather than
comparing partially-trained checkpoints:

```python
from mushin import Study

study = Study(
    methods={"cnn": train_cnn, "mlp": train_mlp},
    load_fn=load_model,
    seeds=[0, 1, 2, 3, 4],
    data=test_loader,
    num_classes=10,
    working_dir="runs/study",
    on_error="nan",  # fail-soft training
    capture_env=True,  # snapshot the environment
)
study.run()  # raises IncompleteSweepError if any (method, seed) failed
# ...fix the cause, then re-run with resume=True on the same working_dir:
study = Study(..., working_dir="runs/study", resume=True)
result = study.run()
```

## See also

- [Workflows & sweeps](workflows.md): the multirun API
- [Comparing methods](compare.md): the `compare` API
- [Studies](study.md): the train → compare motion
