# Resilient & resumable sweeps

A long method x seed sweep is only as reliable as its flakiest cell. One OOM, a
corrupt data shard, or a transient cluster hiccup should not throw away the hours
already spent on the runs that *did* succeed — and it must never let you quietly
compute statistics on a half-finished grid. mushin gives you three tools for
this: **fail-soft** runs, **resume**, and **provenance**.

> **Prefer to follow along?** [Notebook 04 — Resilient sweeps](../notebooks/04_resilience.ipynb)
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
wf.is_complete        # False — at least one cell failed
wf.failures           # [{"combo": "method=mlp,seed=3", "exception": "...", "working_dir": "..."}]
ds = wf.to_xarray()
ds["accuracy"].sel({"method": "mlp", "seed": 3})   # nan
ds.attrs["mushin_failures"]                         # ["method=mlp,seed=3"]
```

The failed cells are also written to a **sweep manifest**,
`<working_dir>/mushin_sweep_manifest.json`, which records every requested cell's
status (`completed` / `failed`) and the job directory it ran in. This manifest is
what makes a resume possible.

## Statistics refuse an incomplete sweep

You cannot compare methods on a grid that has holes in it — the missing cells are
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

This is keyed purely on `ds.attrs["mushin_failures"]` — a plain user dataset, or
a metric that is legitimately `NaN` for some other reason, is unaffected. Only a
sweep that actually recorded failures triggers the guard.

## Resume: fill only the failed cells

Fix the underlying cause, then re-run against the **same `working_dir`** with
`resume=True`. mushin reads the prior manifest and short-circuits every cell that
already `completed` (reusing its cached metrics from disk) — only the failed and
missing cells actually re-execute:

```python
wf = MyWorkflow()
wf.run(
    method=multirun(["cnn", "mlp"]),
    seed=multirun([0, 1, 2, 3, 4]),
    working_dir="runs/experiment",   # same directory as before
    resume=True,
)

wf.is_complete            # True — every cell now present
compare_methods(wf.to_xarray())   # no longer raises
```

`resume=True` requires `working_dir` to be set (there is nothing to resume from
without a stable location). The completed cells are not recomputed, so a resume
of a mostly-successful sweep is cheap.

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
never loses the cells that already finished — resuming re-runs only the unfinished
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
        trainer.fit(model, ckpt_path=ckpt)  # Lightning: default_root_dir=mushin_resume.dir
        return dict(accuracy=...)
```

Write your checkpoint into `mushin_resume.dir` (Lightning: set
`default_root_dir=mushin_resume.dir` with `ModelCheckpoint(save_last=True)`).
Re-running the **same** sweep (same grid) reuses each cell's directory, so a
resumed cell finds its own checkpoint; a cell is never handed a checkpoint from a
different cell. Tasks that don't declare `mushin_resume` are unaffected.

## Provenance

Every run — fail-soft or not — writes a per-job provenance record,
`mushin_provenance.json`, into each job directory *before* the task executes, so
even a failing cell leaves its lineage behind. It captures the git SHA, key
package versions, and the resolved config:

```python
wf.provenance                 # dict: {"git": {"sha": ...}, "packages": {...}, "config": {...}, ...}
ds.attrs["provenance"]        # the same record, JSON-encoded, so a saved dataset carries its lineage
```

For a fuller record, pass `capture_env=True` to write a complete dependency
snapshot (`mushin_env.txt`, via `uv export`/`pip freeze`, falling back to an
`importlib.metadata` dump) alongside the sweep:

```python
wf.run(..., working_dir="runs/experiment", capture_env=True)
```

## Using it from `Study`

`Study` threads the same options through to its training sweep, so you get
fail-soft, resumable *training* runs with the same guarantees — an incomplete
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
    on_error="nan",     # fail-soft training
    capture_env=True,   # snapshot the environment
)
study.run()             # raises IncompleteSweepError if any (method, seed) failed
# ...fix the cause, then re-run with resume=True on the same working_dir:
study = Study(..., working_dir="runs/study", resume=True)
result = study.run()
```

## See also

- [Workflows & sweeps](workflows.md) — the multirun API
- [Comparing methods](compare.md) — the `compare` API
- [Studies](study.md) — the train → compare motion
