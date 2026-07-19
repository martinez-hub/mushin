# Cluster-gated validation runbook

> **Status (2026-07-18): COMPLETE.** All features below were validated on real
> SLURM/GPU hardware (`mantis`: L40S single-node + A100 multi-node) and merged —
> multi-node DDP (#50), HydraFSDP (#58), GPU packing (#59); resume (#83) and
> out-of-process submitit (#86) were covered too. This document is retained as a
> reusable template for validating future HPC features.

**Purpose:** a self-contained checklist to validate the mushin features that are
unit-/adversarially-verified but never run on real GPU/SLURM hardware. Anyone with
HPC access can run this — no mushin knowledge required. Run a test, record the
result, and send back what "Report" asks for; a PR **graduates from cluster-gated
to mergeable** once its tests PASS.

**Who runs it:** you or a colleague with access to a SLURM cluster with multi-GPU
nodes. You need: SSH to the cluster, the ability to `pip install` into a Python
3.10+ env, and (for Phase 2) the ability to submit SLURM jobs.

**What's being validated, and where it lives (branch → PR):**

| Feature | Branch to install | PR | Phase |
|---|---|---|---|
| `HydraDDP` (single-node multi-GPU) | `main` | (on main) | 1 |
| `pin_gpu_round_robin` (GPU packing) | `gpu-packing` | #59 | 1 |
| `HydraFSDP` (single-node) | `hydra-fsdp` | #58 | 1 |
| Resume: hard-kill durability | `main` | #83 | 1 |
| Out-of-process submitit launcher | `main` | #86 | 2 |
| Multi-node DDP | `multinode-ddp` | #50 | 2 |
| `HydraFSDP` (multi-node) | `hydra-fsdp` | #58 | 2 |
| Resume: real SLURM preemption | `main` | #83 | 2 |

> **Phase 1** needs one node with **2+ GPUs** (an interactive/`salloc` GPU session
> is enough). **Phase 2** needs **2+ nodes + the SLURM scheduler** and the submitit
> launcher. Do Phase 1 first — it graduates most of the work at the lowest cost.

---

## 0. One-time setup

```bash
# Clone once; you'll check out a specific branch per test.
git clone https://github.com/martinez-hub/mushin.git && cd mushin

# A clean env (uv recommended; plain venv + pip works too).
python -m venv .venv && . .venv/bin/activate
```

Each test says which **branch** to check out and what to install. To install a
branch's mushin into the current env:

```bash
git checkout <branch> && pip install -e .        # editable install of that branch
```

Two Hydra launcher plugins are used (install when a test calls for them):

```bash
pip install hydra-joblib-launcher       # Phase 1 packing (local multiprocessing)
pip install hydra-submitit-launcher     # Phase 2 (SLURM submission)
```

**What to report for every test:** the test ID, **PASS / FAIL**, the exact command
you ran, the tail of stdout/stderr, and (Phase 2) the SLURM job id + the contents
of the `.out`/`.err` logs. For a FAIL, include the full traceback.

---

## Phase 1 — single-node multi-GPU (one node, 2+ GPUs)

Grab an interactive GPU session first, e.g. `salloc --gres=gpu:2 --time=1:00:00`
(adjust to your cluster), then activate the env.

### T1 — `HydraDDP` runs one training across the node's GPUs

**Branch:** `main` · **PR:** on main · **Needs:** 2+ GPUs on one node.

`HydraDDP` is a Lightning `DDPStrategy` used *inside* a task's `Trainer`. Write
`t1_hydraddp.py`:

```python
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

import mushin
from mushin import HydraDDP


class Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(8, 2)

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.cross_entropy(self.net(x), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


@mushin.sweep
def experiment(seed):
    torch.manual_seed(seed)
    x, y = torch.randn(256, 8), torch.randint(0, 2, (256,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    trainer = pl.Trainer(
        strategy=HydraDDP(), devices=2, accelerator="gpu",   # devices == #GPUs
        max_epochs=1, enable_progress_bar=False, logger=False,
    )
    trainer.fit(Tiny(), loader)
    return dict(loss=float(trainer.callback_metrics.get("train_loss", 0.0)))


if __name__ == "__main__":
    ds = experiment.run(seed=mushin.multirun([0, 1]), working_dir="t1_runs")
    print("DONE", dict(ds.sizes))
```

**Run:** `python t1_hydraddp.py`

**PASS if:** it prints `DONE {'seed': 2}` with no hang and no NCCL/DDP error; both
GPUs show utilization during the run (`nvidia-smi` in another shell); each of the 2
job dirs under `t1_runs/` contains one `mushin_metrics.json`. **FAIL** if it hangs
(rank desync), errors, or only one GPU is used.

**Report:** stdout tail + whether both GPUs were utilized.

### T2 — `pin_gpu_round_robin` packs a sweep across GPUs (PR #59)

**Branch:** `gpu-packing` (`git checkout gpu-packing && pip install -e .`) · **Needs:**
2+ GPUs + `pip install hydra-joblib-launcher`.

`pin_gpu_round_robin` sets `CUDA_VISIBLE_DEVICES` from the Hydra job index so each
parallel job lands on a distinct GPU. Write `t2_packing.py`:

```python
import os

import mushin
from mushin import pin_gpu_round_robin
from mushin.workflows import MultiRunMetricsWorkflow


class Pack(MultiRunMetricsWorkflow):
    @staticmethod
    def task(seed):
        gpu = pin_gpu_round_robin(num_gpus=2)   # this job -> GPU (job_num % 2)
        import torch
        assert torch.cuda.is_available()
        # record which physical GPU this job was pinned to
        return dict(pinned_gpu=float(gpu), cuda_visible=float(int(os.environ["CUDA_VISIBLE_DEVICES"])))


if __name__ == "__main__":
    Pack().run(
        seed=mushin.multirun([0, 1, 2, 3]),
        working_dir="t2_runs",
        launcher="joblib",
        overrides=["hydra.launcher.n_jobs=4"],   # 4 parallel jobs over 2 GPUs
    )
    print("DONE")
```

**Run:** `python t2_packing.py`

**PASS if:** it completes; across the 4 job dirs the recorded `pinned_gpu` /
`cuda_visible` values cycle over `{0, 1}` (round-robin), i.e. jobs land on *both*
GPUs, not all on GPU 0. Check with:
`python -c "import json,glob; print([json.load(open(p))['pinned_gpu'] for p in glob.glob('t2_runs/*/mushin_metrics.json')])"`
**FAIL** if all jobs report the same GPU (packing didn't take effect) or it errors.

**Report:** the list of `pinned_gpu` values across jobs.

### T3 — `HydraFSDP` (single-node) shards one model across GPUs (PR #58)

**Branch:** `hydra-fsdp` (`git checkout hydra-fsdp && pip install -e .`) · **Needs:** 2+ GPUs.

Same shape as T1 but with `strategy=HydraFSDP()` in the `Trainer`. Copy
`t1_hydraddp.py` to `t3_fsdp.py` and change the import + strategy:

```python
from mushin import HydraFSDP
# ...
trainer = pl.Trainer(strategy=HydraFSDP(), devices=2, accelerator="gpu",
                     max_epochs=1, enable_progress_bar=False, logger=False)
```

**Run:** `python t3_fsdp.py`

**PASS if:** completes with `DONE {'seed': 2}`, both GPUs utilized, no FSDP/sharding
error, one `mushin_metrics.json` per job dir. **FAIL** on hang or FSDP error.

**Report:** stdout tail + GPU utilization.

### T4 — Resume survives a hard process kill (PR #83)

**Branch:** `main` · **Needs:** just a multi-cell sweep you can `kill -9`. (No GPUs
required — this validates the durability mechanism, of which SLURM preemption is one
cause. GPUs optional.)

Write `t4_killresume.py`:

```python
import time

import mushin
from mushin.workflows import MultiRunMetricsWorkflow


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(seed):
        open("ran.log", "a").write(f"{seed}\n")
        if seed == 2:
            time.sleep(120)          # hang so you can SIGKILL mid-cell
        return dict(v=float(seed))


if __name__ == "__main__":
    import sys
    resume = "--resume" in sys.argv
    W().run(seed=mushin.multirun([0, 1, 2, 3]), working_dir="t4_runs", resume=resume)
    print("COMPLETE")
```

**Run:**
```bash
rm -f ran.log
python t4_killresume.py &            # let seeds 0,1 finish, then it hangs on seed 2
sleep 20 && kill -9 %1               # hard kill mid-sweep (simulates preemption/OOM)
# edit t4_killresume.py: change `if seed == 2` to `if False` (the cause is "fixed")
: > ran.log
python t4_killresume.py --resume     # resume the same working_dir
```

**PASS if:** the resume run prints `COMPLETE`, and `ran.log` after the resume does
**not** contain `0` or `1` (the completed cells were NOT recomputed — only the
killed/remaining cells ran). Check: `cat ran.log`. **FAIL** if seeds 0/1 re-ran.

**Report:** the contents of `ran.log` after the resume run.

---

## Phase 2 — multi-node / SLURM (2+ nodes + scheduler)

Needs the submitit launcher: `pip install hydra-submitit-launcher`. Replace
`<PARTITION>`/`<ACCOUNT>`/`<G>` (GPUs per node) with your cluster's values.

### T5 — Out-of-process submitit launcher submits cells as SLURM jobs (PR #86)

**Branch:** `main` · **Needs:** SLURM + submitit launcher.

This validates that mushin's picklable dispatch ships cells to SLURM workers. Write
`t5_submitit.py` with a **module-level** `@mushin.sweep` task (importable so the
worker can run it) returning a trivial metric, then:

```python
experiment.run(
    seed=mushin.multirun([0, 1, 2, 3]),
    working_dir="t5_runs",
    launcher="submitit_slurm",
    overrides=[
        "hydra.launcher.timeout_min=15",
        "hydra.launcher.partition=<PARTITION>",
        "hydra.launcher.tasks_per_node=1",
    ],
)
```

**Run:** `python t5_submitit.py`

**PASS if:** submitit submits 4 SLURM jobs (`squeue` shows them), all complete, and
`t5_runs/*/mushin_metrics.json` exists once per cell; no `PicklingError`. **FAIL** on
a pickling error or missing results.

**Report:** `squeue` snapshot, the submitit `.out` logs, and the job-dir listing.

### T6 — Multi-node DDP, the merge gate (PR #50)

**Branch:** `multinode-ddp` (`git checkout multinode-ddp && pip install -e .`) · **Needs:**
2 nodes × `<G>` GPUs + submitit. This is the merge gate documented in
`docs/guides/multinode.md` on that branch. The **one contract**:
`tasks_per_node == gpus_per_node == Trainer devices`, `world_size == nodes × gpus_per_node`.

1. Build the launcher config: `submitit_slurm_config(nodes=2, gpus_per_node=<G>, partition="<PARTITION>", account="<ACCOUNT>")`.
2. Trainer: `devices=<G>`, `num_nodes=2`, `strategy=builds(HydraDDP)`.
3. Launch the hydra-zen workflow with `hydra/launcher=submitit_slurm` and `--multirun`
   (see the branch's `docs/guides/multinode.md` for the full example).

**PASS if:** the 2-node job completes; `metrics.pt`/`mushin_metrics.json` exists
exactly once per job dir; `load_experiment` aggregates results; **and** a
deliberately mismatched `tasks_per_node` (≠ `devices`) raises the fail-fast
world-size error rather than hanging. **FAIL** on hang, world-size desync, or
duplicated/missing metrics.

**Report:** the SLURM job id, `.out`/`.err` logs, the per-job-dir metrics listing,
and confirmation that the mismatch case fails fast.

### T7 — `HydraFSDP` multi-node (PR #58)

**Branch:** `hydra-fsdp` · **Needs:** 2 nodes × `<G>` GPUs + submitit. Same as T6 but
`strategy=builds(HydraFSDP)`.

**PASS if:** the 2-node FSDP job completes, metrics once per job dir, no sharding/
world-size error. **Report:** as T6.

### T8 — Resume survives a real SLURM preemption/requeue (PR #83)

**Branch:** `main` · **Needs:** SLURM (ideally a preemptible/requeue-enabled
partition).

Run a multi-cell sweep as a SLURM job on a `--requeue` partition; `scancel` it (or
let it be preempted) after some cells finish, then resubmit the same script with
`resume=True` against the same `working_dir`.

**PASS if:** the resumed job does **not** recompute the cells that completed before
the cancellation (durable status survived the kill), and the grid completes. **FAIL**
if completed cells re-ran. **Report:** the before/after job-dir listing and which
cells re-executed.

---

## Graduation

| Once these PASS… | …this graduates |
|---|---|
| T4 | #83 kill-durability (T8 adds real-SLURM confirmation) |
| T2 | #59 GPU packing |
| T1 + T6 | #50 multi-node DDP (T1 single-node, T6 multi-node) |
| T3 + T7 | #58 HydraFSDP |
| T5 | #86 submitit out-of-process |

When a PR's tests pass, remove its `do-not-merge:needs-cluster` gate, rebase onto
`main`, and merge. Record the run (cluster, GPUs, date, logs) on the PR so the
"HPC-validated" claim is auditable. See the [[cluster-gated-prs]] memory for the
current parked-PR status and rebase notes.
