# Multi-node DDP support (submitit + SLURM/Elastic) — design

- **Date:** 2026-06-27
- **Issue:** #43
- **Status:** approved (brainstorm complete; pending implementation plan)
- **Merge gate:** this branch must NOT merge until validated end-to-end on a real
  multi-node cluster (see "Cluster-gated validation"). Everything else is
  unit-testable in CI without a cluster.

## Motivation

`HydraDDP` (`src/mushin/lightning/launchers.py`) targets single-node multi-GPU:
its `_HydraDDPLauncher` spawns one `subprocess.Popen` per local GPU, each
re-entering via `python -m mushin.lightning._pl_main` to rebuild the
trainer/module from Hydra's saved `config.yaml`. Issue #43 asks for cluster-scale
multi-node, delegating orchestration to `hydra-submitit-launcher` + Lightning's
`SLURMEnvironment`/`TorchElasticEnvironment` rather than growing HydraDDP into a
cross-node spawner.

## Key architectural finding

`HydraDDP._configure_launcher` only installs the custom subprocess launcher when
the cluster environment does **not** create processes externally:

```python
if not self.cluster_environment.creates_processes_externally:
    self._launcher = _HydraDDPLauncher(...)
```

Under `SLURMEnvironment` (srun) or `TorchElasticEnvironment` (torchrun) —
`creates_processes_externally is True` — HydraDDP **steps aside** and Lightning
uses the externally-launched ranks directly (native DDP). Therefore the entire
reattach path (`_HydraDDPLauncher`, `_subprocess_call`, `_pl_main`, the per-rank
output subdir) is **bypassed** on the real multi-node path.

Consequence: mushin's job for multi-node is **not** to spawn processes. It is to
(a) not corrupt results across ranks, (b) catch the SLURM-tasks↔DDP-ranks
mismatch early, (c) make the submitit config foolproof, and (d) preserve
reproducibility. Two genuine bugs fire under multi-node and are fixed here.

## Scope (this branch)

Build the pieces that are fully unit-testable without a cluster, plus docs and a
gated cluster smoke test. The c10d/rendezvous launcher rewrite (fault tolerance)
is **deferred** to a separate cluster-validated effort.

## Components

All of the following are unit-testable in CI (mock the trainer / cluster
environment); none requires a GPU or a cluster to test the logic.

### 1. `MetricsCallback` saves on global rank 0 only — `lightning/callbacks.py` (bug fix)

`on_validation_end`/`on_test_end` currently call `torch.save(...)` with no rank
guard. Under multi-node DDP every rank fires the callback, so N ranks write the
same `metrics.pt` (clobbering on a shared filesystem; non-zero ranks may hold
unsynced `callback_metrics`). Fix: keep recording on all ranks (harmless,
in-memory) but **only save when `trainer.is_global_zero`**. Single-node behavior
is unchanged (rank 0 is the only rank).

### 2. Scoped `_teardown` — `lightning/launchers.py`

`HydraDDP.teardown()` calls `_teardown()` unconditionally, which pops
`MASTER_ADDR/PORT`, `NODE_RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `PL_GLOBAL_SEED`.
Under SLURM/torchrun these are **scheduler-owned**; clearing them can interfere
with the allocation. Fix: `_teardown` clears only the env vars mushin actually
set itself. The single-node subprocess launcher (`_call_children_scripts`) sets
those vars, so track them (e.g. a module-level set populated when mushin sets
them, or a guard "only pop if mushin set it"); under an external launcher mushin
sets none, so `_teardown` becomes a no-op. Behavior preserved for the single-node
path; no scheduler vars touched under SLURM.

### 3. SLURM tasks↔ranks fail-fast validation — `lightning/launchers.py`

The #1 multi-node footgun: `ntasks_per_node` must equal `gpus_per_node`
(== Trainer `devices` per node), and `world_size == num_nodes × devices_per_node`.
A mismatch hangs at rendezvous (too few tasks), OOMs (too many), or silently runs
single-GPU (`ntasks_per_node == 1` with `devices=N`). Fix: in
`HydraDDP.setup_environment` (after `super().setup_environment()`, when running
under an external launcher), validate the cluster environment's world size /
per-node task count against `num_nodes` × `num_processes` and raise a legible
`RuntimeError` naming the mismatch and the fix, instead of hanging. Read the
counts from `self.cluster_environment` (world_size, local_rank/global_rank) and
`self.num_nodes` / `self.num_processes`.

### 4. `submitit_slurm_config(...)` helper — new, `lightning/` (ergonomics)

A thin hydra-zen config builder that returns a `hydra/launcher` config for
`hydra-submitit-launcher`, deriving `tasks_per_node` from `gpus_per_node` so the
two cannot desync (the foolproof half of the footgun). Signature roughly:
`submitit_slurm_config(*, nodes, gpus_per_node, cpus_per_task=..., partition=None, timeout_min=..., mem_gb=None, **extra)` →
a config with `tasks_per_node = gpus_per_node`, `nodes`, `gpus_per_node`, plus the
passed-through SLURM resources. It does NOT submit anything; the user sets it as
the launcher. Validate inputs (positive ints) and document that the cluster's
partition/account are the user's to supply.

### 5. `seed_everything_per_rank(base, workers=True)` helper — new, `lightning/`

Reproducibility: derive a deterministic per-global-rank seed so a 64-GPU run is
as reproducible as a 1-GPU run. Reads the global rank from the environment
(`RANK`/`SLURM_PROCID`, falling back to 0) and calls Lightning's
`seed_everything(base + global_rank, workers=workers)`. Unit-test the
seed-derivation logic (mock the env).

### 6. (Defensive) global-rank output subdir — `lightning/launchers.py`

`_subprocess_call` keys the per-rank Hydra output subdir on **local** rank
(`.pl_hydra_rank_{local_rank}`). This path only runs in the single-node
subprocess launcher, but for the rare multi-node-without-external-launcher case
two nodes' local-rank-1 collide on a shared FS. Cheap one-line fix: key on
**global rank** (`node_rank × num_processes + local_rank`). Low priority but
included since it's trivial and correct.

## Error handling

- The tasks↔ranks validation raises a clear `RuntimeError` (or `ValueError`) that
  names the observed vs expected counts and the one-line fix
  (`ntasks_per_node == gpus_per_node`).
- `submitit_slurm_config` rejects non-positive `nodes`/`gpus_per_node`.
- The rank-0 save guard and scoped teardown never raise; they no-op off rank 0 /
  when mushin set nothing.

## Testing

### Now — CI, no cluster (the implementation plan's tests)
- `MetricsCallback` saves only when `trainer.is_global_zero` (mock a fake trainer
  with `is_global_zero=False` → assert no file written; `True` → file written).
- `_teardown` clears only mushin-set vars; with scheduler-style vars present and
  mushin having set none, they are preserved (extends
  `tests/test_lightning_launchers.py`).
- tasks↔ranks validation: a fake `cluster_environment` whose world size /
  per-node count disagrees with `num_nodes × num_processes` → `setup_environment`
  raises the legible error; a matching layout passes.
- `submitit_slurm_config`: `tasks_per_node == gpus_per_node`, resources passed
  through, bad inputs rejected.
- `seed_everything_per_rank`: derives `base + global_rank` from the env (mock
  `RANK`/`SLURM_PROCID`).
- global-rank subdir: `_subprocess_call`/the helper computes
  `node_rank × num_processes + local_rank` (unit-test the computation).

### Cluster-gated — the MERGE GATE (you run this)
- A `@pytest.mark.cluster` (deselected by default) end-to-end smoke test: launch a
  2-node × N-GPU job via `submitit_slurm_config` + `SLURMEnvironment`, assert it
  completes, `metrics.pt` is written exactly once (rank 0), and results aggregate
  via `load_experiment`.
- A docs **runbook**: the exact submitit config + `srun`/sbatch invocation to run
  on the user's cluster, with the tasks↔ranks contract spelled out, so the user
  can validate before merge.

### Deferred (separate effort, also cluster-gated)
- c10d/`TCPStore` rendezvous launcher (replacing subprocess+sleep) for fault
  tolerance / node-failure recovery.

## Docs

- A "Multi-node training" guide: the `submitit_slurm_config` example, the
  `SLURMEnvironment` pairing, the tasks↔ranks contract and failure modes, the
  per-rank seeding helper, and the validation runbook.
- Changelog fragment.

## Files touched

- `src/mushin/lightning/callbacks.py` — rank-0 save guard.
- `src/mushin/lightning/launchers.py` — scoped `_teardown`, tasks↔ranks
  validation in `setup_environment`, global-rank output subdir.
- `src/mushin/lightning/` (new module, e.g. `_cluster.py`) —
  `submitit_slurm_config`, `seed_everything_per_rank`; exported via
  `mushin.lightning`/`mushin`.
- `tests/test_lightning_launchers.py`, `tests/test_lightning_callbacks.py`,
  new `tests/test_cluster.py` — unit tests; gated cluster smoke test.
- `docs/` — multi-node guide + runbook; `changes/` fragment.

## Out of scope

- Cross-node process spawning from scratch (delegated to submitit/SLURM/Elastic).
- The rendezvous rewrite (deferred).
- Non-SLURM schedulers beyond what `TorchElasticEnvironment` already covers.

## Open questions

None — resolved in the brainstorm. The only gate is real-cluster validation
before merge.
