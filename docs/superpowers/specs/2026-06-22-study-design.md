# Study (train-sweep → compare composition layer) — Design

*Date: 2026-06-22*

## Goal

Close the composition gap found by dogfooding ([#19](https://github.com/martinez-hub/mushin/issues/19)):
mushin can run a multi-seed training sweep (via `MultiRunMetricsWorkflow` +
Hydra) and can compare trained models (via `mushin.benchmark.compare`), but the
two don't connect — the user manually saves, reloads, and regroups checkpoints
between them. `Study` is the connective tissue that makes
**define → train-sweep → evaluate → report** a single motion, while delegating
training (Lightning/Hydra) and evaluation (`compare`) to the existing pieces.

This is squarely the *owned evaluate+report* spine reaching down to orchestrate
the *inherited* define/execute layers — see
[`2026-06-22-mushin-positioning-design.md`](2026-06-22-mushin-positioning-design.md).

## Design decisions

| Element | Decision |
| --- | --- |
| Orchestration | `Study` wraps `MultiRunMetricsWorkflow` to run the method×seed training sweep through Hydra (inherits launchers), then routes results into `compare`. |
| Model handoff | Checkpoint-based. Training jobs may run in separate processes/machines, so models can't be returned as live objects; each job saves a checkpoint and `Study` loads it afterward. (Also gives reproducibility for free.) |
| Train contract | `train_fn(seed) -> checkpoint_path` (saves a checkpoint, returns its path). |
| Load contract | User-provided `load_fn(path) -> model` (framework-agnostic; Lightning users pass `LitModule.load_from_checkpoint`). |
| Two entry points | (1) full motion from `train_fn`s; (2) `Study.from_checkpoints(...)` to load + compare existing checkpoints with no training. Both share one load+compare core. |
| Launcher (v1) | Local (in-process). Slurm/Ray work via Hydra plugins **when `train_fn`/`load_fn` are importable** — documented, not v1-tested. |
| Significance | Folds in [#17](https://github.com/martinez-hub/mushin/issues/17): warn when the chosen `test` cannot reach `alpha` for the given seed count. |
| Task type | Classification only (inherited from `compare` v1). |

## Owned vs. delegated (north-star check)

- **Owned by `Study`:** the orchestration glue — the method×seed sweep wiring,
  recovering checkpoint paths keyed by `(method, seed)`, loading + regrouping,
  and the call into `compare`.
- **Delegated:** training loop → the user's `train_fn` (typically Lightning);
  sweep execution → Hydra; metrics → torchmetrics (via `compare`); stats → scipy
  (via `compare`).

`Study` adds **no** training, metric, or statistics logic of its own.

## Public API

```python
from mushin import Study

# 1. Full motion: train sweep + compare
study = Study(
    methods={"cnn": train_cnn, "mlp": train_mlp},   # train_fn(seed) -> checkpoint_path
    load_fn=LitClassifier.load_from_checkpoint,       # path -> model
    seeds=[0, 1, 2],
    data=test_loader,                                  # eval data for compare
    task="classification",
    num_classes=10,
    test="welch",
    working_dir=None,                                  # where the sweep runs
)
result = study.run()         # -> BenchmarkResult
study.checkpoints            # {method: [checkpoint_path_per_seed]}
study.working_dir

# 2. Eval-only: load existing checkpoints + compare (no training)
study = Study.from_checkpoints(
    checkpoints={"cnn": ["cnn_0.ckpt", "cnn_1.ckpt", "cnn_2.ckpt"],
                 "mlp": ["mlp_0.ckpt", "mlp_1.ckpt", "mlp_2.ckpt"]},
    load_fn=LitClassifier.load_from_checkpoint,
    data=test_loader,
    task="classification",
    num_classes=10,
    test="welch",
)
result = study.run()         # -> BenchmarkResult
```

`train_fn(seed: int) -> str | Path` trains one model for one seed and returns the
path to its saved checkpoint. It may close over its own training data (resolving
the "data as a module global" friction from #19). `load_fn(path) -> model`
reconstructs a model from a checkpoint.

## Data flow

**Full motion (`run()` with `methods`):**
1. Build an internal `MultiRunMetricsWorkflow` whose task, for `(method, seed)`,
   calls `methods[method](seed)` and records the returned checkpoint path.
2. `wf.run(method=multirun(method_names), seed=multirun(seeds))` — Hydra runs the
   jobs (local launcher in v1).
3. Recover the checkpoint paths keyed by `(method, seed)` from the sweep results
   into `{method: [path_per_seed]}` (stored as `study.checkpoints`).
4. Call the shared load+compare core.

**Eval-only (`from_checkpoints`):** skip steps 1–3; the `{method: paths}` map is
provided directly, then call the shared core.

**Shared load+compare core:** for each method, `load_fn` each path → a list of
models; assemble `{method: [models]}`; call
`compare(methods=..., data=data, task=task, num_classes=num_classes, test=test)`
→ `BenchmarkResult`.

## Components (file structure)

`src/mushin/study/`:

- `_sweep.py` — `run_training_sweep(methods, seeds, working_dir) ->
  dict[str, list[str]]`: build/run the internal workflow and return checkpoint
  paths per method (ordered by seed). Train path only.
- `_load.py` — `evaluate_checkpoints(checkpoints, load_fn, data, task,
  num_classes, test, alpha) -> BenchmarkResult`: the shared core (load → regroup
  → `compare`). Also where the #17 underpowered-test warning is emitted.
- `_study.py` — the `Study` class: `__init__` (full) and `from_checkpoints`
  classmethod, `run()`, and the `checkpoints` / `working_dir` / result attributes.
- `__init__.py` — exports `Study`.
- `src/mushin/__init__.py` — re-export `Study` for `from mushin import Study`.

## Significance warning (#17 fold-in)

In the shared core, before/after calling `compare`, check whether the chosen
`test` can reach `alpha` at the given seed count (e.g. Wilcoxon's minimum
achievable two-sided p-value with `n` paired seeds). If not, emit a
`UserWarning` naming the shortfall, e.g.:

> "test='wilcoxon' cannot reach alpha=0.05 with 3 seeds (min achievable
> p=0.25); use >=6 seeds or a parametric test (e.g. test='welch')."

This is the minimal, honest fix for #17 surfaced through `Study`; a fuller
power-aware default can follow in `compare` itself.

## Return value and state

`run()` returns the `BenchmarkResult` from `compare`. The `Study` instance also
exposes:
- `checkpoints` → `{method: [path_per_seed]}` (lists ordered by seed),
- `working_dir` → the sweep's working directory (full motion; `None` for
  eval-only).

## Error handling

- A `train_fn` that returns a missing/None path, or a method whose seed count
  doesn't match → `ValueError` naming the `(method, seed)`.
- `load_fn` raising on a checkpoint → propagate with the offending path in the
  message.
- Ragged checkpoint counts across methods → caught by `compare`'s existing
  validation (#12), surfaced clearly.

## Testing strategy

- **`_sweep`**: a trivial `train_fn(seed)` that writes a dummy file and returns
  its path; assert `run_training_sweep` returns the correct
  `{method: [path_per_seed]}` mapping for a 2-method × 2-seed sweep (local
  launcher, in a temp dir via the `cleandir` fixture).
- **`_load`**: save two tiny `torch.nn.Linear` checkpoints per method (via
  `torch.save` + a matching `load_fn`); assert `evaluate_checkpoints` returns a
  `BenchmarkResult` with dims `(method, seed)` and the expected metrics; assert
  the #17 warning fires for an underpowered `(test, n)` combo.
- **`Study` e2e (no real data)**: two trivial `train_fn`s that save tiny models
  (one deliberately better on a synthetic loader), a `load_fn`, `Study(...).run()`
  → assert `BenchmarkResult`, dims `(method, seed)`, better method present; and
  `Study.from_checkpoints(...).run()` on the same saved checkpoints → equivalent
  result. CPU, fast, no MNIST/CIFAR download.

## Scope / non-goals (v1)

- **No new training, metric, or statistics logic** — all delegated.
- **Local launcher only is tested.** Distributed (Slurm/Ray) is supported insofar
  as Hydra + importable callables allow, but is documented, not v1-tested.
- **Classification only**, inheriting `compare`'s v1 task support; the `task=`
  seam carries through.
- No hyperparameter search, no multi-dataset studies, no result persistence
  beyond the checkpoints the user's `train_fn` writes.
