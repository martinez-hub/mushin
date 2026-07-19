# Examples

Every example below is a **runnable** script in the
[`examples/`](https://github.com/martinez-hub/mushin/tree/main/examples) directory —
CI-tested, except the multi-GPU scaling examples (which need real GPUs, so they run
on a cluster rather than in CI). The guides embed pieces of them; this page indexes
them all. Clone the repo and run any one with:

```bash
uv run python examples/<name>.py
```

## Sweeps → datasets

| Example | What it shows |
|---|---|
| [`sweep_to_dataset.py`](https://github.com/martinez-hub/mushin/blob/main/examples/sweep_to_dataset.py) | The flagship flow: define `task(...)`, sweep with `multirun`, get results back as a labeled `xarray.Dataset`. |
| [`sklearn_sweep.py`](https://github.com/martinez-hub/mushin/blob/main/examples/sklearn_sweep.py) | The sweep layer is **framework-agnostic** — a scikit-learn `LogisticRegression` sweep (no torch) still returns a labeled dataset. |
| [`parallel_sweep.py`](https://github.com/martinez-hub/mushin/blob/main/examples/parallel_sweep.py) | Submit a sweep **out-of-process** — `run(..., launcher="joblib")` runs cells across worker processes (needs `hydra-joblib-launcher`); the docstring shows the submitit/SLURM variant. |

## Compare & Study, with statistics

| Example | What it shows |
|---|---|
| [`compare_classifiers.py`](https://github.com/martinez-hub/mushin/blob/main/examples/compare_classifiers.py) | `compare(...)` two classifiers across seeds on MNIST with significance (`BenchmarkResult`). |
| [`study_mnist.py`](https://github.com/martinez-hub/mushin/blob/main/examples/study_mnist.py) | `Study` — a multi-seed training sweep routed straight into `compare`, in one call. |
| [`segmentation_demo.py`](https://github.com/martinez-hub/mushin/blob/main/examples/segmentation_demo.py) | `compare(task="segmentation")` on synthetic masks (mIoU, Dice, …). |
| [`compare_llms_demo.py`](https://github.com/martinez-hub/mushin/blob/main/examples/compare_llms_demo.py) | `llm.compare_llms` — compare LLM systems across reproducible seeds with significance. |

## Benchmark batteries

| Example | What it shows |
|---|---|
| [`batteries.py`](https://github.com/martinez-hub/mushin/blob/main/examples/batteries.py) | A runnable toy for **all 7** built-in batteries (classification, segmentation, detection, regression, retrieval, image_quality, audio). |

For the full per-battery walkthrough — real-model recipes (SAM 3.1, YOLO-World, CLIP,
…) alongside each of these toys — see the
[Built-in batteries guide](guides/batteries.md).

## Scaling across GPUs & nodes

These need real multi-GPU / multi-node hardware, so they run on a cluster rather
than in CI. See the linked guides for the full recipe and validation runbook.

| Example / guide | What it shows |
|---|---|
| [`sharding_fsdp_multirun.py`](https://github.com/martinez-hub/mushin/blob/main/examples/sharding_fsdp_multirun.py) | Shard one model across GPUs with `HydraFSDP` under a Hydra `--multirun` sweep (needs ≥2 GPUs). See the [Sharded training guide](guides/sharding.md). |
| [Multi-node training guide](guides/multinode.md) | `HydraDDP` + `submitit_slurm_config` across SLURM nodes — one process per GPU, with a fail-fast world-size guard. |
| [GPU packing guide](guides/packing.md) | `pin_gpu_round_robin` to co-locate small sweep jobs on shared GPUs (or Ray for fractional sharing). |

## See also

- [Quickstart](quickstart.md) — the flagship example, run end-to-end.
- [Guides](guides/workflows.md) — workflows, compare, Study, resilience, and more.
