# Examples

Every example below is a **runnable, CI-tested** script in the
[`examples/`](https://github.com/martinez-hub/mushin/tree/main/examples) directory.
The guides embed pieces of them; this page indexes them all. Clone the repo and run
any one with:

```bash
uv run python examples/<name>.py
```

## Sweeps → datasets

| Example | What it shows |
|---|---|
| [`sweep_to_dataset.py`](https://github.com/martinez-hub/mushin/blob/main/examples/sweep_to_dataset.py) | The flagship flow: define `task(...)`, sweep with `multirun`, get results back as a labeled `xarray.Dataset`. |
| [`sklearn_sweep.py`](https://github.com/martinez-hub/mushin/blob/main/examples/sklearn_sweep.py) | The sweep layer is **framework-agnostic** — a scikit-learn `LogisticRegression` sweep (no torch) still returns a labeled dataset. |

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

## See also

- [Quickstart](quickstart.md) — the flagship example, run end-to-end.
- [Guides](guides/workflows.md) — workflows, compare, Study, resilience, and more.
