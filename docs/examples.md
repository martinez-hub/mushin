# Examples

Every example below is a **runnable** script in the
[`examples/`](https://github.com/martinez-hub/mushin/tree/main/examples) directory —
CI-tested, except the multi-GPU scaling examples (which need real GPUs, so they run
on a cluster rather than in CI). The guides embed pieces of them; this page indexes
them all. Clone the repo and run any one with:

```bash
uv run python examples/<name>.py
```

!!! note "Some examples need the `eval` extra"
    The comparison/Study/battery/LLM examples (`compare_classifiers`,
    `study_mnist`, `segmentation_demo`, `compare_llms_demo`, `batteries`,
    `inspect_ai_compare`, `prompt_injection_eval`, `llm_prompt_sweep`) use
    mushin's optional evaluation layer. On a plain `pip install mushin-py` they
    raise an install hint — run them with `pip install "mushin-py[eval]"`
    (`uv run` in this repo already includes it). `batteries.py` additionally
    wants the `detection`/`image`/`audio` extras for those batteries.

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

| Example | What it shows |
|---|---|
| [`gpu_packing.py`](https://github.com/martinez-hub/mushin/blob/main/examples/gpu_packing.py) | Pack several small sweep jobs onto each GPU with `pin_gpu_round_robin` (needs ≥2 GPUs + `hydra-joblib-launcher`). See the [GPU packing guide](guides/packing.md). |
| [`multinode_ddp.py`](https://github.com/martinez-hub/mushin/blob/main/examples/multinode_ddp.py) | Multi-node DDP with `HydraDDP` + `submitit_slurm_config` — one process per GPU across SLURM nodes, with a fail-fast world-size guard. See the [Multi-node training guide](guides/multinode.md). |
| [`sharding_fsdp_multirun.py`](https://github.com/martinez-hub/mushin/blob/main/examples/sharding_fsdp_multirun.py) | Shard one model across GPUs with `HydraFSDP` under a Hydra `--multirun` sweep (needs ≥2 GPUs). See the [Sharded training guide](guides/sharding.md). |

## See also

- [Quickstart](quickstart.md) — the flagship example, run end-to-end.
- [Guides](guides/workflows.md) — workflows, compare, Study, resilience, and more.

### `inspect_ai_compare.py` — is that eval gap real?

[Inspect AI](https://inspect.aisi.org.uk) tells you gpt-4 scored 60% and claude
scored 56.7%. This tells you whether the gap survives two checks: would a re-run
agree, and would a different set of questions agree.

```bash
python examples/inspect_ai_compare.py --demo      # two scenarios, known ground truth
python examples/inspect_ai_compare.py logs/*.eval # your own Inspect logs
```

Needs `pip install "mushin-py[eval]"`; reading real logs additionally needs
`pip install inspect-ai`.

### `prompt_injection_eval.py` — is your model actually more injection-resistant?

Measures resistance to **indirect** prompt injection: the attack arrives inside
content the model retrieves (a memo, a page, a tool result), not from the user.
Detection uses an inert canary — if the sentinel appears in the output, the
retrieved text steered the model — so the payloads measure the control failure
without carrying a real one.

The metric is the **complement of attack success rate** — the mean is `1 - ASR`,
so higher is better, the opposite of the convention most of the literature uses.

Resistance is binary per document, which is exactly the shape where a headline
score misleads: attacks vary enormously in difficulty, so a different attack set
can reverse the ranking. The example reports whether a gap survives a re-run and a different set of
attacks, and prints a per-attack breakdown. In the demo the "better" system is
15 points *worse* on one attack shape, and — because five attack shapes are five
grouped observations, not forty independent ones — the aggregate gap does not
actually reach significance once `clusters=` accounts for the grouping.

Five clusters is also few enough that mushin warns the interval is unreliable,
and the demo prints that warning rather than hiding it: the fix for a too-narrow
interval is more attack *shapes*, not more topics wrapped around the same five.

```bash
python examples/prompt_injection_eval.py --demo
```

Point it at your own systems (baseline vs. hardened, or two providers) to compare
mitigations with an honest uncertainty on the difference.

### `llm_prompt_sweep.py` — tuning a prompt without paying twice

Sweeps prompt template x temperature x seed and reads the winner off a labelled
dataset. The point is the sweep machinery, not the statistics: transient API
failures become NaN cells instead of discarding the run (`on_error="nan"`), a
re-run reuses what already completed (`resume=True`), and the result is keyed by
the parameters you swept so picking the best configuration is one reduction.

```bash
pip install "mushin-py[eval]"   # the closing significance step needs the extra
python examples/llm_prompt_sweep.py --demo
```

In the demo 8 of 60 cells fail transiently; a strict harness would have discarded
the other 52 completed cells — 2,080 paid calls — along with them. Runs
identically every time (no keys, no network), which is the point: `resume` can
only reuse a cell whose value is reproducible. The demo prints mushin's own
`8 run(s) failed … set to NaN` report rather than filtering it out — a partial
sweep should be impossible to mistake for a complete one.
