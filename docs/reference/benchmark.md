# benchmark

Statistics over a sweep's dataset, and the evaluation tasks and metric
batteries that `compare` runs. Everything here needs the `eval` extra
(`pip install "mushin-py[eval]"`). The [Comparing methods](../guides/compare.md)
and [batteries](../guides/batteries.md) guides show them in use.

::: mushin.benchmark.compare_methods

::: mushin.benchmark.IncompleteSweepError

::: mushin.benchmark.compare

::: mushin.benchmark.BenchmarkResult

## Tasks

A task bundles a metric battery with the code that produces predictions for
it, under a name `compare(task=...)` accepts.

::: mushin.benchmark.Task

::: mushin.benchmark.register_task

::: mushin.benchmark.get_task

::: mushin.benchmark.list_tasks

## Batteries

Each battery is a factory for a `torchmetrics` collection. The detection,
image-quality and audio batteries need their own extras (`detection`, `image`,
`audio`), as their docstrings say.

::: mushin.benchmark.classification_battery

::: mushin.benchmark.regression_battery

::: mushin.benchmark.segmentation_battery

::: mushin.benchmark.detection_battery

::: mushin.benchmark.retrieval_battery

::: mushin.benchmark.image_quality_battery

::: mushin.benchmark.audio_battery
