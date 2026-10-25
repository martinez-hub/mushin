# benchmark

Statistics over a sweep's dataset, and the evaluation tasks and metric
batteries that `compare` runs. Everything here needs the `eval` extra
(`pip install "mushin-py[eval]"`). The [Comparing methods](../guides/compare.md)
guide shows `compare_methods` and `compare` in use; the
[batteries guide](../guides/batteries.md) shows the batteries and the task
registry.

::: mushin.benchmark.compare_methods

::: mushin.benchmark.IncompleteSweepError

::: mushin.benchmark.compare.compare

::: mushin.benchmark.BenchmarkResult

## Tasks

A task bundles a metric battery with the code that produces predictions for
it, under a name `compare(task=...)` accepts.

::: mushin.benchmark.Task
    options:
      heading_level: 3

::: mushin.benchmark.register_task
    options:
      heading_level: 3

::: mushin.benchmark.get_task
    options:
      heading_level: 3

::: mushin.benchmark.list_tasks
    options:
      heading_level: 3

## Batteries

Each battery returns a dict of named `torchmetrics` metrics. The detection,
image-quality and audio batteries need their own extras (`detection`, `image`,
`audio`), as their docstrings say.

::: mushin.benchmark.classification_battery
    options:
      heading_level: 3

::: mushin.benchmark.regression_battery
    options:
      heading_level: 3

::: mushin.benchmark.segmentation_battery
    options:
      heading_level: 3

::: mushin.benchmark.detection_battery
    options:
      heading_level: 3

::: mushin.benchmark.retrieval_battery
    options:
      heading_level: 3

::: mushin.benchmark.image_quality_battery
    options:
      heading_level: 3

::: mushin.benchmark.audio_battery
    options:
      heading_level: 3
