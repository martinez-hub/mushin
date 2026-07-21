# workflows

::: mushin.sweep

## `mushin.multirun`

`multirun(values)` wraps a list (of ints, floats, bools, or strings) to mark it
as a **sweep axis**: Hydra runs one job per value and the values become an
`xarray` dimension. It is hydra-zen's
[`multirun`](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.launch.html)
re-exported; it behaves like a list (`multirun([1, 2, 3])`).

## `mushin.hydra_list`

`hydra_list(values)` wraps a list to pass it as a **single argument value** to
one job (no sweep) — use it when the task parameter itself takes a list.
Re-exported from hydra-zen.

::: mushin.workflows.BaseWorkflow

::: mushin.workflows.MultiRunMetricsWorkflow

::: mushin.workflows.RobustnessCurve
