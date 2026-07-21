`mushin.best(root, metric)` returns the completed cell that optimizes a metric
across a sweep — `mode="max"` (default) or `mode="min"`. Like `mushin.show` it
reads the per-cell sidecars directly (offline, no Hydra/xarray). The returned
`BestResult` carries the winning `combo` (swept params), the optimized `value`,
the full `metrics` dict, the cell `status`, and its job `dir` (for locating
checkpoints/artifacts). Failed/running/skipped cells and non-finite metric
values are ignored; an unknown metric raises a `ValueError` that lists the
available scalar metrics.
