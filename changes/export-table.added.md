`mushin.export.table(root)` exports a sweep as CSV — one row per cell with its
swept parameters, `status`, and metrics. It reads the per-cell sidecars directly
(offline, no Hydra/xarray) and writes full-precision values so pandas parses
numbers numerically. Returns the CSV string, or writes to `path=` and returns the
`Path`; `metrics=` restricts the metric columns. A durable, spreadsheet-friendly
substrate for researchers who prefer pandas to xarray.
