`mushin.show(root)` prints (and returns) a status/metrics table for a sweep
directory. It reads each cell's status and metrics sidecars directly — pure
JSON, so it needs neither Hydra nor xarray and works mid-sweep — making it handy
for watching a live sweep or eyeballing a finished one before the full
`to_xarray` load. Each row carries the cell's swept parameters, its status
(`completed`/`running`/`failed`/`skipped`/…), and its metric values; `metrics=`
restricts the metric columns and `sort=` orders the rows. The returned
`ShowResult` exposes `.rows` (one dict per cell) and `.table`.
