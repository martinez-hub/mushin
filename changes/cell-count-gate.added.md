A pre-launch cell-count gate guards against an accidentally huge grid.
`run(..., confirm_above=N)` refuses to launch a sweep with more than `N` cells,
raising a `ValueError` that reports the count and how to proceed (preview with
`dry_run=True`, or raise the ceiling). The `MUSHIN_MAX_CELLS` environment
variable supplies a default ceiling when a call doesn't set one — so a shared
cluster or CI profile can cap every sweep — and an explicit `confirm_above=`
wins over it. A malformed `MUSHIN_MAX_CELLS` is ignored, and `dry_run=True`
bypasses the gate so an over-limit sweep can still be previewed.
