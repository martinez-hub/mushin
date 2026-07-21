Reloading a sweep offline with `load_from_dir(dir, "mushin_metrics.json")` no
longer crashes. The default `metric_load_fn` now sniffs the file — it reads the
JSON metrics sidecar (written by any task that returns a dict, including
`@mushin.sweep`/decorator sweeps) with `json` and falls back to `torch.load`
only for torch pickle/`.pt` files. Previously it was hard-wired to `torch.load`,
which raised `UnpicklingError` on the JSON sidecar unless you overrode the loader.
