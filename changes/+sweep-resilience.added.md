Sweep resilience and provenance: `run(on_error="nan")` records failed grid cells as
NaN and keeps going (default stays `"raise"`); `run(working_dir=..., resume=True)`
re-runs only the failed/missing cells and fills them in place; `compare`/`Study`
refuse statistics on an incomplete sweep (`IncompleteSweepError`) until you resume;
every run writes per-job provenance (`mushin_provenance.json`: git SHA, versions,
config) with an opt-in `capture_env=True` full dependency snapshot. `Study` accepts
`on_error`/`resume`/`capture_env` for training sweeps.
