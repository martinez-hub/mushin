`compare_methods` now refuses a sweep that has **skipped/never-run** cells, not
just failed ones. Previously it keyed only on `mushin_failures`, so a `sample=`
subset or a `max_total_seconds`-limited sweep — both of which leave NaN cells —
would compute statistics silently over a partial grid. It now also keys on the
`mushin_skipped` completeness signal and raises `IncompleteSweepError`. A new
`allow_incomplete=True` argument bypasses the guard (with a warning) to compute
stats over only the completed cells, for exploratory analysis of a partial
sweep. As before, the guard keys on the completeness signals (attrs), never on
raw NaN values, so a metric that is legitimately NaN for other reasons does not
trigger it.
