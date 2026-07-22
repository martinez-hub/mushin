`run(..., cache_dir=...)` adds a content-addressed cache of completed cells that
is shared across `working_dir`s. A cell whose resolved config AND task source
match a previously-computed cell — keyed on the same fingerprints as `resume` —
reuses that result instead of recomputing, so a cell computed in one sweep is
free in another (a coarse grid refined into a finer one, a re-run in a fresh
directory). Newly-computed cells are stored in the cache automatically. It
complements `resume` (which reuses only within a single `working_dir`); a changed
non-swept config value or an edited task body is a cache miss. Cache writes are
best-effort and never fail the cell.
