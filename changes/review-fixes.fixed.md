Fixed five defects found by an adversarial review of the exploration features:

- **`cache_dir`** is now resolved to an absolute path (`expanduser().resolve()`)
  before launching. A relative or `~`-prefixed `cache_dir` was previously
  dereferenced inside each per-cell job dir (Hydra chdirs into it), so
  cross-cell/cross-directory reuse silently never hit.
- **`resume=True` with `sample=`** is now rejected with a clear error. Combining
  them overwrote already-completed cells (skipped by job index) to `skipped`/NaN,
  silently discarding prior results; resume WITHOUT `sample` fills the rest.
- **The cell count** (used by `sample=`, `confirm_above=`/`MUSHIN_MAX_CELLS`, and
  `dry_run`) now counts sweep axes supplied via the raw `overrides=[...]` list,
  not only `param=multirun(...)` kwargs — so those no longer undercount the grid.
- **`compare_methods(allow_incomplete=True)`** now computes each comparison over
  only the seeds completed for both methods (dropping NaN cells), fulfilling its
  documented promise instead of feeding NaN into the tests and returning all-NaN
  statistics.
- **`mushin.show(sort=...)`** places NaN metric values last deterministically
  (they no longer break the sort's total order) and raises a clear error when
  `sort` names a column that isn't present.
