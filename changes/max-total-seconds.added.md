`run(..., max_total_seconds=T)` adds a graceful wall-clock budget. Once the
budget is exhausted the remaining grid cells are skipped — recorded `'skipped'`
in the manifest and per-cell status, NaN in the dataset, and surfaced in
`self.skipped` (and a `mushin_skipped` dataset attr) — instead of the sweep
running to the end. The clock starts at the first computed cell, so at least one
cell always runs and resume cache hits don't consume the budget; a cell already
running is never interrupted. Because a skipped cell is not `completed`, a later
`resume=True` with more time finishes exactly the skipped cells. The budget is
measured per launcher process, so it's best paired with the default sequential
launcher.
