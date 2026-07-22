Documented the sweep-analysis API. `mushin.show`, `mushin.best`, `mushin.diff`,
and `mushin.export.table` now have an API reference page (`reference/analysis`),
and the exploration-to-paper guide covers `max_total_seconds`, `notes=`, and
`tags=`. The workflows guide also gains a "Using mushin alongside a
hyperparameter search" section documenting the Optuna/Ax/Nevergrad → mushin
two-phase pattern (searcher finds configs; mushin runs the reproducible final
grid), with no new dependency.
