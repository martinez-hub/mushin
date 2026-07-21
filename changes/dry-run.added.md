`run(..., dry_run=True)` (and `@mushin.sweep` `.run(dry_run=True)`) previews a
sweep instead of launching it: it prints the cell count and each swept axis with
its values — so a range typo shows up as an unexpectedly wide axis before any
compute is spent — and returns a summary dict (`num_cells`, `axes`, `fixed`,
`working_dir`) with no jobs run.
