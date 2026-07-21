`mushin.diff(a, b)` compares two sweep directories. It aligns cells by their
swept-parameter combination and, for each shared cell, reports the delta
`b - a` of every metric that is a finite scalar in both — printed as a `Δmetric`
table. Cells present in only one sweep are listed (`only_in_a` / `only_in_b`),
and the two runs' environment provenance is diffed field-by-field
(git/packages/python/accelerator), excluding volatile fields like the per-cell
timestamp. Like `show`/`best` it reads the per-cell sidecars directly (offline,
no Hydra/xarray); the returned `DiffResult` exposes `.rows`, `.only_in_a`,
`.only_in_b`, `.provenance`, and `.table`.
