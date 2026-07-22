`run(..., sample=K)` runs only a random `K`-cell subset of the grid — the rest
are skipped (NaN, and listed in `self.skipped`) — for fast exploration of a large
grid without paying for every cell. The subset is chosen deterministically from
the Hydra job indices (seeded by `sample_seed`, default 0), so it is reproducible
and identical across a resume; resuming *without* `sample` fills in the remaining
cells. `sample >= n_cells` runs everything. Because selection is by job index it
is launcher- and axis-type-agnostic. Note: the full grid is still composed by
Hydra (only the sampled cells run), and a sampled sweep is intentionally
incomplete (`is_complete` is False).
