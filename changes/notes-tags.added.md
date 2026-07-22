`run(..., notes="...", tags=[...])` annotates a sweep with free-form lineage. The
note and tags are recorded in the sweep manifest, exposed on the workflow as
`wf.notes` / `wf.tags` (and preserved when a sweep is reloaded with
`load_from_dir`), and carried on the dataset as the `mushin_notes` /
`mushin_tags` attrs — so "why did I run this?" travels with the results. `tags`
must be a list of strings and `notes` a string. A resume that does not re-pass
`notes`/`tags` preserves the original run's lineage rather than wiping it.
