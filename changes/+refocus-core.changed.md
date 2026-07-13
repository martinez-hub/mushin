`import mushin` is now lightweight: the `benchmark` and `llm` subsystems load on
first use instead of at import time, so a bare import no longer pulls the
battery/eval machinery. Every existing top-level name still resolves. The default
Hydra config/job name is now `mushin_workflow` (was `rai_workflow`), and the new
`mushin.original_cwd()` helper anchors relative paths in `task()` against the
launch directory rather than Hydra's per-job output directory.
