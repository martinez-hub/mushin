`run()` now validates up front that every required task (and `pre_task`)
parameter is satisfied — by the base config, an override, or the raw
`overrides=[...]` list — and raises a clear `ValueError` naming the missing
parameter(s) before launching. Previously a genuinely missing parameter surfaced
as an opaque per-job Hydra `ConfigAttributeError` after the sweep had already
started, once per cell. A parameter supplied only via an override is still valid
(the common decorator case, where the base config is empty), so this adds no
false positives.
