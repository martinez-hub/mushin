Public task API: `Task` dataclass plus `register_task`, `get_task`, and
`list_tasks` make evaluation tasks first-class and reusable. `compare(...)` and
`Study(...)` now accept either a `Task` object or a registered task name, and the
built-in batteries (`classification_battery`, `segmentation_battery`,
`detection_battery`) are exported from `mushin`.
