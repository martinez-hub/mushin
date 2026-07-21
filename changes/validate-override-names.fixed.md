A typo'd sweep-axis name is now caught before launch instead of silently
producing a wrong dataset. Passing `run(lrate=multirun(...))` when the task
parameter is `lr` previously ran to completion and added a phantom `lrate`
dimension of constant values; it now raises a `ValueError` that names the likely
intended parameter. Only near-misses of a real target are rejected — deliberate
overrides beyond the task's own parameters (values consumed by `pre_task`,
config groups, interpolations) and tasks declaring `**kwargs` are unaffected.
