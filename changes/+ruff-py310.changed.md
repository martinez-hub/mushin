Modernized the codebase to Python 3.10+ idioms now that 3.9 is no longer
supported: `ruff` `target-version` is `py310`, and the pyupgrade auto-fixes
(`Optional[X]`/`Union[X, Y]` -> `X | None`/`X | Y`) plus explicit `zip(..., strict=True)`
have been applied. No behavior change.
