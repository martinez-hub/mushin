# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The `@mushin.sweep` decorator: a boilerplate-free entry point that turns a
plain `task`-style function into a runnable sweep, over `MultiRunMetricsWorkflow`."""

from __future__ import annotations

import functools

from .workflows import MultiRunMetricsWorkflow


class Sweep:
    """Handle returned by `@mushin.sweep`. Call `.run(...)` to run the sweep and
    get the labeled `xarray.Dataset`; drop to `.workflow` (last-run instance) or
    `.workflow_cls` (the synthesized class) for power features."""

    def __init__(self, fn, cls):
        functools.wraps(fn)(self)  # name/doc/__wrapped__ (copies fn's qualname)
        # Make `fn` picklable despite the decorator shadowing its module name:
        # re-point its qualname THROUGH this handle (findable at module.<name>) and
        # hang it here, so out-of-process launchers can serialize the task.
        fn.__qualname__ = fn.__qualname__ + ".__mushin_task__"
        self.__mushin_task__ = fn
        self.workflow_cls = cls
        self.workflow = None  # last-run instance (None before the first run)

    def run(self, **kwargs):
        """Run the sweep and return its labeled `xarray.Dataset`. Forwards every
        keyword to `MultiRunMetricsWorkflow.run` (sweep dims via `multirun(...)`
        plus `working_dir` / `on_error` / `resume` / `launcher` / … )."""
        wf = self.workflow_cls()  # fresh per run — no state carryover
        wf.run(**kwargs)
        self.workflow = wf
        return wf.to_xarray()


def sweep(fn):
    """Turn a plain `task(**params) -> dict` function into a runnable `Sweep`."""
    cls = type(fn.__name__, (MultiRunMetricsWorkflow,), {"task": staticmethod(fn)})
    cls.__module__ = fn.__module__
    cls.__qualname__ = fn.__qualname__
    return Sweep(fn, cls)
