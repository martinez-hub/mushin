# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""The `@mushin.sweep` decorator: a boilerplate-free entry point that turns a
plain `task`-style function into a runnable sweep, over `MultiRunMetricsWorkflow`."""

from __future__ import annotations

import functools
import types

from .workflows import MultiRunMetricsWorkflow


def _mangled_task_copy(fn):
    """Return a COPY of `fn` whose qualname is re-pointed through the Sweep handle
    (``<name>.__mushin_task__``). The copy is used as the synthesized workflow's
    ``task`` so out-of-process launchers can pickle it by reference
    (``module.<name>`` is the handle, whose ``__mushin_task__`` is this copy). The
    caller's original ``fn`` is left untouched — its qualname, repr, and its own
    picklability are unchanged."""
    task = types.FunctionType(
        fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    task.__dict__.update(fn.__dict__)
    task.__kwdefaults__ = fn.__kwdefaults__
    task.__annotations__ = dict(fn.__annotations__)
    task.__module__ = fn.__module__
    task.__doc__ = fn.__doc__
    task.__qualname__ = fn.__qualname__ + ".__mushin_task__"
    return task


class Sweep:
    """Handle returned by `@mushin.sweep`. Call `.run(...)` to run the sweep and
    get the labeled `xarray.Dataset`; drop to `.workflow` (last-run instance) or
    `.workflow_cls` (the synthesized class) for power features."""

    def __init__(self, fn, task, cls):
        functools.wraps(fn)(self)  # handle mirrors the ORIGINAL fn (clean name/doc)
        # `task` is a copy of `fn` with a qualname that resolves back through this
        # handle: pickle finds `module.<name>.__mushin_task__` -> this copy, so
        # out-of-process launchers can serialize the task. `fn` itself is pristine.
        self.__mushin_task__ = task
        self.workflow_cls = cls
        self.workflow = None  # last-run instance (None before the first run)

    def run(self, **kwargs):
        """Run the sweep and return its labeled `xarray.Dataset`. Forwards every
        keyword to `MultiRunMetricsWorkflow.run` (sweep dims via `multirun(...)`
        plus `working_dir` / `on_error` / `resume` / `launcher` / … ).

        With `dry_run=True` no jobs launch: the grid-preview summary is returned
        (there is no dataset to build)."""
        wf = self.workflow_cls()  # fresh per run — no state carryover
        result = wf.run(**kwargs)
        self.workflow = wf
        if kwargs.get("dry_run"):
            return result  # the dry-run summary; nothing ran, no dataset
        return wf.to_xarray()


def sweep(fn):
    """Turn a plain `task(**params) -> dict` function into a runnable `Sweep`."""
    task = _mangled_task_copy(fn)  # never mutates the caller's `fn`
    cls = type(fn.__name__, (MultiRunMetricsWorkflow,), {"task": staticmethod(task)})
    cls.__module__ = fn.__module__
    cls.__qualname__ = fn.__qualname__
    return Sweep(fn, task, cls)
