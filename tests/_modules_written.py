# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""An autouse fixture for test files that write a module and then import it
or look it up. Import it into the test module (pytest collects fixtures from
the module's namespace)."""

import importlib
import pathlib

import pytest


@pytest.fixture(autouse=True)
def _new_modules_are_found(monkeypatch):
    """The import system caches each directory's listing and rescans it only
    when the directory's mtime moves, which on Windows can lag a file written
    a moment ago, so the module is not found. ``importlib.invalidate_caches``
    is the documented remedy; every ``.py`` written calls it."""
    write_text = pathlib.Path.write_text

    def write_and_invalidate(self, *args, **kwargs):
        n = write_text(self, *args, **kwargs)
        if self.suffix == ".py":
            importlib.invalidate_caches()
        return n

    monkeypatch.setattr(pathlib.Path, "write_text", write_and_invalidate)
