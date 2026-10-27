# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""resume=True and cache_dir= notice edits to code defined in a notebook.

A notebook has no file to compare. The record follows the names the task's
code looks up, through the objects the kernel holds now: a function's code
names the cell text of the version in force, a class is found through its
methods, and a plain constant is compared by its value (see
docs/superpowers/specs/2026-10-27-notebook-code-record-design.md).
"""

from __future__ import annotations

import hashlib
import json
import linecache
import os
import subprocess
import sys
import textwrap
import types
import warnings

import pytest

from mushin._resume import cell_project_record, project_fingerprints

KERNEL = "mushin_t_kernel"


class Kernel:
    """A kernel's namespace as IPython keeps it: each cell compiled under a
    name that linecache serves from memory (no mtime, as IPython writes it),
    run in a module that has no file."""

    def __init__(self, name: str = KERNEL, legacy: bool = False):
        self.module = types.ModuleType(name)
        sys.modules[name] = self.module
        self.ns = self.module.__dict__
        self.names: list[str] = []
        self.count = 0
        # IPython before 8.6 wrote an mtime, and kept its cells in its own
        # linecache._ipython_cache as well.
        self.legacy = legacy

    def run(self, text: str) -> None:
        self.count += 1
        text = textwrap.dedent(text)
        digest = hashlib.sha1(text.encode()).hexdigest()[:12]
        name = f"<ipython-input-{self.count}-{digest}>"
        if self.legacy:
            entry = (len(text), 1793100000.0, text.splitlines(True), name)
            linecache.__dict__.setdefault("_ipython_cache", {})[name] = entry
        else:
            entry = (len(text), None, text.splitlines(True), name)
        linecache.cache[name] = entry
        self.names.append(name)
        # As IPython compiles a cell: without this module's future flags
        # (`annotations` would hide what 3.14 generates for a class body).
        exec(compile(text, name, "exec", dont_inherit=True), self.ns)

    def close(self) -> None:
        sys.modules.pop(self.module.__name__, None)
        for name in self.names:
            linecache.cache.pop(name, None)
            linecache.__dict__.get("_ipython_cache", {}).pop(name, None)


@pytest.fixture
def kernel():
    k = Kernel()
    yield k
    k.close()


def _notebook(record) -> dict:
    assert record is not None
    return {k: v for k, v in record["code"].items() if k.endswith(" (notebook)")}


def _record(kernel, *names) -> dict:
    return _notebook(project_fingerprints(*(kernel.ns[n] for n in names)))


def _run_cells(kernel, cells) -> None:
    for cell in cells:
        try:
            kernel.run(cell)
        except Exception:  # noqa: BLE001 - a cell that raises, as in a kernel
            pass


TASK = "def task(a):\n    return helper(a)\n"

# Each case: the cells run first, then the cells run after. An EDIT changes
# the code a later call of `task` runs, so the record must change; a NOT_EDIT
# does not, so it must not.
NOTEBOOK_EDITS = {
    "a helper's body": (
        ["def helper(x):\n    return x * 2\n", TASK],
        ["def helper(x):\n    return x * 3\n"],
    ),
    "a constant a helper reads": (
        ["SCALE = 2\n\n\ndef helper(x):\n    return x * SCALE\n", TASK],
        ["SCALE = 3\n"],
    ),
    "a constant the task reads": (
        ["SCALE = 2\n\n\ndef task(a):\n    return a * SCALE\n"],
        ["SCALE = 3\n"],
    ),
    "a tuple constant": (
        ["LRS = (0.1, 0.2)\n\n\ndef task(a):\n    return a * LRS[1]\n"],
        ["LRS = (0.1, 0.3)\n"],
    ),
    "an int replaced by a bool": (
        ["FLAG = 1\n\n\ndef task(a):\n    return a * FLAG\n"],
        ["FLAG = True\n"],
    ),
    "a default captured from a global, its cell run again": (
        [
            "SCALE = 2\n",
            "def helper(x, s=SCALE):\n    return x * s\n",
            TASK,
        ],
        ["SCALE = 3\n", "def helper(x, s=SCALE):\n    return x * s\n"],
    ),
    "a class's method": (
        [
            "class Model:\n    def f(self, x):\n        return x * 2\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        ["class Model:\n    def f(self, x):\n        return x * 3\n"],
    ),
    "a class attribute": (
        [
            "class Model:\n    K = 2\n\n    def f(self, x):\n        return x * self.K\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        ["class Model:\n    K = 3\n\n    def f(self, x):\n        return x * self.K\n"],
    ),
    "a class attribute read from a global when the class ran": (
        [
            "SCALE = 2\n",
            "class Model:\n    K = SCALE\n\n    def f(self, x):\n        return x * self.K\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        [
            "SCALE = 3\n",
            "class Model:\n    K = SCALE\n\n    def f(self, x):\n        return x * self.K\n",
        ],
    ),
    "a dataclass's default": (
        [
            "import dataclasses\n\n\n@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.1\n",
            "def task(a):\n    return a * Cfg().lr\n",
        ],
        ["@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.2\n"],
    ),
    "a helper only a class's method reaches": (
        [
            "def helper(x):\n    return x * 2\n",
            "class Model:\n    def f(self, x):\n        return helper(x)\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        ["def helper(x):\n    return x * 3\n"],
    ),
    "an instance made again from an edited class": (
        [
            "class Model:\n    def f(self, x):\n        return x * 2\n\n\nmodel = Model()\n",
            "def task(a):\n    return model.f(a)\n",
        ],
        [
            "class Model:\n    def f(self, x):\n        return x * 3\n\n\nmodel = Model()\n"
        ],
    ),
    "a callable reached through a registry": (
        [
            "MODELS = {}\n\n\ndef register(fn):\n    MODELS[fn.__name__] = fn\n"
            "    return fn\n",
            "@register\ndef cnn(x):\n    return x * 2\n",
            "def task(a):\n    return MODELS['cnn'](a)\n",
        ],
        ["@register\ndef cnn(x):\n    return x * 3\n"],
    ),
    "a closure's captured value": (
        [
            "def make(k):\n    def helper(x):\n        return x * k\n\n    return helper\n",
            "helper = make(2)\n",
            TASK,
        ],
        ["helper = make(3)\n"],
    ),
    "a method calling super()": (
        [
            "class Base:\n    def f(self, x):\n        return x\n",
            "class Model(Base):\n    def f(self, x):\n        return super().f(x) * 2\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        [
            "class Model(Base):\n    def f(self, x):\n        return super().f(x) * 3\n",
        ],
    ),
    "a helper a library decorator wraps": (
        [
            "import contextlib\n\n\n@contextlib.contextmanager\ndef scaled(x):\n"
            "    yield x * 2\n",
            "def task(a):\n    with scaled(a) as v:\n        return v\n",
        ],
        ["@contextlib.contextmanager\ndef scaled(x):\n    yield x * 3\n"],
    ),
    "a decorator wrapping the task": (
        [
            "import functools\n\n\ndef twice(fn):\n    @functools.wraps(fn)\n"
            "    def inner(*a):\n        return fn(*a) * 2\n\n    return inner\n",
            "@twice\ndef task(a):\n    return a\n",
        ],
        [
            "def twice(fn):\n    @functools.wraps(fn)\n"
            "    def inner(*a):\n        return fn(*a) * 3\n\n    return inner\n",
            "@twice\ndef task(a):\n    return a\n",
        ],
    ),
    "a lambda helper": (
        ["act = lambda x: x * 2\n", "def task(a):\n    return act(a)\n"],
        ["act = lambda x: x * 3\n"],
    ),
    "an int named like a token count": (
        ["MAX_TOKENS = 256\n\n\ndef task(a):\n    return a * MAX_TOKENS\n"],
        ["MAX_TOKENS = 512\n"],
    ),
    "one handler of a closure dispatch table": (
        [
            "def make():\n    table = {}\n\n"
            + "".join(
                f"    def h{i}(x):\n        return table['h{(i + 1) % 12}'](x - 1) "
                f"if x > 0 else {i}\n\n"
                for i in range(12)
            )
            + "    table.update({"
            + ", ".join(f"'h{i}': h{i}" for i in range(12))
            + "})\n    return table\n",
            "TABLE = make()\n",
            "def task(a):\n    return TABLE['h0'](a)\n",
        ],
        [
            "def make():\n    table = {}\n\n"
            + "".join(
                f"    def h{i}(x):\n        return table['h{(i + 1) % 12}'](x - 1) "
                f"if x > 0 else {i if i != 5 else 50}\n\n"
                for i in range(12)
            )
            + "    table.update({"
            + ", ".join(f"'h{i}': h{i}" for i in range(12))
            + "})\n    return table\n",
            "TABLE = make()\n",
        ],
    ),
    "a decorated helper": (
        [
            "import functools\n\n\n@functools.lru_cache\ndef helper(x):\n"
            "    return x * 2\n",
            TASK,
        ],
        ["@functools.lru_cache\ndef helper(x):\n    return x * 3\n"],
    ),
}

NOTEBOOK_NOT_EDITS = {
    "an unchanged helper cell run again": (
        ["def helper(x):\n    return x * 2\n", TASK],
        ["def helper(x):\n    return x * 2\n"],
    ),
    "a comment, a docstring and formatting": (
        ["def helper(x):\n    return x * 2\n", TASK],
        ['def helper(x):\n    """Twice."""\n    # the scale\n    return x*2\n'],
    ),
    "a helper the task does not reach": (
        ["def helper(x):\n    return x * 2\n\n\ndef other(x):\n    return x\n", TASK],
        ["def other(x):\n    return x + 1\n"],
    ),
    "a cell that failed before it redefined the helper": (
        ["def helper(x):\n    return x * 2\n", TASK],
        ["undefined_name\n\n\ndef helper(x):\n    return x * 100\n"],
    ),
    "a task appending to a global list": (
        ["LOG = []\n\n\ndef task(a):\n    LOG.append(a)\n    return a\n"],
        ["task(1)\ntask(2)\n"],
    ),
    "a task counting its calls in a global": (
        [
            "CALLS = 0\n\n\ndef task(a):\n    global CALLS\n    CALLS += 1\n    return a\n"
        ],
        ["task(1)\ntask(2)\n"],
    ),
    "a global named like an attribute the task reads": (
        [
            "import types\n\ncfg = types.SimpleNamespace(lr=0.1)\nlr = 1\n",
            "def task(a):\n    return a * cfg.lr\n",
        ],
        ["lr = 2\n"],
    ),
    "a global a default was captured from, changed after": (
        ["SCALE = 2\n", "def helper(x, s=SCALE):\n    return x * s\n", TASK],
        ["SCALE = 3\n"],
    ),
    "an instance made before its class was edited": (
        [
            "class Model:\n    def f(self, x):\n        return x * 2\n\n\nmodel = Model()\n",
            "def task(a):\n    return model.f(a)\n",
        ],
        ["class Model:\n    def f(self, x):\n        return x * 3\n"],
    ),
    "a mutable config (documented: not compared)": (
        ["CFG = {'lr': 0.1}\n\n\ndef task(a):\n    return a * CFG['lr']\n"],
        ["CFG = {'lr': 0.2}\n"],
    ),
    "a task counting its calls in a class attribute": (
        [
            "class Counter:\n    n = 0\n",
            "def task(a):\n    Counter.n += 1\n    return a\n",
        ],
        ["task(1)\ntask(2)\n"],
    ),
    "a timing decorator keeping a nonlocal total": (
        [
            "import time\n\n\ndef timed(fn):\n    total = 0.0\n\n    def inner(*a):\n"
            "        nonlocal total\n        t = time.perf_counter()\n"
            "        out = fn(*a)\n        total += time.perf_counter() - t + 1.0\n"
            "        return out\n\n    return inner\n",
            "@timed\ndef helper(x):\n    return x * 2\n",
            TASK,
        ],
        ["task(1)\ntask(2)\n"],
    ),
    "a task resetting a helper's defaults": (
        [
            "def helper(x, k=2):\n    return x * k\n",
            "def task(a):\n    helper.__defaults__ = (a,)\n    return helper(a)\n",
        ],
        ["task(5)\n"],
    ),
    "an unchanged registration cell run again into a set": (
        [
            "HOOKS = set()\n\n\ndef hook(fn):\n    HOOKS.add(fn)\n    return fn\n",
            "@hook\ndef log(x):\n    return x\n",
            "def task(a):\n    return [h(a) for h in HOOKS]\n",
        ],
        ["@hook\ndef log(x):\n    return x\n"],
    ),
    "a module the task reads": (
        ["import math\n\n\ndef task(a):\n    return math.sqrt(a)\n"],
        ["import math\n"],
    ),
}


@pytest.mark.parametrize("case", sorted(NOTEBOOK_EDITS))
def test_a_notebook_edit_is_seen(kernel, case):
    before, after = NOTEBOOK_EDITS[case]
    _run_cells(kernel, before)
    first = _record(kernel, "task")
    assert first, "the notebook entries are empty"
    _run_cells(kernel, after)
    assert _record(kernel, "task") != first


@pytest.mark.parametrize("case", sorted(NOTEBOOK_NOT_EDITS))
def test_a_notebook_change_the_task_cannot_see_is_not_an_edit(kernel, case):
    before, after = NOTEBOOK_NOT_EDITS[case]
    _run_cells(kernel, before)
    first = _record(kernel, "task")
    _run_cells(kernel, after)
    assert _record(kernel, "task") == first


def test_a_restart_replaying_the_cells_gives_the_same_record():
    """A new kernel running the same cells, in other cell numbers: the cell
    names are not in the record."""
    cells = [
        "import dataclasses\nSCALE = 2\nNAMES = frozenset({'a', 'b', 'c'})\n",
        "def helper(x, s=SCALE):\n    return x * s * len(NAMES)\n",
        "@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.1\n",
        "class Model:\n    K = 2\n\n    def f(self, x):\n        return helper(x) * self.K\n",
        "class Sub(Model):\n    def f(self, x):\n        return super().f(x) + 1\n",
        "def task(a):\n    return Sub().f(a) * Cfg().lr\n",
    ]
    first = Kernel()
    try:
        first.run("print('a cell before')\n")
        _run_cells(first, cells)
        one = _record(first, "task")
    finally:
        first.close()
    second = Kernel()
    try:
        _run_cells(second, cells)
        two = _record(second, "task")
    finally:
        second.close()
    assert one == two
    assert set(one) == {
        "task (notebook)",
        "Sub (notebook)",
        "Cfg (notebook)",
        "helper (notebook)",
        "NAMES (notebook)",
    }


def test_the_entries_name_what_the_task_reaches(kernel):
    _run_cells(
        kernel,
        [
            "SCALE = 2\nUNUSED = 5\n\n\ndef helper(x):\n    return x * SCALE\n",
            "def task(a):\n    return helper(a)\n",
        ],
    )
    assert set(_record(kernel, "task")) == {
        "task (notebook)",
        "helper (notebook)",
        "SCALE (notebook)",
    }


def test_an_inline_lambda_task_does_not_record_the_grid(kernel):
    """A lambda passed to the call that also holds the grid: only the lambda
    is its definition."""
    kernel.run("SCALE = 2\nF, GRID = (lambda a: a * SCALE), [1, 2]\n")
    one = _record(kernel, "F")
    kernel.run("SCALE = 2\nF, GRID = (lambda a: a * SCALE), [1, 2, 3]\n")
    assert _record(kernel, "F") == one
    assert "SCALE (notebook)" in one
    kernel.run("SCALE = 2\nF, GRID = (lambda a: a * SCALE * 10), [1, 2, 3]\n")
    assert _record(kernel, "F")["<lambda> (notebook)"] != one["<lambda> (notebook)"]


def test_a_value_past_the_bound_is_not_compared(kernel):
    kernel.run("BIG = tuple(range(20000))\n\n\ndef task(a):\n    return BIG[a]\n")
    assert "BIG (notebook)" not in _record(kernel, "task")


def test_a_large_data_global_is_walked_once(kernel, monkeypatch):
    """A container is walked for the notebook functions it holds within one
    bound shared by the whole walk, each container once: an aliased board
    costs its distinct lists, not their product."""
    import mushin._resume as resume

    seen = []
    walk = resume._container_items

    def counting(value):
        items = walk(value)
        seen.append(len(items))
        return items

    monkeypatch.setattr(resume, "_container_items", counting)
    kernel.run(
        "BOARD = [[0] * 3000] * 3000\n\n\ndef task(a):\n    return BOARD[a][0]\n"
    )
    _record(kernel, "task")
    assert sum(seen) <= 2 * 3000 + 10


def test_a_registry_past_the_bound_gives_no_entry(kernel, monkeypatch):
    """Which callables a cut walk finds depends on a set's order, which is per
    process: past the bound the entry is left out, not taken from a part."""
    import mushin._resume as resume

    monkeypatch.setattr(resume, "_CONTAINER_BUDGET", 5)
    kernel.run(
        "".join(f"def h{i}(x):\n    return x + {i}\n\n\n" for i in range(20))
        + "HOOKS = [h0, h1, {"
        + ", ".join(f"h{i}" for i in range(2, 20))
        + "}]\n\n\ndef task(a):\n    return [h(a) for h in HOOKS[:2]] + [h0(a), h1(a)]\n"
    )
    record = _record(kernel, "task")
    assert "task (notebook)" in record and "HOOKS (notebook)" not in record


def test_too_many_objects_give_no_notebook_entries(kernel, monkeypatch):
    """Past the object bound, which objects were taken depends on the walk's
    order: no notebook entry is recorded rather than a part of them."""
    import mushin._resume as resume

    monkeypatch.setattr(resume, "_NOTEBOOK_MAX_OBJECTS", 3)
    _run_cells(
        kernel,
        [
            "".join(f"def h{i}(x):\n    return x + {i}\n\n\n" for i in range(10)),
            "def task(a):\n    return "
            + " + ".join(f"h{i}(a)" for i in range(10))
            + "\n",
        ],
    )
    assert _record(kernel, "task") == {}


def test_ipython_before_8_6_keeps_its_cells_too():
    """IPython 7 to 8.5 wrote an mtime into a cell's linecache entry and kept
    the cell in linecache._ipython_cache: its cells are read the same."""
    legacy = Kernel(legacy=True)
    try:
        _run_cells(legacy, ["def helper(x):\n    return x * 2\n", TASK])
        one = _record(legacy, "task")
        legacy.run("def helper(x):\n    return x * 3\n")
        assert "helper (notebook)" in one
        assert _record(legacy, "task")["helper (notebook)"] != one["helper (notebook)"]
    finally:
        legacy.close()


def test_ipythons_own_names_are_not_compared(kernel):
    kernel.run("_ = 1\n_i3 = 'x'\n\n\ndef task(a):\n    return (_, _i3, a)\n")
    record = _record(kernel, "task")
    assert "_ (notebook)" not in record and "_i3 (notebook)" not in record


def test_a_function_whose_text_is_not_kept_is_not_compared(kernel):
    """`exec` of a string: there is no cell text to compare (as on main)."""
    exec("def helper(x):\n    return x * 2\n", kernel.ns)
    kernel.run(TASK)
    record = _record(kernel, "task")
    assert "helper (notebook)" not in record and "task (notebook)" in record


def test_a_value_that_looks_secret_is_not_compared(kernel):
    """A digest of a short string alone can confirm a guess of it: a name or
    a value the provenance record redacts is left out."""
    kernel.run(
        "DB_PASSWORD = 'hunter2'\nENDPOINT = 'hf_' + 'a' * 12\nNAME = 'resnet'\n\n\n"
        "def task(a):\n    return (DB_PASSWORD, ENDPOINT, NAME, a)\n"
    )
    record = _record(kernel, "task")
    assert "DB_PASSWORD (notebook)" not in record
    assert "ENDPOINT (notebook)" not in record
    assert "NAME (notebook)" in record


@pytest.mark.parametrize(
    "cells",
    [
        # A password read at run time, captured as a default under an
        # innocent parameter name, or under the global's own name.
        [
            "DB_PASSWORD = ''.join(['hun', 'ter{k}'])\n",
            "def task(a, password=DB_PASSWORD):\n    return a\n",
        ],
        [
            "DB_PASSWORD = ''.join(['hun', 'ter{k}'])\n",
            "def task(a, pw=DB_PASSWORD):\n    return a\n",
        ],
        # A provider token in a closure.
        [
            "def make(t):\n    def task(a):\n        return (t, a)\n\n    return task\n",
            "task = make('hf_' + 'abcdefgh{k}')\n",
        ],
        # A class attribute bound from a secret.
        [
            "DB_PASSWORD = ''.join(['hun', 'ter{k}'])\n",
            "class Conn:\n    pw = DB_PASSWORD\n\n    def get(self):\n        return self.pw\n",
            "def task(a):\n    return Conn().get()\n",
        ],
    ],
)
def test_a_captured_secret_is_not_compared(kernel, cells):
    _run_cells(kernel, [c.format(k=2) for c in cells])
    one = _record(kernel, "task")
    _run_cells(kernel, [c.format(k=3) for c in cells])
    assert _record(kernel, "task") == one


def test_a_function_from_a_file_is_not_compared(kernel, tmp_path):
    """A file run into the notebook (`%run`): the file can change after it
    ran, so its text is not the code in force."""
    path = tmp_path / "helpers.py"
    path.write_text("def helper(x):\n    return x * 2\n")
    exec(compile(path.read_text(), str(path), "exec"), kernel.ns)
    linecache.getlines(str(path))  # as a traceback or a warning from it would
    kernel.run(TASK)
    record = _record(kernel, "task")
    path.write_text("def helper(x):\n    return x * 3\n")
    assert "helper (notebook)" not in record
    assert _record(kernel, "task") == record


def test_the_record_does_not_resolve_a_lazy_object(kernel):
    """Objects are inspected by type: a proxy the task reads stays unresolved."""
    kernel.run(
        "RESOLVED = []\n\n\nclass Proxy:\n    def __getattr__(self, name):\n"
        "        RESOLVED.append(name)\n        raise AttributeError(name)\n\n"
        "    @property\n    def __class__(self):\n        RESOLVED.append('__class__')\n"
        "        return Proxy\n\n\nDATA = Proxy()\n\n\ndef task(a):\n    return DATA\n"
    )
    _record(kernel, "task")
    assert kernel.ns["RESOLVED"] == []


def test_a_notebook_target_is_followed(kernel):
    """A config `_target_` naming a class defined in the notebook, as a serial
    run instantiates it."""
    _run_cells(
        kernel,
        [
            "class Model:\n    def f(self, x):\n        return x * 2\n",
            "def task(a):\n    return a\n",
        ],
    )
    cfg = {"model": {"_target_": f"{KERNEL}.Model"}}
    one = _notebook(cell_project_record("t1", (kernel.ns["task"],), (), cfg))
    assert "Model (notebook)" in one
    kernel.run("class Model:\n    def f(self, x):\n        return x * 3\n")
    two = _notebook(cell_project_record("t2", (kernel.ns["task"],), (), cfg))
    assert two["Model (notebook)"] != one["Model (notebook)"]


def test_a_notebook_main_target_is_followed(kernel, monkeypatch):
    """`builds(Cfg)` on a dataclass defined in a cell names `__main__.Cfg`:
    a kernel's `__main__` is the notebook, a class without methods too."""
    _run_cells(
        kernel,
        [
            "import dataclasses\n\n\n@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.1\n",
            "def task(a):\n    return a\n",
        ],
    )
    monkeypatch.setitem(sys.modules, "__main__", kernel.module)
    cfg = {"cfg": {"_target_": "__main__.Cfg"}}
    one = _notebook(cell_project_record("m1", (kernel.ns["task"],), (), cfg))
    kernel.run("@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.2\n")
    two = _notebook(cell_project_record("m2", (kernel.ns["task"],), (), cfg))
    assert one["Cfg (notebook)"] != two["Cfg (notebook)"]


def test_a_target_in_a_module_without_a_file_keeps_mains_entries(tmp_path, monkeypatch):
    """An extension's submodule has no file (pybind11's def_submodule): its
    target is not notebook code, and its package's file is recorded as on
    main. A builtin is neither."""
    from mushin._resume import _record as record_of

    root = tmp_path / "proj"
    (root / "mushin_t_extpkg").mkdir(parents=True)
    (root / "mushin_t_extpkg" / "__init__.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(root))
    import importlib

    pkg = importlib.import_module("mushin_t_extpkg")
    ext = types.ModuleType("mushin_t_extpkg.ext")
    ext.op = len  # a builtin, as an extension's function is
    monkeypatch.setitem(sys.modules, "mushin_t_extpkg.ext", ext)
    monkeypatch.setattr(pkg, "ext", ext, raising=False)
    try:
        code = record_of((), ["mushin_t_extpkg.ext.op"])["code"]
        assert "mushin_t_extpkg (file)" in code
        assert not [k for k in code if k.endswith(" (notebook)")]
        builtin = record_of((), ["builtins.dict"])["code"]
        assert not [k for k in builtin if k.endswith(" (notebook)")]
    finally:
        sys.modules.pop("mushin_t_extpkg", None)


def test_the_driver_ships_the_notebook_entries(kernel):
    """A worker gets the task by value, without the kernel's cell text: the
    driver's entries, shipped with the run, are the ones its cells record."""
    from mushin._resume import driver_script_entries

    _run_cells(kernel, ["def helper(x):\n    return x * 2\n", TASK])
    shipped = driver_script_entries((kernel.ns["task"],))
    taken = _record(kernel, "task")
    kernel.run("def helper(x):\n    return x * 3\n")
    # The run started before the edit: its cells record what it shipped.
    assert _notebook(project_fingerprints(kernel.ns["task"], scripts=shipped)) == taken


# One kernel holding most of the rule's constructs, pinned to one digest on
# every supported Python: the record is compared across versions.
CROSS_VERSION_CELLS = [
    "import dataclasses\nimport functools\n\nSCALE = 2\nLRS = (0.1, 0.2)\n"
    "NAMES = frozenset({'a', 'b', 'c'})\nFLAG = True\nNOTE = b'x'\n",
    "def helper(x, s=SCALE):\n    '''Doc.'''\n    return x * s * len(NAMES)\n",
    "def make(k):\n    def inner(x):\n        return x * k\n\n    return inner\n\n\n"
    "triple = make(3)\n",
    "@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.1\n",
    "class Model:\n    K = SCALE\n\n    def f(self, x):\n        return helper(x) * self.K\n",
    "@functools.lru_cache\ndef cached(x):\n    return x + LRS[0]\n",
    "model = Model()\n",
    "def task(a):\n    return model.f(a) * Cfg().lr + triple(a) + cached(a) + FLAG"
    " + len(NOTE)\n",
]
CROSS_VERSION_DIGEST = "8d626dc73fd4efee"


def test_the_record_reads_alike_on_every_supported_python(kernel):
    _run_cells(kernel, CROSS_VERSION_CELLS)
    record = _record(kernel, "task")
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    assert digest[:16] == CROSS_VERSION_DIGEST, record


def test_the_record_does_not_depend_on_the_hash_seed(tmp_path):
    """A frozenset of strings iterates in an order set by PYTHONHASHSEED."""
    script = tmp_path / "seeded.py"
    script.write_text(
        "import json, sys\n"
        f"sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
        "from test_resume_notebook import Kernel, _record, CROSS_VERSION_CELLS, _run_cells\n"
        "k = Kernel()\n"
        "_run_cells(k, CROSS_VERSION_CELLS)\n"
        "print(json.dumps(_record(k, 'task'), sort_keys=True))\n"
    )
    outs = {
        subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("0", "1", "2", "3")
    }
    assert len(outs) == 1, outs


# --- end to end: what a resume does with the record -------------------------------

SWEEP_CELLS = [
    "import mushin\nfrom mushin import multirun\n\nCALLS = []\nSCALE = 2\n",
    "def helper(x):\n    return x * SCALE\n",
    "@mushin.sweep\ndef task(a):\n    CALLS.append(a)\n    return {'v': helper(a)}\n",
]


def _sweep(kernel, wd, grid, **kwargs):
    kernel.ns["_wd"], kernel.ns["_grid"], kernel.ns["_kw"] = str(wd), grid, kwargs
    kernel.run("ds = task.run(a=multirun(_grid), working_dir=_wd, **_kw)\n")
    return kernel.ns["ds"]


def test_a_resume_after_a_notebook_edit_reruns_and_returns_the_new_values(
    kernel, tmp_path
):
    """The invariant: each cell's value is the one the code in force computes.
    A resume after an edit re-runs the cells, one after no edit reuses them,
    and a longer grid runs only the new cell."""
    _run_cells(kernel, SWEEP_CELLS)
    wd = tmp_path / "wd"
    ds = _sweep(kernel, wd, [1, 2])
    assert list(ds["v"].values) == [2, 4] and kernel.ns["CALLS"] == [1, 2]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = _sweep(kernel, wd, [1, 2, 3], resume=True)
    assert not [w for w in caught if "code the task reaches" in str(w.message)]
    assert list(ds["v"].values) == [2, 4, 6] and kernel.ns["CALLS"] == [1, 2, 3]

    kernel.run("def helper(x):\n    return x * SCALE * 10\n")
    with pytest.warns(UserWarning, match=r"helper \(notebook\)"):
        ds = _sweep(kernel, wd, [1, 2, 3], resume=True)
    assert list(ds["v"].values) == [20, 40, 60]

    kernel.run("SCALE = 3\n")
    with pytest.warns(UserWarning, match=r"SCALE \(notebook\)"):
        ds = _sweep(kernel, wd, [1, 2, 3], resume=True)
    assert list(ds["v"].values) == [30, 60, 90]


def test_cache_dir_keys_on_the_notebook_code(kernel, tmp_path):
    _run_cells(kernel, SWEEP_CELLS)
    cache = str(tmp_path / "cache")
    _sweep(kernel, tmp_path / "one", [1, 2], cache_dir=cache)
    assert kernel.ns["CALLS"] == [1, 2]
    _sweep(kernel, tmp_path / "two", [1, 2], cache_dir=cache)
    assert kernel.ns["CALLS"] == [1, 2]  # reused
    kernel.run("SCALE = 5\n")
    ds = _sweep(kernel, tmp_path / "three", [1, 2], cache_dir=cache)
    assert kernel.ns["CALLS"] == [1, 2, 1, 2]
    assert list(ds["v"].values) == [5, 10]


# --- a real IPython shell and a real kernel ---------------------------------------

IPYTHON_SCRIPT = """
import json
from IPython.core.interactiveshell import InteractiveShell
from mushin._resume import project_fingerprints

shell = InteractiveShell.instance()


def record():
    rec = project_fingerprints(shell.user_ns["task"])["code"]
    return {k: v for k, v in rec.items() if k.endswith(" (notebook)")}


for cell in CELLS:
    shell.run_cell(cell, store_history=True)
one = record()
shell.run_cell("def helper(x):\\n    return x * 2\\n", store_history=True)
same = record()
shell.run_cell("def helper(x):\\n    return x * 3\\n", store_history=True)
print(json.dumps([one, same, record()]))
"""


def test_an_ipython_shell_resolves_its_cells(tmp_path):
    pytest.importorskip("IPython")
    cells = ["SCALE = 2\n", "def helper(x):\n    return x * 2\n", TASK]
    script = tmp_path / "shell.py"
    script.write_text(f"CELLS = {cells!r}\n" + IPYTHON_SCRIPT)
    out = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    ).stdout
    one, same, edited = json.loads(out.strip().splitlines()[-1])
    assert "helper (notebook)" in one and one == same
    assert edited["helper (notebook)"] != one["helper (notebook)"]


def _kernel_session():
    pytest.importorskip("ipykernel")
    manager = pytest.importorskip("jupyter_client.manager")
    km, kc = manager.start_new_kernel(kernel_name="python3")

    def run(code: str) -> str:
        msg_id = kc.execute(code)
        out = []
        while True:
            msg = kc.get_iopub_msg(timeout=120)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            if msg["msg_type"] == "stream":
                out.append(msg["content"]["text"])
            elif msg["msg_type"] == "error":
                raise AssertionError("\n".join(msg["content"]["traceback"]))
            elif (
                msg["msg_type"] == "status"
                and msg["content"]["execution_state"] == "idle"
            ):
                break
        kc.get_shell_msg(timeout=120)
        return "".join(out)

    def close():
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    return run, close


KERNEL_RECORD = (
    "import json\nfrom mushin._resume import project_fingerprints\n"
    "print(json.dumps({k: v for k, v in "
    "project_fingerprints(task)['code'].items() if k.endswith(' (notebook)')}))\n"
)


def test_a_real_kernel_restart_and_edit():
    """Cell files in a real kernel are temp paths named by their content; a
    new kernel replaying the cells records the same, and an edit differs."""
    cells = ["SCALE = 2\n", "def helper(x):\n    return x * SCALE\n", TASK]
    records = []
    for edit in (None, None, "def helper(x):\n    return x * SCALE * 2\n"):
        run, close = _kernel_session()
        try:
            for cell in cells:
                run(cell)
            if edit:
                run(edit)
            records.append(json.loads(run(KERNEL_RECORD).strip().splitlines()[-1]))
        finally:
            close()
    assert records[0] == records[1]
    assert "helper (notebook)" in records[0]
    assert records[2]["helper (notebook)"] != records[0]["helper (notebook)"]


def test_a_real_kernel_resume_under_joblib(tmp_path):
    """A loky worker gets the task by value and has no cell text: the cells it
    runs record the driver's entries, so an edit re-runs them there too (the
    re-run warning is the worker's, so the values and records show it)."""
    pytest.importorskip("hydra_plugins.hydra_joblib_launcher")
    from mushin._resume import STATUS_FILE

    run, close = _kernel_session()
    wd = str(tmp_path / "wd")
    sweep = (
        f"ds = task.run(a=multirun([1, 2]), working_dir={wd!r}, "
        "launcher='joblib', resume=RESUME)\n"
        "print(json.dumps([list(map(float, ds['v'].values))]))\n"
    )
    try:
        run(
            "import json\nimport mushin\nfrom mushin import multirun\n\nSCALE = 2\n"
            "\n\ndef helper(x):\n    return x * SCALE\n\n\n"
            "@mushin.sweep\ndef task(a):\n    return {'v': helper(a)}\n"
        )
        first = json.loads(run("RESUME = False\n" + sweep).strip().splitlines()[-1])
        assert first[0] == [2.0, 4.0]
        run("def helper(x):\n    return x * SCALE * 10\n")
        again = json.loads(run("RESUME = True\n" + sweep).strip().splitlines()[-1])
        assert again[0] == [20.0, 40.0]
    finally:
        close()
    records = [
        json.loads(p.read_text()).get("code_deps") for p in tmp_path.rglob(STATUS_FILE)
    ]
    assert records and all("helper (notebook)" in r["code"] for r in records), records
