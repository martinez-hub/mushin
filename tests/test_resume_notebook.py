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
import subprocess
import sys
import textwrap
import types
import warnings

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
    "one function of a ring of closures": (
        [
            "def make():\n"
            + "".join(
                f"    def h{i}(x):\n        return h{(i + 1) % 12}(x - 1) "
                f"+ h{(i + 2) % 12}(x - 2) if x > 0 else {i}\n\n"
                for i in range(12)
            )
            + "    return h0\n",
            "task = make()\n",
        ],
        [
            "def make():\n"
            + "".join(
                f"    def h{i}(x):\n        return h{(i + 1) % 12}(x - 1) "
                f"+ h{(i + 2) % 12}(x - 2) if x > 0 else {i if i != 5 else 50}\n\n"
                for i in range(12)
            )
            + "    return h0\n",
            "task = make()\n",
        ],
    ),
    "a helper a closure captures": (
        [
            "def make():\n    def inner(x):\n        return x * 2\n\n"
            "    def outer(x):\n        return inner(x) + 1\n\n    return outer\n",
            "task = make()\n",
        ],
        [
            "def make():\n    def inner(x):\n        return x * 3\n\n"
            "    def outer(x):\n        return inner(x) + 1\n\n    return outer\n",
            "task = make()\n",
        ],
    ),
    "a method reached through a bound-method alias": (
        [
            "class Model:\n    def f(self, x):\n        return x * 2\n\n\nscore = Model().f\n",
            "def task(a):\n    return score(a)\n",
        ],
        [
            "class Model:\n    def f(self, x):\n        return x * 3\n\n\nscore = Model().f\n"
        ],
    ),
    "a helper reached through a partial": (
        [
            "import functools\n\n\ndef scale(x, k):\n    return x * k\n\n\n"
            "helper = functools.partial(scale, k=2)\ndel scale\n",
            TASK,
        ],
        [
            "def scale(x, k):\n    return x * k * 10\n\n\n"
            "helper = functools.partial(scale, k=2)\ndel scale\n",
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
    "a lazily set nonlocal implementation": (
        [
            "def make():\n    impl = None\n\n    def helper(x):\n        nonlocal impl\n"
            "        if impl is None:\n            impl = lambda v: v * 2\n"
            "        return impl(x)\n\n    return helper\n\n\nhelper = make()\n",
            TASK,
        ],
        ["task(1)\ntask(2)\n"],
    ),
    "a writer held in a partial": (
        [
            "import functools\n\nBEST = 0\n\n\ndef set_best(x):\n    global BEST\n"
            "    BEST = max(BEST, x)\n\n\nRECORD = functools.partial(set_best)\n",
            "def task(a):\n    RECORD(a)\n    return a + BEST\n",
        ],
        ["task(5)\ntask(7)\n"],
    ),
    "a registry the task fills with closures": (
        [
            "FNS = {}\n\n\ndef make_fn(k):\n    def fn(x):\n        return x * k\n\n"
            "    return fn\n",
            "def task(a):\n    FNS.setdefault(a, make_fn(a))\n    return FNS[a](a)\n",
        ],
        ["task(1)\ntask(2)\n"],
    ),
    "a task setting a helper's keyword defaults": (
        [
            "def helper(x, *, k=2):\n    return x * k\n",
            "def task(a):\n    helper.__kwdefaults__ = {'k': a}\n    return helper(a)\n",
        ],
        ["task(5)\n"],
    ),
    "a task setting a closure cell's contents": (
        [
            "def make(k):\n    def helper(x):\n        return x * k\n\n    return helper\n"
            "\n\nhelper = make(2)\n",
            "def task(a):\n    helper.__closure__[0].cell_contents = a\n"
            "    return helper(a)\n",
        ],
        ["task(5)\n"],
    ),
    "an equal frozenset built in another order": (
        ["NUMS = frozenset([1, 9, 17])\n\n\ndef task(a):\n    return a in NUMS\n"],
        ["NUMS = frozenset([17, 9, 1])\n"],
    ),
    "a class attribute captured from a global (documented: not compared)": (
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
    "a method-less dataclass's default (documented: not compared)": (
        [
            "import dataclasses\n\n\n@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.1\n",
            "def task(a):\n    return a * Cfg().lr\n",
        ],
        ["@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.2\n"],
    ),
    "a callable reached only through a container (documented)": (
        [
            "MODELS = {}\n\n\ndef register(fn):\n    MODELS[fn.__name__] = fn\n"
            "    return fn\n",
            "@register\ndef cnn(x):\n    return x * 2\n",
            "def task(a):\n    return MODELS['cnn'](a)\n",
        ],
        ["@register\ndef cnn(x):\n    return x * 3\n"],
    ),
    "a hook in a tuple default writing a global": (
        [
            "N = 0\n\n\ndef bump(a):\n    global N\n    N += a\n",
            "def task(a, hooks=(bump,)):\n    for h in hooks:\n        h(a)\n"
            "    return N\n",
        ],
        ["task(1)\ntask(2)\n"],
    ),
    "a global written by a hook in a captured list": (
        [
            "N = 0\n\n\ndef bump(a):\n    global N\n    N += a\n",
            "def make(hooks):\n    def task(a):\n        for h in hooks:\n            h(a)\n"
            "        return N\n\n    return task\n\n\ntask = make([bump])\n",
        ],
        ["task(1)\ntask(2)\n"],
    ),
    "a global written through a bound-method alias": (
        [
            "best = 0.0\n\n\nclass Tracker:\n    def update(self, v):\n"
            "        global best\n        best = max(best, v)\n\n\n"
            "record_best = Tracker().update\n",
            "def task(a):\n    record_best(a)\n    return a + best\n",
        ],
        ["task(5)\ntask(7)\n"],
    ),
    "a global written through an instance default": (
        [
            "best = 0.0\n\n\nclass Tracker:\n    def update(self, v):\n"
            "        global best\n        best = max(best, v)\n",
            "def task(a, tracker=Tracker()):\n    tracker.update(a)\n    return a + best\n",
        ],
        ["task(5)\ntask(7)\n"],
    ),
    "a closure caching lambdas in a captured dict": (
        [
            "def make():\n    cache = {}\n\n    def helper(x):\n        if x not in cache:\n"
            "            cache[x] = lambda v: v * x\n        return cache[x](x)\n\n"
            "    return helper\n\n\nhelper = make()\n",
            TASK,
        ],
        ["task(1)\ntask(2)\n"],
    ),
    "an output that held a writer, cleared": (
        [
            "N = 1\n\n\ndef setter():\n    global N\n    N = 5\n\n\n_7 = setter\ndel setter\n",
            "def task(a):\n    return a + N\n",
        ],
        ["del _7\n"],
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
        "import dataclasses\nSCALE = 2\nNAMES = frozenset({1.5, 2, 3})\n",
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


def test_too_many_objects_give_no_notebook_entries(kernel, monkeypatch):
    """Past the object bound, which objects were taken depends on the walk's
    order: no notebook entry is recorded rather than a part of them."""
    import mushin._resume as resume

    monkeypatch.setattr(resume, "_NOTEBOOK_MAX_OBJECTS", 3)
    monkeypatch.setattr(resume, "_NOTEBOOK_TRUNCATION_WARNED", [])
    _run_cells(
        kernel,
        [
            "".join(f"def h{i}(x):\n    return x + {i}\n\n\n" for i in range(10)),
            "def task(a):\n    return "
            + " + ".join(f"h{i}(a)" for i in range(10))
            + "\n",
        ],
    )
    with pytest.warns(UserWarning, match="larger than the record covers"):
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


@settings(max_examples=200)
@given(
    st.integers(min_value=1, max_value=8).flatmap(
        lambda n: st.lists(
            st.lists(st.integers(min_value=0, max_value=n - 1), max_size=n),
            min_size=n,
            max_size=n,
        )
    )
)
def test_components_match_reachability(edges):
    """Two nodes share a component exactly when each reaches the other, and
    an edge between components points to one closed earlier."""
    from mushin._resume import _components

    children = dict(enumerate(edges))
    comp_of = _components(children)

    def reach(a):
        seen, todo = {a}, [a]
        while todo:
            for c in children[todo.pop()]:
                if c not in seen:
                    seen.add(c)
                    todo.append(c)
        return seen

    reached = {a: reach(a) for a in children}
    for a in children:
        for b in children:
            assert (comp_of[a] == comp_of[b]) == (b in reached[a] and a in reached[b])
        for c in children[a]:
            assert comp_of[c] <= comp_of[a]


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


@pytest.mark.parametrize(
    "cells",
    [
        # Strings, under any name and from any source: a digest of one alone,
        # or of a known definition with it, can confirm a guess of a secret.
        ["PW = ''.join(['hun', 'ter{k}'])\n", "def task(a):\n    return (PW, a)\n"],
        [
            "ENV = {{'PW': ''.join(['hun', 'ter{k}'])}}\n",
            "def task(a, pw=ENV['PW']):\n    return a\n",
        ],
        [
            "def make(t):\n    def task(a):\n        return (t, a)\n\n    return task\n",
            "task = make(''.join(['abc', 'defgh{k}']))\n",
        ],
        [
            "SECRET = ''.join(['hun', 'ter{k}'])\n",
            "import dataclasses\n\n\n@dataclasses.dataclass\nclass DBConfig:\n"
            "    user: str = 'admin'\n    pw: str = SECRET\n",
            "def task(a):\n    return DBConfig().user\n",
        ],
        [
            "SECRET = ''.join(['hun', 'ter{k}'])\n",
            "task = lambda a, pw=SECRET: sorted([a], key=lambda r: r)\n",
        ],
        # A string constant: compared only through the code that reads it.
        [
            "MODEL = ''.join(['resnet', '{k}'])\n",
            "def task(a):\n    return (MODEL, a)\n",
        ],
        # A number whose name, or whose parameter's, says it is a secret.
        ["DB_PASSWORD = 123{k}\n", "def task(a):\n    return DB_PASSWORD + a\n"],
        ["PIN = 123{k}\n", "def task(a, password=PIN):\n    return a\n"],
        ["PIN = 482{k}\n", "def task(a):\n    return a + PIN\n"],
    ],
)
def test_a_string_or_a_secret_number_is_never_compared(kernel, cells):
    _run_cells(kernel, [c.format(k=2) for c in cells])
    one = _record(kernel, "task")
    assert one  # the task itself is recorded
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


def test_a_notebook_target_is_not_compared(kernel):
    """A config `_target_` naming a class defined in the notebook gives no
    entry, as on main: the task's record cannot see what the task changes
    in it, and hydra-zen's `builds` puts the class's defaults in the config,
    which is compared."""
    _run_cells(
        kernel,
        [
            "class Model:\n    def f(self, x):\n        return x * 2\n",
            "def task(a):\n    return a\n",
        ],
    )
    cfg = {"model": {"_target_": f"{KERNEL}.Model"}}
    record = _notebook(cell_project_record("t1", (kernel.ns["task"],), (), cfg))
    assert "Model (notebook)" not in record and "task (notebook)" in record


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
    "NAMES = frozenset({1.5, 2, 3})\nFLAG = True\nNOTE = b'x'\n",
    "def helper(x, s=SCALE):\n    '''Doc.'''\n    return x * s * len(NAMES)\n",
    "def make(k):\n    def inner(x):\n        return x * k\n\n    return inner\n\n\n"
    "triple = make(3)\n",
    "@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.1\n",
    "class Model:\n    K = SCALE\n\n    def f(self, x):\n        return helper(x) * self.K\n",
    "@functools.lru_cache\ndef cached(x):\n    return x + LRS[0]\n",
    "model = Model()\n",
    # `k` is a cell of `scaled` on 3.11 (its comprehension is a scope) and a
    # plain local on 3.12+: neither is a rebind of `triple`'s own `k`.
    "def scaled(a):\n    k = 2\n    return sum([triple(x) * k for x in range(a)])\n",
    # A walrus in a comprehension (a store to the comprehension's free variable
    # on 3.11), and a comprehension variable shadowing a captured name that a
    # lambda captures (a store to the function's free variable on 3.12+):
    # neither is a `nonlocal` rebind.
    "def walrus(a):\n    return [(k := x) for x in range(a)]\n",
    "def make2(i):\n    def inner(x):\n        fs = [lambda: i for i in range(x)]\n"
    "        return len(fs) + i\n\n    return inner\n\n\nshadow = make2(3)\n",
    "def task(a):\n    return model.f(a) * Cfg().lr + scaled(a) + cached(a) + FLAG"
    " + len(NOTE) + len(walrus(a)) + shadow(a)\n",
]
CROSS_VERSION_DIGEST = "46d45ec92c9cdde2"


def test_the_record_reads_alike_on_every_supported_python(kernel):
    _run_cells(kernel, CROSS_VERSION_CELLS)
    record = _record(kernel, "task")
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    assert digest[:16] == CROSS_VERSION_DIGEST, record


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
