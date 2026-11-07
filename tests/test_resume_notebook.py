# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""resume=True and cache_dir= notice edits to code defined in a notebook.

A notebook has no file to compare. The record follows the names the task's
code looks up, through the objects the kernel holds now, and compares their
definitions: a function's code names the cell text of the version in force,
and a class is found through its methods. No value is compared (see
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

import mushin._resume
from mushin._resume import cell_project_record, project_fingerprints

#: The scan of the kept cells' imports, kept here so a fixture can turn it
#: back on (``getattr``: the file also runs against an older ``_resume``).
KEPT_CELL_READING = getattr(mushin._resume, "_kept_cell_reading", None)

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
    # A class with no method of its own: its statement is found through the
    # kept cells (and, on 3.14+, a dataclass's `__annotate_func__`).
    "a method-less dataclass's default": (
        [
            "import dataclasses\n\n\n@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.1\n",
            "def task(a):\n    return a * Cfg().lr\n",
        ],
        ["@dataclasses.dataclass\nclass Cfg:\n    lr: float = 0.2\n"],
    ),
    "an Enum member": (
        [
            "import enum\n\n\nclass Color(enum.Enum):\n    RED = 1\n",
            "def task(a):\n    return a + Color.RED.value\n",
        ],
        ["class Color(enum.Enum):\n    RED = 2\n"],
    ),
    "a method-less class's attribute": (
        ["class Config:\n    LR = 0.1\n", "def task(a):\n    return a * Config.LR\n"],
        ["class Config:\n    LR = 0.2\n"],
    ),
    "a method-less class a factory makes": (
        [
            "def make():\n    class C:\n        K = 2\n\n    return C\n\n\nC = make()\n",
            "def task(a):\n    return a * C.K\n",
        ],
        ["def make():\n    class C:\n        K = 3\n\n    return C\n\n\nC = make()\n"],
    ),
    # Every method wrapped by a notebook decorator that keeps no `__wrapped__`:
    # the wrapper's code is the decorator's cell, so no method locates it.
    "a class whose every method a notebook decorator wraps": (
        [
            "def deco(fn):\n    def wrapper(*a):\n        return fn(*a)\n\n    return wrapper\n",
            "class Model:\n    @deco\n    def f(self, x):\n        return x * 2\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        ["class Model:\n    @deco\n    def f(self, x):\n        return x * 3\n"],
    ),
    "a helper's body": (
        ["def helper(x):\n    return x * 2\n", TASK],
        ["def helper(x):\n    return x * 3\n"],
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
    "a helper a notebook decorator wraps": (
        [
            "import functools\n\n\ndef logged(fn):\n    @functools.wraps(fn)\n"
            "    def inner(*a):\n        return fn(*a)\n\n    return inner\n",
            "@logged\ndef helper(x):\n    return x * 2\n",
            TASK,
        ],
        ["@logged\ndef helper(x):\n    return x * 3\n"],
    ),
    "a method a bound method calls through self": (
        [
            "class Model:\n    def f(self, x):\n        return self.g(x)\n\n"
            "    def g(self, x):\n        return x * 2\n\n\nscore = Model().f\n",
            "def task(a):\n    return score(a)\n",
        ],
        [
            "class Model:\n    def f(self, x):\n        return self.g(x)\n\n"
            "    def g(self, x):\n        return x * 3\n\n\nscore = Model().f\n",
        ],
    ),
    "a base class's method": (
        [
            "class Base:\n    def g(self, x):\n        return x * 2\n",
            "class Model(Base):\n    def f(self, x):\n        return self.g(x)\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        [
            # Model's cell runs again, unchanged, so it subclasses the new Base
            # (until then the old Base is what runs, and the record says so).
            "class Base:\n    def g(self, x):\n        return x * 3\n",
            "class Model(Base):\n    def f(self, x):\n        return self.g(x)\n",
        ],
    ),
    "a cached method's body": (
        [
            "import functools\n\n\nclass Model:\n    @functools.cache\n    def f(self, x):\n"
            "        return x * 2\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        [
            "class Model:\n    @functools.cache\n    def f(self, x):\n        return x * 3\n"
        ],
    ),
    "a cached property's body": (
        [
            "import functools\n\n\nclass Model:\n    @functools.cached_property\n"
            "    def k(self):\n        return 2\n",
            "def task(a):\n    return a * Model().k\n",
        ],
        [
            "class Model:\n    @functools.cached_property\n    def k(self):\n"
            "        return 3\n",
        ],
    ),
    "a class reached through its class method as the task": (
        [
            "class Model:\n    FACTOR = 2\n\n    @classmethod\n    def run(cls, x):\n"
            "        return x * cls.FACTOR\n\n\ntask = Model.run\n",
        ],
        [
            "class Model:\n    FACTOR = 3\n\n    @classmethod\n    def run(cls, x):\n"
            "        return x * cls.FACTOR\n\n\ntask = Model.run\n",
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
    # Round 6: an OrderedDict's own iterator looks each key up again, so one
    # whose key's hash changed raised in the write scan and the notebook had
    # no entries at all.
    "a helper defined in an except block": (
        [
            "try:\n    import mushin_no_such_module\nexcept ImportError:\n\n"
            '    def helper(x):\n        """Doc."""\n        return x * 2\n',
            TASK,
        ],
        [
            "try:\n    import mushin_no_such_module\nexcept ImportError:\n\n"
            '    def helper(x):\n        """Doc."""\n        return x * 3\n'
        ],
    ),
    "a helper beside an OrderedDict whose key changed": (
        [
            "import collections\nimport dataclasses\n\n\n"
            "@dataclasses.dataclass(unsafe_hash=True)\nclass Cfg:\n    lr: float\n\n\n"
            "cfg = Cfg(0.1)\nBEST = collections.OrderedDict({cfg: 0.9})\ncfg.lr = 0.2\n",
            "def helper(x):\n    return x\n",
            TASK,
        ],
        ["def helper(x):\n    return x * 2\n"],
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
    "a docstring of a helper defined in a match case": (
        [
            "match 1:\n    case 1:\n\n"
            '        def helper(x):\n            """Old."""\n            return x * 2\n',
            TASK,
        ],
        [
            "match 1:\n    case 1:\n\n"
            '        def helper(x):\n            """New."""\n            return x * 2\n'
        ],
    ),
    "a docstring of a helper defined in an except block": (
        [
            "try:\n    import mushin_no_such_module\nexcept ImportError:\n\n"
            '    def helper(x):\n        """Old."""\n        return x * 2\n',
            TASK,
        ],
        [
            "try:\n    import mushin_no_such_module\nexcept ImportError:\n\n"
            '    def helper(x):\n        """New."""\n        return x * 2\n'
        ],
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
    "a method-less class's cell run again unchanged": (
        [
            "import enum\n\n\nclass Color(enum.Enum):\n    RED = 1\n",
            "class Config:\n    LR = 0.1\n",
            "def task(a):\n    return a * Config.LR + Color.RED.value\n",
        ],
        ["class Config:\n    LR = 0.1\n", "class Color(enum.Enum):\n    RED = 1\n"],
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
    "a constant a helper reads (documented: values are not compared)": (
        ["SCALE = 2\n\n\ndef helper(x):\n    return x * SCALE\n", TASK],
        ["SCALE = 3\n"],
    ),
    "a constant the task reads (documented)": (
        ["SCALE = 2\n\n\ndef task(a):\n    return a * SCALE\n"],
        ["SCALE = 3\n"],
    ),
    "a tuple constant (documented)": (
        ["LRS = (0.1, 0.2)\n\n\ndef task(a):\n    return a * LRS[1]\n"],
        ["LRS = (0.1, 0.3)\n"],
    ),
    "an int replaced by a bool (documented)": (
        ["FLAG = 1\n\n\ndef task(a):\n    return a * FLAG\n"],
        ["FLAG = True\n"],
    ),
    "a default captured from a global, its cell run again (documented)": (
        [
            "SCALE = 2\n",
            "def helper(x, s=SCALE):\n    return x * s\n",
            TASK,
        ],
        ["SCALE = 3\n", "def helper(x, s=SCALE):\n    return x * s\n"],
    ),
    "a closure's captured value (documented)": (
        [
            "def make(k):\n    def helper(x):\n        return x * k\n\n    return helper\n",
            "helper = make(2)\n",
            TASK,
        ],
        ["helper = make(3)\n"],
    ),
    "an int named like a token count (documented)": (
        ["MAX_TOKENS = 256\n\n\ndef task(a):\n    return a * MAX_TOKENS\n"],
        ["MAX_TOKENS = 512\n"],
    ),
    "one function of a ring of closures (documented: closures are not followed)": (
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
    "a helper a closure captures (documented)": (
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
    "a task rebinding a helper to another function": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x\n\n\nhelper = slow\n",
            "def task(a):\n    global helper\n    helper = fast\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a global function named like an attribute the task reads": (
        [
            "import types\n\nns = types.SimpleNamespace(scale=abs)\n\n\n"
            "def scale(x):\n    return x * 2\n",
            "def task(a):\n    return ns.scale(a)\n",
        ],
        ["def scale(x):\n    return x * 3\n"],
    ),
    "a hook in a list rebinding a helper": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "def switch():\n    global helper\n    helper = fast\n\n\nHOOKS = [switch]\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a callback instance rebinding a model": (
        [
            "class Baseline:\n    def f(self, x):\n        return x\n\n\n"
            "class Tuned:\n    def f(self, x):\n        return x + 0\n\n\nmodel = Baseline()\n",
            "class Promote:\n    def __call__(self):\n        global model\n"
            "        model = Tuned()\n\n\nCALLBACKS = [Promote()]\ndel Promote\n",
            "def task(a):\n    for c in CALLBACKS:\n        c()\n    return model.f(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in a nested class rebinding a helper": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "class Outer:\n    class Inner:\n        def go(self):\n            global helper\n"
            "            helper = fast\n",
            "def task(a):\n    Outer.Inner().go()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer bound as a method of another object": (
        [
            "import types\n\n\ndef slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\n"
            "helper = slow\n",
            "def switch(self):\n    global helper\n    helper = fast\n\n\n"
            "HOOKS = [types.MethodType(switch, types.SimpleNamespace())]\ndel switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer registered with a decorator that returns None": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\n"
            "helper = slow\nHOOKS = []\n",
            "@HOOKS.append\ndef switch():\n    global helper\n    helper = fast\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer only a closure holds": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\n"
            "helper = slow\n",
            "def make():\n    def switch():\n        global helper\n        helper = fast\n\n"
            "    def run(a):\n        switch()\n        return helper(a)\n\n    return run\n\n\n"
            "task = make()\ndel make\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in a base class a class decorator hides": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "def registered(cls):\n    return type(cls.__name__, (cls,), {})\n",
            "@registered\nclass Model:\n    def warm(self):\n        global helper\n"
            "        helper = fast\n\n    def f(self, x):\n        self.warm()\n"
            "        return helper(x)\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in a base class no longer bound": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "class Base:\n    def warm(self):\n        global helper\n        helper = fast\n",
            "class Model(Base):\n    def f(self, x):\n        self.warm()\n"
            "        return helper(x)\n\n\ndel Base\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in a class a subclass of the same name shadows": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "class Model:\n    def warm(self):\n        global helper\n        helper = fast\n",
            "class Model(Model):\n    def f(self, x):\n        self.warm()\n"
            "        return helper(x)\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in a defaultdict of hooks": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "import collections\n\nHOOKS = collections.defaultdict(list)\n\n\n"
            "def switch():\n    global helper\n    helper = fast\n\n\n"
            "HOOKS['pre'].append(switch)\ndel switch\n",
            "def task(a):\n    for h in HOOKS['pre']:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer kept only in a partial": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "import functools\n\n\ndef switch(to):\n    global helper\n    helper = to\n\n\n"
            "HOOKS = [functools.partial(switch, fast)]\ndel switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer behind an lru_cache wrapper": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "import functools\n\n\n@functools.lru_cache\ndef switch():\n    global helper\n"
            "    helper = fast\n\n\nHOOKS = [switch]\ndel switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer held only as a default": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "def switch():\n    global helper\n    helper = fast\n\n\n"
            "def run(a, hook=switch):\n    hook()\n    return helper(a)\n\n\ndel switch\n",
            "def task(a):\n    return run(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer held in a dict": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "def switch():\n    global helper\n    helper = fast\n\n\n"
            "HOOKS = {'pre': switch}\ndel switch\n",
            "def task(a):\n    HOOKS['pre']()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer that is a static method": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "class Setup:\n    @staticmethod\n    def switch():\n        global helper\n"
            "        helper = fast\n\n\nHOOKS = [Setup]\ndel Setup\n",
            "def task(a):\n    for h in HOOKS:\n        h.switch()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a task setting a global it reads to an instance": (
        [
            "def mlp_block(x):\n    return x + 1\n\n\nclass MLP:\n    def f(self, x):\n"
            "        return mlp_block(x)\n\n\nMODELS = {'mlp': MLP}\nmodel = None\n",
            "def task(a):\n    global model\n    if model is None:\n"
            "        model = MODELS['mlp']()\n    return model.f(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a task caching a method at run time": (
        [
            "import functools\n\n\nclass Model:\n    def f(self, x):\n        return x * 2\n",
            "def task(a):\n    if not hasattr(Model, '_memo'):\n"
            "        Model.f = functools.lru_cache(Model.f)\n        Model._memo = True\n"
            "    return Model().f(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a method adding a nested function to its class": (
        [
            "class Model:\n    def f(self, x):\n        def g(y):\n            return y\n\n"
            "        type(self).g = g\n        return x\n",
            "def task(a):\n    return Model().f(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a module the task reads": (
        ["import math\n\n\ndef task(a):\n    return math.sqrt(a)\n"],
        ["import math\n"],
    ),
    # Round 6: each place the write scan looks, with the writer held there
    # and nowhere else.
    "a writer in a deque of hooks": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "import collections\n\n\ndef switch():\n    global helper\n    helper = fast\n\n\n"
            "HOOKS = collections.deque([switch])\ndel switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in an OrderedDict of hooks": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "import collections\n\n\ndef switch():\n    global helper\n    helper = fast\n\n\n"
            "HOOKS = collections.OrderedDict(pre=switch)\ndel switch\n",
            "def task(a):\n    HOOKS['pre']()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer passed to a partial as an argument": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "import functools\n\n\ndef switch():\n    global helper\n    helper = fast\n\n\n"
            "def call(f):\n    f()\n\n\nHOOKS = [functools.partial(call, switch)]\ndel switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer held only as a keyword-only default": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "def switch():\n    global helper\n    helper = fast\n\n\n"
            "def run(a, *, hook=switch):\n    hook()\n    return helper(a)\n\n\ndel switch\n",
            "def task(a):\n    return run(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer only a wrapper's __wrapped__ holds": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "def switch():\n    global helper\n    helper = fast\n\n\n"
            "def mark(f):\n    def w():\n        return w.__wrapped__()\n\n"
            "    w.__wrapped__ = f\n    return w\n\n\nHOOKS = [mark(switch)]\ndel switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer reached through a bound method's instance": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "class Holder:\n    def switch(self):\n        global helper\n        helper = fast\n\n"
            "    def run(self):\n        self.switch()\n\n\nHOOK = Holder().run\ndel Holder\n",
            "def task(a):\n    HOOK()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in a tuple of hooks": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "def switch():\n    global helper\n    helper = fast\n\n\n"
            "HOOKS = (switch,)\ndel switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in a set of hooks": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "def switch():\n    global helper\n    helper = fast\n\n\n"
            "HOOKS = {switch}\ndel switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in a frozenset of hooks": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "def switch():\n    global helper\n    helper = fast\n\n\n"
            "HOOKS = frozenset({switch})\ndel switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer passed to a partial as a keyword": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "import functools\n\n\ndef switch():\n    global helper\n    helper = fast\n\n\n"
            "def call(*, f):\n    f()\n\n\nHOOKS = [functools.partial(call, f=switch)]\n"
            "del switch\n",
            "def task(a):\n    for h in HOOKS:\n        h()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
    ),
    "a writer in a grandparent of a class a registry holds": (
        [
            "def slow(x):\n    return x\n\n\ndef fast(x):\n    return x + 0\n\n\nhelper = slow\n",
            "class Base:\n    def switch(self):\n        global helper\n        helper = fast\n\n\n"
            "class Mid(Base):\n    pass\n\n\nclass Leaf(Mid):\n    pass\n\n\n"
            "REG = [Leaf]\ndel Base, Mid, Leaf\n",
            "def task(a):\n    REG[0]().switch()\n    return helper(a)\n",
        ],
        ["task(1)\n"],
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
    # An empty record would pass vacuously: a scan that raised gives none.
    assert first, "the notebook entries are empty"
    _run_cells(kernel, after)
    assert _record(kernel, "task") == first


def test_a_notebook_whose_record_fails_warns_on_every_run(kernel, monkeypatch):
    """A notebook whose entries cannot be taken has none, as on main, but
    says so, once a run, with the exception's type and not its message (the
    notebook's own text): a kernel is one process, and the resume is where
    the edit goes unnoticed."""
    import mushin._resume as resume

    def fail(namespace, roots, *rest):
        raise RuntimeError("the notebook's own text")

    monkeypatch.setattr(resume, "_notebook_entries", fail)
    monkeypatch.setattr(resume, "_NOTEBOOK_FAILURE_WARNED", [])
    _run_cells(kernel, ["def helper(x):\n    return x\n", TASK])
    for _ in range(2):  # a run, then a resume
        match = r"could not be recorded \(RuntimeError\)"
        with pytest.warns(UserWarning, match=match) as got:
            entries = resume.driver_script_entries((kernel.ns["task"],))
            resume.project_fingerprints(kernel.ns["task"])  # a cell: no second
        assert entries[resume.NOTEBOOK_ENTRIES] == {}
        assert len(got) == 1 and "own text" not in str(got[0].message)


def test_a_notebook_metaclass_hook_that_raises_warns(kernel, monkeypatch):
    """The real case: a class's __dict__ is read through its metaclass, whose
    hook can raise. The notebook then has no entries, with the warning."""
    import mushin._resume as resume

    monkeypatch.setattr(resume, "_NOTEBOOK_FAILURE_WARNED", [])
    _run_cells(
        kernel,
        [
            "class Meta(type):\n    def __getattribute__(cls, name):\n"
            "        if name == '__dict__':\n            raise RuntimeError('no')\n"
            "        return super().__getattribute__(name)\n\n\n"
            "class Thing(metaclass=Meta):\n    pass\n\n\nobj = Thing()\n",
            "def helper(x):\n    return x\n",
            TASK,
        ],
    )
    with pytest.warns(UserWarning, match=r"could not be recorded \(RuntimeError\)"):
        assert _record(kernel, "task") == {}


def test_a_metaclass_eq_that_raises_does_not_drop_the_record(kernel):
    """Types are matched by identity, so a metaclass's __eq__ never runs: one
    that raised used to drop every notebook entry."""
    _run_cells(
        kernel,
        [
            "class Meta(type):\n    def __eq__(cls, other):\n        raise RuntimeError('no')\n\n"
            "    __hash__ = type.__hash__\n\n\n"
            "class Thing(metaclass=Meta):\n    def f(self, x):\n        return x\n\n\n"
            "obj = Thing()\n",
            "def helper(x):\n    return x\n",
            "def task(a):\n    return Thing().f(helper(a))\n",
        ],
    )
    assert {"Thing (notebook)", "helper (notebook)"} <= set(_record(kernel, "task"))


def test_a_metaclass_eq_does_not_steer_the_write_scan(kernel):
    """Round 8: types are matched by identity. A notebook metaclass whose
    `__eq__` answers yes to everything made `kind in containers` true, and
    the scan iterated the notebook's own object."""
    _run_cells(
        kernel,
        [
            "class Meta(type):\n    def __eq__(cls, other):\n        return True\n\n"
            "    __hash__ = type.__hash__\n\n\n"
            "class Feed(metaclass=Meta):\n    calls = 0\n\n    def __len__(self):\n"
            "        type(self).calls += 1\n        return 1\n\n    def __iter__(self):\n"
            "        type(self).calls += 1\n        return iter(())\n\n    @property\n"
            "    def __func__(self):\n        type(self).calls += 1\n        return None\n\n\n"
            "class Holder:\n    feed = Feed()\n\n    def f(self, x):\n        return x\n\n\n"
            "feed = Feed()\n",
            "def helper(x):\n    return x\n",
            "def task(a):\n    return Holder().f(helper(a))\n",
        ],
    )
    assert "helper (notebook)" in _record(kernel, "task")
    assert kernel.ns["Feed"].calls == 0


def test_the_write_scan_runs_no_keys_hash():
    """Round 6: walking a container must run no notebook code. An
    OrderedDict's own iterator hashes each key again, so a key whose hash
    changed raised and the notebook had no entries; dict's iterator does not."""
    kernel = Kernel()
    try:
        kernel.run(
            "import collections\n\n\nclass Key:\n    calls = 0\n\n"
            "    def __hash__(self):\n        type(self).calls += 1\n        return 1\n\n\n"
            "k = Key()\nREG = [collections.OrderedDict({k: 1}), "
            "collections.defaultdict(int, {k: 1}), {k: 1}, {k}, frozenset({k}), "
            "collections.deque([k])]\n"
        )
        kernel.run(TASK)
        kernel.run("def helper(x):\n    return x\n")
        calls = kernel.ns["Key"].calls
        assert "helper (notebook)" in _record(kernel, "task")
        assert kernel.ns["Key"].calls == calls
    finally:
        kernel.close()


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
    }


def test_the_entries_name_what_the_task_reaches(kernel):
    _run_cells(
        kernel,
        [
            "SCALE = 2\nUNUSED = 5\n\n\ndef helper(x):\n    return x * SCALE\n",
            "def task(a):\n    return helper(a)\n",
        ],
    )
    assert set(_record(kernel, "task")) == {"task (notebook)", "helper (notebook)"}


def test_an_inline_lambda_task_does_not_record_the_grid(kernel):
    """A lambda passed to the call that also holds the grid: only the lambda
    is its definition."""
    kernel.run("SCALE = 2\nF, GRID = (lambda a: a * SCALE), [1, 2]\n")
    one = _record(kernel, "F")
    kernel.run("SCALE = 2\nF, GRID = (lambda a: a * SCALE), [1, 2, 3]\n")
    assert _record(kernel, "F") == one
    kernel.run("SCALE = 2\nF, GRID = (lambda a: a * SCALE * 10), [1, 2, 3]\n")
    assert _record(kernel, "F")["<lambda> (notebook)"] != one["<lambda> (notebook)"]


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


def test_data_does_not_count_toward_the_write_scan_bound(kernel):
    """A notebook's data (a results dict of number lists) holds no function:
    it must not use up the bound and turn the record off."""
    kernel.run(
        # 40,000 distinct floats, in lists the scan walks (200 items each).
        "RESULTS = {i: [float(j) for j in range(200)] for i in range(200)}\n\n\n"
        "def helper(x):\n    return x\n\n\ndef task(a):\n    return helper(a)\n"
    )
    assert "helper (notebook)" in _record(kernel, "task")


def test_a_sweep_tasks_own_global_write_is_seen(kernel):
    """`@mushin.sweep` binds a handle, not the task: the scan must take the
    task's own `global` writes from the roots. Whole records compared, since
    a key that appears later is not a resume mismatch but changes a
    cache_dir key."""
    _run_cells(
        kernel,
        [
            "import mushin\n\n\ndef mlp_block(x):\n    return x + 1\n\n\n"
            "class MLP:\n    def f(self, x):\n        return mlp_block(x)\n\n\n"
            "MODELS = {'mlp': MLP}\nmodel = None\n",
            "@mushin.sweep\ndef task(a):\n    global model\n"
            "    if model is None:\n        model = MODELS['mlp']()\n"
            "    return {'v': model.f(a)}\n",
        ],
    )
    root = kernel.ns["task"].__mushin_task__
    one = _notebook(project_fingerprints(root))
    root(1)
    assert _notebook(project_fingerprints(root)) == one


def test_a_write_scan_past_its_bound_gives_no_notebook_entries(kernel, monkeypatch):
    """Which writes a cut scan saw depends on its order: no notebook entry is
    recorded rather than a part of them."""
    import mushin._resume as resume

    monkeypatch.setattr(resume, "_WRITE_SCAN_MAX", 3)
    monkeypatch.setattr(resume, "_NOTEBOOK_TRUNCATION_WARNED", [])
    _run_cells(
        kernel,
        [
            "".join(f"def h{i}(x):\n    return x + {i}\n\n\n" for i in range(10)),
            "def task(a):\n    return h0(a)\n",
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
    """An output IPython keeps (`_`, `_3`) changes with every cell."""
    kernel.run(
        "def f(x):\n    return x\n\n\n_ = f\n_i3 = f\n\n\n"
        "def task(a):\n    return (_(a), _i3(a))\n"
    )
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
        # No value is compared, under any name and from any source (a string,
        # a secret, a number), nor what a default or a closure holds: only a
        # definition, whose text the notebook already holds.
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
        [
            "MODEL = ''.join(['resnet', '{k}'])\n",
            "def task(a):\n    return (MODEL, a)\n",
        ],
        ["DB_PASSWORD = 123{k}\n", "def task(a):\n    return DB_PASSWORD + a\n"],
        ["PIN = 123{k}\n", "def task(a, password=PIN):\n    return a\n"],
        ["PIN = 482{k}\n", "def task(a):\n    return a + PIN\n"],
    ],
)
def test_no_value_is_compared(kernel, cells):
    """A value is only safe to compare when every writer of it is found, and
    a digest of a short value alone can confirm a guess of a secret."""
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


# --- lazy imports (PEP 810, 3.15+) ------------------------------------------------

lazy_imports = pytest.mark.skipif(
    sys.version_info < (3, 15), reason="lazy imports are new in 3.15"
)
# U+00B7 is an identifier character that the regex class `\w` does not match.
MIDDLE_DOT = "mushin_t_lazy\u00b7dot"
LAZY_MODULES = ("mushin_t_lazymod", "mushin_t_lazypkg", MIDDLE_DOT)


@pytest.fixture
def lazy_project(tmp_path, monkeypatch):
    """A project module and a package with a submodule, none of them imported.
    The scan of the kept cells' imports is off: these tests pin how the
    namespace's proxies are read, and the cell's statement would record the
    module anyway."""
    import importlib

    monkeypatch.setattr(
        mushin._resume, "_kept_cell_reading", lambda *scan: (set(), {}), raising=False
    )

    root = tmp_path / "proj"
    (root / "mushin_t_lazypkg").mkdir(parents=True)
    (root / "mushin_t_lazymod.py").write_text("def f(x):\n    return x + 1\n")
    (root / f"{MIDDLE_DOT}.py").write_text(
        "def f(x):\n    return x + 1\n", encoding="utf-8"
    )
    (root / "mushin_t_lazypkg" / "__init__.py").write_text("")
    (root / "mushin_t_lazypkg" / "sub.py").write_text("def f(x):\n    return x + 1\n")
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    yield root
    for name in [n for n in sys.modules if n.startswith(LAZY_MODULES)]:
        sys.modules.pop(name)


@lazy_imports
@pytest.mark.parametrize(
    ("imports", "use", "expected"),
    [
        ("import mushin_t_lazymod", "mushin_t_lazymod.f(a)", ["mushin_t_lazymod"]),
        (f"import {MIDDLE_DOT}", f"{MIDDLE_DOT}.f(a)", [MIDDLE_DOT]),
        (f"from {MIDDLE_DOT} import f", "f(a)", [MIDDLE_DOT]),
        ("from mushin_t_lazymod import f", "f(a)", ["mushin_t_lazymod"]),
        (
            "from mushin_t_lazypkg import sub",
            "sub.f(a)",
            ["mushin_t_lazypkg", "mushin_t_lazypkg.sub"],
        ),
    ],
)
def test_a_lazily_imported_project_module_is_recorded(
    kernel, lazy_project, imports, use, expected
):
    """A lazy import binds a proxy that a lookup of its name resolves, which
    is when the task runs, after the record is taken. The record counts the
    module it will import, with the digests the plain import gives, and does
    not import it."""
    expected = {f"{name} (file)" for name in expected}
    kernel.run(f"lazy {imports}\n")
    kernel.run(f"def task(a):\n    return {use}\n")
    lazy = project_fingerprints(kernel.ns["task"])["code"]
    assert expected <= set(lazy), lazy
    assert not [n for n in sys.modules if n.startswith(LAZY_MODULES)]

    plain_kernel = Kernel("mushin_t_kernel_plain")
    try:
        plain_kernel.run(f"{imports}\n")
        plain_kernel.run(f"def task(a):\n    return {use}\n")
        plain = project_fingerprints(plain_kernel.ns["task"])["code"]
    finally:
        plain_kernel.close()
    assert {k: lazy[k] for k in expected} == {k: plain[k] for k in expected}


@lazy_imports
def test_a_resume_after_editing_a_lazily_imported_module_reruns(
    kernel, lazy_project, tmp_path
):
    """The failure this guards: a restarted kernel binds the proxy again, so
    a record that skipped it would compare nothing and reuse every cell."""
    import importlib

    cells = [
        "import mushin\nfrom mushin import multirun\n\nCALLS = []\n",
        "lazy import mushin_t_lazymod\n",
        "@mushin.sweep\ndef task(a):\n"
        "    CALLS.append(a)\n    return {'v': mushin_t_lazymod.f(a)}\n",
    ]
    for cell in cells:
        kernel.run(cell)
    wd = tmp_path / "wd"
    assert list(_sweep(kernel, wd, [1, 2])["v"].values) == [2, 3]
    kernel.close()

    (lazy_project / "mushin_t_lazymod.py").write_text("def f(x):\n    return x + 100\n")
    for name in [n for n in sys.modules if n.startswith(LAZY_MODULES)]:
        sys.modules.pop(name)
    importlib.invalidate_caches()
    restarted = Kernel()
    try:
        for cell in cells:
            restarted.run(cell)
        with pytest.warns(UserWarning, match=r"mushin_t_lazymod \(file\)"):
            ds = _sweep(restarted, wd, [1, 2], resume=True)
        assert list(ds["v"].values) == [101, 102]
        assert restarted.ns["CALLS"] == [1, 2]
    finally:
        restarted.close()


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
CROSS_VERSION_DIGEST = "4fdf8a6535b53657"


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

    kernel.run("def helper(x):\n    return x * SCALE * 100\n")
    with pytest.warns(UserWarning, match=r"helper \(notebook\)"):
        ds = _sweep(kernel, wd, [1, 2, 3], resume=True)
    assert list(ds["v"].values) == [200, 400, 600]


def test_cache_dir_keys_on_the_notebook_code(kernel, tmp_path):
    _run_cells(kernel, SWEEP_CELLS)
    cache = str(tmp_path / "cache")
    _sweep(kernel, tmp_path / "one", [1, 2], cache_dir=cache)
    assert kernel.ns["CALLS"] == [1, 2]
    _sweep(kernel, tmp_path / "two", [1, 2], cache_dir=cache)
    assert kernel.ns["CALLS"] == [1, 2]  # reused
    kernel.run("def helper(x):\n    return x * 5\n")
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


# --- the modules the notebook's code names ----------------------------------------
#
# A cell's `import pkg.sub` binds only `pkg`: the code the task reaches names
# the submodule (`pkg.sub.f`), and an import inside a function binds a local.
# The record follows those names, read from the syntax trees the digests hash
# (bodies, decorators, defaults, bases, class bodies); the driver ships them to
# a worker beside the notebook entries.

SUBPKG = "mushin_t_subpkg"
SUBREL = "mushin_t_subrel"
SUBWIDE = "mushin_t_subwide"
SUBNS = "mushin_t_subns"
SUB_SOURCE = (
    "def f(x):\n    return x + 1\n\n\nclass Base:\n    pass\n\n\nCONST = 1\n\n\n"
    "def deco(fn):\n    return fn\n"
)

# The constructs a definition can name a module through; the sets below pin
# what the syntax tree gives (it reads alike on every supported Python).
CHAIN_SOURCE = """
def f(xs):
    a = pkg.sub.f(1)
    b = pkg.attr
    pkg.store.x = 1
    del pkg.dele.y
    c = [pkg.comp.b(x) for x in xs]
    g = lambda: pkg.lam.v
    def inner():
        return pkg.inner.w
    with pkg.ctx.cm():
        pass
    pkg.star.f(*xs, **{})
    pkg.call()
    if pkg.cond.c:
        pass
    x = pkg.tern.w if xs else other.alt.a
    y = pkg.sub.f is not None and pkg.bool.b
    z = (pkg.gen.g(x) for x in xs)
    try:
        pkg.tr.t()
    except pkg.exc.E:
        pass
    print(f"{pkg.fstr.s}")
    return pkg.ret.r
class C:
    v = pkg.cls.v
    w: int = pkg.ann.v
    def m(self):
        return pkg.meth.m
async def af():
    await pkg.aw.f()
    async with pkg.actx.cm():
        pass
@pkg.deco.d
def decorated(x, m=pkg.default.M, *, k: pkg.ann2.T = None):
    return x
class Model(pkg.base.Base, metaclass=pkg.meta.Meta):
    K = pkg.body.CONST
    s = staticmethod(pkg.sm.f)
@pkg.cdeco.d
class D:
    pass
def kw(*, k=pkg.kwd.K) -> pkg.rann.R:
    pass
ld = lambda x, d=pkg.ldef.D: x
"""
CHAINS = {
    "other.alt.a", "pkg.actx.cm", "pkg.ann.v", "pkg.ann2.T", "pkg.attr", "pkg.aw.f",
    "pkg.base.Base", "pkg.body.CONST", "pkg.bool.b", "pkg.call", "pkg.cls.v",
    "pkg.comp.b", "pkg.cond.c", "pkg.ctx.cm", "pkg.deco.d", "pkg.default.M",
    "pkg.dele.y", "pkg.exc.E", "pkg.fstr.s", "pkg.gen.g", "pkg.inner.w", "pkg.lam.v",
    "pkg.meta.Meta", "pkg.meth.m", "pkg.ret.r", "pkg.sm.f", "pkg.star.f",
    "pkg.store.x", "pkg.sub.f", "pkg.tern.w", "pkg.tr.t", "pkg.cdeco.d",
    "pkg.kwd.K", "pkg.rann.R", "pkg.ldef.D",
}  # fmt: skip
# Imports, and chains on the names they bind; a relative import names none.
IMPORT_SOURCE = """
def imports():
    import pkg.fn_imp
    import pkg.fn_as.deep as alias
    from pkg import fn_from, fn_from2
    from pkg.deep import thing
    from . import rel
    from .relmod import r
    import pkg as top
    from pkg import fn_from3 as fa
    return top.sub.f, alias.x, fn_from.y, r.z, rel.w, fa.q.r
"""
IMPORTS = {
    "pkg", "pkg.deep", "pkg.deep.thing", "pkg.fn_as.deep", "pkg.fn_as.deep.x",
    "pkg.fn_from", "pkg.fn_from.y", "pkg.fn_from2", "pkg.fn_from3",
    "pkg.fn_from3.q.r", "pkg.fn_imp", "pkg.sub.f",
}  # fmt: skip


def test_the_syntax_names_the_pinned_modules():
    import ast

    from mushin._resume import _unit_reading

    tree = ast.parse(CHAIN_SOURCE)
    modules, chains, cells, rebound = _unit_reading(tree, [tree])
    assert modules == set()
    assert {".".join(chain) for chain in chains} == CHAINS
    assert cells == set()
    assert rebound == set()
    tree = ast.parse(IMPORT_SOURCE)
    modules, chains, cells, rebound = _unit_reading(tree, [tree])
    assert modules == IMPORTS
    assert chains == set()
    assert cells == set()  # the relative imports' names are the unit's own
    assert rebound == set()


@pytest.fixture
def sub_project(tmp_path, monkeypatch):
    """A project package with submodules, and a top-level module named like
    a relative import's target; none of them imported. The scan of the kept
    cells' imports is off: most tests here pin how the reached code names a
    module, and a cell's `import pkg.sub` would record it anyway
    (``cell_imports`` turns it back on)."""
    import importlib

    monkeypatch.setattr(
        mushin._resume, "_kept_cell_reading", lambda *scan: (set(), {}), raising=False
    )

    root = tmp_path / "subproj"
    (root / SUBPKG / "a").mkdir(parents=True)
    (root / SUBPKG / "__init__.py").write_text("")
    (root / SUBPKG / "a" / "__init__.py").write_text("")
    for rel in ("other.py", "a/b.py"):
        (root / SUBPKG / rel).write_text("def f(x):\n    return x + 1\n")
    (root / SUBPKG / "sub.py").write_text(SUB_SOURCE)
    # A subpackage named like its package: `lazy from pkg import pkg`.
    (root / SUBPKG / SUBPKG).mkdir()
    (root / SUBPKG / SUBPKG / "__init__.py").write_text("")
    (root / SUBPKG / SUBPKG / "deep.py").write_text("def f(x):\n    return x + 1\n")
    (root / f"{SUBREL}.py").write_text("def f(x):\n    return x + 1\n")
    (root / SUBNS).mkdir()  # a namespace package: no __init__.py
    (root / SUBNS / "mod.py").write_text("def f(x):\n    return x + 1\n")
    # A package whose __init__ imports more files than a small bound holds.
    (root / SUBWIDE).mkdir()
    (root / SUBWIDE / "__init__.py").write_text("from . import x0, x1, x2\n")
    for i in range(3):
        (root / SUBWIDE / f"x{i}.py").write_text(f"X = {i}\n")
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    yield root
    for name in [
        n for n in sys.modules if n.startswith((SUBPKG, SUBREL, SUBWIDE, SUBNS))
    ]:
        sys.modules.pop(name)


@pytest.fixture
def cell_imports(sub_project, monkeypatch):
    """``sub_project`` with the scan of the kept cells' imports on."""
    monkeypatch.setattr(
        mushin._resume, "_kept_cell_reading", KEPT_CELL_READING, raising=False
    )
    return sub_project


def _files(kernel, *names, allow_failure=False, **kwargs) -> set:
    """The record's file keys. Unless ``allow_failure``, a record whose
    notebook entries could not be taken fails the test, so an assertion that
    a file is absent cannot pass because the reading crashed. Other warnings
    pass through."""
    import mushin._resume as resume

    resume._NOTEBOOK_FAILURE_WARNED.clear()  # it warns once per run otherwise
    getattr(resume, "_CELL_SCAN_WARNED", []).clear()  # (absent before #294)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = project_fingerprints(*(kernel.ns[n] for n in names), **kwargs)
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if not allow_failure:
        failed = [
            w
            for w in caught
            if "could not be recorded" in str(w.message)
            or "could not be read" in str(w.message)
        ]
        assert not failed, failed
    return {k for k in record["code"] if k.endswith(" (file)")}


@pytest.mark.parametrize(
    ("imports", "use", "expected"),
    [
        (f"import {SUBPKG}.sub", f"{SUBPKG}.sub.f(a)", ["", ".sub"]),
        (f"import {SUBPKG}.a.b", f"{SUBPKG}.a.b.f(a)", ["", ".a", ".a.b"]),
        # These bind the submodule itself: the same record.
        (f"import {SUBPKG}.sub as s", "s.f(a)", ["", ".sub"]),
        (f"from {SUBPKG} import sub", "sub.f(a)", ["", ".sub"]),
        # A module bound under another name: the chain starts at its name.
        (f"import {SUBPKG}.sub\nimport {SUBPKG} as sp", "sp.sub.f(a)", ["", ".sub"]),
        (
            f"import {SUBPKG}.a.b\nimport {SUBPKG}.a as sa",
            "sa.b.f(a)",
            ["", ".a", ".a.b"],
        ),
    ],
)
def test_a_submodule_the_task_names_through_its_package_is_recorded(
    kernel, sub_project, imports, use, expected
):
    kernel.run(f"{imports}\n")
    kernel.run(f"def task(a):\n    return {use}\n")
    files = _files(kernel, "task")
    assert {f"{SUBPKG}{e} (file)" for e in expected} <= files, files
    assert f"{SUBPKG}.other (file)" not in files


MODEL_TASK = "def task(a):\n    return Model()\n"
HELPER_TASK = "def task(a):\n    return helper(a)\n"


@pytest.mark.parametrize(
    "cells",
    [
        [f"class Model({SUBPKG}.sub.Base):\n    def f(self, x):\n        return x\n"],
        [
            f"class Model:\n    K = {SUBPKG}.sub.CONST\n\n"
            "    def f(self, x):\n        return x * self.K\n"
        ],
        [
            f"class Model:\n    g = staticmethod({SUBPKG}.sub.f)\n\n"
            "    def f(self, x):\n        return self.g(x)\n"
        ],
        [f"@{SUBPKG}.sub.deco\ndef helper(x):\n    return x\n"],
        [f"def helper(x, g={SUBPKG}.sub.f):\n    return g(x)\n"],
        # A class with no method of its own: its statement is found through
        # the kept cells, and its bases count by type too.
        [f"class Model({SUBPKG}.sub.Base):\n    K = 3\n"],
        [f"class Model:\n    K = {SUBPKG}.sub.CONST\n"],
        [
            f"class Mid({SUBPKG}.sub.Base):\n    K = 3\n",
            "class Model(Mid):\n    def f(self, x):\n        return x\n",
        ],
    ],
    ids=[
        "base",
        "class attribute",
        "staticmethod",
        "decorator",
        "default",
        "methodless base",
        "methodless body",
        "methodless notebook base",
    ],
)
def test_a_submodule_a_definition_names_outside_a_body_is_recorded(
    kernel, sub_project, cells
):
    """A cell runs a class's bases and body, a decorator and a default
    itself, not in a function: the definition's text names the module."""
    kernel.run(f"import {SUBPKG}.sub\n")
    task = MODEL_TASK if any("Model" in c for c in cells) else HELPER_TASK
    _run_cells(kernel, [*cells, task])
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_only_the_code_the_task_reaches_names_modules(kernel, sub_project):
    """A cell helper the task calls counts; a cell function it does not
    reach, and a submodule a cell loaded that the reached code never names,
    do not. (That is the reached code's reading: the cells' own imports
    count too, see `test_every_project_module_a_cell_imports_counts`.)"""
    kernel.run(f"import {SUBPKG}.sub\nimport {SUBPKG}.other\n")
    kernel.run(f"def helper(x):\n    return {SUBPKG}.sub.f(x)\n")
    kernel.run(f"def unused(x):\n    return {SUBPKG}.other.f(x)\n")
    kernel.run("def task(a):\n    return helper(a)\n")
    files = _files(kernel, "task")
    assert f"{SUBPKG}.sub (file)" in files
    assert f"{SUBPKG}.other (file)" not in files


@pytest.mark.parametrize(
    "body",
    [
        f"import {SUBPKG}.sub\n    return {SUBPKG}.sub.f(a)",
        f"from {SUBPKG} import sub\n    return sub.f(a)",
        f"from {SUBPKG}.sub import f\n    return f(a)",
        f"import {SUBPKG}\n    return {SUBPKG}.sub.f(a)",
        f"from {SUBPKG} import sub as S\n    return S.f(a)",
        f"from {SUBPKG} import a as A\n    return A.b.f(a)",
        # `import pkg.sub` binds `pkg`, so `pkg.a.b` is the package's.
        f"import {SUBPKG}.sub\n    return {SUBPKG}.a.b.f(a)",
    ],
)
def test_an_import_inside_the_tasks_body_is_recorded(kernel, sub_project, body):
    kernel.run(f"def task(a):\n    {body}\n")
    expected = ".a.b" if ("A.b" in body or ".a.b" in body) else ".sub"
    assert f"{SUBPKG}{expected} (file)" in _files(kernel, "task")


def test_a_relative_import_is_not_read_as_an_absolute_one(kernel, sub_project):
    # Read without its level, the import would name the top-level module.
    kernel.run(f"def task(a):\n    from .{SUBREL} import f\n    return f(a)\n")
    assert f"{SUBREL} (file)" not in _files(kernel, "task")
    # The name it binds is not the namespace's module of that name.
    kernel.run(f"import {SUBPKG}.sub\n")
    kernel.run(
        f"def task(a):\n    from . import {SUBPKG}\n    return {SUBPKG}.other.f(a)\n"
    )
    assert f"{SUBPKG}.other (file)" not in _files(kernel, "task")


def test_a_name_bound_to_a_value_starts_no_chain(kernel, sub_project):
    """Only a module (or a lazy import of one) has submodules: a value named
    like the package gives no seed."""
    kernel.run(
        f"import types\n{SUBPKG} = types.SimpleNamespace("
        "sub=types.SimpleNamespace(f=abs))\n"
    )
    kernel.run(f"def task(a):\n    return {SUBPKG}.sub.f(a)\n")
    assert _files(kernel, "task") == set()


def test_a_reload_helper_rebinding_the_package_keeps_the_submodule(kernel, sub_project):
    """The walk skips a name a notebook function rebinds, since the definition
    it holds can be gone by the next run. A chain does not: a reload helper
    keeps a module there, and skipping it would drop the submodule."""
    kernel.run(f"import importlib\nimport {SUBPKG}.sub\n")
    kernel.run(
        f"def refresh():\n    global {SUBPKG}\n"
        f"    {SUBPKG} = importlib.reload({SUBPKG})\n"
    )
    kernel.run(f"def task(a):\n    refresh()\n    return {SUBPKG}.sub.f(a)\n")
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


SCOPE_CELL = f"import {SUBPKG}.a.b\nfrom {SUBPKG} import a as utils\n"


@pytest.mark.parametrize(
    "task",
    [
        # An import in a nested function binds there, not in the task.
        "def task(a):\n    def stamp():\n        from email import utils\n"
        "        return utils.formatdate()\n\n    return utils.b.f(a)\n",
        # A default is evaluated where the definition stands, not in its body.
        "def task(a, g=utils.b.f):\n    from email import utils\n"
        "    return g(a) if utils else 0\n",
        # A class body's names are not its methods'.
        "class Model:\n    from email import utils\n\n"
        "    def f(self, x):\n        return utils.b.f(x)\n\n\n"
        "def task(a):\n    return Model().f(a)\n",
        # Two bindings of one name: a chain on it resolves through each.
        f"def task(a, fast=True):\n    if fast:\n        from {SUBPKG} import a as h\n"
        f"    else:\n        from {SUBPKG} import sub as h\n    return h.b.f(a)\n",
        f"def task(a, fast=True):\n    if fast:\n        from {SUBPKG} import sub as h\n"
        f"    else:\n        from {SUBPKG} import a as h\n    return h.b.f(a)\n",
    ],
    ids=[
        "nested function",
        "default",
        "class body",
        "two bindings",
        "two bindings, other order",
    ],
)
def test_an_import_binds_only_in_its_own_scope(kernel, sub_project, task):
    kernel.run(SCOPE_CELL)
    kernel.run(task)
    assert f"{SUBPKG}.a.b (file)" in _files(kernel, "task")


def test_a_class_bodys_import_does_not_steer_its_methods_names(kernel, sub_project):
    """A method's `utils` is the global, not the class body's: reading it
    through the class body's import would record a file the method never
    reaches (a recompute on every edit to it)."""
    kernel.run(f"import {SUBPKG}.a.b\nfrom {SUBPKG} import sub as utils\n")
    kernel.run(
        f"class Model:\n    from {SUBPKG} import a as utils\n\n"
        "    def f(self, x):\n        return utils.b.f(x)\n"
    )
    kernel.run("def task(a):\n    return Model().f(a)\n")
    assert f"{SUBPKG}.a.b (file)" not in _files(kernel, "task")


@pytest.mark.parametrize(
    "body",
    [
        "    def inner(x):\n        return utils.b.f(x)\n\n    return inner(a)\n",
        "    g = lambda x: utils.b.f(x)\n    return g(a)\n",
    ],
    ids=["nested function", "lambda"],
)
def test_an_enclosing_functions_import_reaches_the_functions_in_it(
    kernel, sub_project, body
):
    kernel.run(f"import {SUBPKG}.a.b\n")  # `utils` is no global here
    kernel.run(f"def task(a):\n    from {SUBPKG} import a as utils\n{body}")
    assert f"{SUBPKG}.a.b (file)" in _files(kernel, "task")


def test_the_innermost_enclosing_binding_is_in_force(kernel, sub_project):
    kernel.run(f"import {SUBPKG}.a.b\n")
    kernel.run(
        f"def task(a):\n    from {SUBPKG} import sub as utils\n\n    def mid(x):\n"
        f"        from {SUBPKG} import a as utils\n\n        def inner(y):\n"
        "            return utils.b.f(y)\n\n        return inner(x)\n\n"
        "    return mid(a)\n"
    )
    assert f"{SUBPKG}.a.b (file)" in _files(kernel, "task")


FACTORY = (
    f"def outer():\n    import {SUBPKG}.a as s\n\n"
    "    def inner(a):\n        return s.b.f(a)\n\n    return inner\n"
)
CLASS_IMPORT = (
    f"class Model:\n    import {SUBPKG}.a as s\n\n"
    "    def f(self, a, k=s.b.f):\n        return k(a)\n"
)


@pytest.mark.parametrize("name", ["A_fn", "z_fn"])  # before, after the container
@pytest.mark.parametrize(
    ("cell", "bind", "use"),
    [
        (FACTORY, "outer()", "outer()(a)"),
        (CLASS_IMPORT, "Model.f", "Model().f(a)"),
    ],
    ids=["closure", "method"],
)
def test_what_a_unit_names_does_not_depend_on_the_walk_order(
    kernel, sub_project, cell, bind, use, name
):
    """A closure and its factory, or a method bound by name and its class,
    are both reached; which is read first follows the names' order, and the
    record is the same either way."""
    kernel.run(f"import {SUBPKG}.a.b\n")  # the class's default reads `s.b`
    kernel.run(cell)
    kernel.run(f"{name} = {bind}\n")
    kernel.run(f"def task(a):\n    return {name}(a) + {use}\n")
    assert f"{SUBPKG}.a.b (file)" in _files(kernel, "task")


@pytest.mark.parametrize(
    "cells",
    [
        [
            "import functools\n\n\ndef logged(fn):\n"
            f"    import {SUBPKG}.a as s\n\n    @functools.wraps(fn)\n"
            "    def wrapper(*args):\n        s.b.f(1)\n        return fn(*args)\n\n"
            "    return wrapper\n",
            "@logged\ndef task(a):\n    return a\n",
        ],
        [
            f"def make():\n    from {SUBPKG} import a as A\n\n"
            "    def task(a):\n        return A.b.f(a)\n\n    return task\n\n\n"
            "task = make()\n"
        ],
        # A global of the same name is not what the closure holds.
        [
            f"import functools\nimport {SUBPKG}.sub as s\n\n\ndef logged(fn):\n"
            f"    import {SUBPKG}.a as s\n\n    @functools.wraps(fn)\n"
            "    def wrapper(*args):\n        s.b.f(1)\n        return fn(*args)\n\n"
            "    return wrapper\n",
            "@logged\ndef task(a):\n    return a\n",
        ],
    ],
    ids=["decorator's wrapper", "factory's function", "global of the same name"],
)
def test_a_function_made_inside_another_reaches_its_imports(kernel, sub_project, cells):
    """Its definition is the inner `def` only; the module it uses is in its
    closure."""
    _run_cells(kernel, cells)
    assert f"{SUBPKG}.a.b (file)" in _files(kernel, "task")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="type parameters are 3.12+")
def test_a_type_parameters_bound_is_read(kernel, sub_project):
    kernel.run(f"import {SUBPKG}.sub\n")
    kernel.run(f"def task[T: {SUBPKG}.sub.Base](a):\n    return a\n")
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_the_driver_ships_only_project_modules(kernel, sub_project):
    """An installed package's names are dropped before they ship (by the
    top-level module's file), and a built-in one's by name; a project
    namespace package, which has no file, is kept for the record to
    decide."""
    from mushin._resume import NOTEBOOK_MODULES, driver_script_entries

    kernel.run(f"import os\nimport sys\nimport numpy as np\nimport {SUBNS}.mod\n")
    kernel.run(
        "def task(a):\n    return os.path.join(str(np.linalg.norm(a)), "
        f"str({SUBNS}.mod.f(a)), str(sys.path.count(a)))\n"
    )
    shipped = driver_script_entries((kernel.ns["task"],))
    assert shipped[NOTEBOOK_MODULES] == [f"{SUBNS}.mod.f"]
    assert f"{SUBNS}.mod (file)" in _files(kernel, "task", scripts=shipped)


def test_a_foreign_modules_chain_adds_no_entry(kernel):
    kernel.run("import os\nimport numpy as np\n")
    kernel.run("def task(a):\n    return os.path.join('x', str(np.linalg.norm(a)))\n")
    assert _files(kernel, "task") == set()


def test_the_driver_ships_the_modules_the_notebook_code_names(kernel, sub_project):
    """A worker gets the task by value: the modules the driver's walk found
    are the ones its cells follow."""
    from mushin._resume import driver_script_entries

    kernel.run(f"import {SUBPKG}.sub\n")
    kernel.run(f"def task(a):\n    return {SUBPKG}.sub.f(a)\n")
    shipped = driver_script_entries((kernel.ns["task"],))
    kernel.run(f"{SUBPKG} = None\n")  # this process's own walk would find none
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task", scripts=shipped)


@pytest.mark.parametrize(
    "missing",
    [("<notebook modules>", "<notebook later modules>"), ("<notebook later modules>",)],
    ids=["0.17.0", "0.17.0 with #291 and #293"],
)
def test_an_older_drivers_shipment_still_counts(kernel, missing):
    """A runner pickled by 0.17.0 ships the notebook entries without either
    list of modules, and one with #291 and #293 without the later ones. In a
    worker, which keeps no cell text, they are still the record's notebook
    entries."""
    from mushin._resume import (
        NOTEBOOK_LATER_MODULES,
        NOTEBOOK_MODULES,
        driver_script_entries,
    )

    assert {NOTEBOOK_MODULES, NOTEBOOK_LATER_MODULES} >= set(missing)
    _run_cells(kernel, ["def helper(x):\n    return x * 2\n", TASK])
    shipped = driver_script_entries((kernel.ns["task"],))
    taken = _notebook(project_fingerprints(kernel.ns["task"], scripts=shipped))
    for key in missing:
        del shipped[key]
    for name in kernel.names:
        linecache.cache.pop(name, None)  # as in a worker
    assert "helper (notebook)" in taken
    assert _notebook(project_fingerprints(kernel.ns["task"], scripts=shipped)) == taken


def test_a_record_cut_short_keeps_the_submodule_the_task_names(
    kernel, sub_project, monkeypatch
):
    """The modules the task's code names go first: a record cut by its file
    bound keeps them over what the namespace's other packages import."""
    import mushin._resume as resume

    monkeypatch.setattr(resume, "_FILES_MAX", 3)
    monkeypatch.setattr(resume, "_TRUNCATION_WARNED", [])
    kernel.run(f"import {SUBPKG}.sub\nimport {SUBWIDE}\n")
    kernel.run(f"def task(a):\n    return {SUBPKG}.sub.f(a)\n")
    with pytest.warns(UserWarning, match="more project files"):
        files = _files(kernel, "task")
    assert f"{SUBPKG}.sub (file)" in files


def test_a_cut_or_failed_walk_names_no_modules(kernel, sub_project, monkeypatch):
    """Which modules a cut walk found depends on its order, as its entries
    do: neither is recorded."""
    import mushin._resume as resume

    kernel.run(f"import {SUBPKG}.sub\n")
    kernel.run("".join(f"def h{i}(x):\n    return x + {i}\n\n\n" for i in range(10)))
    kernel.run(
        f"def task(a):\n    return {SUBPKG}.sub.f(a) + "
        + " + ".join(f"h{i}(a)" for i in range(10))
        + "\n"
    )
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")

    with monkeypatch.context() as patch:
        patch.setattr(resume, "_NOTEBOOK_MAX_OBJECTS", 3)
        patch.setattr(resume, "_NOTEBOOK_TRUNCATION_WARNED", [])
        with pytest.warns(UserWarning, match="larger than the record covers"):
            assert f"{SUBPKG}.sub (file)" not in _files(kernel, "task")

    def fail(children):
        raise RuntimeError("after the modules were found")

    with monkeypatch.context() as patch:
        patch.setattr(resume, "_components", fail)
        patch.setattr(resume, "_NOTEBOOK_FAILURE_WARNED", [])
        with pytest.warns(UserWarning, match="could not be recorded"):
            files = _files(kernel, "task", allow_failure=True)
        assert f"{SUBPKG}.sub (file)" not in files


def test_a_resume_after_editing_a_submodule_named_through_its_package_reruns(
    kernel, sub_project, tmp_path
):
    """The failure this guards: the record held only `pkg`, so an edit to
    `pkg/sub.py` and a restart reused every cell."""
    import importlib

    cells = [
        "import mushin\nfrom mushin import multirun\n\nCALLS = []\n",
        f"import {SUBPKG}.sub\n",
        "@mushin.sweep\ndef task(a):\n"
        f"    CALLS.append(a)\n    return {{'v': {SUBPKG}.sub.f(a)}}\n",
    ]
    for cell in cells:
        kernel.run(cell)
    wd = tmp_path / "wd"
    assert list(_sweep(kernel, wd, [1, 2])["v"].values) == [2, 3]
    kernel.close()

    (sub_project / SUBPKG / "sub.py").write_text("def f(x):\n    return x + 100\n")
    for name in [n for n in sys.modules if n.startswith(SUBPKG)]:
        sys.modules.pop(name)
    importlib.invalidate_caches()
    restarted = Kernel()
    try:
        for cell in cells:
            restarted.run(cell)
        with pytest.warns(UserWarning, match=rf"{SUBPKG}\.sub \(file\)"):
            ds = _sweep(restarted, wd, [1, 2], resume=True)
        assert list(ds["v"].values) == [101, 102]
        assert restarted.ns["CALLS"] == [1, 2]
    finally:
        restarted.close()


@lazy_imports
@pytest.mark.parametrize(
    ("cell", "use", "expected"),
    [
        (f"lazy import {SUBPKG}.sub", f"{SUBPKG}.sub.f(a)", ".sub"),
        (f"lazy import {SUBPKG}.sub as s", "s.f(a)", ".sub"),
        (f"lazy from {SUBPKG} import sub", "sub.f(a)", ".sub"),
        # `pkg` stands for the package, not the module the proxy names: a
        # chain through it reaches a submodule the cell loaded without
        # binding it.
        (
            f"lazy import {SUBPKG}.a\nimport importlib\n"
            f"importlib.import_module('{SUBPKG}.a.b')",
            f"{SUBPKG}.a.b.f(a)",
            ".a.b",
        ),
        # Bound under another name, the proxy stands for the module it names.
        (
            f"lazy import {SUBPKG}.a as sa\nimport importlib\n"
            f"importlib.import_module('{SUBPKG}.a.b')",
            "sa.b.f(a)",
            ".a.b",
        ),
        # Bound as its first component, it may stand for the full name: both.
        (
            f"lazy from {SUBPKG} import {SUBPKG}\nimport importlib\n"
            f"importlib.import_module('{SUBPKG}.{SUBPKG}.deep')",
            f"{SUBPKG}.deep.f(a)",
            f".{SUBPKG}.deep",
        ),
    ],
)
def test_a_lazy_import_records_alike_before_and_after_it_resolves(
    kernel, sub_project, cell, use, expected
):
    kernel.run(f"{cell}\n")
    kernel.run(f"def task(a):\n    return {use}\n")
    before = project_fingerprints(kernel.ns["task"])["code"]
    if "importlib" not in cell:  # taking the record imported nothing
        assert not [n for n in sys.modules if n.startswith(SUBPKG)]
    assert kernel.ns["task"](1) == 2  # resolves the proxy in the kernel
    after = project_fingerprints(kernel.ns["task"])["code"]
    assert f"{SUBPKG}{expected} (file)" in before
    assert before == after


@lazy_imports
def test_two_cache_dir_runs_with_a_lazy_submodule_reuse(kernel, sub_project, tmp_path):
    """The first run's record is taken before the proxy resolves, the
    second's after: the key is the same."""
    for cell in [
        "import mushin\nfrom mushin import multirun\n\nCALLS = []\n",
        f"lazy import {SUBPKG}.sub\n",
        "@mushin.sweep\ndef task(a):\n"
        f"    CALLS.append(a)\n    return {{'v': {SUBPKG}.sub.f(a)}}\n",
    ]:
        kernel.run(cell)
    cache = str(tmp_path / "cache")
    _sweep(kernel, tmp_path / "one", [1, 2], cache_dir=cache)
    _sweep(kernel, tmp_path / "two", [1, 2], cache_dir=cache)
    assert kernel.ns["CALLS"] == [1, 2]


# --- bindings and where a unit stands (#292) --------------------------------------


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (f"def task(a):\n    m = {SUBPKG}\n    return m.sub.f(a)\n", ".sub"),
        (f"def task(a):\n    s = {SUBPKG}.a\n    return s.b.f(a)\n", ".a.b"),
        (f"def task(a):\n    m = {SUBPKG}\n    n = m\n    return n.a.b.f(a)\n", ".a.b"),
        (f"def task(a, lib={SUBPKG}):\n    return lib.sub.f(a)\n", ".sub"),
        (f"def task(a, {SUBPKG}={SUBPKG}):\n    return {SUBPKG}.sub.f(a)\n", ".sub"),
        # Not a unit, so no value of its default is read: the syntax tells.
        (
            f"def task(a):\n    def inner(x, lib={SUBPKG}):\n"
            "        return lib.sub.f(x)\n\n    return inner(a)\n",
            ".sub",
        ),
        # The closure holds `m`, not `lib`: the alias's own cell counts.
        (
            "def make(lib):\n    m = lib\n\n    def task(a):\n"
            "        return m.sub.f(a)\n\n    return task\n\n\n"
            f"task = make({SUBPKG})\n",
            ".sub",
        ),
        (
            f"def make():\n    import {SUBPKG}.a as s\n\n"
            "    def task(a, lib=s):\n        return lib.b.f(a)\n\n    return task\n\n\n"
            "task = make()\n",
            ".a.b",
        ),
    ],
    ids=[
        "alias",
        "alias of a submodule",
        "alias of an alias",
        "default",
        "default named like the module",
        "a nested function's default",
        "alias in a factory",
        "factory default",
    ],
)
def test_an_alias_of_the_package_is_followed(kernel, sub_project, task, expected):
    kernel.run(f"import {SUBPKG}.sub\nimport {SUBPKG}.a.b\n")
    kernel.run(task)
    assert f"{SUBPKG}{expected} (file)" in _files(kernel, "task")


@pytest.mark.parametrize(
    "helper",
    [
        f"def helper({SUBPKG}, a):\n    return {SUBPKG}.sub.f(a)\n",
        f"def helper(m, a):\n    {SUBPKG} = m\n    return {SUBPKG}.sub.f(a)\n",
    ],
    ids=["parameter", "alias of a parameter"],
)
def test_a_name_the_syntax_cannot_tell_reads_the_namespaces_module(
    kernel, sub_project, helper
):
    """A caller can pass the module itself, so a name the syntax cannot tell
    counts the namespace's module of that name: a recompute if it is
    something else, never a missed edit."""
    kernel.run(f"import {SUBPKG}.sub\n")
    kernel.run(helper)
    kernel.run(f"def task(a):\n    return helper({SUBPKG}, a)\n")
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_a_class_body_reads_the_global_before_rebinding_it(kernel, sub_project):
    kernel.run(f"import {SUBPKG}.sub\n")
    kernel.run(
        f"class Model:\n    v = {SUBPKG}.sub.CONST\n\n"
        f"    def {SUBPKG}(self):\n        pass\n\n"
        "    def f(self, x):\n        return x\n"
    )
    kernel.run("def task(a):\n    return Model().f(a)\n")
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_a_class_bodys_comprehension_reads_its_first_iterable_there(
    kernel, sub_project
):
    kernel.run(f"import {SUBPKG}.a.b\nfrom {SUBPKG} import sub as utils\n")
    kernel.run(
        f"class Model:\n    from {SUBPKG} import a as utils\n\n"
        "    vals = [x for x in [utils.b.f(1)]]\n\n"
        "    def f(self, x):\n        return x\n"
    )
    kernel.run("def task(a):\n    return Model().f(a)\n")
    assert f"{SUBPKG}.a.b (file)" in _files(kernel, "task")


@pytest.mark.parametrize(
    "body",
    [
        "".join(f"    x = x.{'left' if i % 2 else 'right'}\n" for i in range(40)),
        "    a = x.l\n    a = x.r\n    x = a.l\n    x = a.r\n" * 10,
        "    a0 = x.l\n    a0 = x.r\n"
        + "".join(
            f"    a{i} = a{i - 1}.l\n    a{i} = a{i - 1}.r\n" for i in range(1, 40)
        ),
    ],
    ids=["rebinding itself", "a cycle", "a doubling chain"],
)
def test_rebinding_aliases_stay_bounded(kernel, sub_project, body):
    """A local rebound from itself, two locals rebound from each other, and a
    chain whose targets double at each step bind as unknown or stop at the
    cap: resolving them in full would multiply targets each time. The cell
    holds the reached task, so the whole cell is read into scopes."""
    from mushin._resume import _TARGETS_MAX, _cell_lines, _cell_tree, _index_entry

    kernel.run(f"import {SUBPKG}.sub\n")
    kernel.run(
        f"def walk(x):\n{body}    return 0\n\n\n"
        f"def task(a):\n    return {SUBPKG}.sub.f(a)\n"
    )
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")
    tree = _cell_tree(_cell_lines(kernel.ns["task"].__code__.co_filename))
    scopes = _index_entry(tree)[1][0]
    walk = next(sc for sc in scopes if getattr(sc.node, "name", None) == "walk")
    assert max(len(targets) for targets in walk.bound.values()) <= _TARGETS_MAX


def test_a_global_declaration_reads_the_namespace(kernel, sub_project):
    kernel.run(f"import {SUBPKG}.sub\n")
    kernel.run(
        f"def task(a):\n    def {SUBPKG}():\n        pass\n\n    def inner(x):\n"
        f"        global {SUBPKG}\n        return {SUBPKG}.sub.f(x)\n\n"
        "    return inner(a)\n"
    )
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_a_nonlocal_declaration_reads_the_enclosing_function(kernel, sub_project):
    kernel.run(
        f"def task(a):\n    from {SUBPKG} import a as m\n\n    def inner(x):\n"
        "        nonlocal m\n        r = m.b.f(x)\n        m = None\n        return r\n\n"
        "    return inner(a)\n"
    )
    assert f"{SUBPKG}.a.b (file)" in _files(kernel, "task")


@pytest.mark.parametrize(
    "cells",
    [
        [
            f"def make():\n    import {SUBPKG}.sub as s\n\n    class M:\n"
            "        k = s.CONST\n\n        def f(self, x):\n            return x * self.k\n\n"
            "    return M\n\n\nm = make()()\n",
            "def task(a):\n    return m.f(a)\n",
        ],
        [
            f"def factory():\n    import {SUBPKG}.sub as s\n\n    @s.deco\n"
            "    def inner(a):\n        return a\n\n    return inner\n\n\ntask = factory()\n"
        ],
        [
            f"class Model:\n    import {SUBPKG}.sub as s\n\n"
            "    def f(self, a, k=s.f):\n        return k(a)\n\n\nzf = Model.f\n",
            "def task(a):\n    return zf(None, a)\n",
        ],
    ],
    ids=["class made in a factory", "factory's decorator", "class body's default"],
)
def test_a_module_the_enclosing_definition_binds_is_followed(
    kernel, sub_project, cells
):
    """The unit is defined inside another definition whose import names the
    module, and that is the only way to it: the class has no closure, the
    decorator and the default are evaluated where the definition stands."""
    _run_cells(kernel, cells)
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_the_cached_reading_holds_no_namespace_value(sub_project):
    """Two kernels run the same cell text, so they share its cached tree and
    reading; what `s` is comes from each kernel's own namespace."""
    one, two = Kernel("mushin_t_kernel_one"), Kernel("mushin_t_kernel_two")
    try:
        one.run(f"import {SUBPKG}.sub as s\n")
        two.run("import types\ns = types.SimpleNamespace(f=abs)\n")
        for k in (one, two):
            k.run("def task(a):\n    return s.f(a)\n")
        assert f"{SUBPKG}.sub (file)" in _files(one, "task")
        assert f"{SUBPKG}.sub (file)" not in _files(two, "task")
    finally:
        one.close()
        two.close()


def test_a_function_made_inside_another_reaches_what_its_closure_holds(
    kernel, sub_project
):
    """The enclosing function binds the module to a value the syntax cannot
    tell (a call): only the closure holds it."""
    kernel.run(
        "import importlib\n\n\ndef make():\n"
        f"    s = importlib.import_module('{SUBPKG}.sub')\n\n"
        "    def task(a):\n        return s.f(a)\n\n    return task\n\n\ntask = make()\n"
    )
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


@lazy_imports
def test_a_lazy_from_import_of_a_same_named_subpackage_records_it_alike(
    kernel, sub_project
):
    """`lazy from pkg import pkg` binds `pkg` to the subpackage `pkg.pkg`: its
    full name counts, before the proxy resolves and after."""
    kernel.run(f"lazy from {SUBPKG} import {SUBPKG}\n")
    kernel.run(f"def task(a):\n    return a if {SUBPKG} else 0\n")
    before = project_fingerprints(kernel.ns["task"])["code"]
    assert kernel.ns["task"](1) == 1  # resolves the proxy in the kernel
    after = project_fingerprints(kernel.ns["task"])["code"]
    assert f"{SUBPKG}.{SUBPKG} (file)" in before
    assert before == after


@pytest.mark.parametrize(
    "cells",
    [
        [
            f"def _warm():\n    import {SUBPKG}.sub\n\n\n_warm()\ndel _warm\n",
            f"def task(a):\n    global {SUBPKG}\n    import {SUBPKG}\n"
            f"    return {SUBPKG}.sub.f(a)\n",
        ],
        [
            f"def factory():\n    {SUBPKG} = None\n\n    def task(a):\n"
            f"        nonlocal {SUBPKG}\n        import {SUBPKG}\n"
            f"        return {SUBPKG}.sub.f(a)\n\n    return task\n\n\n"
            "task = factory()\n",
        ],
    ],
    ids=["global", "nonlocal"],
)
def test_an_import_under_a_declaration_keeps_its_module(kernel, sub_project, cells):
    """`global pkg; import pkg` binds the namespace's `pkg`, which a fresh
    kernel does not hold yet when the record is taken: the import names it."""
    _run_cells(kernel, cells)
    assert SUBPKG not in kernel.ns
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_a_walrus_in_a_comprehension_is_read_through_the_closure(kernel, sub_project):
    """PEP 572 binds the walrus in the enclosing function, so the task made
    there reads it from its closure."""
    kernel.run(
        "import importlib\n\n\ndef make():\n"
        f"    [s := importlib.import_module('{SUBPKG}.sub') for _ in range(1)]\n\n"
        "    def task(a):\n        return s.f(a)\n\n    return task\n\n\n"
        "task = make()\n"
    )
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


@lazy_imports
@pytest.mark.parametrize(
    "cell",
    [
        f"lazy import {SUBPKG}.sub as {SUBPKG}",
        f"lazy from {SUBPKG} import sub as {SUBPKG}",
    ],
)
def test_a_lazy_submodule_bound_under_its_packages_name_is_recorded(
    kernel, sub_project, cell
):
    """It binds the submodule, though its repr reads like `lazy import
    pkg.sub`: the full name counts, before the proxy resolves and after."""
    kernel.run(f"{cell}\n")
    kernel.run(f"def task(a):\n    return a if {SUBPKG} else 0\n")
    before = project_fingerprints(kernel.ns["task"])["code"]
    assert kernel.ns["task"](1) == 1
    after = project_fingerprints(kernel.ns["task"])["code"]
    assert f"{SUBPKG}.sub (file)" in before
    assert before == after


def test_a_project_namespace_package_named_like_a_stdlib_module_stays_the_projects(
    kernel, tmp_path, monkeypatch
):
    """A loaded package with no file is judged module by module, as on main,
    though the standard library lists its name (one this platform lacks)."""
    import importlib
    import importlib.util

    name = next(
        n
        for n in sorted(sys.stdlib_module_names)
        if n.isidentifier()
        and not n.startswith("_")
        and n not in sys.modules
        and importlib.util.find_spec(n) is None
    )
    (tmp_path / name).mkdir()  # a namespace package: no __init__.py
    (tmp_path / name / "mushin_t_mod.py").write_text("def f(x):\n    return x\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        # Bound directly, so `_file_entries` alone decides (a chain on the
        # package would go through the shipping filter, which judges by name).
        kernel.run(f"from {name} import mushin_t_mod\n")
        kernel.run("def task(a):\n    return mushin_t_mod.f(a)\n")
        assert f"{name}.mushin_t_mod (file)" in _files(kernel, "task")
    finally:
        for mod in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
            sys.modules.pop(mod)


def test_capping_an_alias_keeps_what_its_import_binds(kernel, sub_project):
    """Past the cap a name becomes unknown, but the module its own import
    binds still counts (the namespace need not hold the name)."""
    params = ", ".join(f"a{i}=None" for i in range(70))
    branches = "".join(f"    if x == {i}:\n        m = a{i}\n" for i in range(70))
    kernel.run(
        f"def task(x, {params}):\n    import {SUBPKG} as m\n{branches}"
        "    return m.sub.f(x)\n"
    )
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_an_alias_of_a_global_set_later_reads_the_namespaces_module(
    kernel, sub_project
):
    """`current` is None when the record is taken and set by a call the task
    makes, so the alias named like the package reads the namespace's module
    of its own name too, as main did."""
    kernel.run(f"import {SUBPKG}.sub\ncurrent = None\n")
    kernel.run(f"def init():\n    global current\n    current = {SUBPKG}\n")
    kernel.run(
        f"def task(a):\n    init()\n    {SUBPKG} = current\n"
        f"    return {SUBPKG}.sub.f(a)\n"
    )
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


BOUND = f"import {SUBPKG}.sub\n"  # the package bound in the namespace
LOADED = f"def _warm():\n    import {SUBPKG}.sub\n\n\n_warm()\ndel _warm\n"


@pytest.mark.parametrize(
    ("setup", "task"),
    [
        # A nested initializer imports into a `nonlocal` name the task binds
        # to a placeholder `def`.
        (
            BOUND,
            f"def task(a):\n    def {SUBPKG}():\n        pass\n\n    def init():\n"
            f"        nonlocal {SUBPKG}\n        import {SUBPKG}\n\n    init()\n"
            f"    return {SUBPKG}.sub.f(a)\n",
        ),
        # A comprehension rebinds the imported name to itself, with no global
        # of that name to fall back on.
        (
            LOADED,
            f"def task(a):\n    import {SUBPKG}\n"
            f"    return [{SUBPKG}.sub.f(a) for {SUBPKG} in [{SUBPKG}]][0]\n",
        ),
    ],
    ids=["nonlocal import into a placeholder", "comprehension rebinding"],
)
def test_main_s_reading_is_the_floor(kernel, sub_project, setup, task):
    """Shapes the scope model reads one way and main another (both found by
    Codex): what main named still counts, so the record never holds less
    than main's."""
    kernel.run(setup)
    kernel.run(task)
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


# --- the modules the kept cells import (#292) -------------------------------------


@pytest.mark.parametrize(
    "cells",
    [
        [
            f"import functools\nimport {SUBPKG}.sub\n",
            "def work(a, mod):\n    return mod.f(a)\n",
            f"task = functools.partial(work, mod={SUBPKG}.sub)\n",
        ],
        [
            f"import operator\nimport {SUBPKG}.sub\n\nget = operator.attrgetter('sub')\n",
            f"def task(a):\n    return get({SUBPKG}).f(a)\n",
        ],
        [
            f"import {SUBPKG}.sub\n",
            f"def task(a):\n    m = {SUBPKG} if a else None\n    return m.sub.f(a)\n",
        ],
        [
            f"import {SUBPKG}.sub\n\nMODS = [{SUBPKG}.sub]\n",
            "def task(a):\n    for m in MODS:\n        a = m.f(a)\n    return a\n",
        ],
        [f"import {SUBPKG}.sub\n", "def task(a):\n    return a\n"],
        [
            f"import {SUBPKG}\n\n\ndef setup():\n    import {SUBPKG}.sub\n",
            "def task(a):\n    return a\n",
        ],
    ],
    ids=[
        "an argument",
        "a call's result",
        "a conditional alias",
        "a loop over modules",
        "not named by the task",
        "in a function the task does not reach",
    ],
)
def test_every_project_module_a_cell_imports_counts(kernel, cell_imports, cells):
    """A submodule's code runs only if something imported it, and a cell's
    import is there however the task reaches the module, including ways its
    syntax cannot follow."""
    _run_cells(kernel, cells)
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_a_module_no_cell_imports_by_a_statement_is_not_a_cells(kernel, cell_imports):
    """Load state does not count: a submodule only `importlib` loaded, which
    the task never names, is not recorded; nor is a relative import's
    target, an installed package, or what a file's text imports."""
    kernel.run(
        f"import importlib\nimport numpy\nimport {SUBPKG}\n"
        f"importlib.import_module('{SUBPKG}.other')\n"
    )
    kernel.run(f"def task(a):\n    from .{SUBREL} import f\n    return f(a)\n")
    # A file's text in linecache (a `%run` file, a traceback's): it has an
    # mtime, and the file can have changed since.
    ran = str(cell_imports / "ran.py")
    linecache.cache[ran] = (30, 1793100000.0, [f"import {SUBPKG}.other\n"], ran)
    try:
        files = _files(kernel, "task")
    finally:
        linecache.cache.pop(ran, None)
    assert f"{SUBPKG} (file)" in files
    assert not {f"{SUBPKG}.other (file)", f"{SUBREL} (file)"} & files
    assert not [k for k in files if k.startswith("numpy")]


def test_reading_a_cell_shows_no_warning_again(kernel, cell_imports):
    """The kernel showed a cell's warning (an invalid escape sequence) when
    it ran; the record parses the cell again and shows nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel.run(f"import {SUBPKG}.sub\nPATTERN = '\\d'  # mushin_t_escape_one\n")
        kernel.run(
            f"def task(a):\n    return '\\d' and {SUBPKG}.sub.f(a)  # mushin_t_escape\n"
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = project_fingerprints(kernel.ns["task"])
    assert f"{SUBPKG}.sub (file)" in record["code"]
    assert not [w for w in caught if w.filename == "<cell>"], caught


def test_the_cells_imports_count_when_the_walk_is_cut_or_fails(
    kernel, cell_imports, monkeypatch
):
    """The cells' imports do not depend on the walk, so they count when it
    passes its bound or fails (its own modules do not)."""
    import mushin._resume as resume

    kernel.run(f"import {SUBPKG}.other\n")
    kernel.run("".join(f"def h{i}(x):\n    return x + {i}\n\n\n" for i in range(10)))
    kernel.run(
        "def task(a):\n    return " + " + ".join(f"h{i}(a)" for i in range(10)) + "\n"
    )
    with monkeypatch.context() as patch:
        patch.setattr(resume, "_NOTEBOOK_MAX_OBJECTS", 3)
        patch.setattr(resume, "_NOTEBOOK_TRUNCATION_WARNED", [])
        with pytest.warns(UserWarning, match="larger than the record covers"):
            assert f"{SUBPKG}.other (file)" in _files(kernel, "task")

    def fail(children):
        raise RuntimeError("after the modules were found")

    with monkeypatch.context() as patch:
        patch.setattr(resume, "_components", fail)
        with pytest.warns(UserWarning, match="could not be recorded"):
            files = _files(kernel, "task", allow_failure=True)
        assert f"{SUBPKG}.other (file)" in files


def test_a_scan_of_the_cells_that_fails_warns_and_keeps_the_entries(
    kernel, cell_imports, monkeypatch
):
    import mushin._resume as resume

    def fail(text):
        raise RuntimeError("the cell's own text")

    monkeypatch.setattr(resume, "_cell_imports", fail)
    monkeypatch.setattr(resume, "_CELL_IMPORTS", {})  # nothing cached
    monkeypatch.setattr(resume, "_CELL_SCAN_WARNED", [])
    monkeypatch.setattr(resume, "_NOTEBOOK_FAILURE_WARNED", [])
    _run_cells(
        kernel, [f"import {SUBPKG}.other\n", "def helper(x):\n    return x\n", TASK]
    )
    for _ in range(2):  # a run, then a resume: each warns once
        with pytest.warns(UserWarning) as got:
            shipped = resume.driver_script_entries((kernel.ns["task"],))
            record = project_fingerprints(kernel.ns["task"], scripts=shipped)
            project_fingerprints(kernel.ns["task"])  # a cell: no second
        messages = [str(w.message) for w in got]
        read = [m for m in messages if "cells import could not be read" in m]
        assert len(read) == 1 and "(RuntimeError)" in read[0], messages
        # The entries still count, so the walk's own warning does not fire.
        assert not [m for m in messages if "could not be recorded" in m], messages
        assert "helper (notebook)" in record["code"]
        assert f"{SUBPKG}.other (file)" not in record["code"]


def test_the_driver_ships_the_cells_modules_apart(kernel, cell_imports):
    """Under their own key, with no installed package, so a cell follows them
    after every other seed; a worker, which keeps no cells, follows them
    too."""
    from mushin._resume import (
        NOTEBOOK_LATER_MODULES,
        NOTEBOOK_MODULES,
        driver_script_entries,
    )

    kernel.run(f"import numpy\nimport {SUBPKG}.sub\nimport {SUBPKG}.a.b\n")
    kernel.run(f"def task(a):\n    return {SUBPKG}.sub.f(a)\n")
    shipped = driver_script_entries((kernel.ns["task"],))
    assert shipped[NOTEBOOK_MODULES] == [f"{SUBPKG}.sub.f"]
    assert shipped[NOTEBOOK_LATER_MODULES] == [f"{SUBPKG}.a.b", f"{SUBPKG}.sub"]
    for name in kernel.names:
        linecache.cache.pop(name, None)  # as in a worker
    assert f"{SUBPKG}.a.b (file)" in _files(kernel, "task", scripts=shipped)


def test_a_resume_after_editing_a_module_a_cell_imports_reruns(
    kernel, cell_imports, tmp_path
):
    """The task reaches the submodule through a local name its syntax cannot
    follow; the cell's import still counts, so an edit and a restart re-run
    the cells."""
    import importlib

    cells = [
        "import mushin\nfrom mushin import multirun\n\nCALLS = []\n",
        f"import {SUBPKG}.sub\n",
        "@mushin.sweep\ndef task(a):\n    CALLS.append(a)\n"
        f"    m = {SUBPKG} if a else None\n    return {{'v': m.sub.f(a)}}\n",
    ]
    for cell in cells:
        kernel.run(cell)
    wd = tmp_path / "wd"
    assert list(_sweep(kernel, wd, [1, 2])["v"].values) == [2, 3]
    kernel.close()

    (cell_imports / SUBPKG / "sub.py").write_text("def f(x):\n    return x + 100\n")
    for name in [n for n in sys.modules if n.startswith(SUBPKG)]:
        sys.modules.pop(name)
    importlib.invalidate_caches()
    restarted = Kernel()
    try:
        for cell in cells:
            restarted.run(cell)
        with pytest.warns(UserWarning, match=rf"{SUBPKG}\.sub \(file\)"):
            ds = _sweep(restarted, wd, [1, 2], resume=True)
        assert list(ds["v"].values) == [101, 102]
        assert restarted.ns["CALLS"] == [1, 2]
    finally:
        restarted.close()


@lazy_imports
@pytest.mark.parametrize(
    ("cell", "loaded", "expected"),
    [
        # The proxy names `pkg.sub`, and the task reads `pkg.other`: before it
        # resolves the proxy reads as `pkg.sub`, after as `pkg`.
        (f"lazy import {SUBPKG}.sub", "other", ".sub"),
        # Both readings are files (`pkg/other.py`, `pkg/pkg/other.py`).
        (f"lazy import {SUBPKG}.{SUBPKG}", "other", f".{SUBPKG}.other"),
        # The same, with a plain import that `__lazy_modules__` makes lazy.
        (
            f"__lazy_modules__ = ['{SUBPKG}.{SUBPKG}']\nimport {SUBPKG}.{SUBPKG}",
            "other",
            f".{SUBPKG}.other",
        ),
        # Bound under its package's name, it resolves to the submodule, and
        # the package's reading carries over.
        (f"lazy import {SUBPKG}.{SUBPKG} as {SUBPKG}", f"{SUBPKG}.other", ".other"),
    ],
    ids=[
        "proxy named past",
        "both readings files",
        "made lazy by the namespace",
        "bound under the package's name",
    ],
)
def test_a_lazy_import_the_task_reads_past_records_alike(
    kernel, cell_imports, cell, loaded, expected
):
    (cell_imports / SUBPKG / SUBPKG / "other.py").write_text(
        "def f(x):\n    return x + 1\n"
    )
    kernel.run(
        f"{cell}\nimport importlib\nimportlib.import_module('{SUBPKG}.{loaded}')\n"
    )
    kernel.run(f"def task(a):\n    return {SUBPKG}.other.f(a)\n")
    assert type(kernel.ns[SUBPKG]) is types.LazyImportType
    before = project_fingerprints(kernel.ns["task"])["code"]
    assert kernel.ns["task"](1) == 2  # resolves the proxy in the kernel
    assert type(kernel.ns[SUBPKG]) is types.ModuleType
    after = project_fingerprints(kernel.ns["task"])["code"]
    assert f"{SUBPKG}{expected} (file)" in before
    assert before == after


def test_a_closure_the_run_rebinds_records_alike_before_and_after(kernel, sub_project):
    """`nonlocal s; s = pkg.pkg`: the alias's targets count where the
    declaration sends the name, so the record does not follow the cell's
    value from one run to the next."""
    kernel.run(
        f"import importlib\nimport {SUBPKG}\n"
        f"importlib.import_module('{SUBPKG}.sub')\n"
        f"importlib.import_module('{SUBPKG}.{SUBPKG}.deep')\n"
    )
    kernel.run(
        f"def make():\n    s = {SUBPKG}.sub\n\n    def task(a):\n        nonlocal s\n"
        "        out = s.deep.f(a) if hasattr(s, 'deep') else s.f(a)\n"
        f"        s = {SUBPKG}.{SUBPKG}\n        return out\n\n    return task\n\n\n"
        "task = make()\n"
    )
    before = _files(kernel, "task")
    assert kernel.ns["task"](1) == 2  # rebinds the closure's `s`
    assert f"{SUBPKG}.{SUBPKG}.deep (file)" in before
    assert _files(kernel, "task") == before


@pytest.mark.parametrize(
    "params", ["a, m=mod", "a, *, m=mod"], ids=["positional", "keyword"]
)
def test_a_default_holding_a_module_records_its_submodule(kernel, sub_project, params):
    """The default's value is what the parameter holds when no caller passes
    one, and here the syntax cannot tell it (`m=mod`, in a factory)."""
    kernel.run(
        f"import importlib\n\n\ndef make(mod):\n    def task({params}):\n"
        "        return m.deep.f(a)\n\n    return task\n\n\n"
        f"task = make(importlib.import_module('{SUBPKG}.{SUBPKG}'))\n"
        f"importlib.import_module('{SUBPKG}.{SUBPKG}.deep')\n"
    )
    assert f"{SUBPKG}.{SUBPKG}.deep (file)" in _files(kernel, "task")


def test_a_free_variable_the_unit_also_reads_as_a_global_is_read_both_ways(
    kernel, sub_project
):
    """`g` reads the namespace's `m` (`global m`), the task its closure's:
    the namespace's module counts for the chain."""
    kernel.run(
        f"import importlib\nimport {SUBPKG} as m\nimport types\n"
        f"importlib.import_module('{SUBPKG}.sub')\n"
    )
    kernel.run(
        "def make(m):\n    def task(a):\n        def g():\n            global m\n"
        "            return m.sub.f(a)\n\n        return g() + m.f(a)\n\n"
        "    return task\n\n\ntask = make(types.SimpleNamespace(f=abs))\n"
    )
    assert kernel.ns["task"](1) == 3
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_a_free_variable_reads_the_namespaces_module_too(kernel, sub_project):
    """The task's `m` is its closure's, but a chain on it reads the
    namespace's module of that name too, as main did: a recompute when that
    module changes (Codex on #291), kept for the case below."""
    kernel.run(f"import {SUBPKG} as m\nimport types\n")
    kernel.run(
        "def make(m):\n    def task(a):\n        return m.sub.f(a)\n\n    return task\n\n\n"
        "task = make(types.SimpleNamespace(sub=types.SimpleNamespace(f=abs)))\n"
    )
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


def test_a_closure_a_sibling_fills_during_the_run_reads_the_namespace(
    kernel, sub_project
):
    """The closure is empty when the record is taken, and `setup` fills it
    with what the syntax cannot tell: the namespace's module of that name is
    what records the submodule (round 3 of #294)."""
    kernel.run(
        f"import importlib\nimport {SUBPKG}\nimportlib.import_module('{SUBPKG}.sub')\n"
    )
    kernel.run(
        f"def make():\n    {SUBPKG} = None\n\n    def setup():\n"
        f"        nonlocal {SUBPKG}\n"
        f"        {SUBPKG} = importlib.import_module('{SUBPKG}')\n\n"
        f"    def task(a):\n        if {SUBPKG} is None:\n            setup()\n"
        f"        return {SUBPKG}.sub.f(a)\n\n    return task\n\n\ntask = make()\n"
    )
    before = _files(kernel, "task")
    assert kernel.ns["task"](1) == 2
    assert f"{SUBPKG}.sub (file)" in before
    assert _files(kernel, "task") == before


def _rebinds(indent: str) -> str:
    return "".join(f"{indent}x = x.{'left' if i % 2 else 'right'}\n" for i in range(40))


@pytest.mark.parametrize(
    ("walk", "most"),
    [
        (f"def walk():\n    global x\n{_rebinds('    ')}    return 0\n", 2),
        (
            "def outer():\n    x = None\n\n    def walk():\n        nonlocal x\n"
            f"{_rebinds('        ')}        return 0\n\n    return walk\n",
            4,
        ),
        (
            "def walk():\n    global x\n"
            + "".join(f"    x = x.a{i}\n" for i in range(70))
            + "    return 0\n",
            0,  # the cap
        ),
    ],
    ids=["global", "nonlocal", "past the cap"],
)
def test_rebinding_a_declared_name_stays_bounded(kernel, sub_project, walk, most):
    """`x = x.left`, `x = x.right`, ... under a declaration: each alias's
    targets go on the binding the declaration sends `x` to, and no rebinding
    feeds another's resolution, so they do not compound (fed into one another
    they reach the cap); more distinct targets than the cap stop at it."""
    from mushin._resume import _TARGETS_MAX, _cell_lines, _cell_tree, _index_entry

    kernel.run(f"import {SUBPKG}.sub\n")
    kernel.run(f"{walk}\n\ndef task(a):\n    return {SUBPKG}.sub.f(a)\n")
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")
    tree = _cell_tree(_cell_lines(kernel.ns["task"].__code__.co_filename))
    rebound = _index_entry(tree)[1][3]
    assert rebound and max(map(len, rebound.values())) <= (most or _TARGETS_MAX)


def test_ipython_before_8_6_cells_imports_count(cell_imports):
    """Its cells' linecache entries have an mtime; its own cache keeps them."""
    legacy = Kernel(legacy=True)
    try:
        legacy.run(f"import {SUBPKG}.sub\n")
        legacy.run("def task(a):\n    return a\n")
        assert f"{SUBPKG}.sub (file)" in _files(legacy, "task")
    finally:
        legacy.close()


def test_a_record_cut_short_keeps_the_seeds_main_had_before_the_cells(
    kernel, cell_imports, monkeypatch
):
    """The cells' modules come after every other seed: a cell importing a
    wide project module must not push a namespace module out of the bound
    (main recorded it)."""
    import mushin._resume as resume

    (cell_imports / SUBPKG / "wide.py").write_text(f"import {SUBWIDE}\n")
    monkeypatch.setattr(resume, "_FILES_MAX", 3)
    monkeypatch.setattr(resume, "_TRUNCATION_WARNED", [])
    monkeypatch.setattr(resume, "_LATER_TRUNCATION_WARNED", [])
    kernel.run(f"import {SUBPKG}.wide\nfrom {SUBREL} import f\n")
    kernel.run("def task(a):\n    return f(a)\n")
    for _ in range(2):  # a run, then a resume: each warns once
        with pytest.warns(UserWarning) as got:
            shipped = resume.driver_script_entries((kernel.ns["task"],))
            files = _files(kernel, "task", scripts=shipped)
            project_fingerprints(kernel.ns["task"], scripts=shipped)  # no second
        cut = [w for w in got if "cells import, and those" in str(w.message)]
        assert len(cut) == 1, [str(w.message) for w in got]
        assert not [w for w in got if "the task's code imports" in str(w.message)]
        assert f"{SUBREL} (file)" in files, files


def test_a_cell_the_parser_rejects_does_not_stop_the_scan(kernel, cell_imports):
    """A cell too deep for the parser (3.11 raises MemoryError) never ran,
    so it imports nothing; the other cells' imports still count."""
    deep = "import os\nx = " + "2**" * 3000 + "1\n"
    name = "<ipython-input-999-mushin_t_deep>"
    linecache.cache[name] = (len(deep), None, deep.splitlines(True), name)
    try:
        kernel.run(f"import {SUBPKG}.sub\n")
        kernel.run("def task(a):\n    return a\n")
        assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")
    finally:
        linecache.cache.pop(name, None)


def test_the_cell_cache_holds_the_cells_the_last_scan_saw(kernel, cell_imports):
    """A cell already read is not parsed again, and the cache is bounded by
    what linecache keeps: a cell gone from it leaves at the next scan."""
    import mushin._resume as resume

    kernel.run(f"import {SUBPKG}.other  # mushin_t_cache_probe\n")
    kernel.run("def task(a):\n    return a\n")
    assert f"{SUBPKG}.other (file)" in _files(kernel, "task")
    held = len(resume._CELL_IMPORTS)
    parsed = []
    real = resume._cell_imports

    def counted(text):
        parsed.append(text)
        return real(text)

    resume._cell_imports = counted
    try:
        assert f"{SUBPKG}.other (file)" in _files(kernel, "task")
        assert parsed == []  # nothing changed: every cell a cache hit
        linecache.cache.pop(kernel.names[0])
        assert f"{SUBPKG}.other (file)" not in _files(kernel, "task")
    finally:
        resume._cell_imports = real
    assert len(resume._CELL_IMPORTS) == held - 1


@lazy_imports
def test_only_a_cells_own_imports_bind_a_lazy_name(kernel, cell_imports):
    """An import in a function binds a local, never the namespace's name, so
    it adds no reading of a proxy (here `__lazy_modules__` makes every
    import a candidate)."""
    (cell_imports / SUBPKG / SUBPKG / "other.py").write_text(
        "def f(x):\n    return x + 1\n"
    )
    kernel.run(
        f"__lazy_modules__ = ['mushin_t_none']\nimport importlib\nimport {SUBPKG}\n"
        f"importlib.import_module('{SUBPKG}.other')\n\n\n"
        f"def setup():\n    import {SUBPKG}.{SUBPKG}\n"
    )
    kernel.run(f"def task(a):\n    return {SUBPKG}.other.f(a)\n")
    files = _files(kernel, "task")
    assert f"{SUBPKG}.other (file)" in files
    assert f"{SUBPKG}.{SUBPKG} (file)" in files  # still a cell's import
    assert f"{SUBPKG}.{SUBPKG}.other (file)" not in files


def test_a_rebinding_in_another_function_counts_for_the_binding(kernel, sub_project):
    """`rebind` sets the `s` that `task` reads (`nonlocal`, or `global` for the
    namespace's): what it can hold counts for `task`'s chains, so the record
    reads alike before and after a run rebinds it."""
    kernel.run(
        f"import importlib\nimport {SUBPKG}\n"
        f"importlib.import_module('{SUBPKG}.sub')\n"
        f"importlib.import_module('{SUBPKG}.{SUBPKG}.deep')\n"
    )
    kernel.run(
        f"def make():\n    s = {SUBPKG}.sub\n\n    def rebind():\n        nonlocal s\n"
        f"        s = {SUBPKG}.{SUBPKG}\n\n    def task(a):\n"
        "        out = s.deep.f(a) if hasattr(s, 'deep') else s.f(a)\n"
        "        rebind()\n        return out\n\n    return task\n\n\n"
        "task = make()\n"
    )
    before = _files(kernel, "task")
    assert kernel.ns["task"](1) == 2  # rebinds the closure's `s`
    assert f"{SUBPKG}.{SUBPKG}.deep (file)" in before
    assert _files(kernel, "task") == before


def test_an_import_in_another_function_counts_for_the_binding(kernel, sub_project):
    """`nonlocal s; import pkg.pkg as s` in `rebind`: the module counts for
    `task`'s chains on `s` before the run rebinds it."""
    kernel.run(f"import importlib\nimportlib.import_module('{SUBPKG}.{SUBPKG}.deep')\n")
    kernel.run(
        "def make():\n    s = None\n\n    def rebind():\n        nonlocal s\n"
        f"        import {SUBPKG}.{SUBPKG} as s\n\n    def task(a):\n"
        "        out = a + 1 if s is None else s.deep.f(a)\n"
        "        rebind()\n        return out\n\n    return task\n\n\n"
        "task = make()\n"
    )
    before = _files(kernel, "task")
    assert kernel.ns["task"](1) == 2  # rebinds the closure's `s`
    assert f"{SUBPKG}.{SUBPKG}.deep (file)" in before
    assert _files(kernel, "task") == before


def test_a_global_rebinding_in_the_same_cell_counts(kernel, sub_project):
    """The namespace's `s` starts as a value: a module it held would itself
    be recorded by value, and drop out once rebound (a name a notebook
    function rebinds is a documented limit)."""
    kernel.run(
        f"import importlib\nimport {SUBPKG}\n"
        f"importlib.import_module('{SUBPKG}.{SUBPKG}.deep')\ns = None\n"
    )
    kernel.run(
        f"def rebind():\n    global s\n    s = {SUBPKG}.{SUBPKG}\n\n\n"
        "def task(a):\n    out = a + 1 if s is None else s.deep.f(a)\n"
        "    rebind()\n    return out\n"
    )
    before = _files(kernel, "task")
    assert kernel.ns["task"](1) == 2  # rebinds the namespace's `s`
    assert f"{SUBPKG}.{SUBPKG}.deep (file)" in before
    assert _files(kernel, "task") == before


LATER_CUT_CASES = {
    # A default holding a wide package (the round 2 probe).
    "a default": (
        "import importlib\n\n\ndef make(mod):\n    def task(a, m=mod):\n"
        "        return m.x0.X + f(a)\n\n    return task\n\n\n"
        f"task = make(importlib.import_module('{SUBWIDE}'))\n"
    ),
    # A sibling's import into the name.
    "an import rebinding": (
        "def make():\n    s = None\n\n    def rebind():\n        nonlocal s\n"
        f"        import {SUBPKG}.wide as s\n\n    def task(a):\n"
        "        return f(a) if s is None else s.X\n\n    return task\n\n\n"
        "task = make()\n"
    ),
    # A sibling's rebinding reaching a wide submodule.
    "a rebinding": (
        "def make():\n    s = None\n\n    def rebind():\n        nonlocal s\n"
        f"        s = {SUBPKG}\n\n    def task(a):\n"
        "        return f(a) if s is None else s.wide.X\n\n    return task\n\n\n"
        "task = make()\n"
    ),
}


@pytest.mark.parametrize("case", list(LATER_CUT_CASES))
def test_a_later_reading_cut_short_keeps_the_seeds_main_had(
    kernel, sub_project, monkeypatch, case
):
    """Every reading main's record did not have is followed after main's
    seeds: a wide one cannot push the namespace's module out of the bound."""
    import mushin._resume as resume

    (sub_project / SUBPKG / "wide.py").write_text(f"import {SUBWIDE}\nX = 1\n")
    monkeypatch.setattr(resume, "_FILES_MAX", 3)
    monkeypatch.setattr(resume, "_LATER_TRUNCATION_WARNED", [])
    kernel.run(f"import {SUBPKG}\nfrom {SUBREL} import f\n")
    kernel.run(LATER_CUT_CASES[case])
    with pytest.warns(UserWarning, match="reaches through a default"):
        files = _files(kernel, "task")
    assert f"{SUBREL} (file)" in files, files
    assert len(files) <= 3, files  # the later pass gets what the first left


@lazy_imports
def test_a_lazy_proxys_other_reading_cut_short_keeps_the_seeds_main_had(
    kernel, cell_imports, monkeypatch
):
    import mushin._resume as resume

    (cell_imports / SUBPKG / SUBPKG / "wide.py").write_text(
        f"import {SUBWIDE}\nX = 1\n"
    )
    monkeypatch.setattr(resume, "_FILES_MAX", 3)
    monkeypatch.setattr(resume, "_LATER_TRUNCATION_WARNED", [])
    kernel.run(f"lazy import {SUBPKG}.{SUBPKG}\nfrom {SUBREL} import f\n")
    kernel.run(f"def task(a):\n    return f(a) if a else {SUBPKG}.wide.X\n")
    kernel.run(f"{SUBPKG}\n")  # resolves the proxy
    assert type(kernel.ns[SUBPKG]) is types.ModuleType
    with pytest.warns(UserWarning, match="reach more files"):
        files = _files(kernel, "task")
    assert f"{SUBREL} (file)" in files, files
    assert len(files) <= 3, files


@lazy_imports
def test_a_lazy_reading_does_not_follow_a_name_rebound_to_another_module(
    kernel, cell_imports
):
    """`import other as pkg` after `lazy import pkg.pkg`: `pkg` holds another
    module now, so the proxy's readings do not apply to it."""
    (cell_imports / SUBPKG / SUBPKG / "other.py").write_text(
        "def f(x):\n    return x + 1\n"
    )
    kernel.run(f"lazy import {SUBPKG}.{SUBPKG}\n")
    kernel.run(f"import {SUBREL} as {SUBPKG}\n")
    kernel.run(f"def task(a):\n    return {SUBPKG}.other.f(a)\n")
    files = _files(kernel, "task")
    assert f"{SUBREL} (file)" in files
    assert not {f"{SUBPKG}.other (file)", f"{SUBPKG}.{SUBPKG}.other (file)"} & files


@pytest.mark.parametrize(
    "factory",
    [
        # `rebind` sits in `mid`, which does not bind `s`: the binding is
        # `make`'s.
        f"def make():\n    s = {SUBPKG}.sub\n\n    def mid():\n        def rebind():\n"
        f"            nonlocal s\n            s = {SUBPKG}.{SUBPKG}\n\n"
        "        rebind()\n\n    def task(a):\n"
        "        out = s.deep.f(a) if hasattr(s, 'deep') else s.f(a)\n"
        "        mid()\n        return out\n\n    return task\n\n\n",
        # An outer function binds `s` too; `make`'s, the nearest, is the one.
        f"def outer():\n    s = None\n\n    def make():\n        s = {SUBPKG}.sub\n\n"
        f"        def rebind():\n            nonlocal s\n            s = {SUBPKG}.{SUBPKG}\n\n"
        "        def task(a):\n"
        "            out = s.deep.f(a) if hasattr(s, 'deep') else s.f(a)\n"
        "            rebind()\n            return out\n\n        return task\n\n"
        "    return make\n\n\nmake = outer()\n",
    ],
    ids=["through a function that does not bind it", "the nearest binding"],
)
def test_a_rebinding_goes_on_the_binding_it_rebinds(kernel, sub_project, factory):
    kernel.run(
        f"import importlib\nimport {SUBPKG}\n"
        f"importlib.import_module('{SUBPKG}.sub')\n"
        f"importlib.import_module('{SUBPKG}.{SUBPKG}.deep')\n"
    )
    kernel.run(f"{factory}task = make()\n")
    before = _files(kernel, "task")
    assert kernel.ns["task"](1) == 2  # rebinds the closure's `s`
    assert f"{SUBPKG}.{SUBPKG}.deep (file)" in before
    assert _files(kernel, "task") == before


@pytest.mark.parametrize("shape", ["filled by a sibling", "a parameter"])
def test_a_free_variables_namespace_read_is_one_of_mains_seeds(
    kernel, cell_imports, monkeypatch, shape
):
    """The namespace read of a free variable is main's reading, so it is
    followed in the first pass: later modules filling the rest of the bound
    cannot push it out."""
    import mushin._resume as resume

    monkeypatch.setattr(resume, "_FILES_MAX", 3)
    monkeypatch.setattr(resume, "_TRUNCATION_WARNED", [])
    monkeypatch.setattr(resume, "_LATER_TRUNCATION_WARNED", [])
    kernel.run(
        f"import importlib\nimport {SUBPKG}\nimport {SUBPKG}.a.b\n"
        f"importlib.import_module('{SUBPKG}.sub')\nimport types\n"
    )
    if shape == "a parameter":
        kernel.run(
            f"def make({SUBPKG}):\n    def task(a):\n"
            f"        return {SUBPKG}.sub.f(a)\n\n    return task\n\n\n"
            "task = make(types.SimpleNamespace(sub=types.SimpleNamespace(f=abs)))\n"
        )
    else:
        kernel.run(
            f"def make():\n    {SUBPKG} = None\n\n    def setup():\n"
            f"        nonlocal {SUBPKG}\n"
            f"        {SUBPKG} = importlib.import_module('{SUBPKG}')\n\n"
            f"    def task(a):\n        if {SUBPKG} is None:\n            setup()\n"
            f"        return {SUBPKG}.sub.f(a)\n\n    return task\n\n\ntask = make()\n"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the later pass is cut
        files = _files(kernel, "task")
    assert f"{SUBPKG}.sub (file)" in files, files


# --- classes with no method of their own (#292) ----------------------------------


def _own(kernel, name: str = "Config") -> str:
    return _record(kernel, "task")[f"{name} (notebook)"]


def test_an_ambiguous_class_recomputes_once_then_reuses(kernel):
    """Two kept texts of a factory's class, which nothing tells apart: the
    digest is new, then the same while the object stays, and new again once
    the cell runs again (a new object)."""
    factory = (
        "def make():\n    class C:\n        K = {}\n\n    return C\n\n\nC = make()\n"
    )
    _run_cells(kernel, [factory.format(2), "def task(a):\n    return a * C.K\n"])
    exact = _own(kernel, "C")
    kernel.run(factory.format(3))
    first = _own(kernel, "C")
    assert first != exact
    assert _own(kernel, "C") == first
    kernel.run(factory.format(3))
    assert _own(kernel, "C") != first


@pytest.mark.skipif(sys.version_info < (3, 13), reason="__firstlineno__ is 3.13+")
@pytest.mark.parametrize("decorated", [False, True], ids=["plain", "decorated"])
def test_a_class_s_first_line_tells_its_texts_apart(kernel, decorated):
    """3.13+: a reverted cell's class is found exactly by its first line (a
    decorated class's first decorator's), so the record is the first one
    again. An `Enum` has no annotations, so no `__annotate_func__` finds it."""
    deco = "@enum.unique\n" if decorated else ""
    v1 = f"import enum\n\n\n{deco}class Color(enum.Enum):\n    RED = 1\n"
    v2 = f"{deco}class Color(enum.Enum):\n    RED = 2\n"
    _run_cells(kernel, [v1, "def task(a):\n    return a + Color.RED.value\n"])
    first = _own(kernel, "Color")
    kernel.run(v2)
    assert _own(kernel, "Color") != first
    kernel.run(v1)
    assert _own(kernel, "Color") == first


@pytest.mark.skipif(sys.version_info < (3, 14), reason="__annotate_func__ is 3.14+")
@pytest.mark.parametrize("slots", [False, True], ids=["dataclass", "slots"])
def test_a_class_with_annotations_is_found_through_its_annotate_function(kernel, slots):
    """3.14+: the compiler's `__annotate_func__` names the cell, so two texts
    on the same line are told apart (a `slots=True` dataclass, a new class,
    keeps it)."""
    head = f"@dataclasses.dataclass(slots={slots})\nclass Cfg:\n    lr: float = "
    _run_cells(
        kernel,
        [
            "import dataclasses\n",
            head + "0.1\n",
            "def task(a):\n    return a * Cfg().lr\n",
        ],
    )
    first = _own(kernel, "Cfg")
    kernel.run(head + "0.2\n")
    assert _own(kernel, "Cfg") != first
    kernel.run(head + "0.1\n")
    assert _own(kernel, "Cfg") == first


CELL_WATCH_SCRIPT = """
import importlib
import io
import json
import sys
import weakref
from contextlib import redirect_stderr, redirect_stdout

from IPython.core.interactiveshell import InteractiveShell

shell = InteractiveShell.instance()
out, err = io.StringIO(), io.StringIO()


def run(cell):
    with redirect_stdout(out), redirect_stderr(err):
        shell.run_cell(cell, store_history=True)


def own():
    from mushin._resume import project_fingerprints

    return project_fingerprints(shell.user_ns["task"])["code"]["Config (notebook)"]


run("class Early:\\n    X = 1\\n")  # before the watch starts
run("class Early3:\\n    X = 1\\n\\n\\nsaved3 = Early3\\ndel Early3\\n")
early_inst = "class EarlyI:\\n    X = 1\\n\\n\\nei = EarlyI()\\ndel EarlyI\\n"
run(early_inst)  # held only by an instance, before the watch
run("class Pre7:\\n    X = 1\\n\\n\\nreg7 = [Pre7]\\ndel Pre7\\nreg7[0]\\n")
shown_before = shell.execution_count - 1  # its output is `_<n>` from here on
run("import mushin._resume")  # the watch starts here
v1 = "class Config:\\n    LR = 0.1\\n"
run(v1)
run("def task(a):\\n    return a * Config.LR\\n")
first = own()
run("class Config:\\n    LR = 0.2\\n")
edited = own()
run(v1)  # reverted: the same text as the first cell
reverted = own()
run("raise ValueError('before the statement')\\nclass Config:\\n    LR = 0.3\\n")
raised = own()
run("%%capture\\nclass Config:\\n    LR = 0.4\\n")
magic = own()
run("%%capture\\nclass Config:\\n    LR = 0.4\\n")
magic_again = own()
run("class Config:\\n    LR = 0.4\\n")
plain = own()
run("%%time\\nclass Config:\\n    LR = 0.6\\n")
timed = own()
timed_again = own()  # the same object: the same digest
run("%%time\\nclass Config:\\n    LR = 0.6\\n")  # unchanged
timed_same = own()
ifelse = (
    "FLAG = True\\n\\nif FLAG:\\n\\n    class Config:\\n        LR = 1\\n\\n"
    "else:\\n\\n    class Config:\\n        LR = 2\\n"
)
run(ifelse)
branch = own()
run(ifelse)
branch_again = own()
# A nested class, pinned with the class whose body made it.
outer = "class Outer:\\n    class Inner:\\n        X = {}\\n\\n\\nInner = Outer.Inner\\n"
run(outer.format(1))
run("def task3(a):\\n    return a * Inner.X\\n")
from mushin._resume import project_fingerprints as _fp3


def inner():
    return _fp3(shell.user_ns["task3"])["code"]["Inner (notebook)"]


nested = [inner()]
run(outer.format(2))
nested.append(inner())
run(outer.format(1))
nested.append(inner())
# A nested class reached as a base and recorded inside its own cell, its value
# toggled 1, 2, 1, 2: pinned through its outer class, so each is exact.
run("def task5(a):\\n    return a * D.x\\n")
toggled = []
for value in (1, 2, 1, 2):
    run(
        "from mushin._resume import project_fingerprints as _pf5\\n"
        f"class Outer:\\n    class Inner:\\n        x = {value}\\n\\n\\n"
        "class D(Outer.Inner):\\n    pass\\n\\n\\n"
        "_T = _pf5(task5)['code']['D (notebook)']\\n"
    )
    toggled.append(shell.user_ns["_T"])
# A class `type()` remakes after a watched statement: not that statement's.
run("class Mk:\\n    X = 1\\n")
run("def task4(a):\\n    return a * Mk.X\\n")
remade = [_fp3(shell.user_ns["task4"])["code"]["Mk (notebook)"]]
run("Mk = type('Mk', (), {'X': 2})\\n")
remade.append(_fp3(shell.user_ns["task4"])["code"]["Mk (notebook)"])
# An empty nested cell before the statement fires post without pre.
empty = (
    "get_ipython().run_cell('')\\n"
    "from mushin._resume import project_fingerprints as _pf\\n"
    "class Config:\\n    LR = 0.9\\n\\n"
    "_C = _pf(task)['code']['Config (notebook)']\\n"
)
run(empty)
emptied = [shell.user_ns["_C"], own()]
run(empty)
emptied += [shell.user_ns["_C"], own()]
# A class with one kept text, then a `%%time` cell remaking it: the kept text
# is not this class's statement.
run("class Solo:\\n    X = 1\\n")
run("def task2(a):\\n    return a * Solo.X\\n")
from mushin._resume import project_fingerprints as _fp
solo = _fp(shell.user_ns["task2"])["code"]["Solo (notebook)"]
run("%%time\\nclass Solo:\\n    X = 2\\n")
solo_timed = _fp(shell.user_ns["task2"])["code"]["Solo (notebook)"]
run(
    "from mushin._resume import project_fingerprints as _pf\\n"
    "class Config:\\n    LR = 0.7\\n\\n_A = _pf(task)['code']['Config (notebook)']\\n"
    "class Config:\\n    LR = 0.8\\n\\n_B = _pf(task)['code']['Config (notebook)']\\n"
)
twice = [shell.user_ns["_A"], shell.user_ns["_B"]]
run(
    "from mushin._resume import project_fingerprints as _pf\\n"
    "class Config:\\n    LR = 0.7\\n\\n_A = _pf(task)['code']['Config (notebook)']\\n"
    "class Config:\\n    LR = 0.8\\n\\n_B = _pf(task)['code']['Config (notebook)']\\n"
)
twice += [shell.user_ns["_A"], shell.user_ns["_B"]]
# A `global` factory, called from a cell that only mentions its class: in a
# comment, a string, a multi-line string, a string on a `%time` line, a `%`
# continuation. The cell states no such class, so whether the factory made it
# cannot be told: ambiguous, so an edit to the factory is never missed. A name
# a function rebinds is not followed, so the task reaches it as an instance's.
factory = (
    "def build():\\n    global {0}\\n\\n    class {0}:\\n        LR = {1}\\n\\n"
    "    return {0}()\\n"
)
run("def task8(a):\\n    return a * cfg.LR\\n")


def built():
    return _fp(shell.user_ns["task8"])["code"]["cfg (notebook)"]


mentions = {}
for kind, name, call in (
    ("comment", "Built1", "cfg = build()  # rebuild class Built1\\n"),
    ("string", "Built2", "_S = 'class Built2'; cfg = build()\\n"),
    ("docstring", "Built3", "_S = '''\\nclass Built3:\\n'''\\ncfg = build()\\n"),
    ("magic line", "Built4", "%time cfg = build(); _S = 'a class Built4'\\n"),
    ("continuation", "Built5", "cfg = build()\\n_S = ('%s'\\n      % 'the class Built5')\\n"),
):
    run(factory.format(name, 0.11))
    run(call)
    before = built()
    run(call)  # unchanged
    rerun = built()
    run(factory.format(name, 0.12))
    run(call)
    mentions[kind] = [before, rerun, built()]
run("del build, cfg")  # `own()` follows `Config` again
# A class remade from text that is not kept, after one kept statement: never
# that statement's digest. A `%time` line magic, its automagic form, an `exec`.
remakes = {}
for kind, name, remake in (
    ("%time", "Solo2", "%time class Solo2: X = 2\\n"),
    ("automagic", "Solo3", "time class Solo3: X = 2\\n"),
    ("exec", "Solo4", "exec('class Solo4: X = 2')\\n"),
):
    run(f"class {name}:\\n    X = 1\\n")
    run(f"def task7(a):\\n    return a * {name}.X\\n")
    kept = _fp(shell.user_ns["task7"])["code"][f"{name} (notebook)"]
    run(remake)
    remakes[kind] = [kept, _fp(shell.user_ns["task7"])["code"][f"{name} (notebook)"]]
# A class whose cell assigns its `__module__`, which deletes 3.13+'s
# `__firstlineno__`: still its statement's, so an edit and a revert are exact.
moved = "class Moved:\\n    X = {}\\n\\n\\nMoved.__module__ = __name__\\n"
run("def task9(a):\\n    return a * Moved.X\\n")
module_set = []
for value in (1, 2, 1):
    run(moved.format(value))
    module_set.append(_fp(shell.user_ns["task9"])["code"]["Moved (notebook)"])
# A kept class whose comment line above ends in the word `class`: still
# pinned (the snapshot reads the namespace, not the text), so exact.
run("def task10(a):\\n    return a * Rev.X\\n")
commented = []
for value in (1, 2, 1):
    run(f"# the Rev class\\nclass Rev:\\n    X = {value}\\n")
    commented.append(_fp(shell.user_ns["task10"])["code"]["Rev (notebook)"])
# A nested class remade from text that is not kept, after one kept version:
# never the kept nested statement's digest (it is no factory's).
nested_remakes = {}
for kind, outer, remake in (
    ("%%time", "Box", "%%time\\nclass Box:\\n    class Cfg:\\n        X = 2\\n\\n\\ncfg = Box.Cfg()\\n"),
    ("exec", "Box2", "exec('class Box2:\\\\n    class Cfg:\\\\n        X = 2\\\\n')\\ncfg = Box2.Cfg()\\n"),
):
    run(f"class {outer}:\\n    class Cfg:\\n        X = 1\\n\\n\\ncfg = {outer}.Cfg()\\n")
    run("def task11(a):\\n    return a * cfg.X\\n")
    kept = _fp(shell.user_ns["task11"])["code"]["cfg (notebook)"]
    run(remake)
    nested_remakes[kind] = [kept, _fp(shell.user_ns["task11"])["code"]["cfg (notebook)"]]
run("del cfg")
# A record taken in a nested cell that only mentions the class its outer
# cell just stated: pinned to the outer cell, so an edit and a revert are
# exact.
run("def task12(a):\\n    return a * Nest.X\\n")
in_nested = []
for value in (1, 2, 1):
    run(
        "from mushin._resume import project_fingerprints as _pf\\n"
        f"class Nest:\\n    X = {value}\\n\\n\\n"
        "get_ipython().run_cell(\\"_N = _pf(task12)['code']['Nest (notebook)']  # class Nest\\")\\n"
    )
    in_nested.append(shell.user_ns["_N"])
# A class a later cell derives from a kept statement: that cell states
# none, so the class is ambiguous, and an edit to the statement is seen.
derived = {}
for kind, name, derive in (
    ("slots dataclass", "Der1", "import dataclasses\\nDer1 = dataclasses.dataclass(slots=True)(Der1)\\n"),
    ("type copy", "Der2", "Der2 = type(Der2)(Der2.__name__, Der2.__bases__, dict(Der2.__dict__))\\n"),
    ("exec of In", "Der3", "exec(In[-2])\\n"),
):
    run(f"def task15(a):\\n    return a * {name}().X\\n")
    derived[kind] = []
    for value in (1, 2):
        run(f"class {name}:\\n    X: int = {value}\\n")
        run(derive)
        derived[kind].append(_fp(shell.user_ns["task15"])["code"][f"{name} (notebook)"])
# A one-line cell whose first name an object with a raising `__class__`
# hook binds, remaking a class by `exec`: the watch's transform reads no
# namespace, so no hook runs, and the class is pinned (never the kept text).
run(
    "class Proxy:\\n    def __getattribute__(self, n):\\n        if n == '__class__':\\n"
    "            print('proxy-class-read')\\n            raise RuntimeError(n)\\n"
    "        return object.__getattribute__(self, n)\\n"
)
run("class Plain9:\\n    X = 1\\n")
run("def task16(a):\\n    return a * Plain9.X\\n")
proxy = [_fp(shell.user_ns["task16"])["code"]["Plain9 (notebook)"]]
proxy_mark = len(out.getvalue())
run("pp9 = Proxy(); exec('class Plain9:\\\\n    X = 2\\\\n')\\n")
proxy.append(_fp(shell.user_ns["task16"])["code"]["Plain9 (notebook)"])
proxy.append("proxy-class-read" in out.getvalue()[proxy_mark:])
run("del pp9")
# A class saved under an alias before the watch started, restored after it:
# the namespace held it when that cell started, so the cell did not make it,
# and its kept statement is read, as after an unchanged re-run.
run("def task17(a):\\n    return a * Early3.X\\n")
run("Early3 = saved3")
restored = [_fp(shell.user_ns["task17"])["code"]["Early3 (notebook)"]]
run("class Early3:\\n    X = 1\\n")
restored.append(_fp(shell.user_ns["task17"])["code"]["Early3 (notebook)"])
# A class made under the watch from text that is not kept, left only under
# an alias, then restored: pinned by object to the cell that made it, so
# ambiguous, never the older kept statement's digest.
aliased = {}
for kind, name, make in (
    ("exec", "Ali1", "exec('class Ali1:\\\\n    X = 2\\\\n')\\nsaved_Ali1 = Ali1\\ndel Ali1\\n"),
    ("%%time", "Ali2", "%%time\\nclass Ali2:\\n    X = 2\\n\\n\\nsaved_Ali2 = Ali2\\ndel Ali2\\n"),
):
    run(f"class {name}:\\n    X = 1\\n")
    run(f"def task18(a):\\n    return a * {name}.X\\n")
    kept = _fp(shell.user_ns["task18"])["code"][f"{name} (notebook)"]
    run(make)
    run(f"{name} = saved_{name}\\n")
    aliased[kind] = [kept, _fp(shell.user_ns["task18"])["code"][f"{name} (notebook)"]]
# A class only an instance holds, recorded inside a cell that states its
# name at the same first line before running the statement: the cell that
# made it is found through the instance, so it is read from that cell
# exactly (an unchanged re-run keeps the digest, an edit changes it), never
# from the record cell's statement.
run("def task19(a):\\n    return a * Inst.LR\\n")
run("from mushin._resume import project_fingerprints as _pf\\n")
record_cell = "_I = _pf(task19)['code']['Inst (notebook)']\\nclass Inst:\\n    LR = 0.5\\n"
instance_only = []
for value in (0.1, 0.1, 0.2):
    run(f"x = 0\\nclass Inst:\\n    LR = {value}\\n\\n\\nInst = Inst()\\n")
    run(record_cell)
    instance_only.append(shell.user_ns["_I"])
# A nested class whose outer class a cell keeps only under an alias, recorded
# inside that cell: its outer class is bound (by `savedA`), so both are
# pinned there, and the kept nested statement never stands in.
run("class OutA:\\n    class Inner:\\n        X = 1\\n\\n\\ncfg2 = OutA.Inner()\\n")
run("def task20(a):\\n    return a * cfg2.X\\n")
alias_nested = [_fp(shell.user_ns["task20"])["code"]["cfg2 (notebook)"]]
run(
    "from mushin._resume import project_fingerprints as _pf\\n"
    "exec('class OutA:\\\\n    class Inner:\\\\n        X = 2\\\\n')\\n"
    "savedA = OutA\\ndel OutA\\ncfg2 = savedA.Inner()\\n"
    "_J = _pf(task20)['code']['cfg2 (notebook)']\\n"
)
alias_nested.append(shell.user_ns["_J"])
run("del cfg2")
# A class held only by an instance (or reached as a base), remade from text
# that is not kept: the record reaches it, so the cell that made it is found,
# and it is ambiguous, never the kept statement's digest. Recorded inside the
# remaking cell and after it.
reached = {}
for kind, name, remake in (
    ("%%time singleton", "Inr1", "%%time\\nclass Inr1:\\n    LR = 0.2\\n\\n\\nInr1 = Inr1()\\nobj_Inr1 = Inr1\\n"),
    ("exec singleton", "Inr2", "exec('class Inr2:\\\\n    LR = 0.2\\\\n')\\nInr2 = Inr2()\\nobj_Inr2 = Inr2\\n"),
    ("exec, then del", "Inr3", "exec('class Inr3:\\\\n    LR = 0.2\\\\n')\\nobj_Inr3 = Inr3()\\ndel Inr3\\n"),
    ("nested, outer deleted", "Inr4", "exec('class Inr4:\\\\n    class Cfg:\\\\n        LR = 0.2\\\\n')\\nobj_Inr4 = Inr4.Cfg()\\ndel Inr4\\n"),
):
    if kind.startswith("nested"):
        run(f"class {name}:\\n    class Cfg:\\n        LR = 0.1\\n\\n\\nobj_{name} = {name}.Cfg()\\n")
    else:
        run(f"class {name}:\\n    LR = 0.1\\n\\n\\nobj_{name} = {name}()\\n")
    run(f"def task21(a):\\n    return a * obj_{name}.LR\\n")
    key = f"obj_{name} (notebook)"
    kept = _fp(shell.user_ns["task21"])["code"][key]
    run(remake + f"_R = _pf(task21)['code']['{key}']\\n")  # `_pf`: an earlier cell's
    reached[kind] = [kept, shell.user_ns["_R"], _fp(shell.user_ns["task21"])["code"][key]]
    run(f"del obj_{name}\\n")
# A function's local class (`make.<locals>.Loc`), bound by a cell that calls
# the factory: not that cell's, so it is read from the factory's cell, and an
# unchanged re-run of the call keeps the digest.
run("def make_loc():\\n    class Loc:\\n        K = 1\\n\\n    return Loc\\n")
run("Loc = make_loc()\\n")
run("def task23(a):\\n    return a * Loc.K\\n")
local = [_fp(shell.user_ns["task23"])["code"]["Loc (notebook)"]]
run("Loc = make_loc()\\n")
local.append(_fp(shell.user_ns["task23"])["code"]["Loc (notebook)"])
# A class a notebook decorator wraps (`functools.wraps(cls)(make)`), remade
# by `%%time` and by `exec`: the record follows the wrapper's `__wrapped__`,
# and so does the watch, so it is ambiguous, never the kept statement's.
run(
    "import functools\\n\\n\\ndef register(cls):\\n    def make():\\n"
    "        return cls()\\n\\n    return functools.wraps(cls)(make)\\n"
)
run("@register\\nclass Wr:\\n    LR = 0.1\\n")
run("def task24(a):\\n    return a * Wr().LR\\n")
wrapped = [_fp(shell.user_ns["task24"])["code"]["Wr (notebook)"]]
run("%%time\\n@register\\nclass Wr:\\n    LR = 0.2\\n")
wrapped.append(_fp(shell.user_ns["task24"])["code"]["Wr (notebook)"])
run("exec('@register\\\\nclass Wr:\\\\n    LR = 0.3\\\\n')\\n")
wrapped.append(_fp(shell.user_ns["task24"])["code"]["Wr (notebook)"])
# A record in a nested cell that restates one of two classes its outer cell
# stated: the nested cell claims the class it made before the outer cell's
# pin can, so the class is read from the nested cell's statement, exactly.
run("def task25(a):\\n    return a * Nx.K * Ny.K\\n")
run(
    "class Ny:\\n    K = 1\\n\\n\\nclass Nx:\\n    K = 1\\n\\n\\n"
    "get_ipython().run_cell(\\"class Ny:\\\\n    K = 2\\\\n\\\\n\\\\n"
    "_N2 = _pf(task25)['code']['Ny (notebook)']\\\\n\\")\\n"
)
restated = [shell.user_ns["_N2"]]
run("class Ny:\\n    K = 2\\n")
restated.append(_fp(shell.user_ns["task25"])["code"]["Ny (notebook)"])
# The class made before the watch and held only by an instance: in every
# snapshot, so no later cell takes it, and its kept statement is read, as
# after an unchanged re-run of its cell.
run("def task26(a):\\n    return a * ei.X\\n")
early_i = [_fp(shell.user_ns["task26"])["code"]["ei (notebook)"]]
run(early_inst)
early_i.append(_fp(shell.user_ns["task26"])["code"]["ei (notebook)"])
# An old class displayed (IPython binds `_` and `_N` to it) by a cell that
# also restates it: shown, not made, so that cell's statement is never read
# for it.
run("class Dsp:\\n    LR = 0.1\\n\\n\\nregistry = {'old': Dsp}\\ndel Dsp\\n")
run("class Dsp:\\n    LR = 0.2\\n\\n\\nregistry['old']\\n")
run("old_dsp = registry['old']\\n")
run("def task27(a):\\n    return a * old_dsp.LR * Dsp.LR\\n")
shown = _fp(shell.user_ns["task27"])["code"]
displayed = [shown["old_dsp (notebook)"], shown["Dsp (notebook)"]]
# A class held only in a list and displayed by the cell that made it, then
# subclassed under its name by a later cell: IPython's history held it when
# that cell started, so the later cell is not taken for its maker, and an
# edit to it is never missed (round 13 skipped history in the snapshot too).
run("class A13:\\n    Y = 10\\n\\n\\nreg13 = [A13]\\ndel A13\\nreg13[0]\\n")
run("class A13(reg13[0]):\\n    X = 2\\n")
run("def task29(a):\\n    return a * A13.Y\\n")
shown_base = [_fp(shell.user_ns["task29"])["code"]["A13 (notebook)"]]
run("class A13:\\n    Y = 20\\n\\n\\nreg13 = [A13]\\ndel A13\\nreg13[0]\\n")
run("class A13(reg13[0]):\\n    X = 2\\n")
shown_base.append(_fp(shell.user_ns["task29"])["code"]["A13 (notebook)"])
# A class `%%time` made, kept in a list and shown as the cell's result, then
# named by a later cell: history alone reached it at the `%%time` cell, so
# it is pinned there with no text (ambiguous), and an edit is never missed.
run("class K5:\\n    X = 1\\n")
shown_made = []
for value in (3, 7):
    run(f"%%time\\nclass K5:\\n    X = {value}\\n\\n\\nreg5 = [K5]\\ndel K5\\nreg5[0]\\n")
    run("k5 = _\\n")
    run("def task30(a):\\n    return a * k5.X\\n")
    shown_made.append(_fp(shell.user_ns["task30"])["code"]["k5 (notebook)"])
# An old class only shown by one cell, then bound by a cell that restates it.
shown_then_bound = []
for value in (1, 5):
    run(f"class K6:\\n    X = {value}\\n\\n\\nreg6 = {{'k': K6}}\\ndel K6\\n")
    run("reg6['k']\\n")
    run("class K6:\\n    X = 2\\n\\n\\nhold6 = _\\n")
    run("def task31(a):\\n    return a * hold6.X\\n")
    shown_then_bound.append(_fp(shell.user_ns["task31"])["code"]["hold6 (notebook)"])
# A class shown before the watch started (`_<n>` holds it), then named by a
# cell that states it again, unchanged: every snapshot holds it, so no cell
# is taken for its maker (none pins it with no text), and its kept
# statements, all one text, are read exactly.
run(f"class Pre7:\\n    X = 1\\n\\n\\nhold7 = _{shown_before}\\n")
run("def task32(a):\\n    return a * hold7.X * Pre7.X\\n")
pre_shown = _fp(shell.user_ns["task32"])["code"]
pre_shown = [pre_shown["hold7 (notebook)"], pre_shown["Pre7 (notebook)"]]
# Three nested cells; the outer class sorts first, so the record reaches the
# middle and the inner cells' class through cells it passes over: each cell
# claims its own class, newest first.
shell.user_ns["INNER3"] = (
    "class C3:\\n    K = 3\\n\\n\\n_T3 = _pf(task28)['code']['C3 (notebook)']\\n"
)
shell.user_ns["MIDDLE3"] = "class C3:\\n    K = 2\\n\\n\\nget_ipython().run_cell(INNER3)\\n"
run("def task28(a):\\n    return a * B3.K * C3.K\\n")
run("class B3:\\n    K = 1\\n\\n\\nget_ipython().run_cell(MIDDLE3)\\n")
three = [shell.user_ns["_T3"]]
run("class C3:\\n    K = 3\\n")
three.append(_fp(shell.user_ns["task28"])["code"]["C3 (notebook)"])
# An outer cell binds an old class to `_` as a loop's throwaway name, then a
# nested cell names it: the class was alive before both cells, so neither
# is taken for its maker, and the outer cell's statement is never read.
loop_named = []
for value in (1, 5, 1):
    run(f"class Lpd:\\n    LR = {value}\\n\\n\\nregl = {{'old': Lpd}}\\ndel Lpd\\n")
    run(
        "class Lpd:\\n    LR = 2\\n\\n\\nfor _ in regl.values():\\n    pass\\n\\n\\n"
        "get_ipython().run_cell(\\"old_lpd = regl['old']\\")\\n"
    )
    run("def task33(a):\\n    return a * old_lpd.LR\\n")
    loop_named.append(_fp(shell.user_ns["task33"])["code"]["old_lpd (notebook)"])
# A function from an earlier cell makes a class under `global`, called in a
# later cell: made during the calling cell, which states none of it.
run("def build_g():\\n    global Gcls\\n\\n    class Gcls:\\n        X = 1\\n\\n    return Gcls()\\n")
run("def task34(a):\\n    return a * g_obj.X\\n")
global_made = []
for _ in range(2):
    run("g_obj = build_g()\\n")
    global_made.append(_fp(shell.user_ns["task34"])["code"]["g_obj (notebook)"])
run("del build_g, g_obj\\n")
# A class kept only in a container, subclassed under its own name by a later
# cell that states that name: the base was made by its own cell, so it is
# read from there, and an edit to it is seen.
held_base = []
for value in (1, 5, 1):
    run(f"class Hb:\\n    X = {value}\\n\\n\\nregh = {{'k': Hb}}\\ndel Hb\\n")
    run("class Hb(regh['k']):\\n    Y = 2\\n")
    run("def task35(a):\\n    return a * Hb.X\\n")
    held_base.append(_fp(shell.user_ns["task35"])["code"]["Hb (notebook)"])
# The watch holds its snapshots weakly: a class replaced by a later cell is
# freed by a `gc.collect()` in a cell after that.
run("def task36(a):\\n    return a * len(Leak.BUF)\\n")
run(
    "class Leak:\\n    BUF = bytearray(1000)\\n\\n\\n"
    "_L = _pf(task36)['code']['Leak (notebook)']\\n"
)
_fp(shell.user_ns["task36"])  # recorded inside its cell and after it
leak_ref = weakref.ref(shell.user_ns["Leak"])
run("class Leak:\\n    BUF = b''\\n")
run("import gc\\n\\n\\ngc.collect()\\n")
leaked = leak_ref() is not None
# A base remade by `exec` and deleted, reached through a derived class.
run("class Bse:\\n    X = 1\\n")
run("class Der(Bse):\\n    pass\\n")
run("def task22(a):\\n    return a * Der.X\\n")
based = [_fp(shell.user_ns["task22"])["code"]["Der (notebook)"]]
run("exec('class Bse:\\\\n    X = 2\\\\n')\\nclass Der(Bse):\\n    pass\\n\\n\\ndel Bse\\n")
based.append(_fp(shell.user_ns["task22"])["code"]["Der (notebook)"])
# A class made before the watch started: every later cell saw it when it
# started, so none is taken for its cell; its kept statement is read, as
# after an unchanged re-run under the watch.
run("def task13(a):\\n    return a * Early.X\\n")
run("Early\\n")  # displayed (`_`, `_N`): the namespace held it, so not new
early = [_fp(shell.user_ns["task13"])["code"]["Early (notebook)"]]
run("class Early:\\n    X = 1\\n")
early.append(_fp(shell.user_ns["task13"])["code"]["Early (notebook)"])
mark = len(out.getvalue())  # the callbacks' own output, before any record
run(
    "class M(type):\\n    def __getattribute__(cls, n):\\n"
    "        if n == '__dict__':\\n            print('hook ran')\\n"
    "        return super().__getattribute__(n)\\n\\n"
    "    def __subclasses__(cls):\\n        print('hook ran')\\n"
    "        return type.__subclasses__(cls)\\n"
)
run("class Hooked(metaclass=M):\\n    X = 1\\n")
run(v1)
hook = "hook ran" in out.getvalue()[mark:]
run(
    "from mushin._resume import project_fingerprints as _pf\\n"
    "class Config:\\n    LR = 0.5\\n\\n"
    "_INSIDE = _pf(task)['code']['Config (notebook)']\\n"
)
inside = shell.user_ns["_INSIDE"]
after_cell = own()
callbacks = len(shell.events.callbacks["post_run_cell"])
importlib.reload(sys.modules["mushin._resume"])
run("import mushin._resume")
reloaded = len(shell.events.callbacks["post_run_cell"])
run(v1)
after_reload = own()
print(json.dumps({
    "first": first, "edited": edited, "reverted": reverted, "raised": raised,
    "magic": magic, "inside": inside, "after_cell": after_cell,
    "callbacks": [callbacks, reloaded], "after_reload": after_reload,
    "magic_again": magic_again, "plain": plain, "timed": [timed, timed_again, timed_same],
    "twice": twice, "hook": hook, "solo": [solo, solo_timed],
    "branch": [branch, branch_again], "nested": nested, "remade": remade,
    "emptied": emptied, "toggled": toggled, "mentions": mentions,
    "remakes": remakes, "module_set": module_set, "commented": commented,
    "nested_remakes": nested_remakes, "in_nested": in_nested, "early": early,
    "derived": derived, "proxy": proxy, "restored": restored, "aliased": aliased,
    "instance_only": instance_only, "alias_nested": alias_nested,
    "reached": reached, "based": based, "local": local, "wrapped": wrapped,
    "restated": restated, "early_i": early_i, "displayed": displayed,
    "three": three, "shown_base": shown_base, "shown_made": shown_made,
    "shown_then_bound": shown_then_bound, "pre_shown": pre_shown,
    "loop_named": loop_named, "global_made": global_made, "held_base": held_base,
    "leaked": leaked,
    "errors": [line for line in (out.getvalue() + err.getvalue()).splitlines()
               if "callback" in line.lower()],
}))
"""


def test_an_ipython_shell_pins_each_class_to_the_cell_that_made_it(tmp_path):
    """A real shell: the watch pins a class statement to its cell as it runs,
    so an edit, a revert, a cell that raised before its statement, a magic's
    cell, a record taken inside the defining cell and a reload of mushin all
    give the exact statement (several texts on one line would otherwise read
    as ambiguous). A class remade from text that is not kept (`%%time`,
    `%time`, `exec`) is never taken for an older kept text, and a factory's
    class, called from a cell that only mentions it, is ambiguous (it
    re-runs)."""
    pytest.importorskip("IPython")
    script = tmp_path / "watch.py"
    script.write_text(CELL_WATCH_SCRIPT)
    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    )
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["edited"] != got["first"]
    assert got["reverted"] == got["first"]
    assert got["raised"] == got["first"]  # the statement never ran
    # A magic's nested cell pins the class to its body: exact, and stable.
    assert got["magic"] == got["magic_again"] == got["plain"]
    # A `%%time` cell's class is ambiguous: its object keeps its digest, a
    # re-run of the cell (a new object) recomputes, and an older kept text's
    # digest never stands in.
    timed, timed_again, timed_same = got["timed"]
    assert timed == timed_again != timed_same
    assert timed not in (got["first"], got["edited"], got["magic"])
    assert got["solo"][0] != got["solo"][1]  # its only kept text is the old one
    for kind, (kept, remade) in got["remakes"].items():
        assert kept != remade, kind  # so for `%time`, automagic and `exec`
    # A cell that only mentions a factory's class states none: the factory's
    # statement is not read, as whether it made the class cannot be told:
    # ambiguous, so an edit to the factory is never missed.
    for kind, digests in got["mentions"].items():  # an unchanged re-run too
        assert len(set(digests)) == 3, kind
    for kind, (kept, remade) in got["nested_remakes"].items():
        assert kept != remade, kind
    assert got["early"][0] == got["early"][1]  # never pinned to a later cell
    for kind, (v1, v2) in got["derived"].items():
        assert v1 != v2, kind
    kept, remade, hook_ran = got["proxy"]
    assert kept != remade and not hook_ran
    assert got["restored"][0] == got["restored"][1]  # its statement, read
    for kind, (kept, restored) in got["aliased"].items():
        assert kept != restored, kind
    first, again, edited = got["instance_only"]
    assert first == again != edited  # read from its own cell, exactly
    assert got["alias_nested"][0] != got["alias_nested"][1]  # no stand-in
    for kind, (kept, inside, after) in got["reached"].items():
        assert kept not in (inside, after), kind
    assert got["based"][0] != got["based"][1]  # its base, remade, is seen
    assert got["local"][0] == got["local"][1]  # a local class: its factory's
    assert got["wrapped"][0] not in got["wrapped"][1:]  # through `__wrapped__`
    assert got["restated"][0] == got["restated"][1]  # the nested cell's own
    assert got["early_i"][0] == got["early_i"][1]  # in every snapshot
    assert got["displayed"][0] != got["displayed"][1]  # not the restated one's
    assert got["three"][0] == got["three"][1]  # the inner cell's own
    assert got["shown_base"][0] != got["shown_base"][1]  # its edit, seen
    assert got["shown_made"][0] != got["shown_made"][1]  # pinned with no text
    assert got["shown_then_bound"][0] != got["shown_then_bound"][1]
    assert got["pre_shown"][0] == got["pre_shown"][1]  # read exactly, as stated
    assert got["global_made"][0] != got["global_made"][1]  # ambiguous: re-runs
    for kind in ("loop_named", "held_base"):  # read exactly: an edit, a revert
        v1, v5, v1_again = got[kind]
        assert v1 != v5 and v1_again == v1, kind
    assert not got["leaked"]  # the watch keeps no superseded class alive
    for kind in ("commented", "in_nested"):  # pinned: an edit and a revert
        v1, v2, v1_again = got[kind]
        assert v1 != v2 and v1_again == v1, kind
    m1, m2, m3 = got["module_set"]  # `__module__` assigned: still exact
    assert m1 != m2 and m3 == m1
    assert got["branch"][0] == got["branch"][1]  # an if's branches: one runs
    n1, n2, n3 = got["nested"]  # a nested class, pinned: a revert is exact
    assert n1 != n2 and n3 == n1
    # type() made it: the watch pins it to its cell, which states none, so
    # the older statement never stands in for it, on every Python.
    assert got["remade"][0] != got["remade"][1]
    toggled = got["toggled"]  # a nested class recorded inside its own cell
    assert toggled[0] != toggled[1] and toggled[2:] == toggled[:2]
    inside, after, inside_again, after_again = got["emptied"]
    assert inside == after == inside_again == after_again
    a, b, a_again, b_again = got["twice"]  # two statements in one cell
    assert a != b and a_again != b_again
    if sys.version_info >= (3, 13):  # the first line tells them apart
        assert (a, b) == (a_again, b_again)
    # The callbacks ran no notebook metaclass hook (taking a record still can:
    # a documented residual).
    assert not got["hook"]
    assert got["inside"] == got["after_cell"]
    assert got["callbacks"][0] == got["callbacks"][1]  # not registered twice
    assert got["after_reload"] == got["first"]
    assert got["errors"] == []


def test_a_factory_with_the_class_in_two_branches_is_exact(kernel, monkeypatch):
    """One text holds both branches' statements: they count together (which
    one runs is a value), so the digest is the same in a new process."""
    import itertools

    import mushin._resume as resume

    cells = [
        "def make(kind):\n    if kind == 'a':\n\n        class C:\n            K = 1\n\n"
        "    else:\n\n        class C:\n            K = 2\n\n    return C\n\n\n"
        "C = make('a')\n",
        "def task(a):\n    return a * C.K\n",
    ]
    _run_cells(kernel, cells)
    first = _own(kernel, "C")
    monkeypatch.setattr(resume, "_PROCESS_TOKEN", [])
    monkeypatch.setattr(resume, "_CLASS_GENERATIONS", {})
    monkeypatch.setattr(resume, "_GENERATION_COUNTER", itertools.count())
    again = Kernel("mushin_t_kernel_again")
    try:
        _run_cells(again, cells)
        assert _own(again, "C") == first
    finally:
        again.close()


def test_an_ambiguous_class_s_digest_differs_in_another_process(kernel, monkeypatch):
    """The token is the process's: the same sequence in a new process gives
    another digest, so it never matches what an earlier one recorded."""
    import itertools

    import mushin._resume as resume

    factory = (
        "def make():\n    class C:\n        K = {}\n\n    return C\n\n\nC = make()\n"
    )
    cells = [factory.format(2), "def task(a):\n    return a * C.K\n", factory.format(3)]
    monkeypatch.setattr(resume, "_GENERATION_COUNTER", itertools.count())
    _run_cells(kernel, cells)
    first = _own(kernel, "C")
    kernel.close()
    monkeypatch.setattr(resume, "_PROCESS_TOKEN", [])
    monkeypatch.setattr(resume, "_CLASS_GENERATIONS", {})
    monkeypatch.setattr(resume, "_GENERATION_COUNTER", itertools.count())
    again = Kernel("mushin_t_kernel_again")
    try:
        _run_cells(again, cells)
        assert _own(again, "C") != first
    finally:
        again.close()


def test_identical_class_texts_read_every_cell_they_stand_in(kernel, sub_project):
    """Two factories hold the same class statement, around different imports:
    both are read, so the record does not depend on which the scan met first."""
    for sub, name in (("a.b", "make_one"), ("sub", "make_two")):
        kernel.run(
            f"def {name}():\n    import {SUBPKG}.{sub} as m\n\n    class C:\n"
            "        y = m.f\n\n    return C\n"
        )
    kernel.run(
        "def make():\n    class C:\n        y = 1\n\n    return C\n\n\nC = make()\n"
    )
    kernel.run("def task(a):\n    return a * C.y\n")
    # The class bound is `make.<locals>.C`; the two texts above are other
    # qualnames, so only `make`'s statement counts.
    files = _files(kernel, "task")
    assert f"{SUBPKG}.a.b (file)" not in files
    kernel.run(
        f"def make():\n    import {SUBPKG}.a.b as m\n\n    class C:\n"
        "        y = m.f\n\n    return C\n\n\nC = make()\n"
    )
    kernel.run(
        f"def make():\n    import {SUBPKG}.sub as m\n\n    class C:\n"
        "        y = m.f\n\n    return C\n\n\nC = make()\n"
    )
    files = _files(kernel, "task")
    assert {f"{SUBPKG}.a.b (file)", f"{SUBPKG}.sub (file)"} <= files, files


@pytest.mark.parametrize(
    "annotated",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 14), reason="__annotate_func__ is 3.14+"
            ),
        ),
    ],
    ids=["plain", "annotated"],
)
def test_a_method_less_class_s_reading_is_followed_after_main_s_seeds(
    kernel, sub_project, monkeypatch, annotated
):
    """Main read nothing of a class with no method: what its body names goes
    to the later pass, so it cannot push the namespace's module out."""
    import mushin._resume as resume

    (sub_project / SUBPKG / "wide.py").write_text(f"import {SUBWIDE}\nX = 1\n")
    monkeypatch.setattr(resume, "_FILES_MAX", 3)
    monkeypatch.setattr(resume, "_LATER_TRUNCATION_WARNED", [])
    kernel.run(
        f"import importlib\nimport {SUBPKG}\nfrom {SUBREL} import f\n"
        f"importlib.import_module('{SUBPKG}.wide')\n"
    )
    kernel.run(
        f"class Config:\n    K{': int' if annotated else ''} = {SUBPKG}.wide.X\n"
    )
    if annotated:
        assert "__annotate_func__" in vars(kernel.ns["Config"])
    kernel.run("def task(a):\n    return f(a) + Config.K\n")
    with pytest.warns(UserWarning, match="reach more files"):
        files = _files(kernel, "task")
    assert f"{SUBREL} (file)" in files, files


def test_a_failing_scan_keeps_the_entries_with_a_method_less_class(
    kernel, cell_imports, monkeypatch
):
    """The class index comes from the same guarded scan: when it fails, the
    class is ambiguous and every other entry still counts."""
    import mushin._resume as resume

    def fail(text):
        raise RuntimeError("the cell's own text")

    monkeypatch.setattr(resume, "_cell_imports", fail)
    monkeypatch.setattr(resume, "_CELL_IMPORTS", {})
    monkeypatch.setattr(resume, "_CELL_SCAN_WARNED", [])
    monkeypatch.setattr(resume, "_NOTEBOOK_FAILURE_WARNED", [])
    _run_cells(
        kernel,
        [
            "class Config:\n    LR = 0.1\n",
            "def helper(x):\n    return x\n",
            "def task(a):\n    return helper(a) * Config.LR\n",
        ],
    )
    with pytest.warns(UserWarning, match="cells import could not be read"):
        record = project_fingerprints(kernel.ns["task"])
    assert {"helper (notebook)", "Config (notebook)"} <= set(record["code"])
    kernel.run("class Config:\n    LR = 0.2\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edited = project_fingerprints(kernel.ns["task"])["code"]
    assert edited["Config (notebook)"] != record["code"]["Config (notebook)"]
    assert edited["helper (notebook)"] == record["code"]["helper (notebook)"]


@pytest.mark.skipif(sys.version_info < (3, 13), reason="__firstlineno__ is 3.13+")
def test_a_class_no_statement_made_is_not_an_older_statement(kernel):
    """3.13+: `make_dataclass` makes a class without `__firstlineno__`, so the
    kept `class Config` statement is not taken for it."""
    _run_cells(
        kernel,
        ["class Config:\n    LR = 0.1\n", "def task(a):\n    return a * Config.LR\n"],
    )
    statement = _own(kernel)
    kernel.run(
        "import dataclasses\n\nConfig = dataclasses.make_dataclass("
        "'Config', [('LR', float, 0.2)])\n"
    )
    assert _own(kernel) != statement


def test_two_statements_one_after_the_other_are_not_one_class(kernel):
    """A cell defines the class twice, with a record between: the two objects
    must not share a digest (on 3.11/3.12 they are ambiguous; on 3.13+ the
    first line tells them apart)."""
    kernel.run("def task(a):\n    return a * Config.LR\n")
    kernel.run(
        "from mushin._resume import project_fingerprints as _pf\n\n\n"
        "class Config:\n    LR = 0.7\n\n\n"
        "_A = _pf(task)['code']['Config (notebook)']\n\n\n"
        "class Config:\n    LR = 0.8\n\n\n"
        "_B = _pf(task)['code']['Config (notebook)']\n"
    )
    assert kernel.ns["_A"] != kernel.ns["_B"]


@pytest.mark.parametrize(
    "cell",
    [
        "FLAG = True\n\nif FLAG:\n\n    class Config:\n        LR = 1\n\n"
        "else:\n\n    class Config:\n        LR = 2\n",
        "FLAG = 1\n\nif FLAG == 0:\n\n    class Config:\n        LR = 0\n\n"
        "elif FLAG == 1:\n\n    class Config:\n        LR = 1\n",
        "FLAG = 1\n\nmatch FLAG:\n    case 1:\n\n        class Config:\n"
        "            LR = 1\n\n    case _:\n\n        class Config:\n            LR = 2\n",
    ],
    ids=["if", "elif", "match"],
)
def test_a_class_in_two_branches_of_an_if_is_stable(kernel, cell):
    """Only one branch runs, and which is a value: the cell's statements
    count together, so an unchanged re-run keeps the digest."""
    _run_cells(kernel, [cell, "def task(a):\n    return a * Config.LR\n"])
    first = _own(kernel)
    kernel.run(cell)
    assert _own(kernel) == first


def test_a_continued_class_statement_is_found(kernel):
    """`class \\` and the name on the next line: still a statement."""
    _run_cells(
        kernel,
        ["class \\\nCfg2:\n    LR = 0.1\n", "def task(a):\n    return a * Cfg2.LR\n"],
    )
    first = _own(kernel, "Cfg2")
    kernel.run("class \\\nCfg2:\n    LR = 0.2\n")
    assert _own(kernel, "Cfg2") != first


def test_a_forked_process_gets_its_own_token(kernel, monkeypatch):
    """A forked child inherits the module's state: the token is per process
    id, so its ambiguous digests are its own."""
    import os

    factory = (
        "def make():\n    class C:\n        K = {}\n\n    return C\n\n\nC = make()\n"
    )
    _run_cells(
        kernel,
        [factory.format(2), "def task(a):\n    return a * C.K\n", factory.format(3)],
    )
    parent = _own(kernel, "C")
    real = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: real + 1)
    assert _own(kernel, "C") != parent


@pytest.mark.parametrize(
    "cell",
    [
        "import dataclasses\n\n\n@dataclasses.dataclass\nclass Config:\n    LR: float = {}\n",
        "import enum\n\n\nclass Config(enum.Enum):\n    LR = {}\n",
        "class Config:\n    LR = {}\n",
        "def make():\n    class Config:\n        LR = {}\n\n    return Config\n\n\nConfig = make()\n",
    ],
    ids=["dataclass", "Enum", "plain", "factory"],
)
def test_a_method_less_class_s_statement_is_read_exactly(cell):
    """One kept text: the digest is its statement's, so two kernels running
    the same text agree and another text differs (a token would differ in
    both)."""
    task = "def task(a):\n    return a\n\n\nCONFIG = Config\n"
    digests = []
    for value in (1, 1, 2):
        k = Kernel("mushin_t_kernel_exact")
        try:
            _run_cells(
                k,
                [
                    cell.format(value),
                    task.replace("return a", "return a * 0 + len([CONFIG])"),
                ],
            )
            digests.append(_record(k, "task")["CONFIG (notebook)"])
        finally:
            k.close()
    assert digests[0] == digests[1] != digests[2]


@pytest.mark.parametrize(
    "between",
    [
        "if True:\n\n    class Config:\n        LR = 0.8\n",
        "try:\n\n    class Config:\n        LR = 0.8\n\nexcept ValueError:\n    pass\n",
    ],
    ids=["a later if", "a try"],
)
def test_two_statements_that_both_run_are_not_one_class(kernel, between):
    """A top-level statement and a later one in another block both run: a
    record between them must not reuse the first one's results."""
    kernel.run("def task(a):\n    return a * Config.LR\n")
    kernel.run(
        "from mushin._resume import project_fingerprints as _pf\n\n\n"
        "class Config:\n    LR = 0.7\n\n\n"
        "_A = _pf(task)['code']['Config (notebook)']\n"
        f"{between}\n"
        "_B = _pf(task)['code']['Config (notebook)']\n"
    )
    assert kernel.ns["_A"] != kernel.ns["_B"]


def test_a_factory_stating_its_class_twice_is_stable(kernel):
    """Inside a function's body two statements run as one step: an unchanged
    re-run of the factory keeps the digest."""
    factory = (
        "def make():\n    class C:\n        K = 1\n\n    class C(C):\n        K = 2\n\n"
        "    return C\n\n\nC = make()\n"
    )
    _run_cells(kernel, [factory, "def task(a):\n    return a * C.K\n"])
    first = _own(kernel, "C")
    kernel.run(factory)
    assert _own(kernel, "C") == first


def test_a_try_body_and_its_handler_both_run(kernel):
    """Two statements in one `try`, the body's and the handler's: both run, so
    a record between them must not reuse the first one's results."""
    kernel.run("def task(a):\n    return a * Config.LR\n")
    kernel.run(
        "from mushin._resume import project_fingerprints as _pf\n\ntry:\n\n"
        "    class Config:\n        LR = 0.7\n\n"
        "    _A = _pf(task)['code']['Config (notebook)']\n    raise ValueError\n"
        "except ValueError:\n\n    class Config:\n        LR = 0.8\n\n"
        "    _B = _pf(task)['code']['Config (notebook)']\n"
    )
    assert kernel.ns["_A"] != kernel.ns["_B"]


def test_a_watch_that_cannot_register_leaves_no_callback(monkeypatch):
    """The state is stored on the shell first, and a callback already added
    is removed when the next fails, so no later record registers again."""
    import mushin._resume as resume

    class Events:
        def __init__(self):
            self.callbacks = []

        def register(self, event, callback):
            if event == "post_run_cell":
                raise RuntimeError("refused")
            self.callbacks.append((event, callback))

        def unregister(self, event, callback):
            self.callbacks.remove((event, callback))

    class Shell:
        user_ns: dict = {}

        def __init__(self):
            self.events = Events()

        def transform_cell(self, raw):
            return raw

    shell = Shell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    assert resume._cell_watch() is None
    assert shell.events.callbacks == []
    assert type(getattr(shell, resume._CELL_WATCH)) is dict
    assert resume._cell_watch() is getattr(shell, resume._CELL_WATCH)
    assert shell.events.callbacks == []


class _FakeEvents:
    def __init__(self):
        self.callbacks: dict = {}

    def register(self, event, callback):
        self.callbacks[event] = callback

    def unregister(self, event, callback):
        self.callbacks.pop(event, None)


class _FakeShell:
    def __init__(self):
        self.events = _FakeEvents()
        self.user_ns = {"__name__": "mushin_t_fake_shell"}

    def transform_cell(self, raw):
        return raw


@pytest.mark.parametrize("ending", ["a new info, the same raw text", "no result"])
def test_a_cell_s_end_is_found_without_its_own_info(monkeypatch, ending):
    """An interrupted cell's result carries a new info around the same raw
    text, and ipykernel can pass no result: the cell still ends and pins."""
    import mushin._resume as resume

    shell = _FakeShell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    state = resume._cell_watch()
    outer = types.SimpleNamespace(raw_cell="get_ipython().run_cell(CELL)\n")
    shell.events.callbacks["pre_run_cell"](outer)  # still running after
    raw = "class Config:\n    LR = 0.1\n"
    info = types.SimpleNamespace(raw_cell=raw)
    shell.events.callbacks["pre_run_cell"](info)
    shell.user_ns["Config"] = type("Config", (), {"__module__": "mushin_t_fake_shell"})
    if ending == "no result":
        result = None
    else:
        result = types.SimpleNamespace(info=types.SimpleNamespace(raw_cell=raw))
    shell.events.callbacks["post_run_cell"](result)
    assert [running for running, _ in state["running"]] == [outer]
    assert id(shell.user_ns["Config"]) in state["pins"]


def test_a_class_whose_module_is_assigned_is_still_read(kernel):
    """Assigning a class's `__module__` deletes its `__firstlineno__` on 3.13+;
    a `class` statement still made it, so an edit is seen."""
    cell = "class Config:\n    LR = {}\n\n\nConfig.__module__ = __name__\n"
    _run_cells(kernel, [cell.format(1), "def task(a):\n    return a * Config.LR\n"])
    first = _own(kernel)
    kernel.run(cell.format(2))
    assert _own(kernel) != first


def test_a_class_pinned_to_a_cell_stating_none_is_ambiguous(monkeypatch):
    """A class pinned to a cell that states none of it (`exec`, `%%time`, a
    copy) is ambiguous whatever the kept cells hold: no kept statement stands
    in for it. Unpinned, a kept statement is read, and no scan is
    ambiguous."""
    import weakref

    import mushin._resume as resume

    shell = _FakeShell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    state = resume._cell_watch()
    exec("class Config:\n    LR = 0.1\n", shell.user_ns)
    cls = shell.user_ns["Config"]
    state["pins"][id(cls)] = (weakref.ref(cls), ["exec(SOURCE)\n"])
    no_scan = lambda qualname: None  # noqa: E731
    kept = lambda qualname: ["class Config:\n    LR = 0.1\n"]  # noqa: E731
    for texts in (no_scan, kept):
        assert resume._class_sites(cls, [], shell.user_ns, texts) == ([], True, True)
    del state["pins"][id(cls)]
    assert resume._class_sites(cls, [], shell.user_ns, no_scan) == ([], True, True)
    sites, ambiguous, _ = resume._class_sites(cls, [], shell.user_ns, kept)
    assert sites and not ambiguous


@pytest.mark.skipif(sys.version_info < (3, 13), reason="__static_attributes__ is 3.13+")
def test_a_class_type_makes_has_no_site_without_the_watch(kernel):
    """With no watch (a class made before `import mushin`), 3.13+ still tells a
    class `type()` made from one a statement made: no older kept statement
    stands in for it."""
    _run_cells(
        kernel, ["class Mk:\n    X = 1\n", "def task(a):\n    return a * Mk.X\n"]
    )
    kept = _own(kernel, "Mk")
    kernel.run("Mk = type('Mk', (), {'X': 2})\n")
    assert _own(kernel, "Mk") != kept


def test_a_cell_binding_many_classes_still_pins_a_nested_one(monkeypatch):
    """The nested walk has its own budget: a cell binding 70 classes
    (`%run models.py`) still pins the class one of their bodies made."""
    import mushin._resume as resume

    shell = _FakeShell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    state = resume._cell_watch()
    source = "".join(f"class C{i}:\n    X = {i}\n\n\n" for i in range(70))
    exec(source + "class Outer:\n    class Inner:\n        X = 1\n", shell.user_ns)
    resume._pin_classes(state, shell.user_ns, ("%run models.py\n", {}))
    assert id(shell.user_ns["Outer"].Inner) in state["pins"]


def test_a_kept_text_with_carriage_returns_is_read():
    """`compile` takes a bare `\\r` as a line end, so the parse trigger does."""
    import mushin._resume as resume

    assert resume._cell_imports("x = 1\rclass A:\r    X = 1\r")[2] == ("A",)


def test_a_cell_binding_many_classes_pins_every_nested_one(monkeypatch):
    """The nested walk has no budget: each step is a longer qualname and each
    class is taken once, so 300 models with a nested `Meta` each (`%run
    models.py`) all have their `Meta` pinned."""
    import mushin._resume as resume

    shell = _FakeShell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    state = resume._cell_watch()
    source = "".join(
        f"class M{i}:\n    class Meta:\n        X = {i}\n\n\n" for i in range(300)
    )
    exec(source, shell.user_ns)
    resume._pin_classes(state, shell.user_ns, ("%run models.py\n", {}))
    metas = [shell.user_ns[f"M{i}"].Meta for i in range(300)]
    assert all(id(meta) in state["pins"] for meta in metas)


def test_a_running_cell_with_no_text_stops_the_on_demand_pin(monkeypatch):
    """A record inside a nested cell whose text is unknown does not pin the
    class to the cell around it, whose statement of it is older."""
    import mushin._resume as resume

    shell = _FakeShell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    resume._cell_watch()
    pre = shell.events.callbacks["pre_run_cell"]
    pre(types.SimpleNamespace(raw_cell="class Config:\n    LR = 0.5\n"))
    pre(types.SimpleNamespace(raw_cell=None))  # its text is unknown
    exec("class Config:\n    LR = 0.9\n", shell.user_ns)
    assert resume._pinned_cell(shell.user_ns["Config"], shell.user_ns) == (False, None)


@pytest.mark.skipif(sys.version_info < (3, 13), reason="__firstlineno__ is 3.13+")
def test_a_kept_cell_split_at_a_form_feed_is_ambiguous(kernel, no_shell):
    """IPython keeps a cell as its `splitlines()` lines, so a form feed adds a
    line `compile` never counted. With no text of the cell as compiled (no
    shell: see `test_a_split_class_cell_is_read_as_compiled` for one), no
    kept statement sits at the class's first line: ambiguous, so an edit is
    seen (a constant would miss it)."""
    _run_cells(kernel, ["def task(a):\n    return a * Config.LR\n"])
    digests = []
    for value in (0.1, 0.2):
        kernel.run(f"x = 1\f\nclass Config:\n    LR = {value}\n")
        name = kernel.names[-1]
        size, mtime, lines, filename = linecache.cache[name]
        kept = [line + "\n" for line in "".join(lines).splitlines()]  # as IPython
        linecache.cache[name] = (size, mtime, kept, filename)
        digests.append(_own(kernel))
    assert digests[0] != digests[1]


def test_the_watch_finds_every_live_notebook_class():
    """`_live_classes` finds every class whose module is the notebook's,
    however it is held (a name, an instance, a list, only as a base or a
    nested class) and none of a library's, without running a metaclass's
    own `__subclasses__`. Its own kernel name, so no other test's class
    (dead but not yet collected) can stand in."""
    import gc

    import mushin._resume as resume

    k = Kernel("mushin_t_kernel_live")
    try:
        k.run(
            "import collections\n\n\nclass Named:\n    pass\n\n\n"
            "class Held:\n    pass\n\n\nheld = Held()\n\n\n"
            "class InList:\n    pass\n\n\nkept = [InList]\n\n\n"
            "class Base:\n    pass\n\n\nclass Kid(Base):\n    pass\n\n\n"
            "class Out:\n    class In:\n        pass\n\n\nout_in = Out.In\n\n\n"
            "class Meta(type):\n    def __subclasses__(cls):\n"
            "        raise RuntimeError('hook ran')\n\n\n"
            "class WithMeta(metaclass=Meta):\n    pass\n\n\n"
            "del Held, InList, Base, Out\ncounter = collections.Counter()\n"
        )
        ns = k.ns
        expected = [
            ns["Named"],
            type(ns["held"]),
            ns["kept"][0],
            ns["Kid"].__bases__[0],
            ns["Kid"],
            ns["out_in"],
            ns["Meta"],
            ns["WithMeta"],
        ]
        gc.collect()
        live = resume._live_classes(ns["__name__"])
        assert set(live) == {id(cls) for cls in expected}
    finally:
        k.close()


def test_a_snapshot_tells_a_reused_id_and_an_older_watch_s_snapshot(monkeypatch):
    """`_alive_at` takes a class as alive at a cell's start only if its id's
    weak reference still gives that class: a reused id (another class, or a
    dead one) is not, so the cell pins it. A snapshot an older `_resume`
    wrote before a reload holds the class itself: compared, never called."""
    import gc
    import weakref

    import mushin._resume as resume

    shell = _FakeShell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    state = resume._cell_watch()
    exec(
        "class Config:\n    LR = 0.1\n\n    def __init__(self):\n"
        "        raise RuntimeError('called')\n",
        shell.user_ns,
    )
    cls = shell.user_ns["Config"]
    other = type("Other", (), {})
    gone = type("Gone", (), {})
    dead = weakref.ref(gone)
    del gone
    gc.collect()  # a class sits in reference cycles: only the collector frees it
    assert dead() is None
    for ref in (weakref.ref(other), dead):
        assert not resume._alive_at({id(cls): ref}, cls)
    assert resume._alive_at({id(cls): weakref.ref(cls)}, cls)
    assert resume._alive_at({id(cls): cls}, cls)  # not called: no error
    resume._pin_classes(state, shell.user_ns, ("exec(SOURCE)\n", {id(cls): dead}))
    assert resume._live_pin(state, cls) == ["exec(SOURCE)\n"]
    # A dead pin at a listed class's id is a reused id: not a lost cell.
    exec("class Kept:\n    X = 1\n", shell.user_ns)
    kept = shell.user_ns["Kept"]
    state["pins"][id(kept)] = (dead, ["exec(SOURCE)\n"])
    assert resume._listed(kept)
    assert resume._pinned_cell(kept, shell.user_ns) == (False, None)


def test_a_class_a_finalizer_brings_back_is_ambiguous(monkeypatch):
    """Before 3.15 the collector clears a class's weak references before a
    finalizer (`__del__`) brings an instance, and so the class, back: its
    pin is dead and no base lists it. Its cell is lost, so it is ambiguous,
    never read from an older kept statement (a dead pin alone is a reused
    id). 3.15 keeps them, and the class its pin."""
    import gc
    import weakref

    import mushin._resume as resume

    shell = _FakeShell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    state = resume._cell_watch()
    ns = shell.user_ns
    exec(
        "pool = []\n\n\nclass Pooled:\n    def __del__(self):\n        pool.append(self)\n",
        ns,
    )
    enabled = gc.isenabled()
    gc.disable()
    try:
        exec("class C(Pooled):\n    X = 1\n", ns)
        cls = ns["C"]
        state["pins"][id(cls)] = (weakref.ref(cls), ["exec(SOURCE)\n"])
        exec("c = C()\nc.me = c\ndel c\n\n\nclass C(Pooled):\n    X = 3\n", ns)
        pin = state["pins"][id(cls)][0]
        del cls
        gc.collect()
        back = type(ns["pool"][0])
    finally:
        if enabled:
            gc.enable()
    ns["Back"] = back
    exec("class E(Back):\n    Y = 1\n", ns)  # made on it: no walk reaches it
    made_on = ns["E"]
    assert back.X == 1
    if pin() is None:  # before 3.15: its weak references were cleared
        assert not resume._listed(back)
        assert resume._pinned_cell(back, ns) == (True, [])
        assert resume._pinned_cell(made_on, ns) == (True, [])
        exec("class F(E):\n    Z = 1\n", ns)  # two levels down: the lineage
        assert resume._pinned_cell(ns["F"], ns) == (True, [])
    else:  # 3.15 keeps them, and the class its pin
        assert resume._pinned_cell(back, ns) == (True, ["exec(SOURCE)\n"])
    assert resume._listed(ns["C"])  # a class alive as usual is listed


# --- a cell split where `compile` does not end a line (#296) ----------------------

#: The characters `str.splitlines` ends a line at and `compile` does not.
SPLIT_ONLY = ["\f", "\v", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
#: Where a cell can hold one above a definition; each compiles. In a
#: one-line string, the cell as IPython keeps it does not parse.
SPLIT_HEADS = {
    "comment": "# a{}b\n",
    "triple-quoted": 'S = """a{}b"""\n',
    "string": 'S = "a{}b"\n',
}
SPLIT_CASES = [
    pytest.param(head.format(c), id=f"{where}-{ord(c):04x}")
    for where, head in SPLIT_HEADS.items()
    for c in SPLIT_ONLY
] + [pytest.param("x = 1\f\n", id="page-break")]


class _HistoryShell(_FakeShell):
    """A shell with IPython's input history: each stored cell as compiled."""

    def __init__(self):
        super().__init__()
        self.history_manager = types.SimpleNamespace(input_hist_parsed=[""])


@pytest.fixture
def history(monkeypatch):
    """The input history of a shell the record sees (``_HistoryShell``)."""
    import mushin._resume as resume

    shell = _HistoryShell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    return shell.history_manager.input_hist_parsed


@pytest.fixture
def no_shell(monkeypatch):
    """No shell: neither a history nor a cell watch to read a cell from."""
    import mushin._resume as resume

    monkeypatch.setattr(resume, "_shell", lambda: None)


def _run_kept(kernel, text: str, history=None) -> None:
    """Run a cell, kept as IPython keeps it (``[line + "\\n" for line in
    text.splitlines()]``) and, with ``history``, stored as IPython stores it
    (the text it compiled, without trailing newlines)."""
    kernel.run(text)
    name = kernel.names[-1]
    size, mtime, _, filename = linecache.cache[name]
    kept = [line + "\n" for line in text.splitlines()]
    linecache.cache[name] = (size, mtime, kept, filename)
    if history is not None:
        history.append(text.rstrip("\n"))


def _helper(kernel) -> str:
    return _record(kernel, "task")["helper (notebook)"]


@pytest.mark.parametrize("head", SPLIT_CASES)
def test_a_split_cell_is_read_as_compiled(kernel, history, head):
    """IPython keeps a cell as its `splitlines()` lines, so such a character
    above a definition moves it below the line its code names, or the kept
    cell does not parse. The history holds the cell as compiled: an edit is
    seen, and the same text again is not (exact, not ambiguous)."""
    _run_kept(kernel, TASK, history)
    digests = []
    for k in (2, 3, 3):
        _run_kept(kernel, head + f"def helper(a):\n    return a * {k}\n", history)
        digests.append(_helper(kernel))
    assert digests[0] != digests[1]
    assert digests[1] == digests[2]


@pytest.mark.parametrize("head", SPLIT_CASES)
def test_a_split_cell_not_known_as_compiled_is_ambiguous(kernel, no_shell, head):
    """With no text of the cell as compiled, its definition is not at its
    code's line in the kept text: ambiguous, so an edit is seen, the same
    object keeps its digest, and a re-run of the cell (a new object)
    recomputes. A constant digest would miss every edit."""
    _run_cells(kernel, [TASK])
    digests = []
    for k in (2, 3, 3):
        _run_kept(kernel, head + f"def helper(a):\n    return a * {k}\n")
        digests.append(_helper(kernel))
        assert _helper(kernel) == digests[-1]
    assert len(set(digests)) == 3


def test_lambdas_below_a_page_break_are_told_apart(kernel, history):
    """Two lambdas on consecutive lines below a form feed: in the kept text,
    the second's line holds the first, whose text would hide the second's
    edits."""
    _run_kept(kernel, "def task(a):\n    return second(a)\n", history)
    cell = "x = 1\f\nfirst = lambda a: a * 2\nsecond = lambda a: a * {}\n"
    digests = []
    for k in (3, 4, 4):
        _run_kept(kernel, cell.format(k), history)
        digests.append(_record(kernel, "task")["second (notebook)"])
    assert digests[0] != digests[1]
    assert digests[1] == digests[2]


def test_a_method_below_page_breaks_names_its_own_class(kernel, history):
    """Two versions of a class in one cell below two form feeds: in the kept
    text, the second's method's line falls in the first, whose statement
    would hide an edit to the second's (the method itself is the same)."""
    _run_kept(kernel, "def task(a):\n    return a * C.K\n", history)
    cell = (
        "x = 1\f\f\nclass C:\n    def f(self): return 1\n    K = 0\n"
        "class C:\n    def f(self): return 1\n    K = {}\n"
    )
    digests = []
    for k in (2, 3, 3):
        _run_kept(kernel, cell.format(k), history)
        digests.append(_own(kernel, "C"))
    assert digests[0] != digests[1]
    assert digests[1] == digests[2]


def test_cells_kept_alike_are_ambiguous(kernel, history):
    """Two stored cells that differ only where one holds such a character
    and the other a newline keep the same lines: which one a function's code
    came from cannot be told, so it is ambiguous (a new object recomputes)."""
    _run_kept(kernel, TASK, history)
    plain = 'S = """a\nb"""\ndef helper(a):\n    return a * 2\n'
    split = 'S = """a\u2028b"""\ndef helper(a):\n    return a * 2\n'
    _run_kept(kernel, plain, history)
    _run_kept(kernel, split, history)
    first = _helper(kernel)
    assert _helper(kernel) == first  # the same object
    _run_kept(kernel, split, history)
    assert _helper(kernel) != first


def test_a_split_cell_s_imports_count(kernel, history, cell_imports):
    """A cell whose kept text does not parse (such a character in a string)
    is scanned as compiled: its project import counts."""
    _run_kept(kernel, f'S = "a\u2028b"\nimport {SUBPKG}.sub\n', history)
    _run_kept(kernel, "def task(a):\n    return a\n", history)
    assert f"{SUBPKG}.sub (file)" in _files(kernel, "task")


@pytest.mark.skipif(sys.version_info < (3, 13), reason="__firstlineno__ is 3.13+")
def test_a_split_class_cell_is_read_as_compiled(kernel, history):
    """A method-less class below a form feed, read from the kept cells (the
    watch did not see it run): as compiled, its statement is at its first
    line, so it reads as the same class without the form feed does
    (ambiguous, it would have a digest of its own)."""
    task = "def task(a):\n    return a * Config.LR\n"
    _run_kept(kernel, task, history)
    _run_kept(kernel, "x = 1\f\nclass Config:\n    LR = 0.1\n", history)
    split = _own(kernel)
    kernel.close()  # its cells leave linecache: the plain one is read alone
    plain = Kernel("mushin_t_kernel_plain")
    try:
        _run_kept(plain, task)
        _run_kept(plain, "x = 1\nclass Config:\n    LR = 0.1\n")
        assert _own(plain) == split
    finally:
        plain.close()


def test_a_history_without_such_a_character_is_not_split(monkeypatch):
    """The usual case: no stored or watched text holds such a character, so
    no kept form is computed and nothing is indexed."""
    import mushin._resume as resume

    shell = _HistoryShell()
    shell.history_manager.input_hist_parsed.extend(
        f"x = {i}\ny = x\n" for i in range(200)
    )
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    resume._cell_watch()
    called = []
    monkeypatch.setattr(resume, "_kept_form", lambda text: called.append(text))
    assert resume._compiled_texts() == {}
    assert called == []


def test_the_watch_keeps_the_texts_of_split_cells(monkeypatch):
    """The cell watch keeps the text a cell holding such a character compiles
    as (a cell run without being stored is in no history), each once, the
    newest ``_SPLIT_TEXTS_MAX``; a state an older ``_resume`` made gets the
    store on its first such cell."""
    import mushin._resume as resume

    shell = _HistoryShell()
    monkeypatch.setattr(resume, "_shell", lambda: shell)
    monkeypatch.setattr(resume, "_SPLIT_TEXTS_MAX", 2)
    state = resume._cell_watch()
    del state["texts"]  # as an older `_resume` made it
    pre = shell.events.callbacks["pre_run_cell"]
    post = shell.events.callbacks["post_run_cell"]

    def run(raw):
        info = types.SimpleNamespace(raw_cell=raw)
        pre(info)
        post(types.SimpleNamespace(info=info))

    run("x = 1\n")
    assert "texts" not in state  # a cell without one: nothing kept
    run("a = 1\f\n\n")
    run("b = 1\f\n")
    run("a = 1\f\n")  # again: the newest
    run("c = 1\f\n")
    assert list(state["texts"]) == ["a = 1\f", "c = 1\f"]
    assert resume._compiled_texts() == {
        "a = 1": ["a = 1\f"],
        "c = 1": ["c = 1\f"],
    }


SPLIT_SCRIPT = """
import json

from IPython.core.interactiveshell import InteractiveShell

import mushin._resume as resume

shell = InteractiveShell.instance()
resume._cell_watch()  # as the first record would


def helper_digest(history_only):
    if history_only:  # what the watch kept goes: the history alone tells
        getattr(shell, resume._CELL_WATCH)["texts"].clear()
    rec = resume.project_fingerprints(shell.user_ns["task"])["code"]
    return rec["helper (notebook)"]


out = {}
for n, (case, prefix, store) in enumerate(CASES):
    shell.run_cell("def task(a):\\n    return helper(a)\\n", store_history=store)
    seen = []
    for k in (2, 3, 3):
        # Each case its own text: no other case's stored cell can stand in.
        body = "x = %d\\f\\ndef helper(a):\\n    return a * %d\\n" % (n, k)
        cell = prefix + body
        result = shell.run_cell(cell, store_history=store)
        assert result.success, case
        seen.append(helper_digest(case == "stored"))
    out[case] = [seen[0] != seen[1], seen[1] == seen[2]]
print(json.dumps(out))
"""


def test_an_ipython_shell_reads_a_split_cell_as_compiled(tmp_path):
    """A real shell: a stored cell read from its history alone, a cell run
    without being stored (the watch keeps it), and a magic's inner cell,
    which is never stored."""
    pytest.importorskip("IPython")
    cases = [
        ("stored", "", True),
        ("not stored", "", False),
        ("a magic's inner cell", "%%capture\n", True),
    ]
    script = tmp_path / "split.py"
    script.write_text(f"CASES = {cases!r}\n" + SPLIT_SCRIPT)
    out = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    ).stdout
    got = json.loads(out.strip().splitlines()[-1])
    assert got == {case: [True, True] for case, _, _ in cases}
