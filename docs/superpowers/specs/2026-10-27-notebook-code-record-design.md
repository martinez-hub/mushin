# Notebook code in the resume and cache_dir record: design

Date: 2026-10-27. Status: approved by the maintainer's standing instruction to
take the recommended option. Follows #270 (the record), #276 and #277 (its
follow-ups).

## Problem

`resume=True` and `cache_dir=` compare a record of the code a cell ran
(`code_deps`, see `project_fingerprints`). For a task defined in a notebook
cell, main records two things: the task function's own source
(`code_fingerprint`, through `inspect.getsource`, which reads the cell from
`linecache`), and the project modules the notebook's names come from
(`_namespace_modules`). Nothing else defined in the notebook is compared. A
user who edits a helper, a class or a constant in a cell, re-runs that cell
and resumes gets the results the old code computed, silently.

## Why the history design was dropped

The first design (in #277, three review rounds) rebuilt the code in force
from the kernel's cell history (`history_manager.input_hist_parsed`) with a
reaching-definitions pass, and a later variant resolved names at the `run()`
statement's position. History is not the code in force:

- cells run out of order, and a restart with Run All replays a different
  history than the one that built the state;
- a cell that raised ran only up to the failing line, and history keeps all
  of it;
- magics rewrite cells (`%%capture` runs a nested `run_cell`; `%%time` runs
  its body through `exec`), so history and execution diverge;
- identical cell text run twice cannot be told apart.

Every round found about ten verified regressions, each from reconstructing
state the kernel already holds. Do not retry it.

## The idea: start from the live objects

The kernel's namespace is the code in force. Each object in it says where it
came from:

- A function's code object names the exact cell text that defined the
  version now bound (`co_filename`, `co_firstlineno`; ipykernel keeps every
  cell's text in `linecache` under a temp path named by its content, plain
  IPython under `<ipython-input-N-hash>`). Verified in a real ipykernel 7.4
  on Python 3.11: after a later cell failed before it could redefine
  `helper`, the live `helper` still pointed at the earlier text; a function
  defined in a `%%capture` cell resolved too.
- A class has no source pointer of its own (`inspect.getsource(cls)` raises
  in a kernel on 3.11, and `__firstlineno__` on 3.13+ has no file to go
  with), but its methods do.
- A plain constant's value is the only record of it there is.

So the record follows the names the task's code actually looks up, through
the live objects bound to them, and hashes each one's definition or value.

## The bar, and the kill criterion

Main compares nothing defined in a notebook, so it never re-runs a notebook
cell spuriously. Every edit this catches is a gain; every edit it misses is
only a residual; every spurious re-run it adds is a regression. That makes
the bar asymmetric:

- Merge-blocking: a crash or hang, a broken test, or a spurious re-run in the
  normal loop: extending the grid, re-running an unchanged cell, re-running
  the run cell, resuming, a kernel restart with Run All where every value the
  task reads is reproducible, another launcher, Python 3.11 to 3.14.
- **Kill criterion (written before implementing):** a spurious re-run that a
  reviewer reproduces in that loop is fixed only by comparing less (narrowing
  what is followed or hashed). If narrowing cannot fix it, that part of the
  rule comes out and becomes a documented residual. No fix may compare more.
- Accepted and documented, not a regression: a value the task reads that
  differs per session (`STAMP = time.time()`, `SEED = random.randint(...)`)
  re-runs the cells after a restart. The task reads it, and the record cannot
  tell whether the result depends on it (a random seed does). The re-run
  names the global, so the fix (pass it as a swept parameter, or compute it
  in the task) is clear.

## Design

### Where it applies

A root (the task, `pre_task`, a `Study` training function, a function a
partial binds) whose globals have no `__file__`: a Jupyter or IPython
kernel's `__main__`, and any namespace without a file. This is the branch
that today calls `_namespace_modules`; that call stays exactly as it is, so
nothing main records is lost. The driver-script rule is untouched.

### What is followed

From each root, a worklist over `(namespace, name)` pairs, bounded (at most
2000 objects, a visited set by object identity):

- **Reads come from bytecode, not `co_names`.** `co_names` also lists
  attribute names, so `cfg.lr` would follow an unrelated global `lr`. The
  read-set of a code object is the argument of every `LOAD_GLOBAL`,
  `LOAD_NAME` and `LOAD_FROM_DICT_OR_GLOBALS` instruction in it and in the
  code objects nested in its constants (`dis.get_instructions`, the same on
  3.11 to 3.14). Bytecode chooses what to follow; it is never hashed, since
  records are compared across Python versions.
- **Names resolve in the reading function's own `__globals__`.** Only code
  whose text lives in an in-memory `linecache` entry (one IPython wrote,
  with no mtime) is followed. Generated code (a dataclass's `__init__`,
  `co_filename == "<string>"`) is not.
- **Rebound names are left out.** A name any reached code stores or deletes
  as a global (`STORE_GLOBAL`, `DELETE_GLOBAL`: a task counting its calls in
  `global CALLS`) is not recorded: its value moves during a run.
- **IPython's own names are left out:** `_`, `__`, `___`, `_i`, `_ii`,
  `_iii`, `_iN`, `_N`, `_oh`, `_ih`, `_dh`, `In`, `Out`, `get_ipython`,
  `exit`, `quit`, `_exit_code`, and dunder names.
- **Objects are inspected by type only** (`type(value)`, never
  `isinstance`, never an attribute read on an instance), as
  `_namespace_modules` already is: a lazy proxy must not resolve
  (`test_the_record_never_resolves_a_lazy_object`).

### What each name records

One entry per name the task's code reaches, keyed `"<name> (notebook)"`, so
a re-run says which name changed (`code the task reaches (helper
(notebook))`).

| Bound to | Entry | Followed further |
| --- | --- | --- |
| A function defined in the notebook (through `inspect.unwrap`) | The canonical dump of its definition node, docstrings stripped, from the cell text; plus its captured values (below) | Its read-set; callables among its defaults and closure cells |
| A class defined in the notebook (its `__module__` is the namespace's `__name__`) | Its `ClassDef` dump, located through a method (below), plus its plain class attributes | Its notebook bases; its methods' read-sets |
| An instance of a notebook class | Its class's entry | As its class |
| A plain value | `(type name, canonical repr)` | Nothing |
| A list, tuple, dict or set holding notebook functions or classes (a registry) | The sorted entries of those callables (the container's other contents are not compared) | Those callables |
| A module, anything else | No entry (`_namespace_modules` still seeds project modules) | Nothing |

Details:

- **Locating a definition.** Parse the cell text (cached by text, bounded),
  and find the `FunctionDef`/`AsyncFunctionDef` named `co_name` whose
  `lineno`, or one of whose decorators' `lineno`, is `co_firstlineno` (a
  decorated function's first line is its first decorator's). A lambda: the
  `Lambda` nodes on that line, so a lambda passed inline to `run(...)` does
  not drag the grid in. No match (the text no longer parses, or names
  something else): no entry.
- **Locating a class.** Among the functions in `vars(cls)` (through
  `staticmethod`, `classmethod`, `property`), one whose `__qualname__` starts
  with `cls.__qualname__ + "."` and whose code lives in a cell; the innermost
  `ClassDef` named `cls.__name__` in that cell spanning the method's line. A
  class with no such method (a dataclass with fields only) has no dump; its
  plain attributes are its entry.
- **Plain attributes of a class:** the non-dunder names in `vars(cls)` bound
  to plain values, sorted. This also captures what the class body read when
  it ran (`K = SCALE` holds the value it captured, not today's `SCALE`).
- **Captured values of a function:** its `__defaults__`, `__kwdefaults__`
  and closure cells, each a plain value's repr or else its type name. A
  default holds what the definition captured when it ran, which is what
  runs: `def f(x=SCALE)` keeps the old default until its cell is re-run, so
  following `SCALE` by name would record a value the function does not use.
- **Plain values** are exactly `int`, `float`, `complex`, `bool`, `str`,
  `bytes`, `None`, and `tuple`/`frozenset` of plain values (exact types, so a
  `NamedTuple` or an `IntEnum` is not one), at most 10,000 elements and
  1,000,000 characters of repr in all, else no entry. The type name is part
  of the digest, so `1`, `1.0` and `True` differ. A frozenset's members are
  sorted by their repr (string hashing is randomized per process).
- **Mutable values are not compared** (a dict config, a list), and the
  reason stays written here so nobody "improves" it: a task that appends to a
  global `LOG = []` or fills `RESULTS = {}` changes the value during run 1,
  so the resume would re-run every cell.

### Where it is taken

A parallel worker gets the task by cloudpickle, without the kernel's
`linecache`, so the driver takes the notebook entries when the run starts
(`driver_script_entries`) and ships them in the existing `script_entries`
under the reserved key `"<notebook>"`, which no script path can be (they end
in `.py`). No new `_TaskRunner` field, so old pickles need no default. The
shipped snapshot is used even under the serial launcher, so a task that
mutates a global cannot change later cells' records within one run. When no
snapshot was shipped (the driver could not take it), the cell's process takes
its own; in a worker that has no cell text, only plain values come out, and
records compare on shared keys, so that is never an edit.

A config `_target_` naming `__main__.X` in a notebook is followed the same
way, in the cell's process (`cell_project_record` takes targets per cell).
Only the serial launcher can instantiate one; a worker's `__main__` is not
the kernel.

### No IPython import

The rule uses `linecache`, `dis`, `ast` and `sys`. It never imports IPython,
so the `core-only` and `min-versions` jobs are unaffected. Tests import it
with `pytest.importorskip`.

## Residuals (documented)

- A mutable value's contents (a dict config, a list), an object's state (an
  instance's attributes), data (arrays, frames), a value past the size bound.
- A default or closure value that is not plain (its type only).
- Code whose text the kernel does not keep: a `%%time` body, `exec`, a plain
  `python` REPL, a function from a `%run` file.
- A function wrapped without `__wrapped__` (numba's `@njit`, some
  decorators).
- A function that rebinds a global, and names it rebinds.
- A value that differs per session, which re-runs cells after a restart
  (accepted above).

## Tests (written before the code)

In an in-process `InteractiveShell` (`run_cell(..., store_history=True)`)
and in one real `jupyter_client` kernel, since their cell file names differ
(`<ipython-input-...>` against a temp path) and both must resolve.

Not an edit (the record is unchanged):
- re-running an unchanged helper cell; re-running the run cell; extending the
  grid;
- a comment, a docstring or formatting edit to a helper;
- a reset namespace replaying the same cells (a restart), and a real kernel
  restart replaying them;
- a task that appends to a global list, and one that counts calls in a
  `global`;
- a helper the task does not reach; a cell that failed before it could
  redefine the helper;
- a lambda passed inline to `run(...)` with a longer grid.

An edit (the record changes):
- a helper's body edited and re-run; a constant changed; a default captured
  from a global, after its cell is re-run; a class's method; a class
  attribute; a method-less dataclass's default; a callable reached through a
  registry dict; a closure's captured value; a helper only a class's method
  reaches.

End to end, the invariant: in a shell, run a sweep with `resume=True`, edit a
helper, resume, and read the metrics back: each cell's value is the one the
current code computes, then an unchanged resume reuses every cell. The same
with `cache_dir=`. One notebook pinned to a single digest on Python 3.11 to
3.14. The lazy-proxy test stays green.

## Docs inventory

The `run()` docstring's "Not covered" list, `project_fingerprints`'s
docstring, `docs/guides/resilience.md`, `docs/guides/exploration-to-paper.md`,
`src/mushin/agent_guide.md` (around line 168), `changes/270.changed.md`'s
"code defined in a notebook are left out" (unreleased, so it is amended), and
a fragment for this PR.
