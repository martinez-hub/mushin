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
  `global CALLS`) is not recorded: its value moves during a run. Likewise a
  closure variable it rebinds (`STORE_DEREF` to one of the storing code's own
  free variables: a timing decorator's `nonlocal total`; a store to a local a
  nested scope captures is not one, and 3.12 compiles that differently), all
  defaults when it stores `__defaults__` or `__kwdefaults__`, and every
  closure cell when it stores `cell_contents`, are recorded as moved: not
  their value, not their type, and not as an edge. An attribute name it
  stores leaves that class attribute out.
- **IPython's own names are left out:** `_`, `__`, `___`, `_i`, `_ii`,
  `_iii`, `_iN`, `_N`, `_oh`, `_ih`, `_dh`, `In`, `Out`, `get_ipython`,
  `exit`, `quit`, `_exit_code`, and dunder names.
- **Objects are inspected by type only** (`type(value)`, never
  `isinstance`, never an attribute read on an instance; only a function's or
  an `lru_cache` wrapper's own `__dict__` is read for `__wrapped__`, so a
  library's wrapper is followed), as `_namespace_modules` already is: a lazy
  proxy must not resolve (`test_the_record_never_resolves_a_lazy_object`).
- **Bounds omit, never truncate.** A container walk visits each container
  once, breadth first, within 10,000 items; past that, the entries it feeds
  are left out. Past 2000 functions and classes the record has no notebook
  entries, with a warning. A cut walk's findings depend on a set's
  per-process order, so a part of them would re-run cells after a restart.
- **Digests per component.** A digest covers what its object reaches
  through captured values, methods and bases. The graph is condensed into
  strongly connected components (iterative Tarjan): a component hashes its
  members' own parts and the digests of the components it reaches. That is
  order-free and linear; a method calling `super()` and closures sharing a
  dispatch table are cycles.

### What each name records

One entry per name the task's code reaches, keyed `"<name> (notebook)"`, so
a re-run says which name changed (`code the task reaches (helper
(notebook))`).

| Bound to | Entry | Followed further |
| --- | --- | --- |
| A function defined in the notebook (through `__wrapped__` and a `partial`'s function) | The canonical dump of its definition node, docstrings stripped, from the cell text; plus its captured values (below) | Its read-set; a notebook function or class a default or closure cell holds directly |
| A class defined in the notebook (its `__module__` is the namespace's `__name__`) | Its `ClassDef` dump, located through a method (below), plus its plain class attributes | Its notebook bases; its methods' read-sets |
| An instance of a notebook class | Its class's entry | As its class |
| A plain value (numbers) | `(type name, canonical repr)` | Nothing |
| A list, tuple, dict or set the task reads, holding notebook functions or classes the notebook also binds by name (a registry filled by decorators) | The set of those callables' digests (the container's other contents are not compared) | Those callables |
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
  A captured container is not walked: only a function or class a capture
  holds directly is followed (an edge), so a memo dict of closures the task
  fills cannot move the record.
- **Strings are never compared by value**, as a global, a class attribute, a
  captured value or inside a tuple. A digest of a short string alone, or of a
  known definition with it, can confirm a guess of a secret, and round 2 of
  the review showed no rule over names covers every way one arrives (an
  alias `PW = DB_PASSWORD`, `os.environ[...]`, a factory argument, a config
  dataclass field). A string constant is compared only through the code
  that reads it. A number whose own, parameter or closure variable name
  matches `password`, `passwd`, `secret`, `api_key` or `credential` is
  recorded by type only (`MAX_TOKENS = 512` is a setting).
- **Plain values** are exactly `int`, `float`, `complex`, `bool`, `None`,
  and `tuple`/`frozenset` of plain values (exact types, so a `NamedTuple` or
  an `IntEnum` is not one), at most 10,000 elements and 1,000,000 characters
  of repr in all, else no entry. The type name is part of the digest, so
  `1`, `1.0` and `True` differ. A frozenset's members are sorted by their
  repr: two equal frozensets built in another order iterate differently.
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

A config `_target_` naming a class defined in the notebook is not followed,
as on main. It is taken per cell, after the task has run, and apart from the
roots, so the writes it must exclude are the task's, which its own walk
cannot see; round 2 reproduced spurious re-runs through it. hydra-zen's
`builds` puts a class's defaults in the config, which is compared.

IPython 8.6 and later write a cell's `linecache` entry with no mtime;
earlier versions (7.x to 8.5, the floor of the dev group's ipykernel) write
one and keep their cells in `linecache._ipython_cache`, which is read too.

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
- A function that rebinds a global, and names or closure variables it
  rebinds.
- A string constant (a model name, a prompt), compared only through the code
  that reads it; a number derived from a secret under an innocent name.
- A `_target_` defined in the notebook.
- A lambda or closure kept in a registry (only callables the notebook binds
  by name count), and a registry deeper than 4 levels or past 10,000 items;
  a notebook reaching more than 2000 functions and classes (warns).
- State the task changes while it runs other than by rebinding a name, an
  attribute or a closure variable (`setattr`, `globals()`, `exec`, a
  function's defaults changed in place) re-runs the cells on the next
  resume, naming the entry: keep such state outside the notebook's code.
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
