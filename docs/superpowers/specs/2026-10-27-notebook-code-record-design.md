# Notebook code in the resume and cache_dir record: design

Date: 2026-10-27. Status: implemented in #278, after four review rounds that
each removed part of the first version (see "Review history"). Follows #270
(the record), #276 and #277 (its follow-ups).

## Problem

`resume=True` and `cache_dir=` compare a record of the code a cell ran
(`code_deps`, see `project_fingerprints`). For a task defined in a notebook
cell, main records two things: the task function's own source
(`code_fingerprint`, through `inspect.getsource`, which reads the cell from
`linecache`), and the project modules the notebook's names come from
(`_namespace_modules`). Nothing else defined in the notebook is compared. A
user who edits a helper or a class in a cell, re-runs that cell and resumes
gets the results the old code computed, silently.

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

## The idea: definitions of the live objects

The kernel's namespace is the code in force. A function's code object names
the exact cell text that defined the version now bound (`co_filename`,
`co_firstlineno`; ipykernel keeps every cell's text in `linecache` under a
temp path named by its content, plain IPython under
`<ipython-input-N-hash>`). A class has no source pointer of its own
(`inspect.getsource(cls)` raises in a kernel), but its methods do. The record
follows the names the task's code looks up, through the objects bound to
them, and hashes their definitions.

**It compares no value.** A value is only safe to compare when every writer
of it is found, so that a value the run itself changes (a counter, a cache)
is not taken for an edit. In a notebook a writer can sit wherever a value
can: four review rounds found writers in containers, instances, bound methods
and decorator closures, each one a spurious re-run. Definitions are static:
the run does not rewrite a function's or a class's source.

## The bar, and the kill criterion

Main compares nothing defined in a notebook, so it never re-runs a notebook
cell spuriously. Every edit this catches is a gain, every edit it misses is a
residual, and every spurious re-run it adds is a regression.

- Merge-blocking: a crash or hang, a broken test, a regression against main,
  a record that depends on the walk's order or the hash seed, a record that
  names code that did not run, or a spurious re-run in the normal loop
  (extending the grid, re-running an unchanged cell or the run cell,
  resuming, a restart replaying the cells, another launcher, Python 3.11 to
  3.14).
- Kill criterion: a spurious re-run is fixed only by comparing less. Round 3
  added: "a reproduced spurious re-run through a statically visible write
  the scan misses means numbers come out and only definitions are
  compared." Round 4 reproduced one (a writer wrapped by a notebook
  decorator), and value comparison was removed.

## Design

### What is followed

From each root, a breadth-first walk over the names the reached code looks
up, bounded at 2000 functions and classes:

- **Reads come from bytecode, not `co_names`.** `co_names` also lists
  attribute names, so `ns.scale` would follow an unrelated global `scale`.
  The read-set of a code object is the argument of every `LOAD_GLOBAL`,
  `LOAD_NAME` and `LOAD_FROM_DICT_OR_GLOBALS` instruction in it and in the
  code nested in its constants. Bytecode chooses what to follow; it is never
  hashed, since records are compared across Python versions.
- **One resolver** (`_notebook_object`) says what a value stands for: a
  function defined in a cell, through decorators' `__wrapped__` (a library's
  too: `@contextmanager`, `@torch.no_grad()`), an `lru_cache` wrapper and a
  `functools.partial`'s function; a class defined in the notebook; an
  instance's class; a bound method's instance's class (the method can call
  the others through `self`). Only code whose text lives in an in-memory
  `linecache` entry counts.
- **Edges the run cannot move:** a class to its methods and its notebook
  bases, a wrapper defined in a cell to its `__wrapped__`. Not a closure's or
  a default's contents: following a callable a capture holds needs to know
  that nothing rebinds it, which is the writer discovery above.
- **A name the reached code rebinds is left out** (`global helper; helper =
  fast`). Dropping a key never re-runs a cell (records compare on the keys
  both have), and it is taken from the reached code only, so the key set
  does not depend on code the task cannot reach.
- **IPython's own names are not followed:** `_`, `__`, `___`, `_i`, `_ii`,
  `_iii`, `_iN`, `_N`, `_oh`, `_ih`, `_dh`, `In`, `Out`, `get_ipython`,
  `exit`, `quit`, `_exit_code`, and dunder names.
- **Objects are inspected by type only**; only a function's or an
  `lru_cache` wrapper's own `__dict__` is read, so a lazy proxy is never
  resolved (`test_the_record_never_resolves_a_lazy_object`).

### What each name records

One entry per name, keyed `"<name> (notebook)"`, so a re-run says which name
changed (`code the task reaches (helper (notebook))`); roots sharing a name
fold into one.

- **A function:** the canonical dump of its definition node, docstrings
  stripped, found in its cell: the `def` named `co_name` whose `lineno`, or a
  decorator's, is `co_firstlineno`; for a lambda, the `Lambda` nodes on that
  line, so a lambda passed inline to `run(...)` does not bring the grid.
- **A class:** its `ClassDef` dump, found through a method defined in its
  body (the innermost `ClassDef` of that name spanning the method's line). A
  class with no such method (a dataclass with fields only) has no dump.
  3.14's `__annotate__` (the compiler's) is not a method.
- **Digests per component.** A digest covers what its object reaches through
  the edges above. The graph is condensed into strongly connected components
  (`_components`, iterative Tarjan, checked against brute-force
  reachability): a component hashes its members' own definitions and the
  digests of the components it reaches, so a digest does not depend on the
  walk's order. Without capture edges the graph has no cycle; the components
  keep that true if one ever appears.
- **Bounds omit, never truncate:** past 2000 objects the record has no
  notebook entries, with a warning, since which objects a cut walk took
  depends on its order.

### Where it is taken

A parallel worker gets the task by cloudpickle, without the kernel's
`linecache`, so the driver takes the notebook entries when the run starts
(`driver_script_entries`) and ships them in `script_entries` under the
reserved key `"<notebook>"`, which no script path can be (they end in
`.py`). No new `_TaskRunner` field, so old pickles need no default. The
shipped snapshot is used under the serial launcher too.

IPython 8.6 and later write a cell's `linecache` entry with no mtime;
earlier versions (7.x to 8.5, the floor of the dev group's ipykernel) write
one and keep their cells in `linecache._ipython_cache`, which is read too. A
file-backed entry (a `%run` file) is not: the file can change after it ran.

The rule never imports IPython, so the `core-only` and `min-versions` jobs
are unaffected. Tests import it with `pytest.importorskip`.

## Residuals (documented)

- **Values:** a number, a string, a container's contents, an object's state,
  data, and what a default or a closure holds. Pass a value as a swept
  parameter, or use a new `working_dir` after editing one. A flag or a
  per-session value (a timestamp) the task reads is therefore not an edit
  either.
- A function or class reached only through a container (a registry
  `MODELS["cnn"]`), a closure or a default.
- A class's attribute values beyond its literal text (`K = SCALE`), and a
  method-less dataclass (its class statement cannot be located).
- A `_target_` defined in the notebook (as on main): its per-cell walk could
  not see the task's writes; hydra-zen's `builds` puts a class's defaults in
  the config, which is compared.
- Code whose text the kernel does not keep: a `%%time` body, `exec`, a plain
  `python` REPL, a `%run` file. A wrapper without `__wrapped__` (numba's
  `@njit`).
- A notebook reaching more than 2000 functions and classes (warns).

## Review history

- Round 1 (7 + 13 findings): IPython before 8.6, 3.14's `__annotate__`
  (hidden by the test kernel inheriting `from __future__ import
  annotations`), `nonlocal` counters, factorial digests on cycles, secrets in
  captured values, a `_target_` fallback that dropped main's entries.
- Round 2 (9 + 12): secrets kept arriving by paths no name rule covered
  (aliases, `os.environ`, factory arguments), so strings stopped being
  compared; registries, captured containers and notebook `_target_`s were
  cut back.
- Round 3 (9 + 12): writers hid in instances, bound methods and hooks the
  walk no longer visited; registries and class attribute values went, and
  the writes were taken from the whole namespace.
- Round 4 (7 + 4): the namespace-wide writes let unrelated cells change a
  digest, and a writer behind a notebook decorator was still missed. The
  kill criterion fired: values are not compared.

## Tests

`tests/test_resume_notebook.py`, in a simulated kernel that keeps cell text in
`linecache` as IPython does and compiles cells without the test module's
future flags, plus an `InteractiveShell`, a real ipykernel restart and edit,
and a real kernel resuming under joblib:

- edit cases: a helper's body; a class's method or literal attribute; an
  instance made again from an edited class; a helper only a method reaches; a
  base class's method; a method calling `super()`; a helper reached through a
  `partial`, a library decorator, a notebook decorator, `functools.wraps`, a
  bound-method alias (and a method it calls through `self`); a lambda.
- non-edit cases: unchanged cells, the run cell, a longer grid, comments and
  docstrings, a restart replaying the cells, a failed cell, unreached
  helpers, a global function named like an attribute the task reads, a task
  rebinding a helper, the review rounds' writer and counter reproductions,
  IPython outputs, and each value residual pinned as not compared.
- end to end: a resume after a helper edit re-runs and returns the new
  values, an unchanged resume and a longer grid reuse, and `cache_dir=` keys
  on the definitions.
- one notebook pinned to a single digest on Python 3.11 to 3.14 (with the
  comprehension shapes 3.11 and 3.12 compile differently); `_components`
  against brute force; the bound and its warning; the lazy proxy.

## Docs inventory

The `run()` docstring, `project_fingerprints`'s docstring,
`docs/guides/resilience.md`, `docs/guides/exploration-to-paper.md`,
`src/mushin/agent_guide.md`, `changes/270.changed.md` (its notebook sentence
amended, unreleased), and `changes/278.changed.md`.
