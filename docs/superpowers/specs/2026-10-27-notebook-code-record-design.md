# Notebook code in the resume and cache_dir record: design

Date: 2026-10-27. Status: implemented in #278, after six review rounds; the
first four each removed part of the first version (see "Review history").
Follows #270 (the record), #276 and #277 (its follow-ups).

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
- **A name the notebook's code rebinds is not followed** (`global helper;
  helper = fast`): what it holds when the run starts can be gone by the
  next, and following it would let that value shape the record (a
  `cache_dir` key hashes every entry). The writes are taken before the walk
  from every function the namespace holds or can reach without running
  anything (`_notebook_writes`): through decorators, a `partial` and its
  arguments, a bound method, a class's members and nested classes, an
  instance's class, closure cells, defaults, a class's bases (a writer can
  sit in a base no name holds: `class Model(Model)`, a decorator's
  subclass), and lists, tuples, sets, deques and dicts (`defaultdict` and
  `OrderedDict` too) of at most 256 items (`@HOOKS.append` keeps a writer
  nowhere else). Containers are matched by exact type and a mapping is read
  through `dict`'s own iterator, so walking one runs no notebook code (an
  `OrderedDict`'s iterator hashes each key again). Only those objects count
  toward the bound, not the data beside them, so a results dict of floats
  does not turn the record off; whether a type can lead to a function is
  asked once per type.
  Round 5 found writers the reached code alone missed, in a list of hooks
  and a callback instance. The scan is breadth first, each object at its
  least depth, and a container is walked by its size alone, so it does not
  depend on iteration order; past 20,000 objects there are no notebook
  entries, with a warning.
- **IPython's own names are not followed:** `_`, `__`, `___`, `_i`, `_ii`,
  `_iii`, `_iN`, `_N`, `_oh`, `_ih`, `_dh`, `In`, `Out`, `get_ipython`,
  `exit`, `quit`, `_exit_code`, and dunder names.
- **Objects are inspected by type only**; only a function's or an
  `lru_cache` wrapper's own `__dict__` is read, so a lazy proxy is never
  resolved (`test_the_record_does_not_resolve_a_lazy_object`).

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

- A name any of the notebook's functions rebinds, even one the task never
  calls (not compared). A writer the scan cannot reach, so the name it
  rebinds is still followed and a resume after the run re-runs the cells: in
  a container larger than 256 items, deeper than three steps, in an
  instance's attributes, a dict's keys, a metaclass, a `defaultdict`'s
  `default_factory`, a container subclass, or a class defined outside the
  notebook; a name rebound through `globals()` or `exec`, which no bytecode
  names. Under a multiprocess launcher the rebinding happens in a worker,
  so the kernel's name, and the record, do not change.
- A notebook metaclass's attribute hooks (`__getattribute__`, a property)
  can run while the record is taken, since a class's `__dict__` and
  `__mro__` are read through it; if one raises, the notebook has no entries
  (as on main), with a warning on every run. Types are matched by identity,
  so a metaclass's `__eq__` never runs.
- A task that assigns to a class at run time (`Model.f = other`, `setattr`):
  the class's digest is taken from its live members. Wrapping a method in a
  cache (`functools.lru_cache`) and storing a function a method defines are
  not this: they resolve to the same definition.

- **Values:** a number, a string, a container's contents, an object's state,
  data, and what a default or a closure holds. Pass a value as a swept
  parameter, or use a new `working_dir` after editing one. A flag or a
  per-session value (a timestamp) the task reads is therefore not an edit
  either.
- A function or class reached only through a container (a registry
  `MODELS["cnn"]`), a closure or a default.
- A class's attribute values beyond its literal text (`K = SCALE`), and the
  text of a class whose statement cannot be located: a method-less
  dataclass, or a class whose every method is wrapped by a decorator defined
  in the notebook (the wrapper's code is the decorator's; its methods'
  bodies are still compared, through `__wrapped__`).
- A helper read only by a nested class's methods (`Outer.Inner().f`): the
  walk follows a class's direct methods, so the helper is not followed
  (the nested class's own text is, inside the outer class statement).
- A root that calls another function bound to its own name (a task kept as
  `saved_task` after `task` was redefined, calling the new `task`): the
  other function's entry shares the root's key and is left out, so an edit
  to it is not seen (Codex, after the final review).
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
- Round 5 (final head, 8 + 3, and Codex 3): a writer the walk never reached
  rebound a name bound to a function (a hook in a list), and a rebound
  name's value still shaped the `cache_dir` key. Names the notebook rebinds
  are now taken from the whole namespace before the walk and not followed.
  Cached methods and properties, class methods as roots, and direct members
  only were fixed alongside. Its delta review found the scan skipping a
  class's bases and the stdlib registry types, and counting data toward its
  bound; all three were fixed.
- Round 6 (2 + 10): walking an `OrderedDict` through its own iterator ran
  its keys' `__hash__`, so one whose key changed raised and the notebook lost
  every entry, silently; mappings are now read through `dict`'s iterator. The
  rest were writers in places the scan does not look (residuals above), a
  per-item filter cost, and tests: every scan branch is now pinned by a case
  that fails without it, and a not-edit case must have entries.
- Round 7 (0 + 8, nothing on the bar): a notebook whose record raises now
  warns instead of dropping its entries silently (a metaclass `__eq__`);
  dict keys joined the residuals; the tuple, set, frozenset and `partial`
  keyword branches got their cases; the multiprocess qualifier and the
  docstring wording were fixed.
- Round 8 (0 + 9, nothing on the bar): types are matched by identity (a
  metaclass `__eq__` that answered yes had the scan iterate a notebook
  object); the warning reads the exception's name from its slot; the guide
  lists where the write scan looks.
- Final head (0 + 9 and 0 + 1, nothing on the bar). A per-run reset of the
  warnings was reverted on a reviewer's claim that Python shows a repeated
  warning from one place once per process anyway, then restored: that memory
  is cleared whenever the warning filters change, and a sweep's run changes
  them (checked: a probe warning repeats after `run()`), so without the
  reset a resume in the same kernel would never warn. Tests now pin the identity check
  with a raising `__eq__` and the warning with a real metaclass hook. Two
  residuals were left as such, since no change was allowed past the final
  review except for a blocking finding: a class body whose `__module__` is
  an object with its own `__eq__` (the comparison runs it), and a
  `BaseException` raised by notebook code during the record, which is not
  caught. Codex then found a blocking one, fixed with its tests:
  `_strip_tree` kept the docstrings of functions defined in an `except`
  handler or a `match` case, so a docstring edit there re-ran the cells (in
  a project file too). The class whose methods are all wrapped by a
  notebook decorator joined the residuals.

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

## Follow-up (2026-11-02): submodules a notebook reaches through a package

Plan, written before any edit.

**The gap** (found by `/code-review` on #283, reproduced on 3.15 and 3.11):
`import pkg.sub` binds only `pkg` in the namespace, and `_namespace_modules`
records a module value by its own name. So `pkg/sub.py` is not in the record,
and after an edit and a restart a resume reuses every cell. It is a wrong
result, on every Python. A file task is not affected: `_import_pairs` reads
`import pkg.sub` from the syntax tree. `from pkg import sub` and
`import pkg.sub as s` are not affected either: they bind the submodule
itself. On 3.15 the lazy form (`lazy import pkg.sub`) records `pkg.sub` while
unresolved and only `pkg` once resolved. That makes `cache_dir` miss on a
second run in the same kernel (a residual of #283).

**First design, replaced after review.** It walked the live module graph:
a project package held by the namespace brought every submodule bound under
it. `/code-review`, the adversarial workflow (4 lenses) and Codex on 9bc983d
showed it fails this record's own bar:
- the record depended on what happened to be loaded (an unrelated cell's
  `import pkg.plotting` made an edit there re-run every cell, and changed the
  `cache_dir` key between two runs);
- names it found reached `_find_file`'s `getattr` on a `LazyLoader` module,
  so taking the record executed it (and, if its import failed, swallowed the
  error and left the module half-executed);
- its package-level foreign check misjudged namespace packages both ways.
The gap is syntactic (`import pkg.sub` is in the cell's text), so its fix
inspected the wrong thing. That was my altitude error, against this design's
own "inspect by type only".

**Second design.** The dotted names the kernel's kept cells import with a
plain `import pkg.sub` (no `as`), read from the cells' text, each kept only
while `pkg` is bound in the namespace to a module, or on 3.15 to a lazy
import. That brings `import pkg.sub` to parity with `from pkg import sub` and
`import pkg.sub as s`, which bind the submodule and are recorded already.
- Kept cells: `linecache` entries named like an IPython cell
  (`<ipython-input-N-hash>`, or ipykernel's `ipykernel_<pid>/<n>.py`) with no
  mtime, and IPython 7's `_ipython_cache`. A cell's statements at its top
  level count, including inside `if`/`try`/`with`; a function's or class's
  body does not, since its import binds a local.
- No module object is read, so nothing is loaded and nothing that merely
  happens to be loaded counts.
- 3.15: `lazy import pkg.sub` is the same statement (`is_lazy=1`), and `pkg`
  is bound to a proxy or, once resolved, to the package, so the record is
  stable across resolution.

**Separately: module attributes read hook-free.** The record read
`__name__`, `__file__`, `__path__` and `__spec__` of modules with `getattr`,
which runs an `importlib.util.LazyLoader` module's file. That is reachable on
main without this PR: a cell binding such a module directly, or a file task
importing one. One helper reads them from the module's own `__dict__`
(`module_from_spec` puts them there before the loader swaps the class), at
every site: `_is_foreign_module`, `_find_file`, `_file_entries`,
`_loaded_digest`, `_namespace_modules`, and a class's module dict.

**Residuals**
- A parallel worker has no `linecache`, so under joblib the seeds are absent:
  a missed edit, which the bar tolerates. Notebook records already differ
  between serial and joblib, since cloudpickle ships only the globals the
  task references. Shipping the seeds from the driver beside `"<notebook>"`
  is the complete fix.
- Every kept cell counts, including an older version of a re-run cell, so an
  `import pkg.old` since deleted from the notebook is still a seed while
  `pkg` is bound. A recompute, not a wrong result.
- An import inside a notebook function's body binds a local, so the module it
  names is not recorded (as before this PR).

**Tests**
- `import pkg.sub` and `import pkg.a.b` record every level; parity: `import
  pkg.sub`, `import pkg.sub as s`, `from pkg import sub` (and on 3.15 `lazy
  import pkg.sub`) all record `pkg.sub (file)`.
- End to end: edit `pkg/sub.py`, restart, resume -> re-runs with new values.
- Load state does not count: a submodule loaded by `importlib.import_module`
  and not imported by a statement is not recorded; `pkg` rebound to a value
  drops the seed.
- 3.15: two `cache_dir` runs with `lazy import pkg.sub` in one kernel reuse.
- A real `LazyLoader` module in `sys.modules`, bound directly in a cell: the
  record keeps its file and does not load it (`_LazyModule` still, side effect
  unfired); `_find_file` returns its file without loading it.
- A standard-library dotted import records nothing.
- Mutation-check each by node id.

**Round 2 (2026-11-03): deferred to 0.17.1.** `/code-review` (11) and the
adversarial workflow (7 confirmed, minor) reviewed 420d14c.
- **The cell scan has unresolved design problems.**
  - A parallel worker keeps no cells.
  - The scan costs about 0.8 s per run once a session keeps 500 cells, since
    the 128-tree cache thrashes.
  - Its cell-name whitelist is narrower than how notebook definitions are
    found (Databricks' `<command-N>`, `IPYKERNEL_CELL_NAME`).
  - Compiling every import cell prints stray `SyntaxWarning`s.
- **The hook-free reads opened new cases and missed sites.**
  - `object.__getattribute__(m, "__dict__")` still runs a `__dict__` property
    on the module's type, and apipkg's `ApiModule` (`import py`) has one that
    loads everything, which main never triggered.
  - A `ModuleType` subclass holding `__file__` on its class is no longer found.
  - `isinstance` on a partial's bound argument reads `__class__`, which
    `_LazyModule` intercepts.
  - Two sites (`_file_entries`' top-level read, `_find_file`'s parent walk)
    have no test.

The hook-free commit came from a grep inventory added mid-PR, and round 2
showed that inventory was incomplete, so it is not merged on its own either.

**Plan for 0.17.1**, each part planned before code:
- **Seeds:** take them from what the record already follows. That is the
  bytecode of the reached code (`IMPORT_NAME`, and `LOAD_GLOBAL pkg` followed
  by `LOAD_ATTR sub`), which travels with the task to workers and covers
  imports inside function bodies. Alternatively, the driver ships its seeds
  beside `"<notebook>"`. Either replaces the cell scan.
- **Hook-free reads:**
  - first, a complete inventory of every operation the record performs on a
    module object or a value (attribute reads, `isinstance`, `issubclass` on
    instances, `repr`, `hasattr`);
  - then reads through `types.ModuleType.__dict__["__dict__"].__get__(module)`
    for module instances, `type()` checks instead of `isinstance`, and a
    class-dict fallback for plain values only.

**0.17.0 documents both limits:**
- `import pkg.sub` in a notebook counts only through `pkg`; write
  `from pkg import sub`.
- A module loaded through `importlib.util.LazyLoader` is imported when the
  record reaches it.

(0.17.0 shipped different advice from the last bullets: after editing a
submodule reached that way, use a new `working_dir` (#288).)

## Round 3 (2026-11-03): seeds from the reached bytecode

Plan, written before any edit. It replaces the second design's cell scan. The
hook-free reads are not part of it: they get their own PR, planned from a
complete inventory first.

**What is followed.** The notebook code the record already reaches (the
`codes` of `_notebook_entries`: the roots, the cell functions their names look
up, a reached class's methods, a wrapper's `__wrapped__`), and the code nested
in each. In that bytecode:
- A global read (`LOAD_GLOBAL`, `LOAD_NAME`, `LOAD_FROM_DICT_OR_GLOBALS`) of
  `g` followed directly by attribute reads (`LOAD_ATTR`, and 3.11's
  `LOAD_METHOD`) `a`, `b`: when the namespace binds `g` to a module named `m`,
  the seed `m.a.b`. Only the whole chain is seeded, since `_file_entries`
  follows a name's packages too: `m.a` is found when `m.a.b` is an attribute.
  A name the walk does not follow (a dunder, one of IPython's, one a notebook
  function rebinds) gives no seed, by the same test.
- An absolute import (`IMPORT_NAME`, level 0) seeds its module and, with a
  from-list, each `module.name` (not `*`). A relative import gives none. The
  level and the from-list are the two constants loaded just before it; their
  opcodes differ by version (`LOAD_SMALL_INT` from 3.14,
  `LOAD_COMMON_CONSTANT` on 3.15), so their values are read, not the opcodes.
- 3.15: a lazy proxy's repr names the same `n` for `lazy import pkg.sub`
  (bound as `pkg`, resolving to `pkg`), `lazy import pkg.sub as s` and `lazy
  from pkg import sub` (both resolving to `pkg.sub`). A chain on `g` takes as
  its base `n`'s first component when `g` is that component, else `n`. That is
  exact for every form but `lazy import a.b as a`, whose chains then miss a
  deeper submodule; `n` itself is still seeded by `_namespace_modules`.

**Why bytecode is safe here.** It only chooses what to follow; no digest reads
it (as with `_code_names`). A probe on 3.11.14, 3.12.12, 3.13.12, 3.14.3 and
3.15.0 gave the same chains and imports for 24 constructs (a call, an
attribute, a store, a delete, a comprehension, a generator, a lambda, a nested
function, `with`, `async with`, `await`, `try`/`except`, an f-string, a
conditional expression, a class body with an annotation, a method). A suite
test pins those sets, so every CI Python checks them.

**What it reads.** No module attribute beyond what main reads: a chain's base
is the `__name__` of a module in the namespace, which `_namespace_modules`
already reads for every module there, and `_find_file` reads `__file__`,
`__path__` and `__spec__` as for any other seed. Load state does not count: a
seed is a name in the code, found by its file whether or not it is loaded.

**Where it is taken.** In the driver, in the same walk as the notebook
entries, and shipped beside them under a second key (`NOTEBOOK_MODULES`). A
worker keeps no cells, but it has the same bytecode, so it records the same
seeds; a cell given no shipment takes its own. A walk past its bound, or one
that fails, gives no seeds, as it gives no entries: a cut walk's seeds would
depend on its order.

**Residuals**
- A module reached only dynamically (`importlib.import_module(name)`,
  `getattr(pkg, name)`), or through a chain a call or a subscript breaks
  (`get().sub`, `mods[0].sub`), is not followed, as in a file task.
- 3.15: a cell's `lazy import pkg.sub` whose task never names `pkg.sub`
  records `pkg.sub` before the proxy resolves and only `pkg` after, so a
  second `cache_dir` run in that kernel misses (as in 0.17.0).
- `cache_dir` keys on every entry, so a notebook task whose record gains a
  submodule misses the cache once after the upgrade.

**Tests** (each new one mutation-checked by node id, with the counts checked):
- the extraction on the probe's source gives the pinned chains and imports;
- `import pkg.sub` with `pkg.sub.f`, `import pkg.sub as s`, and `from pkg
  import sub` record `pkg.sub (file)`, and `import pkg.a.b` records every
  level;
- a cell helper the task calls reaches `pkg.sub`; a cell function the task
  does not reach, naming `pkg.other`, adds nothing;
- `import pkg.sub` and `from pkg import sub` inside the task's body are
  recorded; a relative import is not (a top-level module of the same name
  exists, so ignoring the level would record it);
- load state: `pkg.other` loaded by a cell but not named in the reached code
  is not recorded, and `pkg` bound to a non-module gives no chain seed;
- end to end: edit `pkg/sub.py`, restart, resume, and the cells re-run with
  the new values;
- shipping: ship, rebind `pkg` in the kernel, then record with the shipment:
  `pkg.sub (file)` is still there;
- a walk past its bound, and a failed walk, give no `(file)` seed from a
  chain;
- 3.15: the three lazy forms record `pkg.sub (file)` before and after the
  proxy resolves, and two `cache_dir` runs in one kernel reuse;
- `os.path.join` and `np.linalg.norm` add no entry.

**Cost**, measured before review: `project_fingerprints(task)` on a kernel
keeping 500 cells, with a task reaching 50 cell functions that name 10
submodules (3.11 on an Apple M1). Main: 31 ms the first time, 2.5 ms after (median of
20), recording 1 file. The branch: 28 ms, then 2.9 ms, recording 11 (the
package and the 10 submodules). The difference is the 10 files followed.

**Docs:** remove the `import pkg.sub` limit from `docs/guides/resilience.md`,
`src/mushin/agent_guide.md` and the `run()` docstring, and the docstring's
"project modules its functions import inside their bodies". Every LazyLoader
sentence stays. The fragment is `changes/284.fixed.md`; it notes the one-time
`cache_dir` miss.

## Round 4 (2026-11-03): seeds from the definitions' syntax trees

`/code-review` (13 findings) and the adversarial workflow (8 confirmed, none
refuted) reviewed round 3 (a6e7f86). Plan, written before any edit.

**The altitude error.** Round 3 read bytecode because "a worker has it", but
a worker never reads any: the driver ships the modules, and without cell text
a worker's walk reaches no cell function anyway (`_cell_function` needs the
text). Bytecode also misses code a cell runs outside a function: a class's
bases and body, a decorator, a default. `class Model(pkg.sub.Base)` after
`import pkg.sub` still recorded only `pkg`, which is the bug itself.

**The design.** Read the modules from the syntax trees the digests already
hash: `_definition_nodes` for a function (its decorators, defaults and
annotations included) and `_class_node` for a class (its bases, keywords,
decorators and body). In those nodes:
- each maximal attribute chain on a name (`pkg.sub.f`), the name resolved
  through an import inside the same definitions first (`import pkg`, then
  `pkg.sub.f`), else through the namespace by the walk's name test and
  `_chain_bases`;
- each absolute import (`ast.Import`; `ast.ImportFrom` at level 0 with its
  names, not `*`).

The syntax tree reads the same on every supported Python (probed on
3.11.14-3.15.0, with the round 3 constructs plus decorators, defaults,
annotations, bases, a metaclass, a class body and `staticmethod(...)`), so no
opcode differs by version. No parse is added: `own()` already gets these
nodes, and both now come from one lookup per unit, so the digests are
unchanged. It reads only reached definitions, not every kept cell, so the
round 2 objections to the cell scan (all cells, a cell-name whitelist,
`SyntaxWarning`s) do not apply.

**From the reviews**, beyond the design:
- A shipment counts per key (both reviews). An older driver's shipment (a
  0.17.0 runner, unpickled by this version) has no `NOTEBOOK_MODULES`: its
  entries still count, with no modules (that driver took none). A cell walks
  for itself only when the entries are missing.
- An ambiguous lazy proxy gets both bases (both reviews). `lazy from pkg
  import pkg` and `lazy import pkg.pkg` both name `pkg.pkg` bound as `pkg`, so
  when the binding is the proxy's first component and the proxy names more,
  the bases are that component and the full name. Before the proxy resolves,
  the record then holds what it holds after, plus whatever the other reading
  names, which is nothing unless both readings are files.
- The notebook-named modules go first in the seeds, so a record cut by
  `_FILES_MAX` keeps the files the task's own code names.
- One helper says which module a value stands for (`_namespace_modules` and
  `_chain_bases` both use it), and one says which names are IPython's or
  dunders (`_notebook_writes` and the walk both use it), so the planned
  hook-free reads change one place.
- The fragment says what changed and no more: on 3.15 a second `cache_dir`
  run reuses when the task names `pkg.sub`, and cells recorded by 0.17.0 keep
  their record (a resume keeps a reused cell's prior record).
- `project_fingerprints`'s docstring names the modules the notebook code
  names.

**Not changed:** shipping only project modules (`/code-review`'s
efficiency finding). `_file_entries` already skips a loaded foreign package
per top-level name, and the measured cost is within noise.

**Residuals**
- A name used both as a module global and as a local of the same name (a
  parameter `np`) is read as the module: a recompute when that file changes,
  never a missed edit.
- Reached only dynamically (`importlib.import_module(name)`, `getattr(pkg,
  name)`), or through a chain a call or a subscript breaks (`get().sub`), is
  not followed, as in a file task.
- 3.15: `lazy import pkg.sub` with a task that never names `pkg.sub` records
  `pkg.sub` before the proxy resolves and only `pkg` after.

**Tests** (round 3's, adapted, plus):
- a class base, a class attribute, a `staticmethod(...)` in a class body, a
  decorator and a default, each `pkg.sub.X` after `import pkg.sub`, record
  `pkg.sub (file)`;
- `import pkg` inside the task's body, then `pkg.sub.f(a)`;
- a module bound under another name (`import pkg as sp`, then
  `sp.sub.f(a)`), and on 3.15 a proxy bound under another name;
- 3.15: `lazy from pkg import pkg` records the same before and after it
  resolves;
- an old shipment (no `NOTEBOOK_MODULES`) in a worker with no cell text keeps
  the driver's notebook entries;
- a record cut by `_FILES_MAX` keeps the submodule the task names;
- the extraction on the probe's source gives the pinned names.
Each new rule mutation-checked by node id; the cost measured again.

**Results.** 33 new tests (26 run on 3.11, all 33 on 3.15). Thirteen
mutations, one per rule (namespace chains, import-bound chains, the level, both
lazy bases, the aliased proxy, the truncation guard, the per-key shipment, the
shipment used, the seed order, a value starting a chain, a rebound name, inner
attributes, a relative import's names), each caught by node id. Cost, three
runs each on the 500-cell kernel: main 27-37 ms the first time, 2.7-2.9 ms
after, recording 1 file; this branch 29-33 ms, then 4.6-5.0 ms, recording 11.
The 2 ms is the reached definitions' walk and the 10 files; a record is taken
once per run per process, so it is not cached.

Two of my measurements in this round were wrong before they were right: a
cross-version check through bare interpreters compared five identical
tracebacks (mushin was not importable there), and an earlier timing run
differed by 2x from the next. Compare the output itself, and repeat a timing.

## Round 5 (2026-11-03): scopes, methodless classes, installed packages

`/code-review` (10 findings) and the adversarial workflow (5 confirmed, 1
refuted: the LazyLoader import, which main shows too and the docs state)
reviewed round 4 (8a1afb8). Plan, written before any edit.

**Scopes** (both reviews; the worst finding). One flat map of import
bindings per unit let an import in one scope steer chains in another. A
nested function's `from email import utils` turned the task's own
`utils.b.f` into `email.utils.b.f`, so the project submodule was dropped: a
missed edit. `_syntax_modules` now walks scope by scope:
- a function's decorators, defaults and annotations, and a class's
  decorators, bases and keywords, are read in the enclosing scope;
- a function or lambda body is its own scope and sees its enclosing
  functions' bindings; a class body is its own scope, and its bindings are
  not visible to the functions inside it;
- a name can have several bindings in one scope (`if`/`else`, `try`/`except
  ImportError`), and a chain on it resolves through each;
- a name with no binding in its scope or an enclosing function's is the
  namespace's.
The import names come from `_import_pairs` (the file record's reader), so the
two cannot drift; the binding map stays local. A definition node walked once
in a record is not walked again (a class's statement holds its methods' defs,
and each method is a unit too).

**Methodless classes** (both reviews). A class with no method of its own (a
dataclass or `Enum` of fields, `class Model(pkg.sub.Base): K = 3`) has no
statement to read, so its bases named nothing. For each reached class, each
base that is not a notebook class now adds its module, read as
`_notebook_class` reads it (`vars(base).get("__module__")`, no new kind of
read). Notebook bases are units already, so a methodless notebook base's own
bases count. What such a class's body names (`K = pkg.sub.CONST`) is still not
followed: the docs say so ("may", with the new-`working_dir` advice).

**Rebound names** (`/code-review`). The walk skips a name a notebook function
rebinds because the definition it holds can be gone by the next run. A chain
does not need that: a reload helper (`global pkg; pkg = importlib.reload(pkg)`)
keeps a module there, and skipping it drops the submodule. Chains skip only
the kernel's own names. A rebinding to something else changes the record
next run: a recompute, not a missed edit.

**Installed packages** (`/code-review`). The driver now drops a module whose
top-level name resolves to a foreign file before shipping (the rule
`_file_entries` applies to a loaded top-level package, by its file, once per
top-level name), so `np.linalg.norm` is neither shipped nor looked up in each
cell. A top-level name with no file (a namespace package) is kept for
`_file_entries` to decide, which avoids round 1's misjudged namespace packages.

**Unchanged, with reasons**
- Two bases for an ambiguous lazy proxy (`/code-review` asked again): when
  both readings are files (`pkg/utils/utils.py`, `lazy import pkg.utils`),
  the record before the proxy resolves holds one more file than after, so a
  second `cache_dir` run misses. One base would instead miss an edit for
  `lazy from pkg import pkg`. The fragment drops its 3.15 sentence rather
  than claim more.
- A closure factory's import (`def make(): import pkg.sub as s; def
  task(a): return s.f(a)`) is not followed: the reached unit is `task`, and
  the enclosing function's body is not its definition. Main recorded nothing
  there either, and closures are a documented limit.

**Tests** (each mutation-checked by node id):
- an import in a nested function, and one in the body read against a
  default, do not change what the task's own chain names;
- two bindings of one name in one function both count;
- a class body's import is not what its method's name means;
- a methodless class's base, and a methodless notebook base's base, record
  the base's module;
- a reload helper rebinding the package keeps the submodule;
- the driver ships no installed package's names, and keeps a project
  namespace package's;
- `from pkg import a as A`, then `A.b.f`, records `pkg.a.b`;
- the pinned extraction, with `from pkg import x as y` added.

**Docs:** the methodless-class caveat at the three sites; the stale bytecode
comment in the tests; the fragment without the 3.15 sentence, naming bases,
decorators and defaults.

**Results.** 42 new tests (35 run on 3.11, all 42 on 3.15). Twenty-one
mutations, one per rule of rounds 4 and 5, each caught by node id. Two
survived at first, and each needed a different fix. The nested-scope mutation
was ineffective as written (a nested `def` is a statement of the task's own
body), so it was rewritten. The one-binding mutation exposed a test that
passed by walk order, so the swapped case was added. Cost (three runs each,
500-cell kernel): main 23-24 ms first, 2.3-2.4 ms after, 1 file; this branch
26-33 ms, then 4.3-4.4 ms, 11 files.

## Round 6 (2026-11-03): each unit read on its own

The adversarial workflow reviewed round 5 (ba25e95): 4 confirmed. Plan,
written before any edit.
- **Walk order** (important, my regression). Round 5 shared one set of walked
  definitions across units, to save reading a method twice (a
  `/code-review` efficiency finding). When a closure was reached before the
  factory that encloses it, the closure was read alone (its `s.deep.f`
  unresolved), and the factory's read then skipped it, so the factory's
  `import pkg.sub as s` never applied: renaming a variable could drop
  `pkg.sub.deep`. Each unit is read on its own again, and the result is the
  union of the readings, whatever the order. A method is read twice
  (alone, and in its class); the cost stays within the measured range.
- **Docs** (important). A class with no method of its own has no statement
  to read, so its decorators and keywords (`metaclass=`) are not followed
  either, only its bases (by type). The caveat names the statement, not just
  the body, and the fragment says "a function's decorator".
- **Tests** (minor): an enclosing function's import reaching a nested
  function and a lambda (the `visible` rule had no positive case), and the
  pinned extraction gains a class decorator, a keyword-only default, a lambda
  default and a return annotation.

Lesson: the efficiency fix coupled units that were independent, and that
broke "the record does not depend on walk order". Test an optimization
against the invariants it can break, not only the cost it saves.

`/code-review` on round 5 (13 findings; its walk-order finding is the one
above). Added to this round, planned before the edit:
- **Closures** (it probed a decorator's wrapper and a factory's task): a unit
  created inside another function never sees that function's imports, since
  its definition is the inner `def` only. A chain on one of the unit's free
  variables now resolves through its closure cell, by type as
  `_chain_bases` reads any value (a module, or a lazy import). That also
  covers round 5's closure-factory residual.
- **Type parameters** (3.12+): `def f[T: pkg.sub.Base]` and `class C[T: ...]`
  bounds and defaults are read where the definition stands.
- **Standard-library modules with no file** (`sys`, `time`, `math`,
  `_thread`) count as foreign before shipping, by name, as
  `_is_foreign_module` judges them; a top-level name that is a file still
  decides by its file.
- **Docs:** a class with no method of its own is not compared at all (its
  text hashes as unknown, as on main), so the caveat says so; and a submodule
  reached only through something the code does not name (a registry it
  fills when imported, `importlib.import_module`) may not be compared, a
  hedged sentence with the new-`working_dir` advice.
- **Wording:** only imports count as local bindings (a parameter or an
  assignment shadowing a module global is read as the module: a recompute,
  never a missed edit), and the docstring says that. Putting the notebook's
  modules first starts a cut record from them; it does not keep every one,
  since `_file_entries` is depth-first, and the comment says that.
- **Cleanup:** the dead `from m import *` branch (a syntax error inside a
  definition) goes, and the one-use name test is inlined back.
Not changed: caching the reads and shipping only module prefixes (the cost
is measured), and finding a methodless class's statement (that is the cell
scan rounds 1 and 2 rejected).

**Results.** 53 new tests (45 run on 3.11, 52 on 3.15; one needs 3.12+).
Thirty mutations, each caught by node id (the lazy and type-parameter ones on
3.15). One survived for a reason worth keeping: with each unit read alone, a
class body's bindings leaking into its methods can only add files (each
method is read alone too), so its test now checks that no extra file is
recorded. The walk-order test fails on round 5's code (ba25e95) and passes
here.
Cost (three runs each, 500-cell kernel): main 23-26 ms first, 2.3-2.4 ms
after; this branch 25-34 ms, then 4.3-4.4 ms (reading each method twice
added nothing measurable).

## Round 6 review and the merge (2026-11-03)

`/code-review` (12) and the adversarial workflow (6, all minor) reviewed
round 6 (ff31ae9). Nothing records less than main: every miss they found,
main misses too. The maintainer chose to touch up docs and tests and merge,
with the rest as a follow-up issue rather than a seventh round.

**A correction.** Rounds 3 and 4 said no module attribute is read beyond
what main reads. That stopped being true in round 6: a closure-held module's
`__name__` is read, and `_find_file` reads the `__file__` of a top-level module
that only an import inside a definition names. Either can run an
`importlib.util.LazyLoader` module's file. The LazyLoader caveat in the docs
("reading the code a task reaches can import" one) covers both.

**Touch-up:** a general hedged caveat at the three doc sites (a submodule
reached through its package may still not be compared in some cases, with
examples, and the new-`working_dir` advice), and three test pins: `import a.b`
in a body binds `a`, a closure cell wins over a global of the same name, and
the innermost enclosing binding wins.

**Follow-up (not in this PR):**
- a package held in a local name or a bare-name default (`m = pkg`, `lib=pkg`);
- a class made inside a factory, and a factory's or class body's import named
  only in an inner definition's decorator or default;
- a comprehension in a class body read in the class's scope;
- 3.15 recomputes: a proxy read two ways when both readings are files, a proxy
  the task resolves through another attribute, and closure cells the run
  rebinds;
- caching the per-unit reading, and one copy of the foreign-top rule.

## Follow-up #292 (2026-11-03): bindings and where a unit stands

Plan, written before any edit, for the cases #291 left (issue #292).

**The class.** Most of the missed edits share one cause: `_syntax_modules`
treats only imports as bindings, and reads every unit as if it stood at the
top of its cell. So it misses a module bound some other way (`m = pkg`,
`lib=pkg`), and a module bound by the function or class a unit is defined in
(a class made in a factory, a method's default naming a class-body import).
The inverse is the over-recording Codex found: a parameter or local that
shadows a module global is read as the module.

**The design.**
- **Each cell tree is read once into scopes** (cached per tree, the tree
  pinned). A scope is a function, a lambda, a class body or a comprehension.
  For each scope we keep:
  - its own nodes (nested scopes' bodies excluded, their outer parts
    included);
  - its bindings: name -> targets;
  - its `global` and `nonlocal` names.

  For each definition, the tree records where it stands (the scope that
  evaluates its outer parts) and which scope its body is.
- **Bindings, as Python makes them local** (parameters, assignment and `del`
  targets, `for`/`with`/`except`/`match` names, imports, nested `def` and
  `class` names), minus `global`/`nonlocal`. Each target is one of:
  - a module (an import, or an alias whose value resolves to one: `m = pkg`,
    `s = pkg.sub`, `m = n` for an alias `n`, iterated to a fixpoint);
  - a chain on a global (`m = pkg` where `pkg` is the namespace's, resolved
    when the record is taken);
  - unknown.

  A parameter's default is resolved where the definition stands.
- **Resolving a chain:** the innermost scope that binds its name decides. Its
  module and global targets count, and unknown ones name nothing (shadowing).
  - Not bound anywhere, or declared `global`: the namespace's.
  - Bound to an unknown target in a function the unit is defined in (outside
    the unit): the unit's closure cell, never the namespace. Python makes
    that name a free variable.
- **Visibility:** a class body's bindings reach neither its methods nor its
  comprehensions (except the first iterable); a function's, a lambda's and a
  comprehension's reach the scopes inside them.
- **A unit's reading:** its outer parts in the scope it stands in, then every
  scope inside its body. Each unit is still read alone, so the union does not
  depend on walk order. The result is cached per (tree, definition).

**Also in this PR**
- 3.15: a lazy proxy bound as its own first component (`lazy import
  pkg.sub`, bound as `pkg`) records the package only, as the eager import's
  binding does. Before, it recorded `pkg.sub` before the proxy resolved and
  `pkg` after, a `cache_dir` miss when the task does not name `pkg.sub`.
- One foreign-top rule (`_file_entries` and the shipping filter share it).
- The lazy-import type is looked up once.

**Not in this PR, with reasons (they stay in #292 and the docs):**
- A class with no method of its own: nothing pins which cell defined it before
  3.14. A methodless class has no function whose code names its cell, and a
  cell re-run after an edit leaves both texts in the kernel's cache, so
  matching by name would pick the wrong one exactly when it matters.
- 3.15 recomputes when both readings of an ambiguous proxy are files, and
  closure cells a run rebinds (`nonlocal`): rare, and a recompute, not a missed
  edit.
- A walrus inside a comprehension binds in the enclosing function. It is read
  as the comprehension's: a missed alias at most, rare.

**Tests** (each rule mutation-checked by node id; every #291 test still
passes):
- aliases: `m = pkg`, `s = pkg.sub`, an alias of an alias, a parameter default
  (at the top and in a factory);
- shadowing (Codex's case): a parameter, an assignment, `for`/`with`/
  `except`/nested `def` names, all recording nothing extra;
- `global` inside a function whose enclosing function binds the name locally;
- a class made in a factory; a factory's import named only in an inner
  definition's decorator; a class-body import named only in a method's
  default, with only the method reached;
- a class-body comprehension does not see the class's binding;
- 3.15: `lazy import pkg.sub` with a task naming only `pkg.other` records the
  same before and after.

**Stopping rule, set now:** one round of `/code-review` and the adversarial
workflow. Fix a confirmed case that records less than main (77882c4), or a
false claim. A second round only if the first round's fixes changed code;
otherwise merge, with what is left recorded in #292.

**Results.**
- **New tests:** 23 for #292, and 78 module tests in all. On 3.11, 68 run (the
  lazy-import and type-parameter ones skip); on 3.15, all 78 run.
- **Against main (77882c4):** 15 of the new tests fail there. The other four
  pin rules main happens to follow: an alias of a submodule, `global`,
  `nonlocal`, and the cache holding no namespace value.
- **Mutations:** thirty-three, one per rule, each caught by node id (the lazy
  and type-parameter ones on 3.15).
- **Two mistakes during the work:**
  - I named the reading's result `named`, overwriting the walk's own
    variable, the same slip as in round 4.
  - I inserted `_foreign_top` under `_foreign_by_path`'s `@functools.cache`,
    the same slip as in #284. Before adding a definition next to another,
    check the lines above it.
- **Cost,** three runs each on the 500-cell kernel: main 26-32 ms the first
  time, then 4.4-4.5 ms, recording 11 files; this branch 27-28 ms, then
  2.9-3.0 ms, recording the same 11. The per-tree cache pays for the scope
  index.

### #293 review round 1 (2026-11-03): what records less than main, and a hang

`/code-review` (12 findings) and the adversarial workflow (9 confirmed, 1
refuted) reviewed 03f6ee2. Plan, written before any edit. The stopping rule
allows this round because its fixes change code; after it, one more review,
then merge.

**Blocking (a hang, or recording less than main)**
- **The alias fixpoint grows exponentially** (`node = node.left; node =
  node.right` repeated), so a resume hangs and eats memory, even for a
  function no task reaches. Aliases resolve once each, in dependency order.
  An alias in a cycle (`x = x.a`, `a = x.l; x = a.r`) binds something unknown.
  A name's targets are capped (64); past the cap the name is unknown.
- **An alias in an enclosing function loses its own name.** `m = lib` in a
  factory made the inner task's chain start at `lib`, while its closure
  holds `m`. An alias (outside a class body) now also binds its own name's
  cell, so the closure is asked for `m`.
- **A class body's binding is not a shadow.** A class body reads a name with
  `LOAD_NAME`, which falls back to the globals before the binding runs
  (`pkg = pkg.sub.CONST`). A class-body binding's targets now count with the
  namespace's.
- **`lazy from pkg import pkg`** names a subpackage bound as its first
  component, so `_namespace_modules` keeps the full name when the proxy's
  last component is the bound name. It collapses to the package only
  otherwise (`lazy import pkg.sub`).
- **A parameter or local named like a module.** `def helper(pkg, a)` is
  called with the module; main recorded its submodule by the name, and
  shadowing dropped it. A binding the syntax cannot tell (a parameter, a
  plain assignment, a loop or `with` or `except` name) now counts the
  namespace's module of that name too. That brings Codex's over-recording
  back (a recompute, never a missed edit), so it returns to #292. A nested
  `def`/`class` name and a relative import still bind something known not
  to be the namespace's.

**Claims and tests**
- The docs keep "a module passed in as an argument or returned by a call" as
  a case that may not be compared; the fragment drops the shadowing claim,
  and says the 3.15 reuse holds for a task that names `pkg` itself, not one
  that reads `pkg.<attr>` (both readings of the proxy count before it
  resolves, by design).
- Tests: the cycle and the long rebind chain finish quickly; an alias in a
  factory; a class body's read-before-bind; `lazy from pkg import pkg`; a
  parameter named like the module and passed it; a default named like the
  module; a comprehension's first iterable in a class body; the vacuous
  alias-of-a-submodule case replaced. The tests' `_files` fails on the
  record-failure warning unless a test expects it, so an assertion that a
  file is absent cannot pass because the reading crashed.

**Cleanups:** the scope index and its readings share one cache entry per tree,
bounded like the trees (128). `_file_entries` keeps main's loaded-only check
(no `find_spec` per record), and only the shipping filter looks unloaded
names up. The dead `_definition_nodes` and `_class_node` wrappers go.

**Into #292:** a parameter default naming a factory's unknown binding (the
value is in `__defaults__`, never read), a class made in a factory from its
parameters (a class has no closure), and Codex's shadowing case.

**Results (round 2).**
- **Tests:** 81 module tests: 71 run on 3.11, all 82 on 3.15 with the digest
  test.
- **Mutations:** forty, each caught by node id (the lazy and type-parameter
  ones on 3.15). Removing the cycle rule and the cap together makes the suite
  hang, which counts as caught.
- **Five tests were vacuous at first, and each needed fixing:**
  - The hang test put the rebinds in their own cell, which is never indexed,
    so the test checked nothing. The rebinds now sit in the reached task's
    cell.
  - The `global` and class-body-fallback tests were masked by this round's
    own rule that unknown bindings read the namespace too. They now use a
    binding known not to be the namespace's (an alias of another name, a
    `def`).
  - The class-body comprehension test was masked by the same fallback. It now
    uses a global that is not a module.
- **Against round 1's code:** the factory-alias, parameter-named-like-a-module
  and class-read-before-rebind tests fail on 03f6ee2 and pass here.

### #293 review round 2 (2026-11-04): the last fixes before merging

`/code-review` (12) and the adversarial workflow (9 confirmed, 1 refuted)
reviewed 22ae791. By the stopping rule this was the last round: these fixes
go in, then the PR merges. Every one either adds targets (the safe
direction) or goes back to main's rule. Plan, written before any edit:
- **An import under `global` or `nonlocal`** keeps its module: `global pkg;
  import pkg` rebinds the namespace's `pkg`, so the import's targets count
  where the declaration sends the name.
- **The compiler's free variables decide:** a chain whose first name is
  one of the unit's free variables asks its closure, whatever the scope
  model says. This covers a walrus in a comprehension (PEP 572 binds it in
  the enclosing function) and any form the model misses.
- **An import in a function also binds its cell,** so a unit made inside asks
  its closure, as main did.
- **An alias derived from something unknown** (`pkg = a`) also reads the
  namespace's module of its own name, as main did.
- **3.15:** `_namespace_modules` goes back to main's rule (the proxy's full
  name). `lazy import pkg.sub as pkg` binds the submodule, and its repr reads
  like `lazy import pkg.sub`, so collapsing to the package missed an edit.
  The recompute for `lazy import pkg.sub` returns to #292, and the fragment
  drops its 3.15 sentence.
- **`_file_entries`, loaded only:** decided by the module's file alone, as
  on main, with no stdlib-name rule (a project namespace package named like
  a stdlib module stays the project's).
- **Cost:** a unit's outer parts are walked directly instead of filtering
  the standing scope's nodes, and the scopes inside it come from its own
  nodes, so a cell with a large literal costs each unit only its own size.
- **Claims:** the docs keep "a module held in a local name the syntax cannot
  follow"; `_unit_reading`'s docstring says unknown bindings read the
  namespace too; the dead `_scope_index` goes.
- **Tests:** a fresh-kernel `global` import; the walrus closure; an alias of a
  parameter named like the module; `lazy import pkg.sub as pkg`; a doubling
  alias chain that reaches the cap, asserted by target counts, not wall-clock
  time. `_files` resets the once-per-run failure flag, so its guard holds.
- **Memory:** an index entry is evicted with its cell tree.

**Results (the last fixes).**
- **Tests:** 86 module tests: 75 run on 3.11, all 87 on 3.15 with the digest
  test.
- **Mutations:** forty-two. The three that only exist on 3.12+ or 3.15 were
  caught there.
- **One surviving mutation, kept on purpose:** an alias's own cell. Asking the
  closure for every free variable, and reading an alias of something unknown
  as unknown, now cover every case it covered. It stays as a backstop.
- **Two of my test mistakes this round:**
  - The namespace-package test first failed on the unmutated code, so the
    whole mutation table read one extra failure per row until it was fixed.
    The lesson: run the baseline before trusting a table.
  - It went through the shipping filter (main's rule) instead of
    `_file_entries`. It now binds the submodule directly; it passes on main and
    fails on 03f6ee2.
- **Also:** the `global` test needed a binding the syntax tells is
  not a module, because this round's alias rule masked it.

### Codex on a6708f0 (2026-11-04): main's reading becomes the floor

Codex found two more shapes that recorded less than main: a nested
initializer importing into a `nonlocal` name the task binds to a placeholder
`def`, and a comprehension that rebinds the imported name to itself. That made
four rounds in a row of the same class of finding, each patched case by case.
So the class is closed instead. Main's reading (`_syntax_modules` at 77882c4)
is kept unchanged as `_floor_reading`, and every unit's record also takes what
it names, resolved as main resolved it (the closure first, then the
namespace). The scope-aware reading can then only add; "never record less
than main" holds by construction, not by review.

The cost of the floor is precision beyond main: two tests asserted that the
record holds less than main (a nested `def` named like the package, and a
class-body comprehension not seeing the class's names), and both went, since
over-recording is a recompute, not a wrong result. The floor's body is
checked to be identical to main's (an AST comparison), and removing it fails
the two Codex shapes, which pass on main.

With the floor in place, the mutation table changes in a way it predicts.
The rules added to match main survive their mutations, because main's
reading now covers their cases: the declared imports, the free-variable
closure, the alias of something unknown, unknown bindings reading the
namespace, `global`/`nonlocal`, and the class-body fallback. They stay as
written, but the floor is what guarantees those cases. The rules that add
recall beyond main are still caught by their tests: aliases, defaults read
where the definition stands, enclosing scopes, outer parts, bases, the
shipment, and the cap (a hang without it). The rules that only exist on
3.12+ or 3.15 were caught there.

## The rest of #292 (2026-11-04): plan for two PRs

The maintainer asked that every case still open in #292 be resolved before
0.17.1. Plan, written before any edit. Each case is either fixed, or closed
with the reason a fix would trade a missed edit for precision (the bar: a
missed edit is a wrong result; over-recording costs a recompute).

**Most of the list has one cause.** A submodule's code runs only if
something imported it: a cell, a project file (`_file_entries` follows those
already), or a name computed at run time. The reached-definitions reading
looks for the submodule in the task's syntax, so every way of reaching it the
syntax cannot follow (an argument, a call's result, `m = pkg if a else
None`, `for m in mods`) misses it. The import statement is in a kept cell
whichever way the task reaches the module, so reading the cells' imports
closes the class instead of patching shapes. It also brings `import pkg.sub`
to parity with `from pkg import sub` and `import pkg.sub as s`, which bind a
module value and are recorded already (`_namespace_modules`).

### PR 1: the modules the kept cells import

- **The cells' imports** (missed edits: a module passed in, returned by a
  call, held in a local name the syntax cannot follow). Every absolute import
  anywhere in a kept cell (`_cell_lines`' rule: a `linecache` entry with no
  mtime, and IPython 7's `_ipython_cache`) adds what `_import_pairs` names,
  filtered as the other notebook modules are (`_maybe_project`). Round 2's
  objections to a cell scan, each answered:
  - a worker keeps no cells: the driver ships the modules
    (`NOTEBOOK_MODULES`), as it does now;
  - cost: a per-cell index (imports only, no trees), cached by the cell's
    text, bounded; a cell without the word `import` is not parsed;
  - a cell-name whitelist: none, the same rule `_cell_lines` uses;
  - stray `SyntaxWarning`s: the parse runs with warnings ignored (and
    `_cell_tree`'s too).
  The cost is precision, by design: a cell's `import pkg.plotting` now counts
  for every task, as `from pkg import plotting` already did. An old version
  of a re-run cell counts too (a recompute, not a wrong result). The cells'
  modules come after the reached code's in the shipment, so a record cut by
  `_FILES_MAX` still starts from what the task names. They are kept when the
  walk passes its bound or fails, since they do not depend on it.
- **3.15, `lazy import pkg.sub`** (recompute). Before the proxy resolves
  `_namespace_modules` records `pkg.sub`, after only `pkg`. The cell's
  statement now records `pkg.sub` either way. When both readings of an
  ambiguous proxy are files (`pkg/utils/utils.py`), a chain on the name after
  it resolves also takes the readings the proxy had: on 3.15, a kept cell's
  import binding `n` with a dotted name starting at `n` (`import pkg.utils`,
  `import pkg.utils as pkg`, `from pkg import pkg`) adds `n` and that name
  as bases for chains on `n`. Only adds, so the floor is untouched.
- **Closure cells a run rebinds** (recompute). With the cells' imports, the
  module before and after a `nonlocal` rebinding are both recorded when a cell
  imports them. Also, a `nonlocal` or `global` alias (`nonlocal s; s =
  pkg.other`) adds its targets where the declaration sends the name, as a
  declared import does now. Each is resolved without the scope's other
  declared aliases (otherwise `s = s.a` repeated doubles the targets each
  time, round 1's hang) and capped at `_TARGETS_MAX`.
- **A parameter's default value** (missed edit: `def make(mod): def task(a,
  m=mod): return m.sub.f(a)`). The default is in `__defaults__`, never read.
  A chain on one of a function unit's own parameters also reads its default
  value, by type, as a closure cell is read (`_chain_bases`).
- **Codex's over-recording** (recompute). Two parts:
  - the scope reading reads a chain on a free variable through the closure
    and the namespace both. A free variable never reads the namespace, so the
    namespace read is dropped when the unit's own code never loads that name
    as a global (`_code_names`). The floor is untouched (it already reads the
    closure only).
  - a parameter or local named like a namespace module reads that module.
    This is the floor (main's reading), and it is what records a module
    passed in under its own name (`def helper(pkg, a)` called with `pkg`).
    Dropping it can miss an edit, so it stays, as a documented recompute.
- **Cleanups listed in the issue:** the per-unit reading is cached and the
  lazy type is looked up once (both done in #293). The foreign rules stay
  two: `_is_foreign_module` judges one module, `_foreign_top` a whole
  top-level name, and they differ on purpose for a namespace package.

**Tests** (each rule mutation-checked by node id, the baseline run first):
- the four syntax-blind shapes (an argument through a `partial`, a call's
  result, a conditional alias, a loop over a list of modules), each with
  `import pkg.sub` in a cell, record `pkg.sub`; each fails on main;
- an import inside an unreached cell function counts; a module loaded only
  by `importlib.import_module` still does not; a foreign import adds nothing;
- `test_only_the_code_the_task_reaches_names_modules` asserted that a cell's
  `import pkg.other` the task never names is not recorded. That precision is
  what this PR gives up, so the case moves to a module only `importlib` loads;
- a cell with an invalid escape sequence prints no warning;
- the cells' modules survive a walk past its bound;
- 3.15: `lazy import pkg.sub` with a task naming `pkg.other.g`, and the
  `pkg/utils/utils.py` case, record alike before and after the proxy
  resolves;
- a `nonlocal` rebinding records alike before and after the run;
- a default holding a module records its submodule;
- a free variable named like a namespace module adds no file through the
  namespace.

**Where it is taken.** In `_notebook_record`, once per record and apart
from each namespace's walk: `_notebook_entries` returns nothing when its
walk passes its bound or fails, and the cells' imports do not depend on the
walk. The scan enumerates `linecache.cache` (and `_ipython_cache`) with
`_cell_lines`' rule, since `_cell_lines` itself takes one filename. A scan
that raises gives no cell modules and the record-failure warning. The cells
are the process's, not one namespace's: in a kernel that is the same thing.

**Existing tests that encode the precision given up**, each decided:
- `test_only_the_code_the_task_reaches_names_modules`: precision dropped by
  design (a cell's `import pkg.other` now counts). Its "a submodule the
  reached code never names" half moves to one only `importlib` loads.
- `test_a_class_bodys_import_does_not_steer_its_methods_names`: its subject
  (a class body's import does not steer a method's chain) still holds, but its
  setup's `import pkg.a.b` now records the file itself, masking it. The setup
  drops that import.
- `test_a_cut_or_failed_walk_names_no_modules`: its subject (the walk's
  modules go with a cut walk) holds; its setup's `import pkg.sub` now records
  the file through the cells. The setup binds the package without importing
  the submodule by a statement, and a new case checks that a cell's import
  survives the cut.
- `test_the_cached_reading_holds_no_namespace_value`: two simulated kernels
  share one process's `linecache`, so the first's `import pkg.sub as s` is a
  kept cell for the second too. Its subject (the cached reading holds no
  namespace value) holds; the first kernel closes before the second records.
- `test_the_driver_ships_only_project_modules`: the shipment now also lists
  the cells' project module (`SUBNS.mod`); what it checks (no installed or
  built-in name ships) is unchanged.
Not affected, checked: a value bound under the package's name
(`import types` only), a relative import (level 1 is skipped), foreign
imports, and the digest-pinned and restart tests (notebook entries only).

**#292, every bullet's disposition** (for the close-out comment):
- missed edits: an argument, a call's result, a local the syntax cannot
  follow, a factory's default: fixed (the cells' imports; the defaults);
- a class with no method of its own: PR 2;
- 3.15 recomputes (`lazy import pkg.sub`, both readings files), closure
  cells a run rebinds: fixed (the cells' imports; the 3.15 bases; declared
  aliases). A closure rebound to a value the syntax cannot tell, holding a
  module only a computed name imports, can still change the `cache_dir` key;
- Codex's over-recording: kept whole, as a documented recompute. The
  parameter named like a namespace module is the floor's reading; the
  free-variable double read was dropped in round 1 and restored in round 3,
  since main's namespace read is what records a closure a sibling fills
  during the run;
- cleanups: the per-unit cache and the one lazy-type lookup were done in
  #293; the two foreign rules stay two, on purpose.

**Docs:** the caveat at the three sites shrinks to a module imported only by
a name computed at run time (`importlib.import_module(name)`, a plugin
loader), as in a script, and the methodless class (until PR 2). The record's
rule gains a sentence: every project module a cell imports counts.

**Results (PR 1).**
- **Tests:** 23 new test items (245 in the module). On 3.11, 225 run; on
  3.15, all 245 pass. Against main (3a8f6d7), 17 of the new items fail (14 on
  3.11, the three lazy ones on 3.15); the load-state test passes there, as
  it should.
- **Fixtures:** `sub_project` and `lazy_project` turn the scan off, since
  their cells import the very submodules most of their tests assert (round
  2's `import pkg.sub` setups): with the scan on, those tests would pass
  without the rules they pin. `cell_imports` turns it back on for the scan's
  own tests. The two precision tests planned for edits above needed none:
  with the scan off they test the reached code's reading, and their
  docstrings say so.
- **Mutations:** twenty for this PR, run by node id after a baseline. All
  are caught but two: widening the free-variable condition to any name the
  unit does not read as a global (the floor and the defaults read cover what
  it drops, so the record is unchanged), and taking any binding as a lazy
  one's (a base no file has). The three 3.15 ones were caught on 3.15;
  compounding declared aliases without the cap hangs (caught). #293's
  forty-five, run again, match its merge table except one: the new defaults
  read masked "defaults not read where the definition stands", so a nested
  function's default (not a unit, so its value is never read) pins it again.
- **Cost** (three runs each, the 500-cell kernel, 3.11 on an Apple M1):
  main 46-54 ms the first time, then 7.7-7.9 ms; this branch 54-72 ms, then
  7.2-8.8 ms, recording the same 11 files. The first record parses each cell
  with an import once.
- **My mistakes this round, each caught before review:**
  - the mutation runner split node ids on spaces (the new ids have
    parameter names with spaces), so its first table ran no test at all and
    every row read "0.00s". Read the counts before reading a table;
  - the 3.15 bases first applied to every import on 3.15, so an eager
    `import pkg.sub` gave chains bases it never has (a shipping test showed
    it on 3.15 only). They now come only from a statement that can bind a
    proxy (`lazy`, `-X lazy_imports=all`, `__lazy_modules__`);
  - the scan masked most of #291's and #293's positive tests, found by
    reading the floor test before the mutation run; hence the fixtures;
  - the new defaults read masked an older rule's test (above). A new rule
    can mask an old one's tests, not only its own: re-run the old table.

### #294 review round 1 (2026-11-04): plan

`/code-review` (10) and the adversarial workflow (7 confirmed, 0 refuted)
reviewed 55139dc; Codex had none. Plan, written before any edit. Stopping
rule: these fixes change code, so one more round of both reviews, then merge
unless that round finds something blocking.

**Blocking (both reviews, reproduced against main):** the cells' modules
went into the shipment beside the reached ones, and `_record` puts the
shipment first. A cell importing one wide project module could then use up
`_FILES_MAX` before the namespace's modules, which main records next, so an
edit to one of those went unnoticed. My design note said "after the reached
ones" and stopped there; main's next seeds were the namespace's, then the
targets' and the scripts'. Fix: the cells' modules ship under their own key
(`NOTEBOOK_CELL_MODULES`) and `_record` adds them after every other seed. A
shipment without the key (an older driver) counts as none. Test: a wide
cell-imported module and a namespace instance's module, with the bound
lowered; the instance's module stays.

**Tests that pinned less than they claimed (adversarial):**
- the free-variable skip's exception (a name the unit also reads as a global,
  `global m` in a nested function) had no case: a mutant skipping every free
  variable passed and recorded less than main. Add it;
- the declared-name bound test held under compounding too: pin the sizes
  the alternating input gives, and add a case with more distinct chains than
  the cap;
- the 3.15 widening's own name (`lazy import pkg.pkg as pkg`) had no case.

**Docs:**
- "a name computed as the code runs" is wrong for `importlib.import_module`
  with a literal, `__import__` and a magic: say "loaded other than by an
  `import` statement", with the registry example back;
- `run()`'s docstring: the edited sentence is garbled; and "no value is
  compared (... what a default or a closure holds)" now reads against the
  defaults read, so that clause goes there and in the guide (a function a
  default or closure holds is listed as not compared already).

**Minor code:**
- a failing scan says the notebook code is not compared, which is false (the
  entries stay), and shares the walk's once-per-run flag: it gets its own
  message and flag;
- `_parse_cell` does not catch `MemoryError` (3.11's parser depth guard), so
  one such cell turned the scan off for the session. Catch it; a cell that
  does not parse never ran, so it imported nothing, and that is cached;
- the per-cell cache evicted first-in-first-out at 4096, so a session past
  that re-parsed every cell on every record. The cache keeps exactly the
  cells the last scan saw instead (bounded by `linecache` itself);
- the 3.15 bindings came from imports at any depth; only a cell's
  module-level statements bind a namespace name, so only those count;
- one `foreign_tops` per record: `_notebook_entries` returns its modules
  unfiltered and `_notebook_record` filters both lists once.

**Not changed, with reasons:**
- *The `cache_dir` key moves with what the cells import* (an `import
  pkg.plotting` after a sweep, or a restart that drops an old cell's import).
  That is the precision this PR gives up, already true of `from pkg import
  plotting` on main; the guide says so now.
- *`linecache` entries a loader served* (`get_source`, a zip import) also
  have no mtime and are scanned: a recompute at most, rare.
- *A declared alias past the cap is dropped:* the assigned chain itself is
  still read (scope reading and floor), and the closure as before, so nothing
  falls below main.
- *`catch_warnings` is process-wide:* it runs only on a cache miss, and a
  sweep's run changes the filters anyway.

**Results (round 1 fixes).**
- **Tests:** 252 in the module; on 3.11, 230 run; on 3.15, all 252 pass.
  `make check`: 1605 passed.
- **The cut:** with the bound lowered to 3, a cell importing a wide project
  module no longer pushes the namespace's `SUBREL` out; with the cells put
  back in front, the same test fails.
- **Mutations:** twenty-eight now, by node id after a baseline. All are
  caught but one: the free-variable condition widened to any name the unit
  does not read as a global. The floor and the defaults read cover what that
  drops, so the record is unchanged. The five 3.15 ones were caught on 3.15.
  The foreign filter, moved into `_notebook_record`, is still caught by the
  shipping test.
- **One test of mine was order-dependent before it ran:** the scan-failure
  test patched `_cell_imports`, but with the new cache an earlier test's
  identical cell text would have been a hit, and the patch never called. It
  now starts from an empty cache. A cached path can hide a patched function;
  reset the cache in such a test.

### #294 review round 2 (2026-11-04): plan

`/code-review` (10) and the adversarial workflow (5 confirmed, 0 refuted)
reviewed 0c77dad. Plan, written before any edit.

**Blocking, and my mistake again:** round 1 fixed one member of a class, not
the class. The invariant is that the file walk keeps every file main's walk
kept, so everything main did not read must be followed after all of main's
seeds. The cells' modules moved, but three other readings this PR added
still went into `reached`, which `_record` puts first: a parameter's
default, the 3.15 extra bases, and the targets of a `nonlocal` or `global`
alias. A default holding a wide package pushed the namespace's module out
of a bound of 3 (reproduced against main). The fix is by the invariant, not
by site:
- **Two passes over the files.** `_file_entries` takes a shared `seen` and
  a file budget. Pass one is main's seed list in main's order (the reached
  modules main's readings give, then the namespace's, the targets', the
  scripts'). Pass two is everything this PR adds, with what is left. Pass
  one is therefore main's walk, whatever pass two holds. A cut in pass two
  alone gets its own warning (the cells' or the added readings' files, not
  "the task's code").
- **The added readings are kept apart from the start.** `_notebook_entries`
  returns `(entries, modules, added)`: `added` holds what only the defaults
  read and the 3.15 extra bases give. `_notebook_record` ships `added` and
  the cells' modules together as the later modules
  (`NOTEBOOK_LATER_MODULES`, renamed from the cell-only key, which never
  shipped).
- **Rebinds move out of the scope reading** (this also fixes `/code-review`'s
  sibling case: `rebind()` doing `nonlocal s; s = pkg.pkg` while `task`
  reads `s.deep`, so the record still followed the cell's value, and an edit
  to `pkg/pkg/deep.py` went unnoticed after the run). A `nonlocal` or
  `global` alias no longer adds to the declaring scope's `declared`.
  `_index_entry` registers its targets on the binding the declaration sends
  it to: the nearest enclosing function that binds the name, or the
  module's for `global`, keyed `(scope id, name)` and capped. `_unit_reading`
  then adds them for every chain that resolves through that binding, the
  task's own, a sibling's, or any function in the tree. They are returned
  apart, as added, so the scope reading main's floor sits beside is
  unchanged. A `global` rebinding in another cell stays a documented limit
  ("a name one of its functions rebinds").

**Smaller (the reviews' minor findings):**
- the 3.15 extra bases apply only while the namespace's value is that
  name's module or its submodule (an eager `import other as pkg` later does
  not pick up an old `lazy import pkg.sub`'s readings);
- the fragment says, as #291's did, that a resume keeps a reused cell's
  record, so a sweep started earlier does not compare these modules for its
  completed cells;
- tests: an old shipment without the later key (a 0.17.0 runner and a
  #293 one), the scan warning's per-run reset, a cache hit (no parse when
  the cells did not change), `_files` failing on the scan's warning too,
  the nested import still counting as a cell module in the 3.15 binding test,
  the sibling rebind, and a cut test for each added reading;
- `_cell_tree`'s dead `SyntaxError` branch and the unwrapped docstring.

**Not changed, with reasons:**
- *Code a library writes into `linecache`* (sympy's `lambdify`, attrs,
  dynamo repros) is scanned like a cell: its imports are rarely of the
  project, and the cost is a recompute;
- *a default's chain read from the outer parts* (`def task(a, m=m.sub)`)
  can join the parameter's default: a recompute at most, when such a file
  exists;
- *one rule for what an import binds, three readers*: kept, since each
  reads a different structure; the cell scan reads names only;
- *`MemoryError` during a parse* is the 3.11 parser's depth guard in
  practice; a real out-of-memory there loses that cell's imports for the
  run;
- *the fixtures turn the scan off for the reading's tests*: the scan only
  adds modules, and its own tests run with it on.

**Stopping rule:** one more round of both reviews on the delta. If it finds
nothing blocking, merge.

**Results (round 2 fixes).**
- **Tests:** 260 in the module; on 3.11, 237 run; on 3.15, all 260 pass.
  `make check`: 1611 passed. Against 0c77dad, the sibling and same-cell
  rebinding tests and the cut tests for a default, a rebinding and (on 3.15)
  a lazy proxy's other reading all fail.
- **Mutations:** thirty-four for this PR. All are caught: rebinds feeding the
  resolution again hangs; a rebinding registered on the declaring scope, a
  declared import left unregistered, the later modules put back into pass
  one, the added readings put back with the reached ones, a cache never read,
  and the scan warning's reset each fail a test. #293's table matches its
  merge table, its floor still caught (now by one test: the registry covers
  the other floor case too).
- **The reviewers' own probes**, run against this code: the wide cell import
  (`zz_big`) keeps `zz_exp`, and editing `zz_exp.py` changes the record; the
  depth-guard cell no longer stops the scan.
- **Cost** (three runs each, interleaved, the 500-cell kernel): main 22-62 ms
  the first time, then 2.9-3.0 ms; this branch 25-27 ms, then 3.5-3.6 ms,
  recording the same 11 files. The 0.6 ms is the scan hashing each kept
  cell's text.
- **A global rebinding still moves the namespace's own value** (the
  same-cell test starts `s` as a value): `_namespace_modules` records a
  module bound in the namespace by value, as on main, so one a notebook
  function rebinds drops out after the run. That is the documented "a name
  one of its functions rebinds".
- **My mistake this round** was the class one: round 1's plan named the
  cells' modules only, when the invariant covers every reading main lacked.
  Write the invariant down first and list its members from the diff (here:
  every line that adds to `modules` and is not in main's reading).

### #294 review round 3 (2026-11-04): plan

`/code-review` (9, none blocking) and the adversarial workflow (7 confirmed:
1 important, 6 minor) reviewed 0c5646f. Plan, written before any edit.

**Blocking: the free-variable skip records less than main.** Round 1 dropped
the scope reading's namespace read for a unit's free variable, on the
argument that a free variable never reads the namespace. It does not, but
the closure can be empty when the record is taken and filled during the run
by a sibling with a value the syntax cannot follow (`nonlocal zzp; zzp =
importlib.import_module("zzp")`), and then main's namespace read of `zzp`
was what recorded `zzp.sub`. Reproduced against main. Round 2 weighed this
very trade (the advisor asked for the reason in writing) and judged the
shape contrived; a lazy initializer is not. The skip goes, and Codex's
over-recording stays whole, as a documented recompute. Lesson: under "never
less than main", a removal from main's reading needs a proof, not a judgment
that the counterexample is rare. Tests: the reviewers' shape as a
regression; the free-variable test now expects main's reading.

**Minor (tests the reviews showed unpinned):** an import rebinding as a cut
case; the registry's owner search past a direct parent (a `mid` function
that does not bind the name, and an outer binding the nearest one shadows);
the total held to the bound in the cut tests; the later warning's per-run
reset; the cells' cut test patching the right flag.

**Wording:**
- the guide's "every project module a cell imports" leaves out a cell the
  kernel keeps no text for, as the definitions' sentence does. A `%%time`
  cell's import is run through a string, so it joins the "loaded other than
  by an `import` statement" example;
- the later warning names the readings too, not only the cells' imports.

**Cleanup:** `_kept_cell_reading` and `_cell_lines` share one predicate for
a kept cell.

**Not changed, with reasons** (`/code-review`): the registry's cap and its
dropped cell targets (neither is below main, which had no registry); a
`__lazy_modules__` that names other modules still makes every binding a
candidate (a recompute at most); the two passes, kept so each warning names
its cause; hashing each cell's text per record (0.6 ms per 500 cells; a name
can be reused for other text).

**Stopping rule:** both reviews on this delta. Merge if they find nothing
blocking.

**Results (round 3 fixes).** 264 tests in the module (241 run on 3.11, all
264 pass on 3.15); `make check` 1615 passed. The reviewers' shape (a closure
a sibling fills during the run) passes on main and fails on 0c5646f, and
the free-variable test, now expecting main's reading, does the same. Seven
mutations for this round (the owner search taking the nearest function or
the outermost, pass two given a full bound, the later warning's reset gone,
an import rebinding put with the reached modules, the free-variable skip
back, the registry ignored), each caught by node id. The test helper reads
`_CELL_SCAN_WARNED` with `getattr`, so the file runs against main too, which
is how the regression was checked there.

### #294 review round 4 (2026-11-04): nothing blocking

`/code-review` (10) and the adversarial workflow (1 confirmed, minor)
reviewed a59a603. Nothing records less than main, so by the stopping rule
the PR merges after the touch-ups:
- a cut variant of the free-variable tests: the namespace read is main's
  reading, so it must stay in the first pass; a mutant moving it to the
  later pass recorded less than main and passed every test until now;
- the guide and the fragment say the cells' imports count for a task
  defined in a notebook (a run scans the cells only when some root is
  defined there: the task, `pre_task` or a `Study` training function; the
  docs name the common case);
- `_cell_tree` keys its cache as the scan does (`surrogatepass`), so a lone
  surrogate in a cell cannot fail the record;
- the later-cut warning's flag comment says what it is in a worker.
Accepted with reasons, as in earlier rounds: the `cache_dir` key moving with
what cells import, library-generated `linecache` entries, the registry's
cap and its dropped cell targets, `catch_warnings`, a `lazy from pkg import
pkg` that resolves to a non-module (a recompute once), and keeping the
default, rebinding and lazy readings beside the scan (they cover modules
loaded without an import statement).

### PR 2: classes with no method of their own

Plan, written before any edit (#294 merged as 3ca30cc).

**The gap.** A class's statement is found through a method's code: a
function's `co_filename` names its cell. A class with no method of its own
(a dataclass or an `Enum` of fields, `class Config: lr = 0.1`) has none, so
its own digest is a constant and an edit to it is missed. Main misses it
too. So is a class whose every method is wrapped by a notebook decorator
(#278's residual): the wrapper's code is the decorator's cell.

**Where the statement is found, the first that answers:**
1. **A method,** as now.
2. **3.14+: `__annotate_func__`,** the function the compiler makes for a
   class body's annotations: its code names the cell. Only a class with
   annotations has one: a dataclass does, an `Enum` or a plain class of
   values does not (probed on 3.14 and 3.15). Its first line is the first
   decorator's, so the match takes the decorators' lines as the statement's
   start.
3. **A pin taken when the cell ran.** Answering the four objections to it:
   - *Footprint:* nothing imports IPython. The hook is registered only when
     `sys.modules` already holds IPython and `IPython.get_ipython()` returns
     a shell: when `mushin._resume` is imported, and again when a record is
     taken (a shell made later). Objects are read by type, as elsewhere:
     `type(obj)`, `vars(obj).get("__module__")`, the qualname through the type
     slot. Pins are keyed by `id()` with a weak reference checked on lookup,
     so no metaclass `__eq__` or `__hash__` runs.
   - *Import-time side effect:* two IPython event callbacks (`pre_run_cell`,
     `post_run_cell`), each wrapped so it never raises (IPython prints a
     callback's traceback after every cell otherwise). Their cost per cell is
     a search of the raw text for `class` and a dict lookup per name it
     finds; a parse only when a class object changed.
   - *An untransformed cell:* the pin's text is `shell.transform_cell` of the
     raw cell, the same text IPython compiles and keeps in `linecache`. The
     pre-cell search runs on the raw text and may over-collect names, which
     only costs a lookup.
   - *Classes defined before `import mushin`:* not pinned; they go to step 4.
     A notebook usually imports first and defines after, and even when it
     does not, step 4 is exact unless that class's cell was edited before the
     import.
   Before a cell, the objects bound to every name a `class` statement in it
   could bind (`class\s+NAME`); after it, each that changed and is a
   notebook class whose qualname is that name is pinned to the cell's text.
   A cell that raised before the statement leaves the object unchanged and
   pins nothing; one that rebinds the name to another class (`C = make()`)
   has another qualname and pins nothing.
4. **The kept cells' `class` statements,** by syntactic qualname
   (`f.<locals>.C`, `A.C`, an explicit `global`) and, on 3.13+, by first line
   (`__firstlineno__`). One distinct text: exact. Several: which is in force
   cannot be told (ipykernel names cells by content, so a reverted cell keeps
   its old entry), so the digest is a per-process token with a per-object
   counter: it differs from every earlier record and stays while the object
   does. That is a recompute, never a wrong reuse.

**Two refinements to step 3** (review of this plan, before any code):
- *The record taken during the cell that defines the class* (a small
  notebook's `class Config` and `run(...)` in one cell): `post_run_cell` has
  not fired yet, so the new object is unpinned, and with two texts in the
  session it would get a fresh token on every run. So the hook state keeps
  the running cell's raw text and its snapshot, and a lookup during a cell
  pins on demand: the name's object differs from the snapshot, so this cell
  made it. `post_run_cell` does the same at the cell's end.
- *Registered once per shell, `%autoreload` included:* reloading
  `mushin._resume` re-runs its module body, and new callback objects would
  not be recognized as the old ones. The callbacks and the pins live on the
  shell (one attribute holding mushin's state), so a reloaded module finds
  them and registers nothing twice.

**An upgrade cost to state in the fragment and the PR:** a methodless
class's own digest was a constant and becomes its text, under the same
`"<name> (notebook)"` key, so a resume compares them and re-runs the cells
once after upgrading, for a notebook task that reaches such a class. (#291
and #293 only added file keys, which a resume does not compare.)

**The token's cost against the bar.** After a restart, Run All gives one text,
so the exact digest differs from the token one and the cells re-run once,
though nothing changed: a spurious re-run after "a restart replaying the
cells". It can only follow a session where a class's cell was edited and
re-run while no pin was taken: before `import mushin`, for a class a factory
makes (step 3 pins top-level names only), or in a frontend without IPython's
events. Decision: accept it as a documented residual. The alternative, picking
one of several texts, can reuse results the other text computed, a wrong
result.

**Reading.** The statement found by any step is a unit's site like a class
found through its methods: its decorators, keywords and body are read for
modules (`_unit_reading` and the floor). With several candidates, all are
read.

**Tests:**
- simulated kernel (steps 2 and 4): a dataclass field default, an `Enum`
  member, a plain class attribute, a class made in a factory, and a class
  whose methods are all wrapped by a notebook decorator, each edited and re-run;
- the ambiguous case recomputes, then reuses while the object stays;
- a methodless class's body naming `pkg.sub` records it;
- a real `InteractiveShell` in a subprocess (step 3): edit, revert, and an
  unchanged re-run keep the record exact; a cell that raised before its
  `class` statement pins nothing; a magic cell (`%%capture`) still pins;
  the callbacks print nothing;
- 3.13+: two texts at different first lines are told apart, a decorated
  class's first line taken as its first decorator's (as step 2 does);
- 3.14+: a dataclass, `slots=True` too (probed: the new class keeps both
  `__annotate_func__` and `__firstlineno__`), is found with no pin;
- the record taken inside the defining cell, and `%autoreload` of
  `mushin._resume` (one pair of callbacks after it), in the subprocess
  shell; a `raw_cell` of None is skipped;
- the all-wrapped class's decorator without `functools.wraps` (with it,
  `__wrapped__` already reaches the methods);
- the cross-version digest (its cells hold a methodless dataclass) changes
  once and reads alike on 3.11 to 3.15: steps 2 and 4 find the same node.
Each rule mutation-checked by node id, baseline first; the cost per cell of
the callbacks measured.

### Mistakes to avoid this time (from the earlier rounds)

- variable names: `named`, `modules` and `seeds` are taken in
  `_notebook_entries` and `_scope_bindings`; check before naming a new one;
- a new definition goes below the previous one's decorators, not between;
- anything that grows targets is linear by construction, and has a test that
  would hang without it;
- an optimization is tested against the invariants it can break (walk
  order, never less than main);
- mutation tests: run the baseline first, check each mutation string matches
  once, run new tests by node id with their counts, and make sure a test
  fails for the reason it names (not a second path, not walk order);
- measurements compare the output itself and are repeated.

**Results (PR 2).**
- **Tests:** 279 in the module, all passing on 3.11 to 3.15 (the version
  skips differ). The cross-version digest changed once (its cells hold a
  methodless dataclass) and reads the same on all five, since step 2 (3.14+)
  and step 4 (3.11 to 3.13) find the same node.
- **Against main (3ca30cc):** the five new edit cases, the methodless body,
  the ambiguous case and the shell test fail there.
- **Mutations:** thirteen, by node id after a baseline, all caught. The ones
  that exist only on 3.14+ (the annotate function, its first line) were
  caught on 3.15. The first-line filter was at first exercised only by a
  dataclass, which 3.14+ finds through its annotate function first, so its
  mutation survived there; the test now uses an `Enum`, plain and decorated.
- **Cost of the watch per cell** (an `InteractiveShell`, 300 cells each,
  two runs): a cell without `class` 52 us either way; one with a `class`
  statement 93-95 us against 65 us without the watch (the transform and the
  pin).

### #295 review round 1 (2026-11-04): plan

`/code-review` (10) and the adversarial workflow (6 confirmed, 0 refuted)
reviewed df8fc95. Plan, written before any edit.

**My mistake, a third time:** the readings of a class found by steps 2 to 4
went into `modules`, main's first pass, though main read nothing there. #294
made "everything main did not read goes to the later pass" the invariant,
and I did not list this PR's new reading as one of its members. They go to
`added`. The lesson stands from #294: list the invariant's members from the
diff before review, every line that adds to `modules`.

**Blocking:**
- *A `%%capture` cell's pin is overwritten* (both reviews, reproduced): the
  magic runs its body as a nested cell, whose `post_run_cell` pins the class
  to the body; the outer cell's then repins it to the magic's call, which
  holds no `class` statement, and the class falls to a new token each re-run.
  The first pin wins, and a cell whose text holds no statement of the class
  pins it as unreadable instead: a `%%time` cell or `exec` made it, so the
  kept cells' older texts must not be taken for it (`/code-review`: a stale
  match, a missed edit). An unreadable pin is ambiguous.
- *Step 4 judged ambiguity per node, not per text* (adversarial): a factory
  with the class in both branches of an `if` got a token in every session.
  Candidates are grouped by text: each kept text's statements of the
  qualname together (after the first-line filter). One distinct group is
  exact, its nodes read together; several are ambiguous. A text editing
  either branch is a new text, so no edit is missed. Every tree of the
  winning group is read, so which of two identical texts the scan met first
  cannot change the modules (`/code-review`: a set's order chose the tree).
- *A pinned cell with two top-level statements of the class*
  (`/code-review`: `class C` twice in one cell, a record after each): on
  3.13+ the first line tells them apart; on 3.11/3.12 two top-level
  statements in the pinned text are ambiguous. (A factory's branches are
  inside a function: one group, as above.)
- *A failing scan dropped every entry* (both reviews): the class index is
  built once per record, in `_notebook_record`'s guarded scan; when it
  cannot be, an unpinned methodless class is ambiguous, and the scan's own
  warning says so.

**Minor:**
- the callback read a class's `__dict__` through `vars()`, which runs a
  notebook metaclass's `__getattribute__`; it reads through the type slot,
  and `__module__` is compared only when it is a `str`;
- one class index per record (each methodless class re-scanned the cells);
- the parse trigger is the `class NAME` regex, not the substring (a cell
  with `num_classes` parsed for nothing);
- 3.13+: a class without `__firstlineno__` was not made by a `class`
  statement (`type()`, `make_dataclass`), so no kept text is taken for it;
- the docs state the token's recompute with may/can wording at the three
  sites;
- tests: `%%capture` exact and stable across re-runs, `%%time` never matched
  to an older text, the cut test for a class body's reading, a factory's two
  branches stable across processes, `class C` twice in one cell, a metaclass
  hook untouched by the callbacks, and two processes' tokens differing.

**Not changed, with reasons:**
- *registration at import* (`/code-review` asked for the first run): a
  notebook that defines a class in two cells, both run on every Run All, is
  exact only if the watch saw them; registered at the first run, every restart
  would re-run its cells. IPython already transforms each cell twice (once
  to decide whether it runs async), so a third time adds no new kind of side
  effect.
- *the fix depth* (`/code-review`: treat every unpinned class as ambiguous):
  a factory's class is never pinned, so each re-run of `C = make()` would
  recompute; the grouping above keeps it exact.

**Stopping rule:** one more round of both reviews after these fixes; merge if
it finds nothing blocking.

**Results (#295 round 1 fixes).**
- **Tests:** 284 in the module; all pass on 3.11, 3.12, 3.14 and 3.15. On
  3.13, one run in a freshly built environment failed
  `test_the_driver_ships_only_project_modules` once; it passed alone and on
  a full rerun, and I could not reproduce it. Recorded here as unexplained.
- **Against df8fc95:** the shell test (the magic's pin, `%%time`, two
  statements in one cell, the metaclass hook), the factory's two branches,
  the identical texts, the cut and the failing scan each fail there.
- **Mutations:** ten for this round, by node id after a baseline, all
  caught. Two needed a sharper test first. The unreadable pin falling back
  to the kept cells survived, because the shell test's `%%time` class had
  several kept texts and read as ambiguous anyway; a class with one kept
  text, then remade in a `%%time` cell, now pins it. And the no-statement
  check (3.13+) runs only on 3.13+, so it was checked on 3.15.
- **A test of mine was wrong at first:** the metaclass-hook assertion read
  the whole output, which includes a record taken inside a cell later (a
  record can run a metaclass hook: a documented residual). It now reads only
  what the callbacks printed.

### #295 review round 2 (2026-11-04): plan

`/code-review` (10) and the adversarial workflow (7 confirmed) reviewed
93764ca. Plan, written before any edit.

**Blocking:**
- *A class an unreadable cell made got a new token on every re-run*
  (adversarial, reproduced through a resume): round 1 made an unreadable pin
  (`%%time`, `exec`) ambiguous, so an unchanged re-run or a restart re-ran
  the cells, though the docs say such code is not compared. It now has no
  site and is not ambiguous: main's constant digest. The pin still keeps an
  older kept text from standing in for it.
- *Two statements of the class in one text* (`/code-review`): on 3.11/3.12,
  with no first line, the kept-cells path grouped them, so a cell defining
  the class twice gave both objects one digest (a record between them would
  reuse the first one's results). Ambiguous is right for two statements one
  after the other in the same block, where a record can stand between them;
  for statements in different branches (`if`/`else`, `try`/`except`), only
  one runs, which one is a value, and grouping them keeps the digest stable
  (the adversarial review found the token there re-running every unchanged
  cell). `_class_nodes` says which block each statement is in, and both the
  pinned and the kept-cells path use it, replacing the `<locals>` check that
  never fired on the pinned path.

**Minor:**
- the token carries the process id, so forked children never share one;
- one scan per record for the imports and the class index;
- each node's dump computed once in `_class_sites`;
- the `class NAME` pattern allows a backslash continuation (`class \`);
- a stale comment in `visit()`, and the `_cell_imports` docstring;
- docs: the token's re-run named for `resume=True` too, not only
  `cache_dir`; a class no `class` statement makes (`type(...)`,
  `namedtuple(...)`, `make_dataclass`) is not compared, with may wording;
- tests: the `%%time` cell re-run unchanged, the annotate path (3.14+) under
  the cut, the failing scan's ambiguity checked by an edit, the two
  statements exact and stable on 3.13+, siblings vs branches on 3.11/3.12.

**Not changed, with reasons:**
- *what a statement-found class's body names* (a wrapped method's helper,
  the decorator, `staticmethod(scale)`) is not followed as notebook code: a
  function reached only through a closure or a class attribute's value is a
  documented residual on main too; this PR compares the class's own text;
- *a class remade by `exec` or an unwatched cell, with one older kept text*:
  without the watch nothing tells the object from the text, as on main for
  code whose text the kernel does not keep (documented);
- *a `type()`-made class on 3.11/3.12* can take an older kept statement of
  its name: no worse than main's constant, and the docs now say such a class
  is not compared;
- *`transform_cell` run again*: a stateful input transformer could differ;
  IPython's compiled filename is internal.

**Stopping rule:** a third round of both reviews; merge if it finds nothing
blocking.

**Results (#295 round 2 fixes).** 289 tests in the module, passing on 3.11
to 3.15; the whole suite 1634 passed (run in three parts, as the machine was
short of memory), the docs build and the docs tests pass. Seven mutations
for this round, all caught by node id (the annotate path's on 3.15, since it
exists only on 3.14+): an unreadable pin made ambiguous again, siblings
ignored, branches taken for siblings, the token without the process id, the
pattern without a continuation, the annotate path's reading put with main's,
and a failed scan's class not ambiguous.

### #295 review round 3 (2026-11-04): plan

`/code-review` (9) and the adversarial workflow (4 confirmed) reviewed
d49dd89. Plan, written before any edit.

**Blocking, both sides of one rule.** Round 2 made two different
statements of a class in one block ambiguous on 3.11/3.12.
- Too narrow (`/code-review`): a top-level `class Config` and a later `if
  True: class Config` are in two blocks and both run; so do a `try` body's
  and its `except` body's.
- Too wide, and my regression (adversarial, through a real resume): inside
  a function's or a class's body, two statements run as one step, so no
  record stands between them and which object comes out is a value. A
  factory stating its class twice re-ran the cells after every unchanged
  re-run.
The rule: two different statements are ambiguous (3.11/3.12) only when both
stand in the cell's own statements, not in a function's or a class's body,
and do not exclude each other. Only an `if`'s body and its `orelse` (an
`elif` chain is nested `orelse`s) and two cases of a `match` exclude each
other. `_class_nodes` returns each statement's path from the cell's top and
whether it stands at module scope (the bare qualname of a `global`
declared in a function does not make it so).

**Minor:**
- tests the reviews showed pass without the code they name: the `if`/`else`
  case in the real shell (the pinned path, where round 2's fix was), and a
  methodless class read exactly (two kernels with the same single text give
  one digest, another text another; the edit tests passed through the token);
- the post-cell callback no longer parses: the pin keeps the transformed
  lines, and the record reads them;
- `class_texts` is required, so "not given" cannot read as "the scan
  failed";
- a nested class (`Outer.Inner`) is pinned with the class whose body made
  it;
- with the watch on, a kept text the watch saw make another object of the
  qualname is not taken for an unpinned one (`make_dataclass` replacing a
  watched `class Config`, on 3.11/3.12);
- `post_run_cell` pops its own `pre_run_cell`'s entry (matched by the
  execution info): IPython fires post without pre for an empty cell;
- docs and comments: the token's re-run also comes from a cell stating the
  class twice, one statement after the other (3.11/3.12), and the run()
  docstring adds "or after a restart".

**Not changed:** the duplicated id-weakref pruning and the per-text walk of
`_class_nodes` (cleanup and cost only).

**Stopping rule:** a fourth round of both reviews on this delta; merge if it
finds nothing blocking.

**Results (#295 round 3 fixes).** 298 tests in the module, passing on 3.11
to 3.15; the whole suite 1643 passed (in three parts), and the docs build
and docs tests pass. Seven mutations for this round, all caught by node id:
an `if`'s branches not exclusive, any parting exclusive, a body's statements
judged like the cell's own, nested classes not pinned, the watch's texts not
skipped, the post callback popping the top entry, and a `match`'s cases not
exclusive. One test of mine failed at first for the wrong reason: the empty
nested cell's case used `_pf` before the script defined it, so the cell
raised; it now imports it itself.

### #295 review round 4 (2026-11-04): plan

`/code-review` (10) and the adversarial workflow (5 confirmed) reviewed
d52238c. Plan, written before any edit.

**The pattern, and the decision.** Four rounds, and each refinement of the
3.11/3.12 rule for telling a class's statements apart produced the next
blocker: round 3's skip of texts the watch had seen made a legitimate
nested class lose every kept text, so an edit went unnoticed (adversarial,
reproduced through a resume). On 3.11/3.12 a class does not say which of
its statements made it; every rule built from the cell's syntax is a guess
that the next shape breaks. So the guesses go, and the cases they aimed at
become one documented residual: *on Python 3.11 and 3.12, a class stated
more than once in one cell, in a loop or in a function that runs the sweep
may not be told apart by its statement; after editing one, use a new
`working_dir`.* That covers `/code-review`'s statement in a function with
`global`, an `if` in a loop, and a dead branch's statement pinned for a
`make_dataclass`; a class no `class` statement makes is documented already.
On 3.13+ `__firstlineno__` answers exactly, and on every Python the common
shapes (one statement, a factory, an edit and a revert under the watch) are
exact.

**Changes:**
- *the skip of seen texts goes*, with its state (it also removes
  `/code-review`'s `global` factory, key-format and growth findings);
- *a nested class is pinned on demand* through its outer class (the
  running cell's snapshot holds bare names: match the qualname's head);
- *the post callback falls back* to the cell's `raw_cell` when IPython
  passes a new execution info (an interrupted async cell), and ends the
  newest entry when it passes none;
- *registration* stores the state on the shell first and unregisters what
  it added if the second step fails;
- `_exclusive` stays `if`/`match`: widening it to `try` would be one more
  guess;
- the `_pinned_cell` docstring and a dead guard; a test that a `try` body's
  statement and its handler's both run.

**Stopping rule, set now:** round 5 is both reviews on this delta. A
finding that depends on a run between two statements of one class on
3.11/3.12 is the documented residual, not a blocker; the reviewers are told
so. Anything else blocking gets fixed; otherwise merge.

**Results (#295 round 4 fixes).** 302 tests in the module, passing on 3.11
to 3.15; the whole suite 1647 passed (in three parts); the docs build and
docs tests pass. Five mutations for this round, all caught by node id: the
nested on-demand pin by the full qualname, each of the two post-callback
fallbacks, the state stored after registering, and no unregister on a
failed registration. The net change against d49dd89 was read hunk by hunk:
the nested pins, the head match, the post fallbacks and the registration
order add information; the 3.11/3.12 rule for statements is the one guess
left, and its failure modes are the documented residual.


### #295 review round 5 (2026-11-04): plan

Both reviews ran on 26f6243. The adversarial run confirmed two minor
findings; `/code-review` gave ten, one of them blocking and outside the
documented residual.

**Blocking: the snapshot's regex reads comments and strings.** `pre`
snapshots every name that `\bclass\s+NAME` matches anywhere in the raw
cell. So `build(3)  # rebuild class Config`, where `build` declares
`global Config` and states the class, pins the new class to the calling
cell. That cell's text holds no statement, the class gets no site and a
constant digest, and an edit to the factory's class body is missed, on
every Python. Without the pin, the kept texts hold the factory's
statement, and the edit is seen.

*Fix:* match `class` only where a statement can start: at a line's start
after indentation (`[ \t\f]*`), or after a line magic on that line
(`%time class C: ...`, which the old regex caught and which must stay
pinned: unpinned, an older kept text of `C` would stand in for it, and a
resume would reuse that text's results). The same regex serves
`_cell_imports`' parse trigger. A real statement always starts a logical
line, so the trigger loses nothing, and a comment no longer costs a parse.
*Residual, accepted (spec only):* a line of a multi-line string that starts
with `class NAME`, in a cell that also calls a `global NAME` factory.

**Minor (adversarial):**
- the no-result post test has one running entry, so `found = 0` passes it:
  add an older outer entry and assert it is still running afterwards;
- the `_cell_watch` docstring says a refused registration "leaves no
  watch"; since round 4 it leaves a watch without callbacks, which pins
  nothing, and a later call does not register again.

**Cleanups (`/code-review`):**
- the `_CELL_WATCH` comment describes `running` entries as
  `(raw, snapshot)` or None; they are `(info, (raw, snapshot) or None)`;
- the `_scan_kept_cells` docstring calls its pairs `(text, index)`; they
  are `(text, (modules, bindings, classes))`;
- `_exclusive`'s `a[0] != b[0]` cannot be true: two paths from one root
  that agree up to a step reach the same parent, so the step's first field
  is redundant. Drop it and `id(node)` from the steps;
- `import itertools` at the top instead of `__import__`.

**Kept as they are:** `_pinned_cell` keeps scanning every running entry
(newest first). Looking at the newest only would miss a record taken in a
nested cell (`get_ipython().run_cell(...)`) after its outer cell stated
the class. The rest of `/code-review`'s list is the documented residual
(`global` in a function body on 3.11/3.12), accepted cost (a second
`transform_cell`, registration at import, efficiency), or a follow-up:
binding `__build_class__` in the notebook namespace's builtins would pin
every `class` statement exactly, on all versions, but it touches the
user's builtins and belongs in its own issue.

**Tests:**
- in the shell script, a `global Config` factory called from a cell whose
  comment says `class Config`, then from one whose string does: an edit to
  the factory's body changes the digest;
- in the shell script, a `%time class Config: ...` line magic after older
  kept statements: its digest is none of theirs;
- the no-result case with an outer entry, as above.

**Mutations:** the old `\b` regex (the comment and string cases fail);
the line-start regex without the magic branch (the `%time` case fails);
`found = 0` in post (the no-result case fails).

**Results (#295 round 5 fixes).** The test of a factory had to reach the
class through an instance (`cfg = build()`): a name a notebook function
rebinds is not followed at all, so the first draft's `Config (notebook)`
key was never there. 302 tests in the module, passing on 3.11 to 3.15; the
whole suite 1647 passed (in three parts); the docs build and the 165 docs
tests pass. Four mutations, all caught by node id, each on its own
assertion: the old `\b` regex (the comment and the string case each keep a
constant digest), the regex without the magic branch (the `%time` case),
`found = 0` in post, and a raw-text fallback that takes the oldest entry.

### #295 round 6 (2026-11-04): two Codex findings left open

Rereading Codex's inline comments while the final reviews ran turned up two
findings from rounds 2 and 3 that no round's plan recorded. (Its nested
`Outer.Inner` finding was the one round 4 fixed.) *Mistake:* the rounds
planned from `/code-review` and the adversarial run and never listed
Codex's inline comments. From now on, every round's plan lists them too.

**1. On 3.13+, assigning `__module__` deletes `__firstlineno__`** (Codex on
d52238c, P2; checked on 3.13, 3.14 and 3.15). A methodless class followed
by `Config.__module__ = __name__` reaches the rule "no `__firstlineno__`:
no `class` statement made it" and gets no site, so its edits are missed.
That is a silent reuse on 3.13+, outside the documented residuals.
`__static_attributes__`, which the compiler also sets in a class body,
survives the assignment. No class that `type()`, `make_dataclass`,
`types.new_class`, the functional `Enum` or `NamedTuple` makes has
either attribute; a `slots=True` dataclass keeps both.

*Fix:* on 3.13+, the absence of `__static_attributes__` decides "no
`class` statement made it". A class that has it but no `__firstlineno__`
(its `__module__` was assigned) is read without the first-line filter, as
on 3.11/3.12: every statement of its qualname, with the same ambiguity
rule. The pin and the kept texts are unchanged.

*Tests:*
- in the shell script, a class with `__module__` assigned in its own cell,
  edited and reverted: pinned, so the revert is exact on every Python;
- a kernel test (no watch) on 3.13+: the edit is seen.

*Mutation:* the old `__firstlineno__` rule; both tests fail on 3.13+.

**2. A class stated in both branches of an `if` digests differently across
versions** (Codex on d49dd89, P2). On 3.11/3.12 both statements are read;
on 3.13+ only the one at `__firstlineno__`. Records are compared across
Python versions, so a resume on another minor re-runs those cells.
That errs toward recomputing. Making 3.13+ read what 3.11/3.12 reads
would mean a second rule for picking statements on 3.13+, which is the
kind of change rounds 1 to 4 kept getting wrong. *Accepted and documented:*
one clause in the resilience guide, the agent guide and the `run()`
docstring, next to the other re-run cases. The same holds for any cell
stating a methodless class more than once, and for the 3.14 annotation
path.

**3. Codex on 844d027 (P2): a multi-line string's `class Config:` line.**
A triple-quoted example with a line starting `class Config:`, in a cell that
calls a `global Config` factory, still snapshots `Config`. Reproduced by
Codex on 3.14. This is the residual round 5 accepted; Codex calls it a
stale reuse, and it is one.

*And a regression from round 5, found while weighing this:* the line-start
regex no longer snapshots `exec("class Config: LR = 2")` on one line. That
class is then unpinned, and an older kept `class Config` stands in for it,
on 3.11/3.12 and, when the lines match, on 3.13+. Round 4's design said
"no older kept text stands in" for `%%time` and `exec`; round 5 broke it
for `exec`. *Mistake:* round 5 changed what the snapshot catches without
listing what relied on it catching more (the `exec` and `make_dataclass`
pins). The lesson [[plan-fixes-by-invariant]] again: list the members of
the invariant ("a cell that makes the class from text that is not kept is
pinned") before narrowing the rule.

*The regex cannot settle this.* A mention of the class's text (a comment, an
example string) and code that is not kept (`exec`, a magic's body) look the
same in a cell's text. A narrower regex trades one silent reuse for the
other. So the regex goes back to round 4's `\bclass NAME` anywhere (the
protective direction), and the pinned branch decides:

- the pinned cell states the class: its statements, as now;
- it does not: the class came from code the cell runs. If a kept cell
  holds a statement of the qualname inside a function's body (the only way a
  bare qualname gets there is `global NAME`, a factory), read the kept
  texts' statements of that kind only, with the usual groups and ambiguity;
- no such factory statement: no site, as now (`%%time`, `exec`,
  `make_dataclass` in a cell mentioning the class).

*Residual, accepted (spec only):* a class one cell makes from text that is
not kept, while a kept function also states it with `global`: on 3.11/3.12
the factory's statement stands in (3.13+'s first line usually tells them
apart).

*The final adversarial run on 844d027 confirmed two findings, both this
class:* the one-line `exec` above plus automagic `time class C: ...` (the
`%time` statement without `%`, on by default), reproduced end to end with
`cache_dir=` returning the old class's results; and the round-5 `%` branch
still matching a string on a magic line or a `% 'the class Config'`
continuation. Its suggested fix is the rule above. A third finding was
refuted but had a point: the mention tests passed through the ambiguity
token, since many cells in the script state `Config`, so they did not show
that the factory's statement is read.

*Tests:*
- the mention cases each use a class name only their factory states
  (comment, one-line string, multi-line string, a string on a `%time` line,
  a `%` continuation): an unchanged re-run of the call keeps the digest (the
  factory's statement is read, exactly), and an edit to the factory changes
  it;
- a one-line `exec` and an automagic `time class` of a class with one older
  kept text: not that text's digest;
- the `%%time` and `%time` cases keep their assertions;
- the `__module__` cases of item 1.

*Mutations:* no fallback (the mention cases fail); a fallback that reads
every statement (the `%%time` and `Solo` cases fail); the round-5 regex (the
`exec` and automagic cases fail); on 3.14, the `__firstlineno__` rule (the
`__module__` cases fail) and no 3.13+ rule (the `type()` case fails).

**Results (#295 round 6 fixes).** Codex's nested `Outer.Inner` case was
re-run against round 5's head before closing it: exact on 3.11 and 3.14.
The `__module__` case reproduced on 3.14 first (one constant digest for
every edit). 303 tests in the module, passing on 3.11 to 3.15; the whole
suite 1648 passed (in three parts); the docs build and the 165 docs tests
pass. Seven mutations, all caught by node id, each on its own assertion:
no fallback (every mention case keeps a constant digest), a fallback
reading every statement (the `%%time` re-run), the round-5 regex (the
automagic and the `exec` cases, checked one by one), and on 3.14 the old
`__firstlineno__` rule (the shell and the kernel `__module__` tests) and no
3.13+ rule (the `type()` case). One full-module run on 3.14 had one failure
that two reruns, one with the failure summary saved, did not repeat; the
first run's output was not kept, so the test is unknown (like the 3.13
flake in round 2). CI's 3.14 legs are the check.

### #295 round 7 (2026-11-05): the snapshot stops reading names from text

`/code-review` on the round-6 delta (844d027...b17c4b8) gave ten findings,
reproduced in a real shell:

1. *Regression:* `\s+` in the restored regex crosses a newline, so a line
   ending in the word `class` (`# the config class`) right above
   `class Solo5:` captures the keyword `class` as the name, and `findall`
   never sees `Solo5`. Unpinned, a `%%time` remake reads the older kept
   statement (stale on 3.11, and on 3.14 when the lines match); a kept
   cell's edit and revert become a token. Round 4 had the same bug.
2. *Regression:* the factory fallback keeps `not top`, which is also true in
   a class's body, so a kept `Outer.Inner` statement stands in for a
   `%%time` or `exec` remake of it.
3. On 3.11/3.12, a cell that states a class and also calls a kept `global`
   factory that remakes it reads the cell's statement (the factory's edits
   are missed). Older than round 6.
4. A failed scan turns a pinned class with no statement into a token
   (it was a stable no site).
5. `_pinned_cell` stops at the innermost running cell whose snapshot names
   the class, even when nothing was pinned there; a nested cell that only
   mentions the class hides the outer cell that stated it.
6. Efficiency: a pinned class with no statement now builds the class index.
7. The mention test's `rerun != after` holds through the token; its comment
   says more.
8. No tests for 1 and 2.
9. Stale docstrings: `_cell_imports`, `_pin_classes`, `_CELL_WATCH`.
10. *Design:* correctness rests on taking class names out of a cell's text,
    which cannot tell a mention from code. Each round traded one stale reuse
    for another (round 5 broke `exec`, round 6 broke comment-adjacent
    statements).

**The decision: take 10.** Rounds 4 to 6 each moved the regex and each
broke a case the last one held. The text is the wrong source for which
names a cell bound to new classes; the namespace has the fact. So:

- `pre` snapshots `{name: value}` for every class-valued name in the
  namespace (`issubclass(type(value), type)`, no hook runs), plus the raw
  text;
- `post` pins every name that now holds a class that is not the snapshot's
  object, whose qualname is the name and whose module is the notebook's,
  that is not pinned yet (first pin wins), with the classes its body made;
- `_pinned_cell` (a record inside a running cell) walks the running cells
  newest first and pins in the first one whose snapshot does not hold the
  class's outer object (it is new since that cell started), so a nested
  cell that started after the class was made is skipped (finding 5);
- the record keeps round 6's rule: a pinned cell that states the class is
  read; one that does not is a factory's call (statements in a *function's*
  body only, finding 2) or text that is not kept (no site).
- `_cell_imports`' parse trigger needs only "a `class` keyword at a line's
  start" (a kept text is Python, where a statement always starts a line):
  no name capture, so finding 1 cannot recur there.

What this changes beyond the findings: a class a cell binds without any
`class` text (`Config = make_dataclass(...)`, `exec(src)`, `%run`, a
factory called from a cell that does not mention it) is now pinned. With no
statement in the pinned text it gets no site, or a factory's statement,
never an older top-level one. On 3.11/3.12 that removes the stand-in the
`type()` residual describes. The cost: two passes over the namespace's
items per cell (a dict comprehension), holding the old class objects until
the cell ends.

**The rest:**
- 3: the 3.11/3.12 family (several statements one cell runs). The
  documented sentence is reworded to cover it: on 3.11/3.12, a class with
  more than one `class` statement that one cell runs (twice in the cell, in
  a loop, or in a function the cell calls) may not be told apart.
- 4: a pinned class with no statement and a failed scan gets no site, as
  before round 6.
- 6: accepted; the index is built once per record and only for such a
  class.
- 7: the comment says what the test shows (exact while one version is
  kept; an edit changes it).
- 8: tests for 1, 2 and 5, plus `make_dataclass` without a mention.
- 9: docstrings.

The adversarial run on the same delta confirmed one finding, finding 2
(reproduced end to end: `cache_dir=` returned the old nested class's
results). Codex finished on b17c4b8 with none, and CI passed.

**Checks the redesign must keep (from the advisor, written before code):**
- a brand-new name has no key in the snapshot, so the class bound to it is
  new; a comment says so. A later rebinding of the outer name within the
  same cell skips `cls`, as today (the 3.11/3.12 residual), and is not
  improved here;
- hook-free: `issubclass(type(v), type)`, never `isinstance`; the snapshot
  copies `list(ns.items())` once, since a callback thread can grow the
  namespace while it runs;
- every cell is snapshotted, with no text filter: `exec(src)` from a file,
  `%run` and a factory called from a cell that never says `class` are the
  cases a filter would drop back to the stand-in. Measured once: 40 us per
  snapshot of an 800-name namespace holding 371 classes (`from numpy import
  *`, pandas, xarray and 300 notebook classes);
- the `remade` assertion (`Mk = type(...)` after a statement) loses its
  3.13+ gate: with the pin it must hold on 3.11 and 3.12 too;
- finding 4 gets a direct test: a pinned class whose text states nothing,
  with `class_texts` giving None, is `([], False, True)`, while an unpinned
  one stays ambiguous;
- the trigger regex is renamed: it captures no name.

**Mutations, written now:** a snapshot of the regex's names only (the
comment-ending-in-`class` case fails); `_pinned_cell` stopping at the first
entry naming the class (the nested-mention case fails); the factory filter
back to `not top` (the nested `%%time` and `exec` cases fail); a failed scan
ambiguous when pinned (the direct test fails); a snapshot only when the raw
text holds `class` (the `make_dataclass`-without-mention case fails); on
3.14, the two 3.13+ rule mutations again.

**Results (#295 round 7).** The `remade` case (`type()` after a statement)
now passes on 3.11 and 3.12 too, with its gate dropped, which is the check
that the namespace snapshot pins what the text never named. 305 tests in
the module, passing on 3.11 to 3.15 (the logs kept); the whole suite 1649
passed (in three parts); the docs build and the 165 docs tests pass. Two
mutants showed gaps in the plan's list:
- *A snapshot of the regex's names only* survived at first. The new pin
  walks the namespace, so a name missing from the snapshot reads as new and
  is still pinned: that mutant was weak, not the test. The faithful round-6
  mutant (names from the text, and the pin walking only them) fails on the
  comment-above case and on `remade`. The weak one exposed what the
  snapshot is for, which no test covered: telling a class that existed
  before the cell from one it bound. An empty snapshot pins a class made
  before the watch started to the next cell, which states none. A class
  `Early` made before `import mushin` now checks it, and both snapshot
  mutants fail on it.
- *No 3.13+ rule* survived on 3.14: the watch now pins a `type()` class on
  its own, so the rule matters only without the watch (no IPython, or a
  class made before `import mushin`). A kernel test without the watch on
  3.13+ now catches it.
Every other mutant was caught on its own assertion, by node id: a snapshot
only when the raw text says `class` (`remade`), the on-demand pin stopping
at the newest running cell (the nested-cell mention), the factory filter
back to `not top` (the nested `%%time` remake), a failed scan ambiguous
when pinned (the direct test), and on 3.14 the old `__firstlineno__` rule
(the shell and the kernel `__module__` tests).

### #295 round 8 (2026-11-05): a cell that binds a class it does not state

`/code-review` on the round-7 delta (b17c4b8...be85372) gave thirteen
findings; the first three reproduced in a real shell on 3.11 and 3.13:

1. *Regression:* the namespace snapshot pins a class a cell derives from a
   kept statement in another cell (`Config = dataclass(slots=True)(Config)`,
   `type(C)(C.__name__, C.__bases__, dict(C.__dict__))`, `exec(In[n])`) to
   the deriving cell, which states none. Round 6's rule then reads only a
   factory's statements, finds none, and gives no site: a constant digest,
   so an edit to the original statement is missed. Round 6 left these
   unpinned, and the kept statement was read.
2. A failed scan gives a pinned class with no statement no site, where it
   was ambiguous (round 7 made this the rule and the test locks it in).
3. On 3.13+, a cell that states a class and calls a `global` factory that
   remakes it reads the cell's statement when both start on the same line
   of their cells; the docs said only 3.11/3.12.
4-5. No tests for 1 or 3.
6. The nested walk stops once `made` holds 64 classes, and `made` now holds
   every class a cell binds (`%run models.py` with 70 classes skips the
   nested ones, and an older kept nested statement can stand in).
7. `_CLASS_LINE` treats only `\n` as a line start; `compile()` accepts a
   bare `\r`.
8. Comments: `class_texts`' says a failed scan makes such a class
   ambiguous; `_class_sites`' "as on main" (main had no reading of these).
9. The fragment still describes the statement-based watch.
10. The reworded `run()` docstring puts "a `_target_` defined there" after
    "in a function it calls".
11. Every running entry holds strong references to every class until the
    cell ends, or until a stale entry is cleared.
12. With `only=`, the pin walks the whole namespace where a lookup by the
    qualname does.
13. The class-binding test is written twice (the snapshot, the pin).

**What finding 1 shows.** When the pinned cell states no statement of the
class, code it ran made the class, from text that is not kept (`%%time`, an
`exec` string, `%run`) or from a kept statement (a copy, `exec(In[n])`, a
`global` factory). Nothing in a class with no methods tells which. Reading
the kept statements misses `%%time`'s edits (round 5's `exec` regression);
reading none misses the copy's (this finding); reading only a factory's is
one more guess (round 6). Only "ambiguous" is safe both ways: the
per-object token, so the cells re-run when that cell runs again or after a
restart, and an unchanged object keeps its digest within the process.

**Decision:** a class pinned to a cell that states none of it is
ambiguous. The factory fallback (round 6) goes, with `in_function` and the
unscanned special case; findings 1 and 2 go with it. Exact answers stay
where they are facts: a pinned cell's own statements, a method's or 3.14's
`__annotate_func__` cell, and the kept texts for a class the watch did not
see bound. The cost: a `%%time`, `exec`, `%run` or `global`-factory class
re-runs its cells each time its cell runs again, and after a restart,
instead of keeping a constant digest that missed its edits. On 3.13+ a
class `type()` makes keeps its constant digest (the `__static_attributes__`
rule comes first): it has no statement, so a digest that never changes
says "not compared", as documented.

**The rest:**
- 3: the docs say it: on 3.11/3.12, or on any Python when two of the
  statements start on the same line of their cells. A test would lock in a
  miss; none.
- 6: the nested walk gets its own budget.
- 7: the line-start regex accepts `\r`, `\r\n` and `\n`.
- 8, 9, 10: comments, the fragment, the docstring.
- 11: superseded below: the snapshot holds the classes themselves, keyed
  by `id`, while the cell runs (holding them is what keeps an id from being
  reused); see "Kept as they are" and Codex's finding.
- 12: the on-demand pin looks the class up by its qualname.
- 13: one helper decides what a class binding is, for both.
- Tests: the three derivations in a separate cell (an edit and its revert
  never share a digest with the old version's); the `%%time` and mention
  assertions now say "a re-run recomputes"; a cell binding 70 classes
  still pins a nested one; the `\r` text is parsed.

**The adversarial run on the same delta** confirmed one minor gap: no test
told `in_function`'s any-step walk from an innermost-step check (a
`global` factory's nested class). It is moot here, since the factory
fallback goes. A refuted finding (refuted as older than round 7, not as
wrong) still holds: `_pin_classes` runs `shell.transform_cell` after the
cell. For a one-line cell that is IPython's prefilter, against the
namespace the cell left, so `isinstance` can run a notebook object's
`__class__` hook in the callback, and a raising hook drops the cell's pins.
Round 7 made every cell that binds a class reach it. *Fix:* the watch
transforms with IPython's static `input_transformer_manager`, which never
reads the namespace. It handles every form the record parses: cell and
line magics, `!`, `?`. Automagic `time class C` stays raw, does not parse,
and is ambiguous, as the dynamic transform's `run_line_magic` text would
be. A `_FakeShell` without the manager falls back to its `transform_cell`.
Test: a one-line cell binding a raising proxy and remaking a class by
`exec`: no hook runs, and the remade class is not the kept statement's.

**Kept as they are:** 11 (strong references during a cell): a weak
reference would change what "the class the name held" means in the middle
of a review round, for memory held only while a cell runs (a stale entry
until eight more cells start). Accepted.

**Mutations, written now:** pinned-with-no-statement back to no site (the
three derivations fail); the nested budget shared with the top-level count
(the 70-class test fails); the `\n`-only line regex (the `\r` test fails);
the dynamic `transform_cell` (the proxy case fails); the on-demand lookup
by the wrong name (the nested-cell mention fails).

**Codex on be85372 (P2):** two versions of a class saved under aliases
before `import mushin`, then `Config = old1` and `Config = old2`: each
existing object read as new, since only the name's previous object was
compared, and was pinned to its assignment cell (no site under round 7, one
digest for both). Round 8's rule alone makes it safe (a token). The root is
the snapshot's question: *new* must mean no name in the namespace held the
class when the cell started. So the snapshot becomes `{id(class): class}`.
It holds the classes, so no id among them can be reused while the cell
runs. `_pinned_cell` asks the same question.

**Decisions the advisor pressed, made now:**
- *`%run` and `%%time` classes re-run:* neither is a documented pattern
  (both appear only in the limits sentences), so the single rule stands, and
  the docs and the fragment name the cost: such a class re-runs its cells
  when that cell runs again or after a restart.
- *Finding 3:* the limit loses its version qualifier ("may not be told
  apart ... when one cell runs more than one `class` statement of it"),
  true on every Python with "may".
- *Tests also:* the unscanned test flips and is renamed (pinned with no
  statement is ambiguous whatever the scan gives); the `%%time` case also
  checks that a record right after keeps the digest (a token is per
  object, not per record); the proxy case runs once, with a unique name;
  the mention cases keep `before != after`, which the no-site mutant fails.
  A class saved under an alias before the watch and restored after it is
  read from its kept statement (an exact match with an unchanged re-run),
  which only the identity snapshot gives.
- *Docstrings to change:* `_class_sites`' bullet and closing line,
  `_pin_classes`' last sentence, the `_CELL_WATCH` comment, the
  `class_texts` comment.

**Stopping rule for round 8, given to both reviews:** a class pinned to a
cell that states none of it is ambiguous and re-runs; it never takes
another kept text's digest; on 3.13+ a class with no
`__static_attributes__` keeps its constant digest. A finding that such a
re-run costs work is the design. A finding that a class takes another
text's digest, or that an exact read is wrong, blocks. The 3.11/3.12
several-statements limit, and the `__build_class__` follow-up, stay as
they are.

**Results (#295 round 8).** 307 tests in the module, passing on 3.11 to
3.15 (the logs kept); the whole suite 1651 passed (in three parts); the
docs build and the 165 docs tests pass. Every mutant was caught by node id
on its own assertion: pinned with no statement back to no site (the three
derivations, the mentions, the `%%time` re-run and the direct test), the
nested budget shared with the top-level count (the 70-class test), the
`\n`-only line regex (the `\r` test), the dynamic transform (the proxy
one-liner), round 7's snapshot keyed by name (the alias restored), the
on-demand pin at the newest running cell (the nested-cell mention), and on
3.14 the old `__firstlineno__` rule and no 3.13+ rule.

*A mistake in the checks themselves, found here:* one comparison of
mutants reported a mutant's behaviour for the next one. The two had the
same byte size, and a cached `.pyc` was reused. Every round-8 mutant was
rerun with `PYTHONDONTWRITEBYTECODE=1` and the cache cleared before each
run; the results above are from that run. Earlier rounds' runners did not
clear it. Their mutants were written one at a time against a restored
source, which makes a reuse unlikely, but not impossible.

### #295 round 9 (2026-11-05): pin by object; the last stand-ins

The reviews of the round-8 delta (be85372...91f2709) ran with the stopping
rule in their briefs.

**Blocking (adversarial, reproduced end to end):** a class made under the
watch from text that is not kept (`exec`, `%%time`) and left only under an
alias (`saved = Config; del Config`) is never pinned: the pin wanted the
binding name to be the qualname. Restored (`Config = saved`), it is not
new to that cell either, so the older kept `class Config` gives it an exact
digest, and `cache_dir=` returned the old results. Round 7's name-keyed
snapshot had caught it, by the accident of reading the restore as new.
*Fix:* pin a new class by object. Any `str` name binding a class no name
held at the cell's start counts, when its qualname has no dot (a top-level
statement's, or `global`'s; `make.<locals>.C` is still skipped) and its
module is the notebook's; each class once. The on-demand pin passes the
class itself, with no lookup by name (which also removes `/code-review`'s
hash-collision note on that lookup).

**Blocking by the rule (`/code-review`, candidates, checked here):**
1. A cell that states one version of a class and binds another (`C =
   REG['C']`, a copy after an `if False:` statement) has its own statement
   read as exact. Nothing in a class with no methods tells these apart, so
   the documented limit widens: one cell holding more than one version of a
   class (two statements run, or a statement and a class bound from
   elsewhere).
2. The static transform skips `input_transformers_post`, which every round
   before 8 applied (inside `transform_cell`). *Not changed, documented
   here:* applying them would run an extension's code again in the
   callback, the hook class round 8 removed, and a stateful transformer
   would see each cell twice. The pinned text is the static transform of the
   raw cell, so an edit to the cell still changes its digest; on 3.13+ a
   transformer that adds lines shifts the first line, and the class is then
   ambiguous (safe).
3. The nested walk's budget is checked between classes, not within one,
   and leaves the rest unpinned (no kept statement: a constant digest). The
   walk is a tree (each step a longer qualname, each class once), so it has
   no budget now, and an id set keeps it linear.
4. IPython keeps a cell as `splitlines()` lines, so a form feed (or `\v`,
   `\x1c`-`\x1e`, `\x85`, `\u2028`, `\u2029`) adds a line that `compile`
   never counted; on 3.13+ no kept statement then sits at `__firstlineno__`,
   and the class got a constant digest. Kept statements of the qualname, but
   none at its first line, are now ambiguous.
5. The on-demand pin passed over a running cell whose text is unknown and
   could pin to the cell around it; it now stops there.

**Minor:** the docs listed `%%time`/`exec`/`%run` code as not compared next
to the new re-run sentence (the sentence now says such a class is not read
but re-runs); the round-8 spec bullet on weak references contradicted the
paragraphs after it; the shell test's docstring and a script comment still
said a factory's class is read from the factory; the mention cases dropped
the check that an unchanged re-run recomputes.

**Accepted:** the `\r` test stays (a kept text from another source can hold
one; a test with no effect in IPython costs nothing); strong references
during a cell (round 8). **Refuted, the design:** a `namedtuple` or
functional `Enum` re-runs its cells after a restart on 3.11/3.12 (pinned to
a cell stating none of it).

**Tests:** the alias restore after `exec` and after `%%time` (not the kept
statement's digest); a class with 300 nested classes (every one pinned); a
form-feed cell on 3.13+ without the watch (ambiguous, not constant); a
running cell with no text before the one that bound the class (not pinned);
a display cell before a class made before the watch is read (no post
transformer test: item 2 is documented, not changed).

**Mutations:** the pin by name again (the alias cases fail); the nested
budget back (the 300 case); no ambiguity for kept statements off the first
line (the form feed, on 3.14); passing over a textless entry (its test).

**Decided with the advisor:** IPython's history names (`_`, `_N`) are not
excluded from the pin by object. A class the namespace already held under
some name is in the snapshot, so a display cell does not take it. A new
class that only a history name holds is pinned to a cell that states none
of it, which makes it ambiguous, and the first pin still wins. The
dedupe on a second name is a cost guard: pins are keyed by id, so a mutant
without it changes nothing observable (an equivalent mutant, not run).
*Stop after this round* if both reviews report only kept-text edges or
costs: no path is left where a class takes another text's digest, and the
exact fix for the rest is the `__build_class__` follow-up.

**Results (#295 round 9).** 310 tests in the module, passing on 3.11 to
3.15 (the logs kept); the whole suite 1653 passed (in three parts); the
docs build and the 165 docs tests pass. Mutants, with the bytecode cache
cleared before each: the pin by name (the `exec` alias restore), the
nested budget (the 300 models), passing over a textless running cell (its
test), and on 3.14 no ambiguity off the first line (the form feed); all
caught. The nested-budget mutant first *survived*: the test held 300
classes nested in one class, which a single pass of the inner loop takes
whatever the budget, so it could not see the reviewer's case (many classes
with one nested class each). The test now has that shape. Each new test
also fails on 91f2709's source (`PYTHONPATH` at an archive of it), the
form-feed one on 3.14.

### #295 round 10 (2026-11-05): the record inside a cell pins only what is bound

Both reviews of the round-9 delta (91f2709...ecdffae) found the same
blocking regression, in five adversarial lenses and in `/code-review`.
Round 9 gave the on-demand pin the class itself (`only`) with no lookup by
name, and with it dropped the check that any name holds the class. A class
only an instance holds (`cfg = Config()` with `Config` rebound since, or
`Config = Config()`) is in no snapshot. A record inside a running cell
pinned it there and read that cell's statement of the name exactly: one
digest for two versions, a stale reuse. *Mistake:* round 9 replaced a check
that did two jobs (this is the name; some name holds it) with one that kept
neither, and the plan named only the first job.

`/code-review` also found the on-demand pin still looking up a nested
class's outer class by name. An outer class kept only under an alias left
the record inside its cell reading an older kept nested statement.

**Fix, one rule for both:** the record inside running cells pins, newest
cell first, what each has bound so far (the end-of-cell pin, by object, with
no `only`). It stops once the class is pinned, or at a cell whose text is
unknown. Only a class some name holds is ever pinned, an alias's nested
classes come with their outer class, and a class an outer cell made before
a nested cell started is pinned to the outer cell. `only` goes; the pin's
list and id set become one dict.

**Kept-text edges, accepted (the stopping rule):** characters that
`str.splitlines` breaks a line at and `compile` does not (a form feed, a
vertical tab, ` `) shift IPython's kept text, or split a string so it
does not parse. For a class made before `import mushin`, that can read an
older kept version's statement on 3.13+, or give a constant digest. Main's
function path has the same flaw. It is documented as a limit, and proposed
as a follow-up issue (fix the line mapping where kept text is read, for
functions and classes alike). `/code-review`'s textless-cell case (a
running cell whose `raw_cell` is not a string, or whose snapshot raised)
stays as is: IPython always passes a string.

**Docs:**
- the re-run sentence holds for cells run after `import mushin` (the watch
  starts there), so it says so;
- the separators and IPython input transformers join the limits, written in
  words (a docstring would turn a backslash escape into the character).

**Tests:**
- a class only an instance holds, recorded inside a cell that states its
  name at the same first line: an edit to its real cell changes the digest;
- a nested class whose outer class a cell keeps only under an alias,
  recorded inside that cell: not the kept nested statement's digest.

Each must fail on ecdffae's source.

**Mutants:**
- the on-demand pin forcing a pin to the running cell when its walk found
  nothing (the instance case);
- the on-demand pin at the newest cell only (the nested-cell mention).

**Results (#295 round 10).** Both new cases fail on ecdffae's source and
pass now (each checked on its own: the instance-only class kept one digest
across its edit, and the alias's nested class took the kept statement's).
*A slip caught before running:* the first draft of the instance case put
the record after the cell's statement, where it reads that cell's own
class correctly, so it could not fail; the record now comes first, with
the statement at the same first line as the earlier cell's. Mutants, with
the bytecode cache cleared: a forced pin to the running cell when its walk
found nothing (the instance case), and the walk at the newest cell only
(the nested-cell mention); both caught. 310 tests in the module, passing on
3.11 to 3.15; the whole suite 1653 passed (in three parts); the docs build
and the 165 docs tests pass.

### #295 round 11 (2026-11-05): the watch sees what the record reaches

Both reviews of the round-10 delta (ecdffae...cc1d5a2) found the mirror
image of round 9's regression. Round 10's pin takes only classes a name
holds, but the record also reaches a class through an instance
(`_notebook_object`: an instance's class, a bound method's owner) and
through notebook bases. So a class a post-import cell made from text that
is not kept and left only under an instance (`Config = Config()`, `cfg =
Config(); del Config`, an instance of a nested class whose outer class was
deleted) was never pinned. It took an older kept statement's digest, or a
constant one, and `cache_dir=` returned old results (reproduced end to end,
3.11 and 3.14). *Mistake, the same one twice:* rounds 9 and 10 each patched
one side of a mismatch at record time. Round 9 forced a pin (wrong exact
reads); round 10 dropped it (lost re-runs). The mismatch itself, between
what the watch counts as a cell's classes and what the record can reach,
was never named.

**Fix: one reach for both.** `_cell_classes(namespace)` gives every
notebook class the record can reach from the namespace's names, as
`_notebook_object` resolves them: a class a name binds, an instance's
class, a bound method's owner (or its class). From each it adds the
class's notebook bases and the classes its body made, all read by type and
through the type's own slots. The snapshot is that set when a cell starts.
The pin at a cell's end takes what is in it now and was not then, whose
qualname is not a function's local one (`<locals>`), first pin winning.
The record inside running cells walks them newest first. It passes over a
cell whose snapshot already held the class (it started after the class was
made), pins in the first one that did not, and stops at a cell whose text
is unknown. The nested walk and the name rules go: a nested class is in
reach through its outer class, an instance or a base, and its own qualname
says it is not a local one.

What this does beyond the findings: an instance-only class from a plain
`class` cell is now pinned to that cell and read exactly (round 10 made it
ambiguous through the kept texts). The record's other reach, a task passed
as a bound method of an unbound instance, stays outside the snapshot and
goes to the kept texts, as before.

**Also:**
- `/code-review`'s efficiency finding goes with the walk: a class every
  running cell's snapshot held is never re-scanned;
- the separator limit names the characters (a form feed, a vertical tab, a
  Unicode line separator) instead of `str.splitlines`, which also ends a
  line at `\n`;
- the fragment gets the "after `import mushin`" qualifier;
- one helper reads a live pin;
- the long lines in the edited paragraphs are reflowed.

**Tests:**
- a class held only by an instance, remade by `exec` or `%%time` (the
  singleton and `del` forms), recorded inside its cell and after it: never
  the kept statement's digest;
- a base class remade by `exec` and deleted, reached through a derived
  class: the derived class's digest changes;
- an instance of a nested class whose outer class was deleted;
- the instance-only case from round 10, now exact (an unchanged re-run
  keeps the digest; an edit changes it);
- `_cell_classes` directly: a name, an instance, a bound method, a base
  and a nested class are reached, and a library class is not.

Each must fail on cc1d5a2's source.

**Mutants:**
- no instance's class (the remakes);
- no bases (the base case);
- no nested classes (the 300 models);
- `<locals>` not skipped (the factory exactness tests);
- the walk without passing over (the nested-cell mention).

**Results (#295 round 11).** Every new shell case fails on cc1d5a2's source
and passes now (checked one by one: the four remakes, the base, the
instance case's exactness). The reach test is new. Mutants, with the
bytecode cache cleared: no instance's class, no bases, no nested classes,
`<locals>` not skipped, and the walk without passing over; all caught. Two
needed a sharper test first. "No bases" passed the reach test, whose `Base`
was also bound by name; the test now deletes it. "`<locals>` not skipped"
had no test under the watch (the factory tests use the kernel fixture,
which has none); a shell case now re-runs a factory call and keeps the
digest. *Slips, both caught before any result:* the remake cases were first
written with single `\n` escapes in a raw patch string, which broke the
script's literals; and a `%%time` remake was prefixed with an import line,
which pushed the magic off the cell's first line. 311 tests in the module,
passing on 3.11 to 3.15; the whole suite 1654 passed (in three parts); the
docs build and the 165 docs tests pass.

**Before the round-11 reviews (decided now):**
- *Cost, measured:* `_cell_classes` takes about 0.65 ms on a namespace of
  901 names that reaches 600 notebook classes (`from numpy import *`,
  pandas, xarray, 300 classes with a nested class each, 100 instances), and
  runs twice per cell. Accepted.
- *Codex:* its one inline finding since round 8 (on ecdffae, instance-only
  classes) is the case this round fixes. It had none on cc1d5a2.
- *A function's local class from a factory that `exec` defined* gets a
  constant digest: no kept text states it, and it is never pinned (a
  `<locals>` class is read from its factory's cell). That is "`exec` code is
  not compared", as documented; accepted.
- *The task root:* a class the record reaches only as the task (a bound
  method of an instance no name holds) is outside the snapshot and goes to
  the kept texts; accepted.

**Stopping rule for round 11:** code changes only for (a) a class the
record reaches from the namespace's names that `_cell_classes` misses,
made from text that is not kept, taking another kept text's digest; or
(b) a wrong exact read. Cost, kept-text edges, the task-root case and
docstrings are documented, and then the PR merges. No round 12 for an edge.

### #295 round 12 (2026-11-05): the whole walk, and the passed-over cells

The adversarial run on the round-11 delta (cc1d5a2...83db877) confirmed
two findings under the stopping rule, and three minor ones.

**(a) A missed reach, four lenses.** The record follows more than
`_notebook_object`: its walk (`visit`) also follows a cell function's
`__wrapped__` (resolved again), and a class's methods (`_class_methods`)
with their `__wrapped__`. A class held only through a notebook decorator
(`functools.wraps(cls)(make)`, a singleton factory) was never in reach, so
a `%%time` or `exec` remake took the kept statement's digest. It is not a
regression from cc1d5a2 (the same probe was stale there), but round 11
claimed the watch reaches what the record reaches. *Mistake:* round 11
mirrored the resolver and not the walk, and the plan named only the
resolver. *Fix:* `_cell_classes` follows each value as the record does,
reusing its hook-free readers: `_cell_function` (library wrappers,
`lru_cache`, `partial`) for a cell function, whose `__wrapped__` is
followed again; a class's methods through `_method_functions` and
`_cell_function`, with theirs; and bases and nested classes, as before.

**(b) A wrong exact read, introduced by round 11.** The walk passed over an
inner running cell whose snapshot held the class, then pinned the outer
cell, and that pin took the classes the inner cell had made so far. The
outer cell's older statement was then read for them, inside the nested cell
and after it. *Fix:* the cells passed over are pinned first, newest first,
so each claims its own classes before an outer one can. This happens only
when the class is new to some running cell, so a class every snapshot held
still costs no scan.

**Minor:**
- the snapshot side of the shared reach had no test (the round-10 by-name
  snapshot passed every test): a class made before the watch and held only
  by an instance, read exactly after an unchanged re-run of its statement;
- the bound-classmethod branch had no test: the reach test adds one;
- the separator list named three of the eight characters: the docs now say
  "a form feed, a vertical tab or another character, other than a newline
  or a carriage return, that ends a line for `str.splitlines`", in words.

**Tests:**
- a class a notebook decorator wraps (`functools.wraps(cls)(make)`),
  remade by `%%time` and by `exec`: never the kept statement's digest;
- the nested-cell case: an outer cell states two classes and runs a nested
  cell that restates one and records; the restated class is read from the
  nested cell, exactly;
- the reach test moves to the kernel fixture, so its functions are cell
  functions, and adds a decorator's wrapped class, a method's wrapped class
  and a bound classmethod;
- the instance-only class made before the watch.

**Mutants:**
- no `__wrapped__` (the decorator cases);
- no methods' `__wrapped__` (the reach test);
- the passed-over cells not pinned first (the nested case);
- the by-name snapshot (the class made before the watch);
- no `take(owner)` (the reach test).

**`/code-review` on the same delta** found the same two blocking findings
and added:
- *a foreign class's notebook metaclass:* `_notebook_object` reaches it
  (`_notebook_class(type(value))` when the class is another module's), and
  the reach now takes a class's metaclass too;
- *a parity test, not a hand-written list:* a test now spies on the
  classes the record visits for a task reaching every path above, and
  asserts each is within `_cell_classes` (a `<locals>` class aside). The
  hand-written reach test stays, for paths the record does not take today.
  *Fixed the test's own premise on the way:* a method that
  `functools.wraps(Target)` rewrote is not a method to the record (its
  qualname is `Target`'s), so the record does not visit `Target`; the watch
  reaching more is safe;
- *a cost, accepted:* a class that becomes reachable only in a later cell
  (stored in a dict at its own cell's end, then instantiated) is pinned to
  that later cell, which states none of it: ambiguous, so its cells re-run,
  where cc1d5a2 read the statement exactly;
- *altitude, a follow-up:* the watch's reach and the record's walk are two
  implementations; one hook-free resolver for both would make the parity
  structural instead of tested.

*Cost, measured again:* the walk's extra edges put `_cell_classes` at about
1.3 ms on the same 901-name namespace (2.6 ms per cell); accepted.

**Results (#295 round 12).** The parity test and the decorator and nested
shell cases fail on 83db877's source and pass now. The instance-only case
made before the watch passes on 83db877 too (it already snapshotted the
full reach), and its mutant, the round-10 by-name snapshot, fails it.
Mutants, with the bytecode cache cleared: no `__wrapped__`, no methods'
`__wrapped__`, the passed-over cells not pinned first, the by-name
snapshot, no `take(owner)`, and no metaclass; all caught, each on its own
assertion. 312 tests in the module, passing on 3.11 to 3.15; the whole
suite 1655 passed (in three parts); the docs build and the 165 docs tests
pass.

### #295 round 13 (2026-11-05): kernel names, and the last small gaps

The reviews of the round-12 delta (83db877...5f80f40) found no regression
from it. `/code-review` found four cases under the stopping rule, all
already present at 83db877; the adversarial run found three minor test and
docstring gaps.

**Fixed:**
- *(b) IPython's history names:* the watch followed `_`, `__`, `___` and
  `_N`, which the record never follows (`_kernel_name`). Displaying an old
  class as a cell's last expression made it newly reachable there, and the
  cell's newer statement was read for it. The reach now skips kernel
  names, as the record does;
- *(a) a `str` subclass as a namespace key or as a class's `__module__`:*
  the reach wanted an exact `str`, while the record finds the entry by its
  code name and compares the module by equality. Both are now read through
  `str`'s own slots (`str.__str__`, `str.__eq__`), still hook-free;
- *cost:* a record inside nested cells computes the reach once per lookup,
  not once per cell pinned; `follow` resolves a function once;
- *docs:* the `_pinned_cell` docstring says the passed-over cells are pinned
  only once a cell that could not reach the class is found; the separator
  sentence uses parentheses.

**Accepted, documented here:** a nested cell run with `silent=True` fires
no cell events, so the watch counts its classes as the outer cell's. If the
outer cell states the class too, that cell has run two `class` statements of
it: the documented several-versions limit.

**Tests:**
- a cell displaying an old class after a cell that restates it;
- three nested levels, with the outer class sorting first (the order of the
  passed-over pins);
- a self-wrapped cell function (`functools.wraps(fn)(fn)`), whose walk must
  end (in a thread with a time limit, so a regression fails instead of
  hanging);
- a `str`-subclass key and module in the parity test.

**Mutants:**
- kernel names followed;
- exact-`str` keys, and exact-`str` modules;
- the passed-over cells pinned oldest first;
- the function guard removed.

**Results (#295 round 13).** The display case and the parity test's
`str`-subclass key and module fail on 5f80f40's source and pass now. The
three-level and self-wrapped cases pass on both: they guard code round 12
had right, and their mutants fail them. Mutants, with the bytecode cache
cleared: kernel names followed, exact-`str` keys, exact-`str` modules, the
passed-over cells pinned oldest first, and the function guard removed (the
walk hangs, and the test's thread times out); all caught. 313 tests in the
module, passing on 3.11 to 3.15; the whole suite 1656 passed (in three
parts); the docs build and the 165 docs tests pass.

**Stopping rule for round 13's reviews, bounded (written before them).**
The unbounded form ("any (a) or (b) blocks") has no end: a reviewer can
always build one more shape. So:
- a regression from 5f80f40 (the same probe gives a worse answer on the
  new head) blocks, whatever its shape;
- a pre-existing (a) or (b) case blocks only through shapes the tests
  already treat as ordinary: plain `class` statements, instances,
  decorators, bases, nested classes, `exec`/`%%time`/`%run` remakes, and
  `run_cell` nesting that fires events;
- a pre-existing case that needs a `str` subclass as a key or a module,
  `silent=True`, a metaclass hook, a colliding `__hash__`, or a
  `__module__` set to a non-`str` goes on the accepted list, and the PR
  ships.

### #295 round 14 (2026-11-05): shown is not made

Both reviews of round 13 (5f80f40...e5f91b6) found one regression, from
several angles: round 13 skipped kernel names in the snapshot and the pin
alike. That fixed one direction (an old class shown by a later cell that
restates it) and broke the mirror: a class kept only in a list and shown
as its own cell's result reached no snapshot, so a later cell that
subclassed it, or named it from `_` while stating the name, was taken for
its maker, and that cell's statement was read for it. *Mistake:* round 13
treated "the record skips kernel names" as "the watch may ignore them".
The record's reach decides what is compared; it says nothing about which
classes already existed.

`/code-review` also probed the first fix tried here (history in the
snapshot only): a class `%%time` made, kept in a list and shown, then named
by `k = _`, was never pinned, and the older kept statement was read.

**Fix:** the snapshot is everything in reach, IPython's history included,
so nothing that existed reads as new. At a cell's end, a class newly in
reach through the names the record follows is pinned to the cell's text,
as before. One newly in reach only through history was *shown* by the
cell, which may not have made it: it is pinned with no text, so it is
ambiguous. `_cell_reach` walks once and returns both: everything, and
what was reached without kernel names. The parity test keeps comparing
the latter with the record.

**Cost, accepted:** a class a cell makes and only shows (bound to no name)
is ambiguous where 5f80f40 read it exactly (it re-runs).

**Accepted under the bounded rule:** a class kept only in a container and
subclassed under its own name by a later cell takes that cell's statement
for the base (it was stale on 5f80f40 too). That cell holds a statement of
the name and a class of that name from elsewhere: the documented
several-versions limit.

**Tests:**
- a class shown by its own cell, then subclassed under its name;
- a class `%%time` made and shown, then named from `_`;
- an old class shown by one cell, then bound in a cell that restates it;
- a class shown before the watch started, then named from its `_<n>` in a
  cell that states it unchanged: read exactly. A snapshot without history
  makes it ambiguous, which is safe but loses the exact read.

**Mutants:**
- (a) the snapshot without history;
- (b) history-only classes left unpinned;
- (c) history-only classes pinned with the cell's text.

Each is caught by one of the four cases. The first form of the last case
asserted a difference, and (a) passed it: history-only classes were
already pinned with no text at the first watched cell, so both answers
were ambiguous. The case now asserts the exact read.

**Results (#295 round 14).** The four cases pass. Mutants (a), (b) and (c)
are each caught through pytest by node id, with the bytecode cache
cleared. The round-13 head fails the shown-base and shown-then-bound cases.
313 tests in the module, passing on 3.11 to 3.15; the whole suite 1656
passed (in three parts); the docs build and the 165 docs tests pass.

### #295 round 15 (2026-11-05): made means alive at the end, not at the start

Both reviews of round 14 (e5f91b6...7ec564f) found a regression in nested
cells: an outer cell binds an old class to a kernel name partway through
(the plainest form is `for _ in registry.values(): pass`), then starts a
nested cell that names it. The nested cell's snapshot already held it
through history, so it did not pin it, and the outer cell pinned it with
its own text and read its own statement for it. The adversarial run's
patch (pin the running parent when a nested cell starts) would be one more
reach rule. `/code-review` named the cause: rounds 11 to 14 approximated
"made during this cell" by reachability (named, then history, then
passed-over cells), and each round's special case interacted with the
next. *Mistake, the third of its kind:* text, then the namespace, then
reach were each a proxy for the fact. The fact is liveness.

**Design.** A cell's snapshot is every live notebook class, found from
`object` through `type.__dict__["__subclasses__"]` (the type's own slot, so
no metaclass hook runs), with the module compared through `str`'s slot.
The objects are held, so no id is reused while the cell runs. At the end,
every notebook class alive now and not in the snapshot was made during the
cell, and is pinned to its text (first pin wins; a `<locals>` class is
skipped and read from its factory's cell). The record inside running cells
keeps its walk: cells whose snapshot holds the class are passed over, and
pinned first, newest first. Everything else stays: the static transform, a
pinned cell stating none of the class making it ambiguous, the kept texts
for a class made before the watch, the 3.13+ rules. `_cell_reach`,
`_cell_classes`, the history rules, the kernel-name logic and the reach and
parity tests go: the pin no longer depends on what the record reaches.

**The existing shell cases under liveness, by hand (expected before
running):**
- *More exact, still passing:* a class kept only in a container is pinned
  to the cell that made it (`shown_base`'s base, `Early`-style list cases).
  A shown class is never pinned to the cell that shows it (`displayed`,
  `shown_then_bound`: the old class keeps its making cell's pin, so it is
  exact and still differs from the restating cell's).
- `shown_made` (`%%time` made it): pinned to the `%%time` cell, which
  states none of it: ambiguous, as before.
- `pre_shown`, `Early`, `early_i`, `restored`: made before the watch, alive
  at every snapshot, never pinned: read from the kept texts, as before.
- *The instance, alias, factory, derived, nested and remake cases:* each
  class is pinned to the cell that ran its creation, as before.

**Accepted, decided now:**
- *Threads:* a class a background thread or a widget callback makes while
  a cell runs is credited to that cell. That is wrong only if the cell
  states the same qualname. One "may" clause goes into the limits.
- *Cost:* two walks per cell, about 5.5 ms each with torch, Lightning,
  numpy, pandas and xarray loaded (12,664 live classes). The one-walk
  version (reuse the previous end's set) would credit classes made between
  cells to the next cell; a follow-up.

**Kill criterion:** if an existing shell case regresses under liveness (an
exact read becomes ambiguous, or any read is wrong), stop and rethink
before patching. Liveness is meant to need no special cases.

**Tests to add:**
- the `for _ in registry.values(): pass` case with a nested cell naming the
  class (the regression);
- a function from an earlier cell that makes a class under `global`, called
  in a later cell (pinned to the caller, ambiguous);
- a container-held class subclassed under its own name: was accepted, now
  exact, so the base's edit is seen.

**Mutants:**
- an empty snapshot (`Early` and `pre_shown`);
- the `<locals>` skip removed (`local`);
- the passed-over cells not pinned first (`restated`);
- the walk through `cls.__subclasses__()` instead of the slot: a metaclass
  that overrides it, and the existing check that no hook ran.

**Stopping rule for round 15's reviews:** a regression from 7ec564f blocks;
a pre-existing wrong read or stand-in through ordinary shapes blocks. A
case that needs a thread, `silent=True`, a `str`-subclass key, a metaclass
hook, or cost is accepted.

**Results (#295 round 15).** The kill criterion held: every existing shell
case passed under liveness on the first run, with no special case added.
The three new cases: the loop-variable case and the container-held base
fail on 7ec564f and pass now; the `global` factory call is ambiguous on
both. The reach and parity tests went with `_cell_reach`; a liveness test
replaces them, and the shell's no-hook check now also covers a metaclass's
own `__subclasses__`. Mutants, with the bytecode cache cleared: an empty
snapshot, the `<locals>` skip removed, the passed-over cells not pinned
first, and the walk through the attribute instead of the slot; all
caught. 311 tests in the module, passing on 3.11 to 3.15; the whole suite
1654 passed (in three parts); the docs build and the 165 docs tests pass.

### #295 round 16 (2026-11-05): the snapshot holds nothing alive

Both reviews of round 15 (7ec564f...128fd22) found no wrong read and no
stand-in. Both found one regression outside the read rule: the snapshot
held every live notebook class strongly, superseded versions that were
not yet collected included. A user's `gc.collect()` always runs inside a
cell, after the next snapshot has taken them again, so a replaced class
and what it holds (class attributes, `functools.cache` contents) were never
freed, and `Base.__subclasses__()` kept listing stale versions. *Mistake:*
round 8's reason for holding the classes (no id reused while a cell runs)
was carried into a design where the snapshot is no longer the namespace's
own objects but everything alive.

**Fix:** the snapshot holds weak references. A class counts as alive at a
cell's start only if its id's reference still gives that class
(`_alive_at`), so a reused id is still never taken for it.

**Also:**
- the liveness test was order-dependent, since another test's kernel class
  of the same module name could still be uncollected. It now has its own
  kernel name, collects first, and asserts by identity;
- the two round-15 cases assert an exact read (an edit, then a revert);
- "bound" became "made" in two comments;
- the thread limit says "a background thread or a callback" at all three
  sites.

**Accepted:** a record that reaches a class the pin never takes (a
`<locals>` class made in the running cell) walks again on each record (a
cost).

**Results.** The leak case fails on 128fd22 and passes now; the mutant
that holds the snapshot strongly behind the same interface is caught.
311 tests in the module, passing on 3.11 to 3.15; the whole suite 1654
passed (in three parts); the docs build, the 165 docs tests and the lazy
import tests pass.

### #295 round 17 (2026-11-05): a class a finalizer brings back, and reloads

`/code-review` on round 16 (128fd22...5a6586f) found nothing blocking. The
adversarial run found one worse answer than 128fd22 (rule (1)): with weak
snapshots, an in-cell `gc.collect()` can collect a superseded class whose
instance a finalizer (`__del__`) then brings back. CPython clears every
weak reference to the class first, so its pin and its base's
`__subclasses__` entry are gone, and the kept texts stood in for it (an
`exec`-made methodless class). The same happens on 128fd22 when the
collection falls between cells; round 16 made an in-cell collection reach
it.

**Fixes:**
- *A class brought back:* a dead pin at the class's id while no base lists
  the class means it was collected and brought back, so it is ambiguous.
  The first form of this check used the dead pin alone and failed the
  `local` shell case: a `<locals>` class (never pinned) at the address of a
  freed pinned class has a dead pin at its id. The base listing tells that
  reused id from a class that was brought back;
- *a reload:* the watch's state survives a reload of `_resume`, and an
  older module's `pre` writes strong snapshots, which `_alive_at` would
  have called, running each class's `__init__` in a callback. A value that
  is not a weak reference is compared by identity;
- *tests:* a deterministic `_alive_at` test (a reused id, a dead reference,
  an older snapshot's class), a real finalizer resurrection with the
  collector disabled, and the leak case now taken through a record first;
- *cleanups:* one `weakref` import, the unused key, the agent guide's long
  line.

**Mutants:** no reload guard (the class is called), no identity check, no
resurrection check, and the dead pin alone (the `local` case); all caught.

**Accepted:** a superseded class that was never pinned (one made before
the watch) and is brought back by a finalizer is read from the kept texts,
as without a watch.

**Results (#295 round 17).** The first matrix run failed the resurrection
test on 3.15 only. 3.15 no longer clears the weak references of an object
a finalizer brings back (probed: the pin's reference stays alive and the
base still lists the class), so on 3.15 the class keeps its pin and the
case cannot arise. The test now branches on what it observes, not on the
version: pinned in both, with no text when the references were cleared.
313 tests in the module, passing on 3.11 to 3.15; the whole suite 1656
passed (in three parts); the docs build and the 165 docs tests pass.

### #295 round 18 (2026-11-05): a lost cell is a lineage, not a pin

Both reviews of round 17 (5a6586f...e440e24) found no regression.
`/code-review` found one case under rule (2), behind a finalizer
resurrection: a class made *on* a class a finalizer brought back is out of
the liveness walk's reach (the walk descends from `object` through each
class's live subclass list, and the brought-back class is unlisted). It is
never pinned, and the kept texts stood in. It also found that the dead-pin
gate could lapse once the pins were pruned, and could be read after the
walk pruned it. The adversarial run found only test gaps: the `_alive_at`
test's "dead" reference was alive (a class is in reference cycles, freed
only by the collector), and no test told a reused id from a lost cell.

**Fix (the altitude both suggested):** no dead-pin gate. A class still
unpinned after the walk, with any class in its lineage (its MRO, `object`
aside) that no base lists, has lost its cell: ambiguous. On CPython only a
class a finalizer brought back is unlisted, so a class alive as usual, at a
reused id or not, is never affected (the `local` case). Pruned pins, a
class made on a brought-back one, and a brought-back class that was never
pinned are all covered.

**Tests:**
- a really dead reference in the `_alive_at` test;
- a listed class with a dead pin at its id (a reused id: not a lost cell);
- the brought-back test now uses the pin's own reference as its oracle,
  not `_listed`, and adds a class made on the brought-back one.

**Mutants:**
- a dead reference counted as alive;
- the dead-pin gate back;
- the class checked without its lineage;
- no lost-cell check.

All caught.

**Results (#295 round 18).** 313 tests in the module, passing on 3.11 to
3.15; the whole suite 1656 passed (in three parts); the docs build and the
165 docs tests pass.
