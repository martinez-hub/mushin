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
