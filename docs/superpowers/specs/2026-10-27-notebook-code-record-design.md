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
  defined in a notebook (a task from a file never scans the cells);
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

Planned in its own PR, after this one merges.

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
