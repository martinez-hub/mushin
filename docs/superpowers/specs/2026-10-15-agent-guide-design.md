# Design: a usage guide for AI coding agents

**Date:** 2026-10-15
**Status:** Revised after review pass 1 (12 confirmed findings). Revised on
2026-10-22 for #253 and #249 (below), then after review pass 2 (an adversarial
review, `/code-review` and `/security-review`) and pass 3 (`/code-review`).
Each finding used here was reproduced or checked against the code. The user asked
for design questions to be decided rather than asked, so each decision records
its rationale and the alternatives it rejected.
**Sequencing:** this branch is rebased onto the following before its own PR:
- **#253** (merged), the docs snippet-include fix. D4 relies on its include
  guard and its CI docs-job step.
- **#249** (open), which adds a contributor `AGENTS.md`, a `CLAUDE.md` that
  imports it, and `docs/llms.txt` with tests. D5 extends that `llms.txt`. If
  #249 is closed unmerged, D5 falls back to adding the file itself (see D5).
- **D9's bug fix**, its own PR, first: a reloaded sweep keeps its record of
  failed cells.

## Problem

Scientists and engineers increasingly ask a coding agent (Claude Code, Codex,
Cursor and the like) to set up their experiments. mushin is a niche library
that agents know little about from training. Nothing in the repo is written for
them:

- There is no agent-oriented page. #249 (open) adds `docs/llms.txt`, an index
  of the docs pages with a short block of install and API facts and a minimal
  sweep, and a root `AGENTS.md` for agents that contribute to mushin. Neither
  tells an agent how to set up, run, check and report a scientist's
  experiment.
- `help(mushin)` gives an agent nothing useful:
  - On a full install it shows the copyright header comment as the
    description.
  - On a core install (`pip install mushin-py`, the README's first install
    line) it fails. It prints only the eval-extra install hint, and
    `python -m pydoc mushin` exits 1. Cause: `__dir__` lists the lazy eval
    names, pydoc `getattr`s each one, the lazy import raises `ImportError`,
    and `inspect.getmembers` catches only `AttributeError`.
- There is no `python -m mushin`. An agent that installed the package has no
  version-matched entry point.
- What an agent needs is spread across the quickstart, concepts, workflows,
  exploration-to-paper, resilience and statistics pages. No page says which
  API to use for which job, shows one runnable end-to-end template, lists the
  on-disk output layout, or says what to check before reporting results.
- Pitfalls that produce wrong results or wasted compute are not collected
  anywhere (each verified on 2026-10-15; see Facts).
- The docs' example includes were broken from 2026-07-27 to 2026-10-22
  (ruff 0.16 respaced them; #253 fixed them). An agent reading the Tutorial
  in that window saw a directive instead of code. Agent-facing code must be executed by
  tests, not just rendered.

## Goal

An agent that has installed mushin, or can fetch its docs, can set up a
correct sweep, run it safely, analyze it, and report it, without reading the
source. Success criteria:

1. **One guide, reachable everywhere.** It ships in the installed package
   (`python -m mushin guide`), offline. It is also reachable from
   `help(mushin)` on both core and full installs, the README (which is also
   the PyPI description), a docs page, and the site's `llms.txt`.
2. **Tested code.** The guide's Python runs as written, and the `run()`
   keywords it uses exist. CI checks both, so API drift fails a test instead
   of misleading agents.
3. **The installed version.** The packaged copy matches the installed
   version. The web copies track `main`.

## Non-goals

- Changing mushin's sweep or analysis behavior. `best()` being per-cell,
  `show()` printing and returning, `compare_methods`' requirements, the
  underpowered warning ignoring Holm correction, `max_total_seconds` under
  joblib, and the non-ASCII console output (Facts) are documented, not
  changed. Each is a follow-up. The behavior changes are D3's `__dir__` fix,
  which only affects introspection, and D9's reload fix, which lands first as
  its own PR because it fixes wrong results.
- MCP changes. The server stays analysis-only, and the guide points to it.
- Vendor-specific packaging (a Claude Code skill, MCP prompts).
- An `AGENTS.md` for agents that *contribute* to mushin (#249 adds one).
- API reference pages for the exports that have none. The guide tells agents
  to use `help(obj)`, which matches the installed version.

## Facts established (2026-10-15, mushin 0.15.1; extended and re-verified 2026-10-22)

- **Install:** torch and pytorch-lightning are core dependencies. The `eval`
  extra adds torchmetrics and scipy. A task need not use torch.
- **Sweep output layout:** each sweep writes `multirun/<date>/<time>/` under
  the launch directory unless `working_dir=` is given. That directory holds
  `multirun.yaml`, `mushin_sweep_manifest.json` and one directory per cell
  (`0/`, `1/`, ...). Each cell directory holds:
  - `mushin_cell_status.json`, `mushin_workflow.log` and `.hydra/`
  - `mushin_provenance.json`, unless the cell was skipped (`sample=` or an
    exhausted budget): a skipped cell returns before provenance is written
  - `mushin_metrics.json` if the cell completed
  - `mushin_error.txt` (the traceback) instead, if it failed under
    `on_error="nan"`

  `sweep.workflow.working_dir` is the sweep root.
- **`dry_run=True`** prints the cell count and every swept axis and returns
  `{"num_cells", "axes", "fixed", "working_dir"}` without launching a job.
- **Swept values:** a bare list raises
  `TypeError: ... wrap them: lr=mushin.multirun([...])`.
- **`@mushin.sweep` replaces the function with a `Sweep` handle.** The handle
  has no `__call__`, so decorating the scientist's own training function in
  place breaks their other call sites.
  - The call form, `swept = mushin.sweep(fn)`, leaves `fn` callable. It also
    runs under `launcher="joblib"` (re-verified 2026-10-22): stdlib `pickle`
    cannot pickle its task, but joblib's loky backend uses cloudpickle.
  - The template still uses a new module-level function, decorated, that
    calls the scientist's code: it adapts their signature and return value
    to a dict of numbers without editing their module.
- **Statistics without the torch eval layer:** a sweep over `method` and
  `seed` returns a `(method, seed)` dataset.
  - `mushin.benchmark.compare_methods(ds)` returns the pairwise table
    (Wilcoxon by default, Holm-corrected).
  - It reads the dims named exactly `method` and `seed`. Another name raises
    a `KeyError` (`No variable named 'method'`), and an extra swept dim
    raises an opaque `TypeError`.
  - The `method` values must be strings: it looks each one up as `str(m)`,
    so integer method ids raise
    `KeyError: "not all values found in index 'method'"`.
  - A method whose metric is identical across seeds gets a NaN `p_corrected`
    and `significant=False`, with a `UserWarning` that it "produced identical
    scores across all N seeds".
  - It raises `IncompleteSweepError` on failed or skipped cells unless
    `allow_incomplete=True`, but only on the dataset `.run()` returns (or a
    netCDF round-trip of it). `load_from_dir(...).to_xarray()` drops the
    `mushin_failures` / `mushin_skipped` attrs the check reads, so a reloaded
    sweep with failed cells is compared as if complete, without a warning.
    Holm then corrects over fewer p-values, because NaN rows leave the
    family. In a 3-method sweep with one failed cell, the reloaded dataset
    reported a pair as significant (`p_corrected = 0.031`) that the dataset
    `.run()` returned does not reach with `allow_incomplete=True` (`0.094`).
    A reload also misses cells that never finished: a cell still `running`,
    or one a killed sweep never started, which has no job dir. `mushin.show`
    lists only dirs with a status file, so it cannot show a never-started
    cell either. D9 fixes the reload.
  - `mean_diff` is `method_a - method_b`, in coordinate order.
  - It needs scipy (the `eval` extra).
- **Seeds and Holm correction:** Wilcoxon's best possible two-sided p with `n`
  seeds is `2 ** (1 - n)`. Holm multiplies the smallest of `m` p-values by
  `m`, so with `m` pairwise comparisons, significance at alpha needs
  `m * 2 ** (1 - n) < alpha` (`significant` is `p_corrected < alpha`).
  - 2 methods (1 pair) need 6 seeds.
  - 3 methods (3 pairs) need 7: with 6 seeds, perfectly separated data gives
    `p_corrected = 0.094`.
  - 4 methods (6 pairs) need 8.

  `warn_if_underpowered` checks only the uncorrected floor.
- **`resume=True`** re-runs incomplete cells, and completed cells whose
  resolved config or task source changed. It does not detect edits to helper
  functions, module constants or the environment. Those cells are silently
  reused. `cache_dir=` has the same blind spot. The template's task is a
  wrapper, so the scientist's own code counts as a helper: editing it is not
  detected either.
- **`max_total_seconds`** applies to the default sequential launcher only.
  Under `launcher="joblib"` each cell reaches its worker with a fresh deadline,
  so the budget never fires and nothing warns. With 12 two-second cells and a
  3 s budget, all 12 ran.
- **joblib** is a Hydra plugin (`pip install hydra-joblib-launcher`), not a
  mushin extra.
  - Its loky workers are reused once the cells outnumber `n_jobs`, so
    `pin_gpu_round_robin` raises in every later cell. `docs/guides/packing.md`
    says each job needs a fresh process and recommends Ray for packing.
  - Killing the main process does not stop the loky workers. They finish
    their cells and start ones already dispatched. Re-verified 2026-10-22:
    after a `kill -9` at 7 s (4 s cells, `n_jobs=2`), a cell started at about
    8 s and completed, and the workers were still alive afterwards. A cell
    `mushin.show` lists as `running` may therefore still be computing.
- **Reloading:**
  - `MultiRunMetricsWorkflow().load_from_dir(root, "mushin_metrics.json")`
    then `.to_xarray()` reloads a finished sweep. Passing `None` as the
    filename silently gives an empty dataset.
  - `mushin.show`, `mushin.best` and `mushin.diff` read a sweep root
    directly. `show()` prints its table and also returns it, and it lists
    each cell's status, failed cells included. `best()` returns the single
    best cell (one seed), not the best configuration averaged over seeds.
- **`help()` covers the launcher parameters:** `MultiRunMetricsWorkflow.run`'s
  docstring defers to `BaseWorkflow.run`, which documents `working_dir` and
  `launcher`.
- **Provenance** records the git SHA and whether the tree was dirty, where
  dirty means `git status --porcelain` printed anything in the launch
  directory. Untracked files count, so a log redirected into the repo, or an
  earlier sweep's output directory, marks every later run dirty.
- **Released versions:** no release has `mushin.__main__`, so
  `python -m mushin` fails on 0.15.1 (`No module named mushin.__main__`). The
  docs site and `llms.txt` deploy from `main` on merge, before the release.
- **PyPI:** the bare name `mushin` is unregistered on PyPI (404 on
  2026-10-22). An agent told to run `python -m mushin` in an environment
  without mushin may try `pip install mushin`, a name anyone could register.
- **The docs site** is rebuilt from `main` on every push (`docs.yml`), not per
  release.
- **CI:**
  - The `test` job runs only when code paths change.
  - The `docs` job runs on every PR. #253 added `tests/test_docs_snippets.py`
    to it (with `--noconftest`). #249 adds `tests/test_llms_txt.py` and the
    existing `tests/test_docs_claims.py`, in a step run with `--extra eval`.
    `test_docs_claims.py` (from #228) already pins the uncorrected Wilcoxon
    floor through its `_seed_floor` helper. The job's base environment
    includes the dev group (which pulls scipy through scikit-learn), but not
    the eval extra.
  - `core-only` installs `.` non-editable with no extras.
  - `batteries-clean-install` builds the wheel and installs it in a clean
    venv.
- **Packaging:** `[tool.setuptools.package-data]` ships only `py.typed` today.
- **#249's `tests/test_llms_txt.py`** requires `llms.txt` to list every nav
  page exactly once, and to link to no docs-site URL outside the nav. It
  recognizes site links by the `site_url` in `mkdocs.yml` and ignores links
  to other hosts. It also pins the install name, Python range and extras
  list to `pyproject.toml`, and runs the first ```python block.
- **`mkdocs-llmstxt` 0.5.1** would cap `markdown-it-py < 4`, downgrading it
  from 4.2.0 to 3.0.0 across the shared `uv.lock`.
- **ruff** formats Python code blocks in Markdown, but `ruff check` does not
  lint `.md` files.
- **Windows pipes:** Python on Windows writes a piped stdout in the ANSI code
  page (cp1252 on Western locales, cp932 or cp949 on Japanese or Korean
  ones), so printing a character outside it raises `UnicodeEncodeError`.
  mushin itself prints such characters: `mushin.diff()` prints `Δ<metric>`
  headers (also the column names of the table it returns), which cp1252
  lacks, and `dry_run=True` prints an em dash, which cp932 and cp949 lack.
  UTF-8 mode (`PYTHONUTF8=1`, or `python -X utf8`) avoids both.

## Approaches considered

1. **Docs-site page plus `llms.txt` only.** Least work, but agents without web
   access can't reach it, and the site tracks `main`, not the installed
   release.
2. **A guide shipped in the package, surfaced everywhere (recommended).**
   - One Markdown file inside `mushin/`, printed by `python -m mushin guide`.
   - Included verbatim in a docs page, linked first from `llms.txt`, and
     pointed to by the README and `help(mushin)`.
   - CI executes its code and checks the built wheel contains it.
3. **Approach 2 plus agent integrations** (a skill package, MCP prompts or
   resources, setup tools in the MCP server). More reach for specific agents,
   but more surface to maintain, and vendor formats are still moving. It can be
   built on approach 2 later, since the guide is the content either way.

## Decisions

### D1. One guide file, shipped in the package

`src/mushin/agent_guide.md`, added to `[tool.setuptools.package-data]`.

**Why:**
- An agent with no network still gets it.
- It matches the installed version.
- An agent browsing `site-packages/mushin/` finds it.

**Name:** not `AGENTS.md`. Tools such as Codex treat an `AGENTS.md` in a
directory as instructions for working on that directory's code, so an agent
editing mushin's own `src/mushin/` would take user-facing guidance as
contributor instructions. For the same reason no user-facing file is named
`agents.md` either: on a case-insensitive filesystem (the macOS and Windows
defaults) a lookup for `AGENTS.md` opens it (see D4).

**ASCII only.** The text is ASCII (D7 enforces it), so every consumer can
print it, including a Windows pipe in cp1252.

### D2. `python -m mushin guide` prints it

Add `src/mushin/__main__.py`, an `argparse` CLI:
- `python -m mushin guide` writes one line, `<!-- mushin X.Y.Z -->` (the
  installed version, from `importlib.metadata.version("mushin-py")`, or
  `unknown` when mushin is importable but not installed, e.g. with
  `PYTHONPATH=src` in a clone),
  followed by the guide. The HTML comment keeps the output valid Markdown and
  invisible when rendered.
- `python -m mushin guide --path` prints the file's absolute path, for agents
  that read files with their own tools.
- `python -m mushin` with no command prints help and exits 0.

**Why `python -m` and not a console script:**
- It works wherever mushin is importable, including notebooks and `uv run`.
- It needs no `[project.scripts]` entry.
- It is what an agent tries first.

**Rejected:** a `mushin.guide()` function. It is less discoverable than the
CLI, and D3 covers interactive discovery.

**From 0.16.0.** The command ships in the release that contains this change,
0.16.0 (a minor bump, for a new feature). Every place that tells an agent to
run it (D4, D5, D6) also says what to do on an older mushin: read the raw guide
at `https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md`.
It also gives the install name, `mushin-py`, because the bare `mushin` on PyPI
is not this project (Facts).

### D3. `help(mushin)` works and points to the guide

- **A module docstring** in `src/mushin/__init__.py`:
  - one paragraph on what mushin does
  - the main entry points (`@mushin.sweep`, `MultiRunMetricsWorkflow`,
    `mushin.benchmark.compare_methods`, `Study` / `compare`)
  - the line "AI coding agents: run `python -m mushin guide` for a usage
    guide matching this version"

  The copyright comment stays a comment.
- **`__all__` and `__dir__` list only names that resolve:**
  - When the eval extra is missing, both omit the lazy eval-layer names and
    the lazy `llm` submodule. On a core install `mushin.llm` raises the same
    install-hint `ImportError` (it imports `mushin.benchmark`), so dropping
    only the benchmark names still leaves pydoc failing. Filtering `__all__`
    as well also fixes `from mushin import *` and
    `inspect.getmembers(mushin)`, which raise on a core install today.
  - "Missing" means `importlib.util.find_spec` returns None for one of the
    modules `mushin.benchmark`'s guard imports. Those modules are listed once,
    `_EVAL_MODULES = ("scipy", "torchmetrics")`, and both the guard (which
    imports them) and this check use the list, so they cannot drift.
    `find_spec` does not import: IPython's tab completion calls `dir()`, and
    importing torchmetrics would import torch. torchmetrics always arrives
    with pytorch-lightning, so in practice scipy decides. A scipy that is
    installed but fails to import still breaks `help()`; that case is out of
    scope.
  - `__dir__` always omits the deprecated top-level names (`_DEPRECATED`).
    They are not in `__all__`, so pydoc does not document them, but
    `__dir__` lists them and `help(mushin)` touches each one: 8
    `DeprecationWarning`s per render, and an error under `-W error`. They
    stay accessible.

  `help()` and pydoc then render on a core install. Attribute access is
  unchanged: `mushin.compare` still raises the install hint.

**Rejected:** making the lazy failure raise a class that is both `ImportError`
and `AttributeError`. `from mushin import compare` would then swallow the
install hint, because CPython replaces an `AttributeError` raised during
`from ... import` with its own "cannot import name" error.

### D4. A docs page that includes the guide verbatim

`docs/for-agents.md` ("For AI agents", served at `/for-agents/`), in the
top-level nav right after Quickstart. Not `docs/agents.md`: on a
case-insensitive filesystem, a coding agent looking for `docs/AGENTS.md` (the
nearest-file convention) would open it and take the scientist-facing
copy-paste block as contributor instructions. It has two parts:
- **A short intro for the human:**
  - how to point an agent at the guide, with a copy-paste block for the
    project's `AGENTS.md` / `CLAUDE.md`. The block says to install with
    `pip install mushin-py` (the PyPI package named `mushin` is not this
    project), to run `python -m mushin guide` first, and to read the raw
    guide URL instead on mushin older than 0.16 (D2)
  - the MCP server, for analysis
  - a note that the web copy tracks `main`, while `python -m mushin guide`
    prints the installed version's copy
- **Then the guide,** included with a one-line snippet include outside any
  fence: `--8<-- "src/mushin/agent_guide.md"`. That is the form
  `docs/changelog.md` uses. #253's guard allows it there and checks that the
  file exists.

**Links inside the guide** are absolute docs-site URLs
(`https://martinez-hub.github.io/mushin/...`), so they work in a terminal, in
the package file and on the site. D7 checks that each one maps to a page.

### D5. Point #249's `docs/llms.txt` at the guide

#249's hand-written `docs/llms.txt` (llmstxt.org format, copied to the site
root) is the agents' index. This change adds two things to it:
- **A line ending its opening facts list:** "To set up, run and report an
  experiment, read the agent guide first: on mushin 0.16 or later,
  `python -m mushin guide` prints the installed version's copy; the latest
  is at" the raw Markdown URL
  (`https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md`).
  Raw Markdown is better input for an agent than the rendered page. The
  file's first fact already gives the install name, `pip install mushin-py`.
- **An entry for the new `docs/for-agents.md` page** in its "Docs" list,
  right after Quickstart, matching the nav. #249's test requires it, since
  the page is in the nav.

Neither addition touches the lines #249's tests parse ("Supported Python: X
to Y.", "Other extras: ...", the first ```python block).

Nothing else in #249's file changes. Its facts block and example stay the
short version, and the guide is the long one.

**If #249 is closed unmerged,** this change adds a hand-written
`docs/llms.txt` itself. It would have the H1 and summary, a "Start here"
section (the raw guide, then the For AI agents, Quickstart and Core concepts
pages) and a "Guides" section. It also ports #249's nav helpers and its
`llms.txt` tests (every nav page listed once, no site link outside the nav)
into `tests/_docs_site.py` and `tests/test_llms_txt.py`, so D7 item 4 has them
either way.

**Why not mkdocs-llmstxt:** it would downgrade `markdown-it-py` across the
shared lock (Facts) for one generated file. #249 reached the same choice.

D7 checks that the raw-GitHub link names a file that exists.

### D6. README pointer

A short "Using mushin with an AI agent" section right after the README
Quickstart. It covers `python -m mushin guide` (0.16 or later, with the raw
URL for older versions), the docs page, `llms.txt`, and the copy-paste block. The README is also the PyPI description, which is
the first page an agent fetches.

### D7. Tests keep the guide honest

A new test file, `tests/test_agent_guide.py`:

1. **It ships.** `importlib.resources.files("mushin") / "agent_guide.md"` is a
   file, and its text is ASCII.
2. **Its code runs.** The test concatenates the guide's ```python blocks, in
   order, into `tmp_path/guide_example.py` and runs
   it with `sys.executable` in a subprocess with `cwd=tmp_path`.
   - It runs as a real script, as an agent would run it, so `inspect.getsource`
     works for mushin's fingerprints and Hydra's global state stays out of the
     test process.
   - Every ```python block runs, in order, as one script. Shell commands go
     in ```bash blocks and output in ```text blocks, which are not executed.
     Any other Python fence tag (`py`, `python3`, `pycon`) fails the test, so
     no example can dodge it. An example of an error is written to run: it
     catches the error (`except TypeError as e: print(e)`).
   - The template's `working_dir` is relative (under `runs/`, D8), so each
     test run writes into its own temporary directory.
   - The subprocess runs with `PYTHONUTF8=1`, as the guide tells agents to
     run mushin on Windows (D8), so a block may call `mushin.diff`.
   - The test skips without scipy.
3. **Its `run()` keywords exist.** The test AST-parses the blocks and checks
   only mushin's `run()` calls: `.run(...)` on a name the guide binds to a
   `@mushin.sweep` function or to a `MultiRunMetricsWorkflow`. Calls such as
   `subprocess.run` are left alone. Each keyword must be a parameter of that
   task, or a named parameter of `MultiRunMetricsWorkflow.run` or
   `BaseWorkflow.run`, which `run()` forwards to. A renamed safety keyword
   therefore fails, even though `run()` accepts arbitrary task overrides.
   `Sweep.run` is not consulted: its signature is `(self, **kwargs)`, so it
   would accept anything. Bindings include the call form
   (`swept = mushin.sweep(fn)`). Keywords passed as `**name` are checked when
   the guide binds `name` to a dict literal or a `dict(...)` call; any other
   `**` unpacking in a mushin `run()` call fails the test, so no keyword can
   hide from it.
4. **Its links resolve.**
   - Every docs-site URL in the guide must be the URL of a page in
     `mkdocs.yml`'s nav, computed the way #249's `tests/test_llms_txt.py`
     computes it (`_nav_pages` and `_site_url`, which handle notebooks,
     `index.md` and URL nav entries). Only nav pages count, so a link into
     the excluded `docs/superpowers/` fails even though the file exists.
     Those helpers move into a shared `tests/_docs_site.py` that both tests
     import. Anchors are stripped.
   - The guide must contain at least one docs-site URL. If `site_url`
     changes, every old link stops matching it, so a guide with no matching
     link fails instead of passing vacuously.
   - Every raw-GitHub URL in the guide, `docs/for-agents.md`, the README and
     `llms.txt` names a file that exists in the repo: it is the only fallback
     D2 gives agents on older versions. #249's tests already cover the site
     links in `llms.txt`.
5. **The CLI works.**
   - `main(["guide"])` prints the version line, then the guide.
   - `main(["guide", "--path"])` prints an existing path.
   - `main([])` exits 0.
   - A subprocess run of `python -m mushin guide` with
     `PYTHONIOENCODING=cp1252` exits 0.
6. **Its numbers are pinned,** in the existing `tests/test_docs_claims.py`
   rather than a second file, next to its uncorrected Wilcoxon floor. The
   seed floors the guide states (6 seeds for 2 methods, 7 for 3, 8 for 4) are
   checked against `compare_methods`: significant at the stated floor, not
   significant one seed below it.
   - The synthetic data must vary across seeds within each method, e.g.
     method `i` scores `arange(n) + 10 * i`. A metric constant within a
     method gives a NaN p-value (Facts), so the naive fixture of 1.0 for one
     method and 0.0 for the other would fail at every seed count.
   - The test also checks that the guide's text states those three numbers.

**In CI:**
- The `docs` job also runs `tests/test_agent_guide.py`, so a docs-only PR that
  breaks a link or the guide's code fails. It needs scipy and torchmetrics, so
  it goes in the step #249 adds with `--extra eval` (if #249 is closed
  unmerged, this change adds a step like it).
- `core-only` (non-editable, no extras) checks
  `"python -m mushin guide" in pydoc.render_doc(mushin)` and runs
  `python -m mushin guide --path`.
- `batteries-clean-install` (built wheel, clean venv) runs
  `python -m mushin guide --path` and checks the file is non-empty. That
  proves the wheel carries it, which an editable install cannot.

### D8. Guide content

Compact: 250 to 400 lines. Prose stays short and the template carries the
detail. In order:

1. **What mushin is.** Sweep a Python function over a grid of parameters and
   seeds, and get a labeled `xarray.Dataset` / `pandas.DataFrame` back, with
   provenance and paired statistics. The task can use any framework. torch and
   Lightning are installed with mushin and power the optional eval layer.
2. **Install.** `pip install mushin-py` (the PyPI package `mushin` is not
   this project). The extras that matter (`eval`, `viz`, `netcdf`, the
   batteries, `mcp`) in a small table, plus the launcher plugins, which are
   Hydra packages rather than extras (`hydra-joblib-launcher`,
   `hydra-ray-launcher`).
3. **Choosing an entry point.** A decision table:

   | Job | Entry point |
   |---|---|
   | A function over a grid | `@mushin.sweep` |
   | Custom collection or Lightning | `MultiRunMetricsWorkflow` |
   | (method x seed) significance | `mushin.benchmark.compare_methods` |
   | torch models on a battery | `mushin.compare` |
   | Train, then compare | `Study` |
   | LLMs | `mushin.llm` |
   | Analyzing finished runs | `show` / `best` / `diff` / MCP |
   | Cells in parallel on one machine | `launcher="joblib"` (a Hydra plugin) |
   | Several runs per GPU | the GPU packing guide (Ray; `pin_gpu_round_robin` needs a fresh process per job) |
   | Cluster runs | the multinode guide |

4. **The template (runnable, tested), as one experiment script:**
   - A new module-level `@mushin.sweep` task that calls the scientist's code
     (stubbed in the guide) and returns a dict of floats. Never decorate the
     scientist's function in place.
   - `dry_run=True`, with an assertion on `num_cells`.
   - A run with a fixed, relative `working_dir` under `runs/`, which the guide
     tells the agent to add to the project's `.gitignore` (an ignored
     directory does not mark provenance dirty); string-valued `method` and
     `seed` axes; and one extra swept axis that is selected away before the
     statistics.
   - Read the results as a dataset and a DataFrame.
   - `compare_methods` on the dataset `.run()` returned, in the same script,
     with the sign of `mean_diff` explained.
   - Where the outputs went.
5. **Rules that prevent wrong results or wasted compute:**
   - `multirun([...])` for swept values; `hydra_list([...])` for a list-valued
     argument.
   - Return a dict of numbers. Values are JSON-coerced into
     `mushin_metrics.json`.
   - Each cell runs in its own directory: use `mushin.original_cwd()` for
     input paths.
   - `dry_run=True` before anything expensive, plus `confirm_above` /
     `MUSHIN_MAX_CELLS` and `max_total_seconds`. The time budget works only
     with the default launcher; under joblib, bound the run some other way.
   - `on_error="nan"` keeps sweeping and lists the failures, and
     `compare_methods` then needs `allow_incomplete=True` or a complete grid.
     `sample=` explores a subset.
   - `resume=True` with the same `working_dir` re-runs incomplete cells and
     cells whose config or task function changed. The task is a wrapper, so
     after editing the scientist's code, a helper, a constant or the
     environment, use a fresh `working_dir` (and don't rely on `cache_dir`).
   - `compare_methods` needs dims named exactly `method` and `seed`, and
     string `method` values: rename, select or convert first. A method whose
     metric is the same for every seed gives NaN, not a result: heed the
     warning `compare_methods` prints.
   - Seeds: `m * 2 ** (1 - n) < alpha` with `m` comparisons, so 6 seeds for 2
     methods, 7 for 3 and 8 for 4 at alpha 0.05. These are floors, not power.
   - Commit code before final runs, and start them with `git status
     --porcelain` empty. Provenance records the git SHA and whether the tree
     was dirty, and untracked files count, so keep the sweep output and logs
     in the gitignored `runs/` (or outside the repository).
   - On Windows, run mushin with `PYTHONUTF8=1` (or `python -X utf8`):
     `diff` and the dry-run preview print characters a piped console's code
     page may lack.
   - `best()` is per-cell: average over seeds to pick a configuration.
     `show()` prints and returns.
6. **Running long sweeps from an agent.** Agent shell commands time out, so:
   - put the sweep, and its `compare_methods` call, in a script with a fixed
     `working_dir` under the gitignored `runs/`
   - start it in the background with output redirected to a log, also under
     `runs/`
   - poll with `python -c "import mushin; mushin.show('<root>')"`
   - prefer the default launcher for a sweep the agent runs in the
     background. If the main process died with cells still `running`, check
     that nothing from that run is still alive before relaunching the same
     script with `resume=True`: under joblib, worker processes outlive the
     main one and keep computing cells.
7. **Where results go and how to reload them.** The layout (including
   `mushin_error.txt`), `load_from_dir(root, "mushin_metrics.json")`,
   `show` / `best` / `diff`, and the MCP server. Before comparing a reloaded
   sweep, check that `mushin.show(root)` lists a `completed` row for every
   cell of the grid (the count `dry_run` reported): it cannot list a cell a
   killed sweep never started. With D9 the comparison itself refuses an
   incomplete reload, but the web copy of this guide also reaches older
   versions, where it does not.
8. **Before reporting to the scientist, a checklist:**
   - every cell of the grid `completed` in `mushin.show(root)`, or the
     missing ones reported
   - enough seeds for the number of comparisons
   - corrected p-values and effect sizes with the sign stated, not raw means
   - a clean provenance (`dirty: false`)
   - the sweep root path, so the run can be found again
9. **Going further.** Links to the specific guides (LLM eval, batteries,
   Study, launchers and GPU packing, multi-node, resilience, statistics), and
   `help(obj)` for exact signatures.

Each claim in sections 3 to 7 is checked against Facts when the guide is
written. The D7 tests exercise the code, the keywords and the numbers.

### D9. A reloaded sweep reports its cells without a result (its own PR, first)

`load_from_dir` walks the sweep's grid and records every cell that has no
result, as `.run()` does. `to_xarray()` then sets `mushin_failures` and
`mushin_skipped` on a reloaded sweep exactly as on the dataset `.run()`
returned, and `compare_methods` raises `IncompleteSweepError` on it.

- **The grid** comes from the sweep's overrides (`multirun.yaml`), keyed as
  `to_xarray` keys a loaded sweep. A cell's status comes from its
  `mushin_cell_status.json`. The end-of-run manifest is missing after a kill
  or an `on_error="raise"` crash, so it is not the source.
- **The record:**
  - `failed` cells go in `failures`, with the manifest's error text when the
    manifest lists this dir, else the last line of the cell's
    `mushin_error.txt`.
  - `skipped` cells (sampled out, out of budget) go in `skipped`, and so do
    cells that never finished: still `running` or `pending`, or with no job
    dir at all. compare_methods' message already says "not run/skipped".
  - Job dirs left by an earlier sweep of a reused `working_dir` are ignored,
    by the rule `mushin.show` uses (the manifest's combo-to-dir map, with an
    exception for a status newer than the manifest). The rule moves into a
    shared `is_stale_cell` helper. A dir whose config lacks a swept axis is
    such residue too, not a crash.
- **Around it:**
  - `load_metrics` NaN-fills a `running` or `pending` cell instead of
    loading a stale sidecar or raising for missing files.
  - `is_complete` on a loaded workflow reflects the record, instead of
    always being `True`.
  - `load_from_dir` resets the metrics, jobs and manifest a reused workflow
    object kept from an earlier `.run()`.
  - Paths in the record are absolute, like `.run()`'s.
- **Tests:** the record equals `.run()`'s for a failed and for a sampled-out
  cell; a complete sweep gets none; a reused `working_dir` with a narrower
  grid over other axes neither reports the old cells nor crashes; a sweep
  killed between cells and one killed mid-cell are recorded as not run; a
  reused workflow object returns the loaded sweep; paths are absolute; and
  `compare_methods` refuses a reloaded sweep with a failed cell, and one
  killed before its last cell.
- **A changelog fragment:** `changes/<PR>.fixed.md`.

**Why first, and separate:** it is a wrong-results bug in the released 0.15.1,
independent of the guide. A separate PR gives it its own "Fixed" entry and
lets it ship in a patch release. The guide (D8 item 7) then needs one rule,
which also covers older versions.

**Rejected for now:** having `compare_methods` treat any NaN cell as
incomplete. That would also catch a task that returns NaN, but it changes
results for tasks that return NaN on purpose. It is a follow-up decision.

## Verification

- **`tests/test_agent_guide.py`** passes locally. Mutating the API in a scratch
  copy makes it fail: renaming the `dry_run` keyword of `run()` fails item 3,
  and changing `compare_methods`' default test fails item 2 or 6.
- **The docs build:** `mkdocs build --strict` passes. The built
  `for-agents/index.html` contains the guide's code, and `llms.txt` is at the
  site root.
- **D9:** its tests pass, and they fail on main before the fix.
- **The wheel:** `uv build --wheel` installed into a clean venv gives
  `python -m mushin guide --path` an existing file.
- **A core install:** a non-editable core install (no extras) renders
  `pydoc.render_doc(mushin)` with the guide pointer.
- **The usual gates:**
  - `ruff format --check` (which covers Markdown code blocks)
  - `ruff check` (Python files; the guide's code is covered by D7 item 2
    instead)
  - codespell
  - #253's include guard and #249's `tests/test_llms_txt.py`
  - the full test suite
- **A changelog fragment:** `changes/<PR>.added.md`.

## Risks

- **The guide drifts from the API.** Mitigated by D7 items 2, 3 and 6, and by
  keeping prose claims few and checkable.
- **The web copies (docs page, `llms.txt`, absolute links) track `main`, not
  the installed release.** This is acceptable: the package carries the
  version-matched guide, and the docs page says so.
- **The `__dir__` change.** `dir(mushin)` on a core install no longer lists
  the eval names. Nothing in the repo relies on `dir(mushin)` listing them,
  and the plan's grep checks this.
- **Import cost of `python -m mushin`.** It imports the package (about 1 s),
  which is acceptable for an occasional command.
- **The command reaches the web before it reaches a release.** The docs page
  and `llms.txt` say to run `python -m mushin guide` as soon as this merges,
  and on 0.15.1 it fails. Mitigated by the fallback each of them gives (D2).
- **The install name.** An agent that cannot import mushin may reach for
  `pip install mushin`, which is unregistered and could be claimed by anyone.
  Mitigated by stating `mushin-py` wherever the command appears (D2). A
  placeholder registration is a maintainer follow-up.

## Follow-ups (not in this change)

- API reference pages for the exports that have none.
- Behavior questions surfaced here, each its own decision:
  - `best()` averaging over seeds
  - `show()` printing
  - `compare_methods` accepting other dim names, non-string method values,
    or reducing extra dims; treating NaN cells as incomplete (D9)
  - `warn_if_underpowered` accounting for Holm correction
  - `resume` / `cache_dir` not detecting helper or constant edits
  - `max_total_seconds` under joblib: warn, or enforce across workers
  - console output that survives a non-UTF-8 console (`diff`'s `Δ` is also
    a column name of its returned table, so renaming it is an API change)
- Maintainer action: register `mushin` on PyPI as a placeholder that points
  to `mushin-py`.
- Agent-specific packaging (a skill, MCP prompts) on top of the guide.
