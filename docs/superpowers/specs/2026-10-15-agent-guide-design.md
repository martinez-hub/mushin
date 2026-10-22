# Design: a usage guide for AI coding agents

**Date:** 2026-10-15
**Status:** Revised after spec review (12 confirmed findings, all
incorporated; three refuted findings still contributed guide content). Revised
again on 2026-10-22: the snippet fix is #253, and #249 (opened from another
session) already adds `docs/llms.txt`, so D5 now extends that file. The
user asked for design questions to be decided rather than asked, so each
decision records its rationale and the alternatives it rejected.
**Sequencing:** lands after two open PRs, and this branch is rebased onto
both before its own PR:
- **#253**, the docs snippet-include fix. D4 relies on its include guard and
  its CI docs-job step.
- **#249**, which adds a contributor `AGENTS.md`, a `CLAUDE.md` that imports
  it, and `docs/llms.txt` with tests. D5 extends that `llms.txt`. If #249 is
  closed unmerged, D5 falls back to adding the file itself (see D5).

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
- The docs' example includes have been broken since 2026-07-27 (ruff 0.16
  respaced them; #253 fixes them). An agent reading the Tutorial in that
  window saw a directive instead of code. Agent-facing code must be executed by
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
  `show()` printing and returning, `compare_methods`' naming requirements, the
  underpowered warning ignoring Holm correction, and the call-form pickling bug
  (Facts) are documented, not changed. Each is a follow-up. The one behavior
  change is D3's `__dir__` fix, which only affects introspection on a core
  install.
- MCP changes. The server stays analysis-only, and the guide points to it.
- Vendor-specific packaging (a Claude Code skill, MCP prompts).
- An `AGENTS.md` for agents that *contribute* to mushin (#249 adds one).
- API reference pages for the exports that have none. The guide tells agents
  to use `help(obj)`, which matches the installed version.

## Facts established (2026-10-15, mushin 0.15.1)

- **Install:** torch and pytorch-lightning are core dependencies. The `eval`
  extra adds torchmetrics and scipy. A task need not use torch.
- **Sweep output layout:** each sweep writes `multirun/<date>/<time>/` under
  the launch directory unless `working_dir=` is given. That directory holds
  `multirun.yaml`, `mushin_sweep_manifest.json` and one directory per cell
  (`0/`, `1/`, ...). Each cell directory holds:
  - `mushin_cell_status.json`, `mushin_provenance.json`,
    `mushin_workflow.log` and `.hydra/`
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
  - The call form, `mushin.sweep(fn)` bound to a different name, runs with
    the default local launcher. Its task does not pickle, though, so
    out-of-process launchers fail: the qualname points at
    `<module>.<fn>.__mushin_task__`.
  - Hence: a new module-level function, decorated, that calls the
    scientist's code.
- **Statistics without the torch eval layer:** a sweep over `method` and
  `seed` returns a `(method, seed)` dataset.
  - `mushin.benchmark.compare_methods(ds)` returns the pairwise table
    (Wilcoxon by default, Holm-corrected).
  - It reads the dims named exactly `method` and `seed`. Other names raise
    `KeyError: 'seed'`, and an extra swept dim raises an opaque `TypeError`.
  - It raises `IncompleteSweepError` on failed or skipped cells unless
    `allow_incomplete=True`.
  - `mean_diff` is `method_a - method_b`, in coordinate order.
  - It needs scipy (the `eval` extra).
- **Seeds and Holm correction:** Wilcoxon's best possible two-sided p with `n`
  seeds is `2 ** (1 - n)`. Holm multiplies the smallest of `m` p-values by
  `m`, so with `m` pairwise comparisons, significance at alpha needs
  `m * 2 ** (1 - n) <= alpha`.
  - 2 methods (1 pair) need 6 seeds.
  - 3 methods (3 pairs) need 7: with 6 seeds, perfectly separated data gives
    `p_corrected = 0.094`.
  - 4 methods (6 pairs) need 8.

  `warn_if_underpowered` checks only the uncorrected floor.
- **`resume=True`** re-runs incomplete cells, and completed cells whose
  resolved config or task source changed. It does not detect edits to helper
  functions, module constants or the environment. Those cells are silently
  reused. `cache_dir=` has the same blind spot.
- **Reloading:**
  - `MultiRunMetricsWorkflow().load_from_dir(root, "mushin_metrics.json")`
    then `.to_xarray()` reloads a finished sweep. Passing `None` as the
    filename silently gives an empty dataset.
  - `mushin.show`, `mushin.best` and `mushin.diff` read a sweep root
    directly. `show()` prints its table and also returns it. `best()` returns
    the single best cell (one seed), not the best configuration averaged over
    seeds.
- **`run()`'s docstring** documents `dry_run`, `confirm_above`,
  `max_total_seconds`, `on_error`, `resume`, `sample` and `cache_dir`, but not
  `working_dir` or `launcher`.
- **The docs site** is rebuilt from `main` on every push (`docs.yml`), not per
  release.
- **CI:**
  - The `test` job runs only when code paths change.
  - The `docs` job runs on every PR. #253 adds `tests/test_docs_snippets.py`
    to it, and #249 adds `tests/test_llms_txt.py` and
    `tests/test_docs_claims.py`. Its environment includes the dev group
    (which pulls scipy through scikit-learn), but not the eval extra.
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
  page (cp1252), so printing a non-cp1252 character such as `→` raises
  `UnicodeEncodeError`.

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
contributor instructions.

**ASCII only.** The text is ASCII (D7 enforces it), so every consumer can
print it, including a Windows pipe in cp1252.

### D2. `python -m mushin guide` prints it

Add `src/mushin/__main__.py`, an `argparse` CLI:
- `python -m mushin guide` writes one line, `<!-- mushin X.Y.Z -->` (the
  installed version, from `importlib.metadata.version("mushin-py")`),
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

### D3. `help(mushin)` works and points to the guide

- **A module docstring** in `src/mushin/__init__.py`:
  - one paragraph on what mushin does
  - the main entry points (`@mushin.sweep`, `MultiRunMetricsWorkflow`,
    `mushin.benchmark.compare_methods`, `Study` / `compare`)
  - the line "AI coding agents: run `python -m mushin guide` for a usage
    guide matching this version"

  The copyright comment stays a comment.
- **`__dir__` omits the lazy eval-layer names when the eval extra is not
  installed**, i.e. when `importlib.util.find_spec("scipy") is None`. This
  matches the check `mushin.benchmark` makes. `help()` and pydoc then render
  on a core install. Attribute access is unchanged: `mushin.compare` still
  raises the install hint.

**Rejected:** making the lazy failure raise a class that is both `ImportError`
and `AttributeError`. `from mushin import compare` would then swallow the
install hint, because CPython replaces an `AttributeError` raised during
`from ... import` with its own "cannot import name" error.

### D4. A docs page that includes the guide verbatim

`docs/agents.md` ("For AI agents"), in the top-level nav right after
Quickstart. It has two parts:
- **A short intro for the human:**
  - how to point an agent at the guide, with a copy-paste block for the
    project's `AGENTS.md` / `CLAUDE.md` telling the agent to run
    `python -m mushin guide` first
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
  experiment, read the agent guide first: `python -m mushin guide` prints the
  installed version's copy, and the latest is at" the raw Markdown URL
  (`https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md`).
  Raw Markdown is better input for an agent than the rendered page.
- **An entry for the new `docs/agents.md` page** in its "Docs" list, right
  after Quickstart, matching the nav. #249's test requires it, since the page
  is in the nav.

Nothing else in #249's file changes. Its facts block and example stay the
short version, and the guide is the long one.

**If #249 is closed unmerged,** this change adds a hand-written
`docs/llms.txt` itself. It would have the H1 and summary, a "Start here"
section (the raw guide, then the agents, Quickstart and Core concepts pages)
and a "Guides" section. D7 item 4 would then check its docs-site links too.

**Why not mkdocs-llmstxt:** it would downgrade `markdown-it-py` across the
shared lock (Facts) for one generated file. #249 reached the same choice.

D7 checks that the raw-GitHub link names a file that exists.

### D6. README pointer

A short "Using mushin with an AI agent" section right after the README
Quickstart. It covers `python -m mushin guide`, the docs page, `llms.txt`,
and the copy-paste block. The README is also the PyPI description, which is
the first page an agent fetches.

### D7. Tests keep the guide honest

A new test file, `tests/test_agent_guide.py`:

1. **It ships.** `importlib.resources.files("mushin") / "agent_guide.md"` is a
   file, and its text is ASCII.
2. **Its code runs.** The test concatenates the guide's ```python blocks, in
   order, into `tmp_path/guide_example.py` and runs
   `python guide_example.py` in a subprocess with `cwd=tmp_path`.
   - It runs as a real script, as an agent would run it, so `inspect.getsource`
     works for mushin's fingerprints and Hydra's global state stays out of the
     test process.
   - Code goes only in ```python blocks; shell commands go in ```bash blocks
     and output in ```text blocks, neither of which is executed.
   - The test skips without scipy.
3. **Its `run()` keywords exist.** The test AST-parses the blocks. Every
   keyword in a `.run(...)` call that is not a parameter of a task defined in
   the guide must be in `inspect.signature(MultiRunMetricsWorkflow.run)` (or
   the `Sweep.run` signature). A renamed safety keyword therefore fails, even
   though `run()` accepts arbitrary task overrides.
4. **Its links resolve.**
   - Every docs-site URL in the guide maps to `docs/<path>.md` or
     `docs/<path>/index.md` (anchors stripped). Site URLs are recognized by
     the `site_url` in `mkdocs.yml`, as #249's test does, so moving the site
     to a new host does not make the check pass vacuously. #249's tests
     already cover the site links in `llms.txt`.
   - The raw-GitHub link in `llms.txt` names a file that exists in the repo.
5. **The CLI works.**
   - `main(["guide"])` prints the version line, then the guide.
   - `main(["guide", "--path"])` prints an existing path.
   - `main([])` exits 0.
   - A subprocess run of `python -m mushin guide` with
     `PYTHONIOENCODING=cp1252` exits 0.
6. **Its numbers are pinned.** The seed floors the guide states (6 seeds for 2
   methods, 7 for 3, 8 for 4) are checked against `compare_methods` on
   perfectly separated synthetic data: significant at the stated floor, not
   significant one seed below it.

**In CI:**
- The `docs` job also runs `tests/test_agent_guide.py`, next to the snippet
  guard, so a docs-only PR that breaks a link or the guide's code fails.
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
2. **Install.** The extras that matter (`eval`, `viz`, `netcdf`, the
   batteries, `mcp`) in a small table.
3. **Choosing an entry point.** A decision table:

   | Job | Entry point |
   |---|---|
   | A function over a grid | `@mushin.sweep` |
   | Custom collection or Lightning | `MultiRunMetricsWorkflow` |
   | (method × seed) significance | `mushin.benchmark.compare_methods` |
   | torch models on a battery | `mushin.compare` |
   | Train, then compare | `Study` |
   | LLMs | `mushin.llm` |
   | Analyzing finished runs | `show` / `best` / `diff` / MCP |
   | Parallel or packed GPU runs | `launcher="joblib"`, `pin_gpu_round_robin` |
   | Cluster runs | the multinode guide |

4. **The template (runnable, tested), as one experiment script:**
   - A new module-level `@mushin.sweep` task that calls the scientist's code
     (stubbed in the guide) and returns a dict of floats. Never decorate the
     scientist's function in place.
   - `dry_run=True`, with an assertion on `num_cells`.
   - A run with a fixed `working_dir` and `method` and `seed` axes, plus one
     extra swept axis that is selected away before the statistics.
   - Read the results as a dataset and a DataFrame.
   - `compare_methods`, with the sign of `mean_diff` explained.
   - Where the outputs went.
5. **Rules that prevent wrong results or wasted compute:**
   - `multirun([...])` for swept values; `hydra_list([...])` for a list-valued
     argument.
   - Return a dict of numbers. Values are JSON-coerced into
     `mushin_metrics.json`.
   - Each cell runs in its own directory: use `mushin.original_cwd()` for
     input paths.
   - `dry_run=True` before anything expensive, plus `confirm_above` /
     `MUSHIN_MAX_CELLS` and `max_total_seconds`.
   - `on_error="nan"` keeps sweeping and lists the failures, and
     `compare_methods` then needs `allow_incomplete=True` or a complete grid.
     `sample=` explores a subset.
   - `resume=True` with the same `working_dir` re-runs incomplete cells and
     cells whose config or task body changed. After editing a helper, a
     constant or the environment, use a fresh `working_dir` (and don't rely on
     `cache_dir`).
   - `compare_methods` needs dims named exactly `method` and `seed`: rename or
     select others first.
   - Seeds: `m * 2 ** (1 - n) <= alpha` with `m` comparisons, so 6 seeds for 2
     methods, 7 for 3 and 8 for 4 at alpha 0.05. These are floors, not power.
   - Commit code before final runs: provenance records the git SHA and whether
     the tree was dirty.
   - `best()` is per-cell: average over seeds to pick a configuration.
     `show()` prints and returns.
6. **Running long sweeps from an agent.** Agent shell commands time out, so:
   - put the sweep in a script with a fixed `working_dir`
   - start it in the background with output redirected to a log
   - poll with `python -c "import mushin; mushin.show('<root>')"`
   - if a cell is still `running` but the process is gone, it was killed:
     relaunch the same script with `resume=True`
7. **Where results go and how to reload them.** The layout (including
   `mushin_error.txt`), `load_from_dir(root, "mushin_metrics.json")`,
   `show` / `best` / `diff`, and the MCP server.
8. **Before reporting to the scientist, a checklist:**
   - no failed or skipped cells, or they are reported
   - enough seeds for the number of comparisons
   - corrected p-values and effect sizes with the sign stated, not raw means
   - a clean provenance
   - the sweep root path, so the run can be found again
9. **Going further.** Links to the specific guides (LLM eval, batteries,
   Study, launchers and GPU packing, multi-node, resilience, statistics), and
   `help(obj)` for exact signatures.

Each claim in sections 4 to 7 is checked against Facts when the guide is
written. The D7 tests exercise the code, the keywords and the numbers.

## Verification

- **`tests/test_agent_guide.py`** passes locally. Mutating the API in a scratch
  copy makes it fail: renaming the `dry_run` keyword of `run()` fails item 3,
  and changing `compare_methods`' default test fails item 2 or 6.
- **The docs build:** `mkdocs build --strict` passes. The built
  `agents/index.html` contains the guide's code, and `llms.txt` is at the site
  root.
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

## Follow-ups (not in this change)

- API reference pages for the exports that have none.
- Docstring entries for `run()`'s `working_dir` and `launcher`.
- Behavior questions surfaced here, each its own decision:
  - `best()` averaging over seeds
  - `show()` printing
  - `compare_methods` accepting other dim names or reducing extra dims
  - `warn_if_underpowered` accounting for Holm correction
  - the `mushin.sweep(fn)` call form not pickling
  - `resume` / `cache_dir` not detecting helper or constant edits
- Agent-specific packaging (a skill, MCP prompts) on top of the guide.
