# Refocus mushin on the Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink mushin's public surface and import weight so it reads as "one thing done well" — a boilerplate-free sweep→dataset tool — without removing any functionality researchers rely on.

**Architecture:** Split `src/mushin/__init__.py` into an eager core (workflows/Study/utils/lightning) plus a package-level `__getattr__` (PEP 562) that lazily loads the `benchmark` and `llm` peripheries on first attribute access. De-RAI the core `run()` defaults, add a Hydra cwd helper, demote two legacy workflow classes with a deprecation shim, and close the out-of-scope HPC PRs.

**Tech Stack:** Python 3.10+, pytest, hypothesis, PyTorch Lightning, hydra-zen, xarray. Package uses `uv` (`uv run pytest ...`).

**Scope note:** This plan covers spec items **1, 3, 4, 5** on the `refocus-core` branch. Spec **item 2 (strip auto-tuning)** lives on the separate `auto-tuning` branch and reshapes the open PR #60 — it is a *separate plan* (`2026-07-13-strip-auto-tuning.md`) to avoid mid-plan branch switching. Do not attempt item 2 here.

**Spec:** `docs/superpowers/specs/2026-07-13-refocus-core-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/mushin/__init__.py` | Eager core imports + lazy `__getattr__` for benchmark/llm + deprecation shim | Modify (rewrite) |
| `src/mushin/_utils.py` | Add `original_cwd()` helper | Modify |
| `src/mushin/workflows.py` | `config_name`/`job_name` default `"rai_workflow"` → `"mushin_workflow"` (3 signatures + docstrings) | Modify |
| `tests/test_lazy_imports.py` | Assert light import + back-compat name resolution + deprecation warnings | Create |
| `tests/test_utils.py` | Test `original_cwd()` | Modify |
| `tests/test_workflows.py` | Assert new default config/job names via signature introspection | Modify |
| `docs/quickstart.md`, `docs/concepts.md` | Hydra per-job cwd callout | Modify |

**Correction vs. spec §1 acceptance:** `pytorch_lightning` imports `torchmetrics`
transitively, so `torchmetrics` is loaded by the eager core regardless. The
honest, testable win is: `import mushin` does **not** import `mushin.benchmark`
or `mushin.llm`. Tests assert on those two, not on `torchmetrics`.

---

## Task 1: Lazy-load benchmark + llm; keep eager core

**Files:**
- Modify: `src/mushin/__init__.py`
- Test: `tests/test_lazy_imports.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lazy_imports.py
"""The top-level import stays light; peripheral subsystems load on first use."""
import subprocess
import sys


def _fresh_import_probe(code: str) -> str:
    """Run `code` in a fresh interpreter and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_import_mushin_does_not_load_benchmark_or_llm():
    out = _fresh_import_probe(
        "import sys, mushin;"
        "print('mushin.benchmark' in sys.modules, 'mushin.llm' in sys.modules)"
    )
    assert out == "False False"


def test_benchmark_names_resolve_lazily():
    # Accessing a benchmark export triggers the lazy import and returns the object.
    out = _fresh_import_probe(
        "import sys, mushin;"
        "obj = mushin.compare;"
        "print(callable(obj), 'mushin.benchmark' in sys.modules)"
    )
    assert out == "True True"


def test_llm_submodule_resolves_lazily():
    out = _fresh_import_probe(
        "import sys, mushin;"
        "mod = mushin.llm;"
        "print(mod.__name__, 'mushin.llm' in sys.modules)"
    )
    assert out == "mushin.llm True"


def test_unknown_attribute_raises_attribute_error():
    import mushin

    try:
        mushin.does_not_exist  # noqa: B018
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lazy_imports.py -v`
Expected: FAIL — `test_import_mushin_does_not_load_benchmark_or_llm` fails because the current `__init__.py` eagerly does `from . import llm` and `from .benchmark import ...`.

- [ ] **Step 3: Rewrite `src/mushin/__init__.py`**

Replace the whole body below the copyright header with:

```python
import importlib
from typing import TYPE_CHECKING

from ._utils import load_experiment, load_from_checkpoint, original_cwd
from .lightning import HydraDDP, MetricsCallback
from .study import Study  # keep last of eager block: avoids circular import via .study -> _sweep
from .workflows import MultiRunMetricsWorkflow, hydra_list, multirun

# Benchmark exports are loaded on first attribute access (see __getattr__), so a
# bare `import mushin` does not pull torchmetrics-heavy battery code.
_LAZY_BENCHMARK = frozenset(
    {
        "BenchmarkResult",
        "Task",
        "compare",
        "register_task",
        "get_task",
        "list_tasks",
        "audio_battery",
        "classification_battery",
        "segmentation_battery",
        "detection_battery",
        "regression_battery",
        "retrieval_battery",
        "image_quality_battery",
    }
)

# Legacy names kept importable from the top level for one release, with a warning
# pointing at their new home. They are NOT advertised in __all__.
_DEPRECATED = {
    "BaseWorkflow": "mushin.workflows",
    "RobustnessCurve": "mushin.workflows",
}

if TYPE_CHECKING:  # help static analysers/IDEs see the lazy names
    from . import llm
    from .benchmark import (
        BenchmarkResult,
        Task,
        audio_battery,
        classification_battery,
        compare,
        detection_battery,
        get_task,
        image_quality_battery,
        list_tasks,
        register_task,
        regression_battery,
        retrieval_battery,
        segmentation_battery,
    )
    from .workflows import BaseWorkflow, RobustnessCurve


def __getattr__(name: str):
    if name == "llm":
        module = importlib.import_module("mushin.llm")
        globals()["llm"] = module
        return module
    if name in _LAZY_BENCHMARK:
        value = getattr(importlib.import_module("mushin.benchmark"), name)
        globals()[name] = value  # cache so later lookups skip __getattr__
        return value
    if name in _DEPRECATED:
        import warnings

        warnings.warn(
            f"mushin.{name} is deprecated and will be removed in a future "
            f"release; import it from {_DEPRECATED[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Intentionally NOT cached in globals(): re-import is cheap and keeps the
        # warning firing on each top-level access during the deprecation window.
        return getattr(importlib.import_module(_DEPRECATED[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(_DEPRECATED))


__all__ = [
    "llm",
    "load_experiment",
    "load_from_checkpoint",
    "original_cwd",
    "MetricsCallback",
    "MultiRunMetricsWorkflow",
    "HydraDDP",
    "multirun",
    "hydra_list",
    "Study",
    "compare",
    "BenchmarkResult",
    "Task",
    "register_task",
    "get_task",
    "list_tasks",
    "audio_battery",
    "classification_battery",
    "segmentation_battery",
    "detection_battery",
    "regression_battery",
    "retrieval_battery",
    "image_quality_battery",
]
```

Note: `original_cwd` is imported eagerly but is not defined yet — Task 3 adds it.
To keep this task self-contained and green, temporarily also add the helper stub
now (Task 3 replaces its body with the real implementation and its own test):

In `src/mushin/_utils.py`, add near the top-level functions:

```python
def original_cwd() -> Path:
    """Placeholder — real implementation added in Task 3."""
    return Path.cwd()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lazy_imports.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Run the full suite to catch import regressions**

Run: `uv run pytest -q`
Expected: PASS. If anything imported `from mushin import BaseWorkflow`/`RobustnessCurve` and now trips the `DeprecationWarning`-into-error filter, note it for Task 2 (the shim test) — but existing internal code imports those from `.workflows` directly, so no failures are expected.

- [ ] **Step 6: Commit**

```bash
git add src/mushin/__init__.py src/mushin/_utils.py tests/test_lazy_imports.py
git commit -m "refactor: lazy-load benchmark/llm; keep import mushin light"
```

---

## Task 2: Deprecation shim for `BaseWorkflow` / `RobustnessCurve`

**Files:**
- Modify: `src/mushin/__init__.py` (already has the `_DEPRECATED` path from Task 1)
- Test: `tests/test_lazy_imports.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lazy_imports.py`:

```python
import warnings

import pytest


@pytest.mark.parametrize("name", ["BaseWorkflow", "RobustnessCurve"])
def test_deprecated_names_warn_but_resolve(name):
    import mushin

    with pytest.warns(DeprecationWarning, match=name):
        obj = getattr(mushin, name)
    # Same object as the canonical home.
    from mushin import workflows

    assert obj is getattr(workflows, name)


@pytest.mark.parametrize("name", ["BaseWorkflow", "RobustnessCurve"])
def test_deprecated_names_absent_from_all(name):
    import mushin

    assert name not in mushin.__all__
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/test_lazy_imports.py -k deprecated -v`
Expected: PASS already — Task 1 implemented `_DEPRECATED` and excluded the names
from `__all__`. If `test_deprecated_names_warn_but_resolve` FAILS because the name
was cached in `globals()` and no longer warns, confirm the Task 1 code does **not**
cache deprecated names (it returns without assigning to `globals()`).

- [ ] **Step 3: Fix only if red**

If the warning did not fire, ensure the `_DEPRECATED` branch in `__getattr__`
returns `getattr(importlib.import_module(...), name)` **without** writing to
`globals()[name]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lazy_imports.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mushin/__init__.py tests/test_lazy_imports.py
git commit -m "refactor: demote BaseWorkflow/RobustnessCurve from top-level API"
```

---

## Task 3: `original_cwd()` helper for the Hydra per-job cwd footgun

**Files:**
- Modify: `src/mushin/_utils.py`
- Test: `tests/test_utils.py`

Context: `task()` runs inside Hydra's per-job working directory, so relative paths
resolve against the wrong directory. `original_cwd()` returns the directory the
sweep was launched from, or the current directory when not inside a Hydra run.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_utils.py`:

```python
from pathlib import Path

import mushin
from mushin._utils import original_cwd


def test_original_cwd_outside_hydra_returns_process_cwd():
    # No active Hydra run -> falls back to the process working directory.
    assert original_cwd() == Path.cwd()


def test_original_cwd_is_exported():
    assert mushin.original_cwd is original_cwd


def test_original_cwd_uses_hydra_when_available(monkeypatch):
    import mushin._utils as utils

    monkeypatch.setattr(utils, "_hydra_original_cwd", lambda: "/tmp/launch-dir")
    assert original_cwd() == Path("/tmp/launch-dir")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_utils.py -k original_cwd -v`
Expected: FAIL — `test_original_cwd_uses_hydra_when_available` fails (no
`_hydra_original_cwd` attribute) and the helper is still the Task 1 stub.

- [ ] **Step 3: Replace the stub with the real implementation**

In `src/mushin/_utils.py`, replace the placeholder `original_cwd` with:

```python
def _hydra_original_cwd() -> str | None:
    """Return Hydra's launch cwd if a run is active, else None.

    Isolated for monkeypatching in tests. Hydra raises if no run is active, so we
    swallow that and let the caller fall back to the process cwd.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        from hydra.utils import get_original_cwd

        if HydraConfig.initialized():
            return get_original_cwd()
    except Exception:  # hydra not initialised / not installed in this context
        return None
    return None


def original_cwd() -> Path:
    """Directory the experiment was launched from.

    Inside a Hydra job the process cwd is the per-job output directory, so relative
    paths in a ``task()`` silently resolve against the wrong place. Use this to
    anchor paths against the launch directory instead::

        data = load(mushin.original_cwd() / "data" / "train.csv")

    Outside a Hydra run this is just the current working directory.
    """
    launch = _hydra_original_cwd()
    return Path(launch) if launch is not None else Path.cwd()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -k original_cwd -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/_utils.py tests/test_utils.py
git commit -m "feat: add mushin.original_cwd() for the Hydra per-job cwd footgun"
```

---

## Task 4: De-RAI the `run()` config/job name defaults

**Files:**
- Modify: `src/mushin/workflows.py` (lines ~287-288, ~612-613, ~1063-1064, plus docstrings ~332-335)
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflows.py`:

```python
import inspect

from mushin.workflows import (
    BaseWorkflow,
    MultiRunMetricsWorkflow,
    RobustnessCurve,
)


@pytest.mark.parametrize("cls", [BaseWorkflow, MultiRunMetricsWorkflow, RobustnessCurve])
@pytest.mark.parametrize("param", ["config_name", "job_name"])
def test_run_defaults_are_not_rai_branded(cls, param):
    default = inspect.signature(cls.run).parameters[param].default
    assert default == "mushin_workflow"
```

(If `pytest` is not already imported at the top of `tests/test_workflows.py`, add
`import pytest`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workflows.py -k rai_branded -v`
Expected: FAIL — defaults are still `"rai_workflow"`.

- [ ] **Step 3: Change the three signatures + docstrings**

In `src/mushin/workflows.py`, replace every occurrence of the defaults in the
three `run` signatures (lines ~287-288, ~612-613, ~1063-1064):

```python
        config_name: str = "mushin_workflow",
        job_name: str = "mushin_workflow",
```

And in the `BaseWorkflow.run` docstring (lines ~332 and ~335), change
`(default: "rai_workflow")` to `(default: "mushin_workflow")`.

Verify none remain:

```bash
grep -rn 'rai_workflow' src/mushin/
```

Expected: no matches.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workflows.py -k rai_branded -v && uv run pytest tests/test_workflows.py -q`
Expected: PASS. (The full-file run confirms the renamed config label doesn't break any run/round-trip test.)

- [ ] **Step 5: Commit**

```bash
git add src/mushin/workflows.py tests/test_workflows.py
git commit -m "refactor: rename rai_workflow config/job default to mushin_workflow"
```

---

## Task 5: Document the Hydra cwd footgun

**Files:**
- Modify: `docs/concepts.md`, `docs/quickstart.md`

No test (docs only). Keep the callout tight and point at `mushin.original_cwd()`.

- [ ] **Step 1: Add a callout to `docs/concepts.md`**

Find the section describing how `task()` runs (per-job working directory / Hydra
execution). Immediately after it, add:

```markdown
!!! warning "Relative paths inside `task()`"
    Each run executes inside Hydra's own per-job output directory, **not** the
    directory you launched from. A relative path like `open("data/train.csv")`
    will silently resolve against the wrong place. Anchor paths to the launch
    directory with [`mushin.original_cwd()`][mushin.original_cwd]:

    ```python
    import mushin
    path = mushin.original_cwd() / "data" / "train.csv"
    ```
```

If `docs/concepts.md` has no such section, add the callout under a new
`## Working directories` heading near the end.

- [ ] **Step 2: Add a one-line pointer to `docs/quickstart.md`**

After the first runnable sweep example, add:

```markdown
> **Heads up:** your `task()` runs in a per-job directory. If it reads or writes
> files by relative path, wrap them with `mushin.original_cwd() / "..."` — see
> [Concepts](concepts.md#working-directories).
```

- [ ] **Step 3: Build the docs to check for broken references**

Run: `uv run mkdocs build --strict`
Expected: build succeeds with no warnings. If the `[mushin.original_cwd]`
cross-reference errors, ensure `original_cwd` is documented in
`docs/reference/utils.md` (add `::: mushin.original_cwd` there if the file uses
mkdocstrings per-symbol blocks; otherwise the module block already covers it).

- [ ] **Step 4: Commit**

```bash
git add docs/concepts.md docs/quickstart.md docs/reference/utils.md
git commit -m "docs: warn about the Hydra per-job cwd and mushin.original_cwd()"
```

---

## Task 6: Changelog fragment + open the refocus PR

**Files:**
- Create: `changes/+refocus-core.changed.md` (towncrier fragment — this repo uses news fragments; see `changes/` and existing `+auto-tuning.added.md`)

- [ ] **Step 1: Write the changelog fragment**

```markdown
`import mushin` is now lightweight: the `benchmark` and `llm` subsystems load on
first use instead of at import time. All existing top-level names still resolve.
`BaseWorkflow` and `RobustnessCurve` are deprecated at the top level — import them
from `mushin.workflows`. The default Hydra config/job name is now
`mushin_workflow`, and `mushin.original_cwd()` helps anchor relative paths in
`task()`.
```

- [ ] **Step 2: Run the whole suite once more**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 3: Commit and push**

```bash
git add changes/+refocus-core.changed.md
git commit -m "changelog: refocus-core"
git push -u origin refocus-core
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "refactor: refocus mushin on the core (lazy periphery, de-RAI, cwd helper)" \
  --body "Implements docs/superpowers/specs/2026-07-13-refocus-core-design.md (items 1, 3, 4). Auto-tuning strip (item 2) and HPC closures (item 5) are tracked separately."
```

Then check CI is green and read the Codex review before merging (per repo policy).

---

## Task 7 (administrative): Close the out-of-scope HPC PRs

Not a code change — a repo decision from the spec (item 5). Preserve the branches.

- [ ] **Step 1: Close #50, #58, #59 with an explanatory comment**

```bash
for pr in 50 58 59; do
  gh pr close "$pr" --comment "Closing as part of the 'refocus on the core' decision (see docs/superpowers/specs/2026-07-13-refocus-core-design.md): multi-node DDP / FSDP / GPU-packing are HPC infrastructure orthogonal to the sweep→dataset core. HydraDDP (single-node multi-GPU) stays on main. The branch is preserved; if demand appears we'll revisit as an opt-in mushin[distributed] extra or a separate package."
done
```

- [ ] **Step 2: Verify they are closed and branches still exist**

```bash
gh pr list --state closed --json number,title,headRefName --jq '.[] | select(.number==50 or .number==58 or .number==59)'
git ls-remote --heads origin multinode-ddp hydra-fsdp gpu-packing
```

Expected: the three PRs show as closed; all three branches still listed on the remote.

---

## Self-Review

**Spec coverage:**
- Item 1 (lazy periphery + light import) → Tasks 1, 2. ✅
- Item 3 (trim top-level exports + de-RAI) → Task 2 (BaseWorkflow/RobustnessCurve), Task 4 (config/job names). ✅
- Item 4 (Hydra cwd) → Task 3 (helper) + Task 5 (docs). ✅
- Item 5 (close HPC PRs) → Task 7. ✅
- Item 2 (strip auto-tuning) → explicitly out of scope; separate plan. ✅ (noted at top)

**Placeholder scan:** The Task 1 `original_cwd` stub is intentional and replaced by
Task 3 with a real, tested implementation — flagged in both tasks, not a dangling TODO.
No other placeholders.

**Type/name consistency:** `original_cwd` / `_hydra_original_cwd` names match across
Tasks 1, 3, and 5. `_LAZY_BENCHMARK` / `_DEPRECATED` names match across Tasks 1-2.
`config_name`/`job_name` default `"mushin_workflow"` matches across Task 4 code and test.
The eager-core `__all__` list matches the spec's §3 acceptance set (plus `original_cwd`).
