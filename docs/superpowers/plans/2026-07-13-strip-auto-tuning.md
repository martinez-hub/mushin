# Strip Auto-Tuning to Find-Once-Pin + Exact Divisor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink `src/mushin/_tuning.py` (385 LOC / 15 raises / ~15 lines of real work) to a lean find-once-and-pin helper that keeps the reproducibility-valuable *fixed effective batch across hardware* feature but computes it **exactly** via divisor selection — eliminating the drift machinery, `safety_margin`, the scale-out-clamp special case, and the Lightning-callback-conflict guards.

**Architecture:** The identity `effective = device_batch × accumulate × num_devices` means, with `target_per_device = effective / num_devices`, we need `device_batch × accumulate = target_per_device`. If we pick `device_batch` = **the largest divisor of `target_per_device` that is ≤ the max batch that fits**, then `accumulate = target_per_device / device_batch` is an exact integer and the realized effective batch always equals the requested one — no drift, ever. The sidecar pin stores only the raw hardware probe (`found_max_device_batch`); `device_batch`/`accumulate` are re-derived on every call, so reuse across different GPU counts is automatic and needs no clamp logic.

**Tech Stack:** Python 3.10+, PyTorch Lightning `Tuner`, omegaconf (sidecar YAML), pytest. Run tests with `uv run pytest ...`.

**Branch:** Execute on `auto-tuning` (reshapes open PR #60). This plan doc lives on that branch.

**Design source:** conversation of 2026-07-13; supersedes the "strip entirely" framing of item 2 in `docs/superpowers/specs/2026-07-13-refocus-core-design.md`. Related: [[../specs/2026-07-13-refocus-core-design]].

---

## Design decisions (the what-stays / what-goes contract)

**Kept guards (prevent a *silent wrong result*):**
1. `effective_batch_size >= 1`
2. `effective_batch_size % num_devices == 0` (a real DDP constraint — the divisor math needs it)
3. `Tuner.scale_batch_size` returned `None`
4. pinned `found_max_device_batch` is invalid (`< 1`)
5. neither module nor datamodule exposes `batch_arg` (else `_set_attr` creates a dead attribute the loader never reads)
6. inside a Hydra `--multirun`, `pin_path` must be explicit (shared default dir would clobber pins across jobs) — unchanged `_default_pin_path` behavior
7. LR helper: module missing `lr_attr`; `lr_find` gave no suggestion; pinned LR invalid/non-finite

**Cut guards (merely pre-empt a Lightning error for a rare combination, or obviated by the divisor rule):**
- `safety_margin` param + its validation (divisor reduction already backs off the max)
- `GradientAccumulationScheduler` conflict raise
- `BatchSizeFinder` conflict raise
- `LearningRateFinder` conflict raise
- both-owners-expose-`batch_arg` ambiguity raise — **replaced** by fixing the apply-target order to match Lightning (module first, then datamodule), so we always apply to the same owner the tuner scaled
- `num_devices < 1` raise (subsumed by `max(1, ...)`)

**Removed mechanics:** the `drift` field on `BatchPin`, the drift `warnings.warn`, the `round()` "closest possible" accumulation, and the separate scale-out clamp (`min(device_batch, per_device_total)` at two sites) — all obviated by exact divisor selection.

**Pin schema change:** `{device_batch, effective_batch_size, num_devices}` → `{found_max_device_batch}` (a pure hardware probe; the effective/num_devices context is a per-call input now, not stored).

**`BatchPin` fields:** `device_batch`, `accumulate_grad_batches`, `effective_batch_size` (always the exact requested value now), `num_devices`. Drop `drift`.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/mushin/_tuning.py` | Helpers: `_largest_divisor_leq`, trimmed `tune_batch_size`/`tune_learning_rate`, `BatchPin`/`LRPin`, pin I/O | Modify (major trim) |
| `src/mushin/__init__.py` | Keep `tune_batch_size`/`tune_learning_rate` as top-level exports after the main-branch `__getattr__` rewrite | Modify (conflict resolution, Task 1) |
| `tests/test_tuning.py` | Unit tests — prune cut-guard/drift tests, add divisor/exactness tests | Modify |
| `tests/test_tuning_integration.py` | Real-`fit` tests — adapt "capped" wording to divisor | Modify |
| `docs/guides/auto-tuning.md` | Remove drift/`safety_margin` prose; explain exact divisor behavior | Modify |
| `changes/+auto-tuning.added.md` | Update the news fragment to describe the shipped (simplified) behavior | Modify |

---

## Task 1: Rebase onto main and resolve the `__init__.py` conflict

The merged refocus PR (#64) rewrote `src/mushin/__init__.py` into an eager-core + lazy `__getattr__` structure. This branch still adds `tune_batch_size`/`tune_learning_rate` to the *old* eager `__init__`. Rebase and reconcile before touching `_tuning.py`.

**Files:**
- Modify (conflict): `src/mushin/__init__.py`

- [ ] **Step 1: Rebase auto-tuning onto the latest main**

```bash
git fetch origin
git rebase origin/main
```
Expect a conflict in `src/mushin/__init__.py` (and possibly none elsewhere).

- [ ] **Step 2: Resolve `src/mushin/__init__.py`**

Take **main's** version (the `__getattr__` structure) and add the two tuning symbols to the eager core. `_tuning.py` imports only stdlib at module scope (pytorch_lightning is imported lazily *inside* the functions), so eager top-level import is cheap and correct. In the eager import block add:

```python
from ._tuning import tune_batch_size, tune_learning_rate
```

and add `"tune_batch_size"` and `"tune_learning_rate"` to `__all__` (near the other function exports). Do **not** add them to `_LAZY_BENCHMARK` or `_DEPRECATED`. Leave the `__getattr__`/`__dir__`/deprecation logic from main intact.

- [ ] **Step 3: Verify the import contract still holds**

```bash
git add src/mushin/__init__.py && git rebase --continue
uv run python -c "import sys, mushin; print(callable(mushin.tune_batch_size), callable(mushin.tune_learning_rate), 'mushin.benchmark' in sys.modules, 'mushin.llm' in sys.modules)"
```
Expected: `True True False False` (tuning symbols resolve eagerly; benchmark/llm still lazy).

- [ ] **Step 4: Green baseline**

```bash
uv run pytest tests/test_tuning.py tests/test_tuning_integration.py -q
uv run ruff check .
```
Expected: PASS with the *current* (un-simplified) tuning behavior — this is the pre-change baseline. Do not commit a separate rebase commit (the rebase rewrites history); just ensure the tree is green before Task 2.

---

## Task 2: Add the `_largest_divisor_leq` helper (pure, TDD)

**Files:**
- Modify: `src/mushin/_tuning.py`
- Test: `tests/test_tuning.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tuning.py`:

```python
import pytest

from mushin._tuning import _largest_divisor_leq


@pytest.mark.parametrize(
    "target, cap, expected",
    [
        (128, 100, 64),   # divisors of 128 <=100 -> 64
        (128, 128, 128),  # cap == target -> target itself
        (512, 600, 512),  # cap > target -> target (accumulate would be 1)
        (512, 300, 256),  # largest power-of-two divisor <=300
        (100, 7, 5),      # divisors of 100 <=7 -> 5
        (17, 4, 1),       # prime target, small cap -> 1
        (1, 1, 1),        # degenerate
        (128, 1, 1),      # cap 1 -> 1 divides everything
    ],
)
def test_largest_divisor_leq(target, cap, expected):
    d = _largest_divisor_leq(target, cap)
    assert d == expected
    assert target % d == 0  # it is always an exact divisor
    assert 1 <= d <= cap
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_tuning.py -k largest_divisor_leq -v`
Expected: FAIL — `_largest_divisor_leq` is not defined.

- [ ] **Step 3: Implement the helper**

In `src/mushin/_tuning.py`, add near the other private helpers:

```python
def _largest_divisor_leq(target: int, cap: int) -> int:
    """Largest ``d`` with ``d`` dividing ``target`` and ``1 <= d <= cap``.

    Used to pick a device batch that divides the per-device target exactly, so the
    realized effective batch equals the requested one with no drift. ``d == 1``
    always divides ``target``, so a value in ``[1, cap]`` always exists. For the
    round batch sizes researchers use (256/512/1024, many divisors) this returns a
    value at or just below ``cap``; only pathological (near-prime) targets fall far.
    """
    d = min(int(target), int(cap))
    while target % d != 0:
        d -= 1
    return d
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_tuning.py -k largest_divisor_leq -v`
Expected: PASS (8 cases).

- [ ] **Step 5: Commit**

```bash
git add src/mushin/_tuning.py tests/test_tuning.py
git commit -m "feat(tuning): add _largest_divisor_leq for exact effective-batch selection"
```

---

## Task 3: Rewrite `tune_batch_size` around exact divisor selection

**Files:**
- Modify: `src/mushin/_tuning.py`
- Test: `tests/test_tuning.py`

- [ ] **Step 1: Write the new/updated tests first**

Add these to `tests/test_tuning.py` (they encode the exact-divisor contract; they will fail against the current drift-based implementation). Reuse the existing `_DM`/module fakes already in the file where possible; the snippet below defines a minimal local fake to be self-contained:

```python
from mushin._tuning import BatchPin, tune_batch_size


class _Owner:
    """Minimal batch-size owner (module or datamodule stand-in)."""

    def __init__(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size


class _FakeTrainer:
    def __init__(self):
        self.num_devices = 1
        self.num_nodes = 1
        self.default_root_dir = None
        self.accumulate_grad_batches = 1
        self.callbacks = []


def _patch_scale(monkeypatch, found_max):
    """Make Tuner.scale_batch_size return a fixed found_max without real training."""
    from pytorch_lightning.tuner.tuning import Tuner

    monkeypatch.setattr(
        Tuner, "scale_batch_size", lambda self, *a, **k: found_max, raising=True
    )


def test_batch_exact_via_divisor_when_max_not_a_divisor(monkeypatch, tmp_path):
    _patch_scale(monkeypatch, found_max=100)
    trainer = _FakeTrainer()
    trainer.num_devices = 4  # -> per_device_total = 512/4 = 128
    module = _Owner(batch_size=1)
    pin = tune_batch_size(
        trainer, module, effective_batch_size=512,
        pin_path=tmp_path / "pin.yaml",
    )
    assert pin.device_batch == 64          # largest divisor of 128 <= 100
    assert pin.accumulate_grad_batches == 2
    assert pin.num_devices == 4
    # ALWAYS exact — no drift field, effective equals the request.
    assert pin.effective_batch_size == 512
    assert not hasattr(pin, "drift")
    assert module.batch_size == 64
    assert trainer.accumulate_grad_batches == 2


@pytest.mark.parametrize(
    "effective, num_devices, found_max",
    [(512, 1, 300), (512, 4, 100), (256, 2, 50), (1024, 8, 33), (128, 1, 128)],
)
def test_batch_effective_is_always_exact(monkeypatch, tmp_path, effective, num_devices, found_max):
    _patch_scale(monkeypatch, found_max=found_max)
    trainer = _FakeTrainer()
    trainer.num_devices = num_devices
    module = _Owner(batch_size=1)
    pin = tune_batch_size(
        trainer, module, effective_batch_size=effective,
        pin_path=tmp_path / "pin.yaml",
    )
    assert pin.device_batch * pin.accumulate_grad_batches * pin.num_devices == effective
    assert pin.device_batch <= found_max


def test_batch_pin_stores_found_max_and_rederives_on_reuse(monkeypatch, tmp_path):
    # First call on 1 device pins the raw hardware probe (found_max=200).
    _patch_scale(monkeypatch, found_max=200)
    pin_path = tmp_path / "pin.yaml"
    t1 = _FakeTrainer()
    tune_batch_size(t1, _Owner(batch_size=1), effective_batch_size=512, pin_path=pin_path)

    from mushin._tuning import _read_pin

    stored = _read_pin(pin_path)
    assert stored == {"found_max_device_batch": 200}

    # Reuse on 4 devices: NO new search (scale would raise if called), re-derive
    # from the pinned found_max against the new per-device target (512/4 = 128).
    def _boom(self, *a, **k):  # pragma: no cover - must not be called
        raise AssertionError("scale_batch_size must not run on pin reuse")

    from pytorch_lightning.tuner.tuning import Tuner

    monkeypatch.setattr(Tuner, "scale_batch_size", _boom, raising=True)
    t4 = _FakeTrainer()
    t4.num_devices = 4
    m4 = _Owner(batch_size=1)
    pin = tune_batch_size(t4, m4, effective_batch_size=512, pin_path=pin_path)
    assert pin.device_batch == 128  # largest divisor of 128 <= 200
    assert pin.accumulate_grad_batches == 1
    assert pin.effective_batch_size == 512


def test_batch_poor_utilization_warns(monkeypatch, tmp_path):
    # target_per_device = 130 (=2*5*13); found_max 64 -> largest divisor is 26,
    # far below 64 -> warn about wasted headroom.
    _patch_scale(monkeypatch, found_max=64)
    trainer = _FakeTrainer()
    module = _Owner(batch_size=1)
    with pytest.warns(UserWarning, match="below the .* that fits"):
        pin = tune_batch_size(
            trainer, module, effective_batch_size=130, pin_path=tmp_path / "pin.yaml"
        )
    assert pin.effective_batch_size == 130  # still exact


def test_batch_effective_not_divisible_by_num_devices_raises(tmp_path):
    trainer = _FakeTrainer()
    trainer.num_devices = 3
    with pytest.raises(ValueError, match="divisible by num_devices"):
        tune_batch_size(
            trainer, _Owner(batch_size=1), effective_batch_size=512,
            pin_path=tmp_path / "pin.yaml",
        )
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_tuning.py -k "divisor or always_exact or found_max_and_rederives or poor_utilization or not_divisible_by_num_devices" -v`
Expected: FAIL (current impl still has `drift`, stores the old pin schema, has `safety_margin`, and does not warn on poor utilization).

- [ ] **Step 3: Replace `BatchPin` and `tune_batch_size`**

In `src/mushin/_tuning.py`, replace the `BatchPin` dataclass with:

```python
@dataclass(frozen=True)
class BatchPin:
    """Result of :func:`tune_batch_size`. The effective batch is always exact."""

    device_batch: int
    accumulate_grad_batches: int
    effective_batch_size: int  # always == the requested value (exact divisor selection)
    num_devices: int
```

Replace the entire `tune_batch_size` function with:

```python
def tune_batch_size(
    trainer,
    module,
    datamodule=None,
    *,
    effective_batch_size: int,
    pin_path=None,
    num_devices: int | None = None,
    batch_arg: str = "batch_size",
    retune: bool = False,
    **scale_kwargs,
) -> BatchPin:
    """Pin the effective batch; find the largest fitting device batch, set accumulation.

    Runs ``Tuner.scale_batch_size`` once to find the largest device batch that
    fits (``found_max``), then chooses ``device_batch`` = the largest divisor of
    ``effective_batch_size / num_devices`` that is ``<= found_max`` and sets
    ``trainer.accumulate_grad_batches`` accordingly. Because ``device_batch``
    divides the per-device target exactly, the realized effective batch always
    equals ``effective_batch_size`` on any hardware — no drift.

    ``found_max`` (the raw hardware probe) is written to ``pin_path``; a later run
    reads it and skips the search, re-deriving ``device_batch``/accumulation for
    that run's ``effective_batch_size``/``num_devices``. ``retune=True`` forces a
    fresh search.

    Parameters
    ----------
    effective_batch_size : int
        The pinned, hardware-independent quantity
        ``device_batch * accumulate_grad_batches * num_devices``. Must be
        divisible by ``num_devices``.
    pin_path : str, Path, or None
        Sidecar YAML storing ``found_max_device_batch``. Defaults to
        ``<trainer.default_root_dir>/mushin_batch_pin.yaml``. Inside a Hydra
        ``--multirun`` an explicit ``pin_path`` is required (the default dir is
        shared across jobs).
    num_devices : int or None
        Defaults to ``trainer.num_devices * trainer.num_nodes``.
    batch_arg : str
        Attribute on the module (or datamodule) the tuner scales and this helper
        sets; forwarded to the tuner as ``batch_arg_name``.
    retune : bool
        Ignore any existing pin and search again.
    **scale_kwargs
        Forwarded to ``Tuner.scale_batch_size`` (e.g. ``mode``, ``steps_per_trial``).

    Returns
    -------
    BatchPin
    """
    from pytorch_lightning.tuner.tuning import Tuner

    if effective_batch_size < 1:
        raise ValueError(
            f"effective_batch_size must be >= 1; got {effective_batch_size}"
        )
    if num_devices is None:
        per_node = int(getattr(trainer, "num_devices", 1) or 1)
        nodes = int(getattr(trainer, "num_nodes", 1) or 1)
        num_devices = max(1, per_node * nodes)
    if effective_batch_size % num_devices != 0:
        raise ValueError(
            f"effective_batch_size={effective_batch_size} must be divisible by "
            f"num_devices={num_devices}; choose an effective batch that divides evenly."
        )
    per_device_total = effective_batch_size // num_devices

    # Validate the batch owner up front so we never write a pin for a call that
    # would then apply the value to a dead attribute. Apply to the SAME owner
    # Lightning's finder scales (module first, then datamodule).
    module_has = _has_attr(module, batch_arg)
    dm_has = datamodule is not None and _has_attr(datamodule, batch_arg)
    if not module_has and not dm_has:
        raise ValueError(
            f"neither the module nor the datamodule exposes '{batch_arg}', so the "
            "tuned batch size cannot be applied to anything the dataloader reads. "
            f"Expose '{batch_arg}' on your module (or datamodule), or set batch_arg=."
        )

    if pin_path is None:
        pin_path = _default_pin_path(trainer, "mushin_batch_pin.yaml")

    pin = None if retune else _read_pin(pin_path)
    if pin is not None:
        found_max = int(pin["found_max_device_batch"])
        if found_max < 1:
            raise ValueError(
                f"pin file {pin_path} has an invalid found_max_device_batch="
                f"{found_max} (must be >= 1); delete it or pass retune=True."
            )
    else:
        found_max = Tuner(trainer).scale_batch_size(
            module, datamodule=datamodule, batch_arg_name=batch_arg, **scale_kwargs
        )
        if found_max is None:
            raise RuntimeError(
                "tune_batch_size: Tuner.scale_batch_size returned no batch size. "
                f"Check that the model or datamodule exposes the '{batch_arg}' "
                "attribute, or pass an explicit pin file."
            )
        found_max = int(found_max)
        _write_pin(pin_path, {"found_max_device_batch": found_max})

    device_batch = _largest_divisor_leq(per_device_total, found_max)
    accumulate = per_device_total // device_batch  # exact: device_batch divides target
    if device_batch < found_max and device_batch * 2 <= found_max:
        warnings.warn(
            f"tune_batch_size: chosen device_batch={device_batch} is well below the "
            f"{found_max} that fits, because effective_batch_size={effective_batch_size} "
            f"(per-device target {per_device_total}) has no larger divisor that fits. "
            "Pick a rounder effective_batch_size (more divisors) for better GPU use.",
            UserWarning,
            stacklevel=2,
        )

    target = module if module_has else datamodule
    _set_attr(target, batch_arg, device_batch)
    trainer.accumulate_grad_batches = accumulate

    return BatchPin(
        device_batch=device_batch,
        accumulate_grad_batches=accumulate,
        effective_batch_size=effective_batch_size,
        num_devices=num_devices,
    )
```

Also remove the now-unused `math` import if nothing else uses it (LR still uses `math.isfinite` — keep `import math`).

- [ ] **Step 4: Remove the obsolete `tune_batch_size` tests**

Delete these test functions from `tests/test_tuning.py` (they exercise removed behavior):
- `test_batch_drift_warns_when_not_divisible` (drift removed)
- `test_batch_safety_margin_backs_off_found_max` (`safety_margin` removed)
- `test_batch_pin_context_mismatch_warns` (pin no longer stores context)
- `test_batch_pin_clamped_on_scale_out` (replaced by `test_batch_pin_stores_found_max_and_rederives_on_reuse`)
- `test_batch_rejects_ambiguous_owners` (guard cut; target order fix makes it moot)
- `test_batch_rejects_existing_accumulation_scheduler` (guard cut)
- `test_batch_rejects_existing_batch_size_finder` (guard cut)
- `test_batch_pin_invalid_device_batch_raises` (pin key renamed — its replacement is covered by the new reuse test; if you want an explicit invalid-pin test, rename it to read `found_max_device_batch: 0` and assert the new message)

Update these to the new schema/fields:
- `test_batchpin_and_lrpin_are_frozen_dataclasses` — assert `BatchPin` has fields `{device_batch, accumulate_grad_batches, effective_batch_size, num_devices}` and NOT `drift`.
- `test_batch_pin_roundtrip_skips_search` — the pinned dict is now `{"found_max_device_batch": ...}`.
- `test_batch_invalid_inputs` — drop any `safety_margin=` cases; keep `effective_batch_size<1` and non-divisible cases.
- `test_batch_accumulation_clean_divisor`, `test_batch_exact_when_max_meets_target`, `test_batch_exact_hparams_updated`, `test_batch_hparams_only_datamodule_is_target`, `test_batch_num_devices_divides_effective`, `test_batch_num_devices_multiplies_nodes`, `test_batch_none_from_tuner_raises`, `test_batch_pin_no_owner_raises`, `test_batch_no_pin_written_when_owner_invalid` — keep; adjust any assertions that referenced `drift`, `safety_margin`, or the old pin keys.

- [ ] **Step 5: Run the batch-size tests green**

Run: `uv run pytest tests/test_tuning.py -k batch -v`
Expected: PASS. Investigate any red before moving on.

- [ ] **Step 6: Commit**

```bash
git add src/mushin/_tuning.py tests/test_tuning.py
git commit -m "refactor(tuning): exact divisor selection; drop drift/safety_margin/clamp + callback guards"
```

---

## Task 4: Trim `tune_learning_rate` guards

**Files:**
- Modify: `src/mushin/_tuning.py`
- Test: `tests/test_tuning.py`

- [ ] **Step 1: Delete the cut-guard test**

Remove `test_lr_rejects_existing_lr_finder` from `tests/test_tuning.py`.

- [ ] **Step 2: Add a test asserting the LR-finder callback is NO LONGER rejected**

```python
def test_lr_allows_existing_lr_finder_callback(monkeypatch, tmp_path):
    """The LearningRateFinder-conflict guard was removed; a pre-existing callback
    must not raise. (We still pin/apply our own found value.)"""
    from pytorch_lightning.callbacks import LearningRateFinder
    from pytorch_lightning.tuner.tuning import Tuner

    class _Sugg:
        def suggestion(self):
            return 0.01

    monkeypatch.setattr(Tuner, "lr_find", lambda self, *a, **k: _Sugg(), raising=True)

    class _Mod:
        def __init__(self):
            self.lr = 0.1

    trainer = _FakeTrainer()
    trainer.callbacks = [LearningRateFinder()]
    module = _Mod()
    pin = tune_learning_rate(trainer, module, pin_path=tmp_path / "lr.yaml")
    assert pin.learning_rate == 0.01
    assert module.lr == 0.01
```

(Import `tune_learning_rate` at the top of the test file alongside `tune_batch_size`.)

- [ ] **Step 3: Run to verify the new test fails**

Run: `uv run pytest tests/test_tuning.py -k lr_allows_existing -v`
Expected: FAIL — the current code raises `ValueError` on a pre-existing `LearningRateFinder`.

- [ ] **Step 4: Remove the guard from `tune_learning_rate`**

In `src/mushin/_tuning.py`, delete the `LearningRateFinder` import and the block that raises when one is present (the `if any(isinstance(cb, LearningRateFinder) ...)` guard). Keep everything else in `tune_learning_rate` unchanged (owner check, `lr_find` None → RuntimeError, invalid/non-finite pin, `_set_attr`, pin roundtrip).

- [ ] **Step 5: Run the LR tests green**

Run: `uv run pytest tests/test_tuning.py -k lr -v`
Expected: PASS (including the kept `test_lr_*` tests and the new allow test).

- [ ] **Step 6: Commit**

```bash
git add src/mushin/_tuning.py tests/test_tuning.py
git commit -m "refactor(tuning): drop LearningRateFinder-conflict guard"
```

---

## Task 5: Update the integration tests

**Files:**
- Modify: `tests/test_tuning_integration.py`

These run a real `Tuner`/`fit`. Behavior is unchanged for the common exact case, but the "capped at per-device target" wording/asserts must become "exact divisor".

- [ ] **Step 1: Read and adjust**

Read `tests/test_tuning_integration.py`. For each of `test_applied_accumulation_and_device_batch_take_effect`, `test_real_tuner_fit_uses_capped_batch`, `test_real_fit_uses_applied_learning_rate`:
- Ensure the chosen `effective_batch_size` and the datamodule's fitting batch make the expected `device_batch` a clean divisor of `per_device_total` (pick round numbers, e.g. effective 16 with per-device target 16 and a found max ≥ some power of two), so the assertions match exact divisor selection.
- Replace any assertion on `pin.drift` or "capped" semantics with the exact identity `pin.device_batch * pin.accumulate_grad_batches * pin.num_devices == effective_batch_size`.
- Remove any `safety_margin=` argument.
- Rename `test_real_tuner_fit_uses_capped_batch` → `test_real_tuner_fit_uses_divisor_batch` for accuracy.

- [ ] **Step 2: Run the integration tests green**

Run: `uv run pytest tests/test_tuning_integration.py -v`
Expected: PASS. These are slower (real fit) — allow a minute.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tuning_integration.py
git commit -m "test(tuning): integration tests assert exact divisor identity"
```

---

## Task 6: Update the guide and changelog

**Files:**
- Modify: `docs/guides/auto-tuning.md`
- Modify: `changes/+auto-tuning.added.md`

- [ ] **Step 1: Rewrite the `tune_batch_size` section of `docs/guides/auto-tuning.md`**

Replace the paragraph that says the realized effective batch "can differ slightly from the target … records the actual value … and warns when it drifts" with an accurate description:

```markdown
The helper finds the largest device batch that fits, then reduces it to the
largest value that **divides the per-device target exactly**, so the realized
effective batch always equals your target on any hardware — there is no drift.
The raw hardware probe (the largest batch that fit) is written to
`<trainer.default_root_dir>/mushin_batch_pin.yaml` (override with `pin_path=`);
commit it to make re-runs deterministic. On a later run the probe is read, the
search is skipped, and `device_batch`/accumulation are re-derived for that run's
`effective_batch_size`/`num_devices` — so the same pin works unchanged across
different GPU counts. Pass `retune=True` to search again.

Pick a rounder `effective_batch_size` (256/512/1024 — many divisors) for the best
GPU utilization; a near-prime target may force a small device batch, and the
helper warns when that happens.
```

Remove the sentence documenting `safety_margin=` (the parameter no longer exists); keep the `num_devices=` note. Leave the `tune_learning_rate` and Caveats sections as-is (the multirun/`pin_path` caveat still holds).

- [ ] **Step 2: Update the changelog fragment**

Rewrite `changes/+auto-tuning.added.md` to describe the shipped behavior (find-once-and-pin with exact divisor selection), e.g.:

```markdown
`tune_batch_size` / `tune_learning_rate`: opt-in, reproducibility-preserving
auto-tuning. Lightning's batch/LR finder runs once, the result is pinned to a
sidecar YAML, and later runs reuse it. `tune_batch_size` pins a hardware-
independent effective batch, choosing the largest device batch that both fits and
divides the per-device target exactly, so the effective batch is identical on any
GPU count with no drift.
```

- [ ] **Step 3: Build docs strictly**

Run: `uv run --group docs mkdocs build --strict`
Expected: clean build (ignore the unrelated Material announcement banner).

- [ ] **Step 4: Commit**

```bash
git add docs/guides/auto-tuning.md changes/+auto-tuning.added.md
git commit -m "docs(tuning): describe exact-divisor behavior; drop drift/safety_margin"
```

---

## Task 7: Full verification and update PR #60

- [ ] **Step 1: Full suite + lint + format**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```
Expected: all PASS. (The format check is what tripped CI on the refocus PR — run it here before pushing.)

- [ ] **Step 2: Confirm the size reduction**

```bash
wc -l src/mushin/_tuning.py
grep -c 'raise ' src/mushin/_tuning.py
```
Expected: `_tuning.py` roughly ~200 LOC (down from 385) and ~8 `raise`s (down from 15). These are sanity checks, not hard gates.

- [ ] **Step 3: Force-push the reshaped branch and update the PR body**

```bash
git push --force-with-lease origin auto-tuning
```
Then update PR #60's description to note the simplification: exact divisor selection replaces the drift/`safety_margin`/scale-out machinery; the callback-conflict guards were dropped; the fixed-effective-batch feature is retained. Check CI is green and read any Codex review before merging (Codex may be out of credits).

---

## Self-Review

**Spec/design coverage:**
- Divisor selection (exact, no drift) → Tasks 2, 3. ✅
- Drop `safety_margin`, drift field/warning, scale-out clamp, `round()` → Task 3. ✅
- Drop callback-conflict guards (GradAccumScheduler, BatchSizeFinder, LearningRateFinder) + both-owner ambiguity (via target-order fix) → Tasks 3, 4. ✅
- Keep genuine guards (effective≥1, divisibility, tuner-None, invalid pin, neither-owner, multirun pin_path, LR owner/None/invalid) → Tasks 3, 4 (kept intact). ✅
- Pin schema → `found_max_device_batch`; reuse re-derives across GPU counts → Task 3 (`test_batch_pin_stores_found_max_and_rederives_on_reuse`). ✅
- `__init__.py` reconciliation after #64 → Task 1. ✅
- Docs + changelog → Task 6. ✅
- Verification incl. `ruff format --check` → Task 7. ✅

**Placeholder scan:** none — every code step shows the full code; test remove/keep lists name exact functions.

**Type/name consistency:** `_largest_divisor_leq(target, cap)` used identically in Tasks 1–3. `found_max_device_batch` pin key consistent across impl (Task 3) and tests (`test_batch_pin_stores_found_max_and_rederives_on_reuse`, roundtrip). `BatchPin` fields (no `drift`) consistent across Task 3 impl and the frozen-dataclass test. `_FakeTrainer`/`_Owner`/`_patch_scale` helpers introduced in Task 3 Step 1 and reused in Task 4.

**Open judgment calls flagged for the reviewer/user:** the guard keep/cut split is a deliberate line ("prevent silent wrong result" = keep; "pre-empt a rare Lightning error" = cut). If the user wants any cut guard retained (e.g. the `GradientAccumulationScheduler` conflict), it is a one-block re-add and does not affect the divisor core.
