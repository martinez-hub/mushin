from mushin._sweep_io import (
    Manifest,
    combo_key,
    read_metrics_sidecar,
    write_metrics_sidecar,
)


def test_combo_key_is_canonical_and_order_stable():
    assert combo_key({"lr": 0.1, "seed": 2}) == combo_key({"seed": 2, "lr": 0.1})
    assert combo_key({"lr": 0.1, "seed": 2}) == "lr=0.1,seed=2"


def test_metrics_sidecar_roundtrip(tmp_path):
    write_metrics_sidecar(tmp_path, {"accuracy": 0.9, "loss": 0.1})
    assert read_metrics_sidecar(tmp_path) == {"accuracy": 0.9, "loss": 0.1}
    assert (tmp_path / "mushin_metrics.json").exists()
    assert read_metrics_sidecar(tmp_path / "nope") is None  # absent -> None


def test_read_metrics_sidecar_corrupt_returns_none(tmp_path):
    # A corrupt (non-JSON) sidecar must be treated like a missing one (None),
    # not raise, so a resume can proceed and simply re-run that cell.
    (tmp_path / "mushin_metrics.json").write_text("{not: valid json,,,")
    assert read_metrics_sidecar(tmp_path) is None


def test_metrics_sidecar_coerces_numpy_and_tensors(tmp_path):
    import numpy as np

    write_metrics_sidecar(tmp_path, {"a": np.float32(0.5), "b": np.array([1, 2])})
    got = read_metrics_sidecar(tmp_path)
    assert got == {"a": 0.5, "b": [1, 2]}


def test_manifest_tracks_and_replaces_cells(tmp_path):
    m = Manifest.load_or_new(tmp_path, params=["lr", "seed"])
    m.mark({"lr": 0.1, "seed": 0}, dir="0", status="completed")
    m.mark({"lr": 0.1, "seed": 2}, dir="5", status="failed", error="OOM")
    m.save()

    m2 = Manifest.load_or_new(tmp_path, params=["lr", "seed"])
    assert m2.status({"lr": 0.1, "seed": 0}) == "completed"
    assert m2.status({"lr": 0.1, "seed": 2}) == "failed"
    assert m2.status({"lr": 1.0, "seed": 0}) == "pending"  # unseen -> pending
    # a re-run REPLACES in place (no duplicate entry)
    m2.mark({"lr": 0.1, "seed": 2}, dir="9", status="completed")
    assert m2.status({"lr": 0.1, "seed": 2}) == "completed"
    assert m2.dir({"lr": 0.1, "seed": 2}) == "9"
    assert m2.failed_cells() == []
    m2.mark({"lr": 1.0, "seed": 1}, dir="3", status="failed", error="boom")
    assert not m2.is_complete()
    assert {"lr=1.0,seed=1"} == {c["key"] for c in m2.failed_cells()}


def test_manifest_load_or_new_corrupt_returns_fresh(tmp_path):
    """A manifest truncated by a mid-write kill must not make the sweep
    un-resumable: treat it like a missing manifest (the per-cell status
    sidecars are the durable source of truth)."""
    from mushin._sweep_io import MANIFEST_FILE, Manifest

    (tmp_path / MANIFEST_FILE).write_text('{"schema": 1, "cells": {tru')
    m = Manifest.load_or_new(tmp_path, ["a", "b"])
    assert m.cells == {}
    assert m.params == ["a", "b"]


def test_from_cell_status_survives_corrupt_manifest(tmp_path):
    from mushin._resume import write_cell_status
    from mushin._sweep_io import MANIFEST_FILE, Manifest, combo_key

    cell = tmp_path / "0"
    cell.mkdir()
    write_cell_status(cell, status="completed", combo={"a": 1}, attempt=1)
    (tmp_path / MANIFEST_FILE).write_text("not json at all")
    m = Manifest.from_cell_status(tmp_path, ["a"])
    assert m.cells[combo_key({"a": 1})]["status"] == "completed"


def test_atomic_write_uses_unique_temp_names(tmp_path, monkeypatch):
    """Two writers of the same file must not share a temp name -- a fixed
    <path>.tmp lets concurrent processes truncate each other mid-write and
    rename half-written content over the target."""
    from pathlib import Path

    from mushin._sweep_io import _atomic_write_json

    seen = []
    orig = Path.replace

    def spy(self, target):
        seen.append(self.name)
        return orig(self, target)

    monkeypatch.setattr(Path, "replace", spy)
    _atomic_write_json(tmp_path / "m.json", {"a": 1})
    _atomic_write_json(tmp_path / "m.json", {"a": 2})
    assert len(seen) == 2 and seen[0] != seen[1]
    assert not list(tmp_path.glob("*.tmp"))  # no residue


def test_metrics_sidecar_roundtrips_infinities(tmp_path):
    """±Inf metric values must survive the sidecar round-trip with their sign
    (not collapse to NaN), including inside nested containers, while the file
    stays strict JSON (no Infinity/NaN literals)."""
    import json
    import math

    from mushin._sweep_io import (
        METRICS_FILE,
        read_metrics_sidecar,
        write_metrics_sidecar,
    )

    write_metrics_sidecar(
        tmp_path,
        {
            "pos": float("inf"),
            "neg": float("-inf"),
            "nan": float("nan"),
            "nested": {"v": [1.0, float("inf")]},
        },
    )
    back = read_metrics_sidecar(tmp_path)
    assert back is not None
    assert back["pos"] == math.inf
    assert back["neg"] == -math.inf
    assert back["nan"] != back["nan"]  # NaN still round-trips to NaN
    assert back["nested"]["v"] == [1.0, math.inf]

    def _reject_constant(name):  # Infinity/NaN literals are not strict JSON
        raise AssertionError(f"non-strict JSON literal in sidecar: {name}")

    json.loads((tmp_path / METRICS_FILE).read_text(), parse_constant=_reject_constant)


def test_user_dict_resembling_nonfinite_tag_is_not_corrupted(tmp_path):
    """Only the exact internal marker values decode to ±Inf — a user metric
    dict that merely shares the tag key must round-trip unchanged."""
    from mushin._sweep_io import read_metrics_sidecar, write_metrics_sidecar

    write_metrics_sidecar(tmp_path, {"weird": {"__mushin_nonfinite__": "hello"}})
    back = read_metrics_sidecar(tmp_path)
    assert back == {"weird": {"__mushin_nonfinite__": "hello"}}


# Option names that YAML 1.1 would not read as strings but OmegaConf does: it
# drops the date resolver. Quoted ones are how Hydra writes a name that looks
# like a number.
_STRING_CHOICES_YAML = """\
hydra:
  job:
    num: 7
  runtime:
    choices:
      date: 2024-01-01
      quoted_sci: '1e3'
      quoted_bool: 'yes'
      plain: adam
      unset: null
"""

# Unquoted scalars that YAML 1.1 and OmegaConf type differently (OmegaConf reads
# `1e3` as a float), or that are not strings at all. Hydra never writes these as
# option names; a hand-edited file goes to OmegaConf.
_NON_STRING_CHOICES_YAML = """\
hydra:
  job:
    num: 7
  runtime:
    choices:
      sci: 1e3
      dotted: 1.5
      under: 1_000
      hex: 0x1f
      sexagesimal: 1:30
      yes_word: yes
      on_word: on
      inf: .inf
      nan: .nan
      plain: adam
"""


def _forbid_omegaconf(monkeypatch):
    """Make the OmegaConf fallback of read_hydra_job fail, so a test proves the
    fast path produced the result. Skips where PyYAML has no libyaml, which
    has no fast path."""
    import hydra_zen
    import pytest

    from mushin import _sweep_io

    if _sweep_io.CParser is None:
        pytest.skip("PyYAML built without libyaml: reloads take OmegaConf's path")

    def _slow(path):
        raise AssertionError(f"read_hydra_job fell back to OmegaConf for {path}")

    monkeypatch.setattr(hydra_zen, "load_from_yaml", _slow)


def test_read_hydra_job_types_values_as_omegaconf_does(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from mushin._sweep_io import read_hydra_job

    p = tmp_path / "hydra.yaml"
    p.write_text(_STRING_CHOICES_YAML, encoding="utf-8")
    ref = OmegaConf.load(p).hydra
    expected = (ref.job.num, dict(ref.runtime.choices))

    _forbid_omegaconf(monkeypatch)
    # repr compares the types too: the date stays the string '2024-01-01'.
    assert repr(read_hydra_job(p)) == repr(expected)


def test_read_hydra_job_reads_a_real_sweep_without_omegaconf(tmp_path, monkeypatch):
    """Hydra's own hydra.yaml (interpolations, `???` fields) must parse on the
    fast path; a silent fallback on every cell would lose the speedup."""
    from omegaconf import OmegaConf

    from mushin import multirun
    from mushin._sweep_io import read_hydra_job
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    W().run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    files = sorted((tmp_path / "s").glob("*/.hydra/hydra.yaml"))
    assert len(files) == 2
    expected = []
    for f in files:
        ref = OmegaConf.load(f).hydra
        expected.append((ref.job.num, dict(ref.runtime.choices)))

    _forbid_omegaconf(monkeypatch)
    got = [read_hydra_job(f) for f in files]
    assert got == expected
    assert sorted(num for num, _ in got) == [0, 1]
    assert all(choices for _, choices in got)  # Hydra records its own groups


def test_read_hydra_job_falls_back_to_omegaconf(tmp_path, monkeypatch):
    """What the plain read cannot judge goes to OmegaConf, which resolves it or
    raises as it always did."""
    import pytest
    from omegaconf.errors import MissingMandatoryValue

    from mushin import _sweep_io
    from mushin._sweep_io import read_hydra_job

    interp = tmp_path / "interp.yaml"
    interp.write_text("base: 3\nhydra:\n  job:\n    num: ${base}\n", encoding="utf-8")
    assert read_hydra_job(interp) == (3, None)  # resolved; no choices recorded

    choice = tmp_path / "choice.yaml"
    choice.write_text(
        "db_name: mysql\nhydra:\n  job:\n    num: 0\n"
        "  runtime:\n    choices:\n      db: ${db_name}\n",
        encoding="utf-8",
    )
    assert read_hydra_job(choice) == (0, {"db": "mysql"})

    missing = tmp_path / "missing.yaml"
    missing.write_text("hydra:\n  job:\n    num: ???\n", encoding="utf-8")
    with pytest.raises(MissingMandatoryValue):
        read_hydra_job(missing)

    with pytest.raises(FileNotFoundError):
        read_hydra_job(tmp_path / "absent.yaml")

    # Without a plain loader (OmegaConf's private helper moved), the result
    # is the same, only slower.
    p = tmp_path / "hydra.yaml"
    p.write_text(_STRING_CHOICES_YAML, encoding="utf-8")
    fast = read_hydra_job(p)
    monkeypatch.setattr(_sweep_io, "_plain_yaml_loader", lambda: None)
    assert repr(read_hydra_job(p)) == repr(fast)


def test_plain_yaml_loader_parses_in_c_and_composes_in_python():
    """libyaml parses; PyYAML's Python composer builds the tree, so deep
    nesting raises RecursionError instead of crashing (see below)."""
    import pytest
    import yaml
    from yaml.composer import Composer

    from mushin._sweep_io import _plain_yaml_loader

    if not yaml.__with_libyaml__:
        pytest.skip("PyYAML built without libyaml: reloads take OmegaConf's path")
    from yaml.cyaml import CParser

    loader = _plain_yaml_loader()
    assert loader is not None
    assert issubclass(loader, CParser)
    assert loader.compose_node is Composer.compose_node
    assert not issubclass(loader, yaml.CSafeLoader)


def test_read_hydra_job_survives_deep_nesting(tmp_path):
    """libyaml's C composer recurses on the C stack with no limit, so a deeply
    nested file killed the process (SIGSEGV) past any `except`. It must raise
    RecursionError, as OmegaConf.load does. A subprocess keeps a regression
    from taking pytest down with it."""
    import subprocess
    import sys

    p = tmp_path / "hydra.yaml"
    p.write_text(
        "hydra:\n  job:\n    num: 0\n  junk: " + "[" * 100_000 + "]" * 100_000 + "\n",
        encoding="utf-8",
    )
    code = (
        "import sys\n"
        "from mushin._sweep_io import read_hydra_job\n"
        "try:\n"
        "    read_hydra_job(sys.argv[1])\n"
        "except RecursionError:\n"
        "    print('RecursionError')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code, str(p)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
    assert r.stdout.strip() == "RecursionError"


def _read_as_before(path):
    """What load_from_dir and the MCP server did before the plain read: a full
    OmegaConf load, then the two fields."""
    from hydra_zen import load_from_yaml

    hc = load_from_yaml(path)
    num = hc.hydra.job.num
    try:
        ch = hc.hydra.runtime.choices
        choices = None if ch is None else dict(ch)
    except Exception:  # noqa: BLE001 - as the old code did
        choices = None
    return num, choices


def _outcome(read, path):
    try:
        return "returns", repr(read(path))
    except Exception as e:  # noqa: BLE001 - the error type is the outcome
        return "raises", type(e).__name__


_SAME_AS_BEFORE = {
    "non-string options": _NON_STRING_CHOICES_YAML,
    "empty file": "",
    "no hydra node": "x: 1\n",
    "missing job number": "hydra:\n  job:\n    name: a\n",
    "??? job number": "hydra:\n  job:\n    num: ???\n",
    "bool job number": "hydra:\n  job:\n    num: true\n",
    "duplicate key in the job": "hydra:\n  job:\n    num: 1\n    num: 5\n",
    "duplicate key elsewhere": "hydra:\n  job:\n    num: 1\n  run:\n    dir: a\n    dir: b\n",
    "??? choice": "hydra:\n  job:\n    num: 0\n  runtime:\n    choices:\n      db: ???\n",
    "interpolated choice": (
        "name: mysql\nhydra:\n  job:\n    num: 0\n"
        "  runtime:\n    choices:\n      db: ${name}\n"
    ),
    "interpolated choices node": (
        "x:\n  db: mysql\nhydra:\n  job:\n    num: 1\n  runtime:\n    choices: ${x}\n"
    ),
    "interpolated runtime node": (
        "x:\n  choices:\n    db: mysql\nhydra:\n  job:\n    num: 1\n  runtime: ${x}\n"
    ),
    "null runtime": "hydra:\n  job:\n    num: 2\n  runtime: null\n",
    "null choices": "hydra:\n  job:\n    num: 2\n  runtime:\n    choices: null\n",
    "null option": "hydra:\n  job:\n    num: 2\n  runtime:\n    choices:\n      db: null\n",
    "null group name": "hydra:\n  job:\n    num: 0\n  runtime:\n    choices:\n      null: a\n",
    "nested choice": "hydra:\n  job:\n    num: 0\n  runtime:\n    choices:\n      db: {a: 1}\n",
    "pathlib tag": (
        "hydra:\n  job:\n    num: 3\n  runtime:\n"
        "    cwd: !!python/object/apply:pathlib.Path\n    - /tmp\n"
    ),
    # Fields a reload never reads: OmegaConf.load still checks them.
    "malformed interpolation elsewhere": "hydra:\n  job:\n    num: 4\n    name: ${broken\n",
    "malformed interpolation in a list": (
        "hydra:\n  job:\n    num: 4\n  overrides:\n    task:\n    - a=${\n"
    ),
    "valid interpolations elsewhere": (
        "hydra:\n  job:\n    num: 4\n    name: ${x}_${oc.env:HOME}\n"
    ),
    "null key elsewhere": "hydra:\n  job:\n    num: 4\n  run:\n    null: a\n",
    "set value elsewhere": "hydra:\n  job:\n    num: 4\n  run: !!set {a: null}\n",
    "explicit timestamp elsewhere": (
        "hydra:\n  job:\n    num: 4\n  run:\n    at: !!timestamp 2001-01-01\n"
    ),
    "binary value elsewhere": "hydra:\n  job:\n    num: 4\n  run:\n    raw: !!binary aGk=\n",
    "int, float and bool keys": (
        "hydra:\n  job:\n    num: 4\n  run:\n    2: a\n    1.5: b\n    true: c\n"
    ),
    "omap value": "hydra:\n  job:\n    num: 4\n  run: !!omap [a: 1, b: 2]\n",
    # libyaml accepts these where OmegaConf's Python scanner raises.
    "tab after the job number's colon": "hydra:\n  job:\n    num:\t5\n",
    "tab in a choice": "hydra:\n  job:\n    num: 5\n  runtime:\n    choices:\n      db:\tmysql\n",
    "tab before a comment": "hydra:\n  job:\n    num: 5\t# c\n",
    "byte-order mark at a line start": "hydra:\n  job:\n\ufeff    num: 5\n",
    "byte-order mark at the start": "\ufeffhydra:\n  job:\n    num: 5\n",
    "shared alias": "a: &x {k: 1}\nhydra:\n  job:\n    num: 5\n  run: *x\n",
    "150 levels deep": (
        "hydra:\n  job:\n    num: 0\n  junk: " + "[" * 150 + "]" * 150 + "\n"
    ),
    "20 levels deep": "hydra:\n  job:\n    num: 0\n  junk: "
    + "[" * 20
    + "]" * 20
    + "\n",
}


def test_read_hydra_job_matches_the_old_read(tmp_path):
    """Every file reads to the same result, or fails with the same error, as
    the full OmegaConf load it replaces: what the plain read cannot settle goes
    to OmegaConf instead of returning something else."""
    from mushin._sweep_io import read_hydra_job

    for name, text in _SAME_AS_BEFORE.items():
        p = tmp_path / f"{name.replace(' ', '_')}.yaml"
        p.write_text(text, encoding="utf-8")
        assert _outcome(read_hydra_job, p) == _outcome(_read_as_before, p), name


def test_read_hydra_job_takes_a_str_path_and_pathlib_tags_on_the_fast_path(
    tmp_path, monkeypatch
):
    from mushin._sweep_io import read_hydra_job

    p = tmp_path / "hydra.yaml"
    p.write_text(_SAME_AS_BEFORE["pathlib tag"], encoding="utf-8")
    _forbid_omegaconf(monkeypatch)
    assert read_hydra_job(str(p)) == (3, None)


_EDGE_STRINGS = [
    "${a}",
    "${broken",
    "x${y}z",
    "${oc.env:HOME}",
    "\\${x}",
    "${a.b}",
    "${",
    "}${",
    "???",
    "2024-01-01",
    "1e3",
    "yes",
    "",
    "adam",
]


def _hydra_docs():
    """Random hydra.yaml documents: the two fields a reload reads, plus other
    fields, built from the keys and values that OmegaConf and YAML treat
    specially."""
    import hypothesis.strategies as st

    keys = st.one_of(
        st.sampled_from(["a", "b", "job", "1e3", "yes", ""]),
        st.integers(-3, 3),
        st.none(),
        st.booleans(),
        st.floats(allow_nan=False, allow_infinity=False, width=16),
    )
    leaves = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(-10, 10),
        st.floats(allow_nan=False, width=16),
        st.sampled_from(_EDGE_STRINGS),
        st.binary(max_size=3),
        st.dates(),
        st.sets(st.integers(0, 3), max_size=2),
    )
    trees = st.recursive(
        leaves,
        lambda inner: (
            st.lists(inner, max_size=3) | st.dictionaries(keys, inner, max_size=3)
        ),
        max_leaves=8,
    )
    num = st.one_of(
        st.integers(0, 50),
        st.sampled_from(["???", "${a}", "abc"]),
        st.booleans(),
        st.none(),
    )
    choices = st.one_of(
        st.none(),
        st.dictionaries(
            st.one_of(
                st.sampled_from(["db", "hydra/launcher"]), st.none(), st.integers(0, 2)
            ),
            st.one_of(st.none(), st.sampled_from(_EDGE_STRINGS), st.integers(0, 2)),
            max_size=3,
        ),
    )
    return st.builds(
        lambda n, c, extra, top: {
            "a": top,
            "hydra": {"job": {"num": n}, "runtime": {"choices": c}, "extra": extra},
        },
        num,
        choices,
        trees,
        trees,
    )


def test_read_hydra_job_matches_the_old_read_on_random_files():
    """Whatever a hydra.yaml holds, the reload returns what the full OmegaConf
    load returned, or fails with the same error type."""
    import tempfile
    from pathlib import Path

    import yaml
    from hypothesis import given, settings

    from mushin._sweep_io import read_hydra_job

    @settings(max_examples=300, deadline=None)
    @given(_hydra_docs())
    def check(doc):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "hydra.yaml"
            p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
            assert _outcome(read_hydra_job, p) == _outcome(_read_as_before, p)

    check()


def test_read_hydra_job_on_a_cyclic_alias_raises_as_before(tmp_path):
    """A YAML alias inside its own anchor builds a cyclic tree. OmegaConf.load
    raises RecursionError on it; a tree walk without a seen-set never ends.
    A subprocess with a timeout keeps a regression from hanging pytest."""
    import subprocess
    import sys

    files = {
        "list": "hydra:\n  job:\n    num: 5\n  junk: &a [*a, *a]\n",
        "mapping": "hydra:\n  job:\n    num: 5\n  junk: &h\n    me: *h\n",
    }
    for name, text in files.items():
        p = tmp_path / f"{name}.yaml"
        p.write_text(text, encoding="utf-8")
        code = (
            "import sys\n"
            "from mushin._sweep_io import read_hydra_job\n"
            "try:\n"
            "    read_hydra_job(sys.argv[1])\n"
            "except RecursionError:\n"
            "    print('RecursionError')\n"
        )
        r = subprocess.run(
            [sys.executable, "-c", code, str(p)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, (name, r.returncode, r.stderr[-2000:])
        assert r.stdout.strip() == "RecursionError", name
        assert _outcome(_read_as_before, p) == ("raises", "RecursionError"), name


def test_plain_yaml_loader_steps_aside_if_omegaconf_loader_changes(monkeypatch):
    """The plain loader copies the tables of OmegaConf's loader and its
    duplicate-key check. If that loader defines anything more, the copy could
    type values differently, so reloads take OmegaConf's path instead."""
    import omegaconf._utils

    from mushin._sweep_io import _plain_yaml_loader

    original = omegaconf._utils.get_yaml_loader

    def grown():
        loader = original()
        loader.construct_yaml_str = lambda self, node: "changed"
        return loader

    monkeypatch.setattr(omegaconf._utils, "get_yaml_loader", grown)
    assert _plain_yaml_loader() is None


# Characters YAML gives meaning to, the two libyaml and PyYAML's Python scanner
# disagree on (tab, byte-order mark), and some it refuses.
_MUTATIONS = [
    "\t",
    "\ufeff",
    "\r",
    "\r\n",
    "\x85",
    "\u2028",
    "\x00",
    "\x07",
    "\xa0",
    "\U0001f600",
    "%",
    "!",
    "&a ",
    "*a",
    "|",
    ">",
    '"',
    "'",
    "#",
    "@",
    "`",
    ":",
    ": ",
    "- ",
    "? ",
    "[",
    "]",
    "{",
    "}",
    ",",
    " ",
    "\n",
    "\\",
    "---",
    "...",
    "!!str ",
    "!!int ",
    "!!null ",
    "<<: ",
    "~",
    "${",
    "}",
]


def test_read_hydra_job_matches_the_old_read_on_mutated_hydra_files(tmp_path):
    """A real hydra.yaml with a few characters inserted or replaced reads the
    same as through OmegaConf, or fails with the same error type. This is what
    found the tab and byte-order-mark differences of libyaml's scanner."""
    import tempfile
    from pathlib import Path

    import hypothesis.strategies as st
    from hypothesis import given, settings

    from mushin import multirun
    from mushin._sweep_io import read_hydra_job
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    W().run(a=multirun([1]), working_dir=str(tmp_path / "s"))
    (real,) = (tmp_path / "s").glob("*/.hydra/hydra.yaml")
    base = real.read_text(encoding="utf-8")
    edits = st.lists(
        st.tuples(
            st.integers(0, len(base) - 1), st.sampled_from(_MUTATIONS), st.booleans()
        ),
        min_size=1,
        max_size=3,
    )

    @settings(max_examples=300, deadline=None)
    @given(edits)
    def check(changes):
        text = base
        for pos, piece, replace in changes:
            pos = min(pos, len(text) - 1)
            text = text[:pos] + piece + text[pos + replace :]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "hydra.yaml"
            p.write_text(text, encoding="utf-8")
            assert _outcome(read_hydra_job, p) == _outcome(_read_as_before, p)

    check()


def test_read_hydra_job_finishes_odd_fields_without_rereading(tmp_path, monkeypatch):
    """A `???` or `${...}` job number (Hydra writes `???` for a single run) goes
    to OmegaConf.create on the tree already parsed, not to a second parse of
    the file through OmegaConf.load. The result and the error stay the same."""
    import pytest
    from omegaconf.errors import MissingMandatoryValue

    from mushin._sweep_io import read_hydra_job

    single = tmp_path / "single.yaml"
    single.write_text("hydra:\n  job:\n    num: ???\n", encoding="utf-8")
    interp = tmp_path / "interp.yaml"
    interp.write_text("base: 3\nhydra:\n  job:\n    num: ${base}\n", encoding="utf-8")
    choice = tmp_path / "choice.yaml"
    choice.write_text(
        "db_name: mysql\nhydra:\n  job:\n    num: 0\n"
        "  runtime:\n    choices:\n      db: ${db_name}\n",
        encoding="utf-8",
    )
    expected = [_outcome(_read_as_before, p) for p in (single, interp, choice)]

    _forbid_omegaconf(monkeypatch)
    with pytest.raises(MissingMandatoryValue):
        read_hydra_job(single)
    assert [_outcome(read_hydra_job, p) for p in (single, interp, choice)] == expected
    assert expected[1:] == [
        ("returns", "(3, None)"),
        ("returns", "(0, {'db': 'mysql'})"),
    ]
