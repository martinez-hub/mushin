# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""`_hydra_yaml.read_hydra_job`, the fast read of a job's hydra.yaml that
`load_from_dir` and the MCP server use: it must give what the full
`OmegaConf.load` read gave, or fail with the same error type, for every file.
Files Hydra wrote take the fast path; everything else goes to OmegaConf."""

import subprocess
import sys
import tempfile
from pathlib import Path

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import given, settings
from omegaconf import OmegaConf

from mushin import _hydra_yaml
from mushin._hydra_yaml import read_hydra_job

# These tests are what guarantees the fast read matches OmegaConf.load, and the
# repo's `fast` profile (10 examples) would barely exercise them.
_PARITY_EXAMPLES = 100


def _hydra_text(tree) -> str:
    """`tree` written as Hydra writes hydra.yaml (`OmegaConf.to_yaml`)."""
    return OmegaConf.to_yaml(OmegaConf.create(tree))


def _read_as_before(path):
    """What load_from_dir and the MCP server did before the fast read: a full
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


def _types(x):
    """The type of every container and leaf in ``x``, which repr hides: a
    DictConfig and a dict print alike."""
    if isinstance(x, (tuple, list)) or type(x).__name__ == "ListConfig":
        return type(x).__name__, tuple(_types(v) for v in x)
    if hasattr(x, "items"):
        return type(x).__name__, tuple((k, _types(v)) for k, v in x.items())
    return type(x).__name__


def _outcome(read, path):
    # _types iterates any config returned, which resolves its interpolations
    # and can raise: inside the try, so that counts as the outcome too.
    try:
        result = read(path)
        return "returns", repr(result), _types(result)
    except Exception as e:  # noqa: BLE001 - the error type is the outcome
        return "raises", type(e).__name__


def _forbid_rereading(monkeypatch):
    """Make a full OmegaConf.load of the file fail. Skips where PyYAML has no
    libyaml, which has no fast path."""
    import hydra_zen

    if _hydra_yaml.CParser is None:
        pytest.skip("PyYAML built without libyaml: reloads take OmegaConf's path")

    def _slow(path):
        raise AssertionError(f"read_hydra_job fell back to OmegaConf.load for {path}")

    monkeypatch.setattr(hydra_zen, "load_from_yaml", _slow)


def _forbid_omegaconf(monkeypatch):
    """Also make building an OmegaConf config fail: the step the fast read
    exists to skip. Compute any OmegaConf reference before calling this."""
    _forbid_rereading(monkeypatch)

    def _create(*args, **kwargs):
        raise AssertionError("read_hydra_job built an OmegaConf config")

    monkeypatch.setattr(OmegaConf, "create", staticmethod(_create))


def _read_in_subprocess(path) -> subprocess.CompletedProcess:
    """read_hydra_job(path) in a fresh interpreter, printing the error type,
    so that a crash or a hang cannot take pytest down with it."""
    code = (
        "import sys\n"
        "from mushin._hydra_yaml import read_hydra_job\n"
        "try:\n"
        "    print(read_hydra_job(sys.argv[1]))\n"
        "except Exception as e:\n"
        "    print(type(e).__name__)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )


# --- The fast path -----------------------------------------------------------


def test_reads_a_real_sweep_without_omegaconf(tmp_path, monkeypatch):
    """Hydra's own hydra.yaml (interpolations, `???` fields, long help text)
    takes the fast path: a silent fallback would lose the speedup."""
    from mushin import multirun
    from mushin.workflows import MultiRunMetricsWorkflow

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(m=float(a))

    W().run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    files = sorted((tmp_path / "s").glob("*/.hydra/hydra.yaml"))
    assert len(files) == 2
    expected = [_read_as_before(f) for f in files]

    _forbid_omegaconf(monkeypatch)
    got = [read_hydra_job(f) for f in files]
    assert got == expected
    assert sorted(num for num, _ in got) == [0, 1]
    assert all(choices for _, choices in got)  # Hydra records its own groups


def test_types_values_as_omegaconf_does(tmp_path, monkeypatch):
    """Option names that YAML 1.1 would read as a date, a float or a bool stay
    the strings OmegaConf reads, on the fast path."""
    tree = {
        "hydra": {
            "job": {"num": 7},
            "runtime": {
                "choices": {
                    "date": "2024-01-01",
                    "sci": "1e3",
                    "bool": "yes",
                    "plain": "adam",
                    "unset": None,
                }
            },
        }
    }
    p = tmp_path / "hydra.yaml"
    p.write_text(_hydra_text(tree), encoding="utf-8")
    expected = _read_as_before(p)

    _forbid_omegaconf(monkeypatch)
    # repr compares the types too: '2024-01-01' stays a string.
    assert repr(read_hydra_job(p)) == repr(expected)


def test_takes_files_the_two_emitters_write_differently(tmp_path, monkeypatch):
    """libyaml's emitter writes a character outside the BMP escaped, and a
    long key as `? key`, where Hydra's (OmegaConf's Python dumper) writes them
    plainly. Such files still take the fast path, through the second check."""
    long_key = "\xe9" * 130
    tree = {
        "hydra": {
            "job": {"num": 4, "override_dirname": "label=rocket\U0001f680"},
            "runtime": {"choices": {"db": "mysql"}},
            "extra": {long_key: 1, "": 2},
        }
    }
    p = tmp_path / "hydra.yaml"
    p.write_text(_hydra_text(tree), encoding="utf-8")
    expected = _read_as_before(p)

    _forbid_omegaconf(monkeypatch)
    assert read_hydra_job(p) == expected == (4, {"db": "mysql"})


def test_takes_a_str_path_and_pathlib_values(tmp_path, monkeypatch):
    tree = {"hydra": {"job": {"num": 3}, "runtime": {"cwd": Path("/tmp")}}}
    p = tmp_path / "hydra.yaml"
    p.write_text(_hydra_text(tree), encoding="utf-8")

    _forbid_omegaconf(monkeypatch)
    assert read_hydra_job(str(p)) == (3, None)


def test_plain_codecs_parse_in_c_and_compose_in_python():
    """libyaml parses; PyYAML's Python composer builds the tree, so deep
    nesting raises RecursionError instead of crashing (see below)."""
    from yaml.composer import Composer

    if _hydra_yaml.CParser is None:
        pytest.skip("PyYAML built without libyaml: reloads take OmegaConf's path")
    from omegaconf._utils import get_omega_conf_dumper

    codecs = _hydra_yaml._plain_codecs()
    assert codecs is not None
    loader, libyaml_dumper, omegaconf_dumper = codecs
    assert issubclass(loader, _hydra_yaml.CParser)
    assert issubclass(loader, Composer)
    assert loader.compose_document is Composer.compose_document
    assert not issubclass(loader, yaml.CSafeLoader)
    # It refuses any alias, which Hydra never writes.
    with pytest.raises(yaml.composer.ComposerError, match="alias"):
        yaml.load("a: &x 1\nb: *x\n", Loader=loader)
    assert issubclass(libyaml_dumper, _hydra_yaml.CDumper)
    assert omegaconf_dumper is get_omega_conf_dumper()


def test_plain_codecs_follow_pyyaml_tables(monkeypatch):
    """OmegaConf builds its loader afresh from PyYAML's tables on each load;
    the cached codecs are rebuilt when a library changes those tables."""
    if _hydra_yaml.CParser is None:
        pytest.skip("PyYAML built without libyaml: reloads take OmegaConf's path")
    before = _hydra_yaml._plain_codecs()
    grown = {k: list(v) for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
    grown.setdefault("~", []).append(
        ("tag:example,2026:x", __import__("re").compile("x"))
    )
    monkeypatch.setattr(yaml.SafeLoader, "yaml_implicit_resolvers", grown)
    after = _hydra_yaml._plain_codecs()
    assert after is not None and after is not before


def test_plain_codecs_follow_a_replaced_constructor(tmp_path, monkeypatch):
    """Replacing an existing constructor keeps each table's identity and size;
    the fast read must still follow it, as OmegaConf.load does."""
    if _hydra_yaml.CParser is None:
        pytest.skip("PyYAML built without libyaml: reloads take OmegaConf's path")
    p = tmp_path / "hydra.yaml"
    p.write_text(_hydra_text({"hydra": {"job": {"num": 3}}}), encoding="utf-8")
    assert read_hydra_job(p) == (3, None)  # builds and caches the codecs

    def plus_100(loader, node):
        return int(loader.construct_scalar(node)) + 100

    monkeypatch.setitem(
        yaml.SafeLoader.yaml_constructors, "tag:yaml.org,2002:int", plus_100
    )
    assert _outcome(read_hydra_job, p) == _outcome(_read_as_before, p)
    assert read_hydra_job(p) == (103, None)


def test_a_helper_of_another_shape_steps_aside(tmp_path, monkeypatch):
    """If OmegaConf's loader helper returns something other than a loader
    class, the fast read steps aside (the file then goes to OmegaConf.load,
    which uses the same helper) instead of raising itself."""
    import omegaconf._utils

    p = tmp_path / "hydra.yaml"
    p.write_text(_hydra_text({"hydra": {"job": {"num": 3}}}), encoding="utf-8")
    monkeypatch.setattr(omegaconf._utils, "get_yaml_loader", lambda: object())
    assert _hydra_yaml._plain_codecs() is None
    assert _hydra_yaml._parse_plain(p) is None


@pytest.mark.parametrize("helper", ["get_yaml_loader", "get_omega_conf_dumper"])
def test_plain_codecs_step_aside_if_omegaconf_changes(monkeypatch, helper):
    """The codecs copy the tables of OmegaConf's loader and dumper and the
    loader's duplicate-key check. If either class defines anything more, the
    copy could read or write values differently, so OmegaConf decides."""
    import omegaconf._utils

    original = getattr(omegaconf._utils, helper)

    def grown():
        cls = original()
        return type(cls.__name__, (cls.__bases__[0],), {**vars(cls), "extra": 1})

    monkeypatch.setattr(omegaconf._utils, helper, grown)
    assert _hydra_yaml._plain_codecs() is None


def test_finishes_odd_fields_without_rereading(tmp_path, monkeypatch):
    """A `???` or `${...}` job number (Hydra writes `???` for a single run), or
    an interpolated choice, goes to OmegaConf.create on the tree already
    parsed, not to a second parse of the file. Same result, same error."""
    from omegaconf.errors import MissingMandatoryValue

    trees = {
        "single": {"hydra": {"job": {"num": "???"}}},
        "interp": {"base": 3, "hydra": {"job": {"num": "${base}"}}},
        "choice": {
            "db_name": "mysql",
            "hydra": {"job": {"num": 0}, "runtime": {"choices": {"db": "${db_name}"}}},
        },
    }
    paths = {}
    for name, tree in trees.items():
        paths[name] = tmp_path / f"{name}.yaml"
        paths[name].write_text(_hydra_text(tree), encoding="utf-8")
    expected = {name: _outcome(_read_as_before, p) for name, p in paths.items()}
    assert expected["interp"][:2] == ("returns", "(3, None)")
    assert expected["choice"][:2] == ("returns", "(0, {'db': 'mysql'})")

    _forbid_rereading(monkeypatch)
    with pytest.raises(MissingMandatoryValue):
        read_hydra_job(paths["single"])
    assert {name: _outcome(read_hydra_job, p) for name, p in paths.items()} == expected


# --- Falling back --------------------------------------------------------------


def test_falls_back_to_omegaconf(tmp_path, monkeypatch):
    """What the fast read cannot settle goes to OmegaConf, which resolves it or
    raises as it always did."""
    from omegaconf.errors import MissingMandatoryValue

    interp = tmp_path / "interp.yaml"
    interp.write_text("base: 3\nhydra:\n  job:\n    num: ${base}\n", encoding="utf-8")
    assert read_hydra_job(interp) == (3, None)  # resolved; no choices recorded

    missing = tmp_path / "missing.yaml"
    missing.write_text("hydra:\n  job:\n    num: ???\n", encoding="utf-8")
    with pytest.raises(MissingMandatoryValue):
        read_hydra_job(missing)

    with pytest.raises(FileNotFoundError):
        read_hydra_job(tmp_path / "absent.yaml")

    # Without the codecs (no libyaml, or OmegaConf changed), the result is
    # the same, only slower.
    p = tmp_path / "hydra.yaml"
    p.write_text(_hydra_text({"hydra": {"job": {"num": 1}}}), encoding="utf-8")
    fast = read_hydra_job(p)
    monkeypatch.setattr(_hydra_yaml, "_plain_codecs", lambda: None)
    assert read_hydra_job(p) == fast == (1, None)


def test_survives_deep_nesting(tmp_path):
    """libyaml's C composer recurses on the C stack with no limit, so a deeply
    nested file killed the process (SIGSEGV) past any `except`. It must raise
    RecursionError, as OmegaConf.load does."""
    p = tmp_path / "hydra.yaml"
    p.write_text(
        "hydra:\n  job:\n    num: 0\n  junk: " + "[" * 100_000 + "]" * 100_000 + "\n",
        encoding="utf-8",
    )
    r = _read_in_subprocess(p)
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
    assert r.stdout.strip() == "RecursionError"


def test_a_cyclic_alias_raises_as_before(tmp_path):
    """A YAML alias inside its own anchor builds a cyclic tree. OmegaConf.load
    raises RecursionError on it; a tree walk without a seen-set never ends."""
    files = {
        "list": "hydra:\n  job:\n    num: 5\n  junk: &a [*a, *a]\n",
        "mapping": "hydra:\n  job:\n    num: 5\n  junk: &h\n    me: *h\n",
    }
    for name, text in files.items():
        p = tmp_path / f"{name}.yaml"
        p.write_text(text, encoding="utf-8")
        r = _read_in_subprocess(p)
        assert r.returncode == 0, (name, r.returncode, r.stderr[-2000:])
        assert r.stdout.strip() == "RecursionError", name
        assert _outcome(_read_as_before, p) == ("raises", "RecursionError"), name


# --- Same as before, on every kind of file -----------------------------------

_SAME_AS_BEFORE = {
    "empty file": "",
    "no hydra node": "x: 1\n",
    "missing job number": "hydra:\n  job:\n    name: a\n",
    "??? job number": "hydra:\n  job:\n    num: ???\n",
    "bool job number": "hydra:\n  job:\n    num: true\n",
    "duplicate key in the job": "hydra:\n  job:\n    num: 1\n    num: 5\n",
    "duplicate key elsewhere": "hydra:\n  job:\n    num: 1\n  run:\n    dir: a\n    dir: b\n",
    "??? choice": "hydra:\n  job:\n    num: 0\n  runtime:\n    choices:\n      db: ???\n",
    "interpolated choice": (
        "name: mysql\nhydra:\n  job:\n    num: 0\n  runtime:\n    choices:\n      db: ${name}\n"
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
    "non-string options": (
        "hydra:\n  job:\n    num: 7\n  runtime:\n    choices:\n      sci: 1e3\n"
        "      dotted: 1.5\n      hex: 0x1f\n      sexagesimal: 1:30\n      yes_word: yes\n"
        "      inf: .inf\n      nan: .nan\n"
    ),
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
    "shared alias": "a: &x {k: 1}\nhydra:\n  job:\n    num: 5\n  run: *x\n",
    "20 levels deep": "hydra:\n  job:\n    num: 0\n  junk: "
    + "[" * 20
    + "]" * 20
    + "\n",
    "150 levels deep": "hydra:\n  job:\n    num: 0\n  junk: "
    + "[" * 150
    + "]" * 150
    + "\n",
    # As an emitter writes them, so they pass the round trip and reach the
    # depth, leaf-type and option-name checks.
    "150 levels deep, as emitted": (
        "hydra:\n  job:\n    num: 0\n  junk:\n"
        + "".join("  " * (i + 2) + "k:\n" for i in range(149))
        + "  " * 151
        + "k: x\n"
    ),
    "set value, as emitted": "hydra:\n  job:\n    num: 4\n  run: !!set\n    a: null\n",
    "nested choice, as emitted": (
        "hydra:\n  job:\n    num: 0\n  runtime:\n    choices:\n      db:\n        a: 1\n"
    ),
    "aliased string": "a: &s abc\nb:\n- *s\n- *s\nhydra:\n  job:\n    num: 5\n",
    # libyaml reads these where PyYAML's Python scanner (OmegaConf's) raises or
    # reads something else; none is text an emitter writes.
    "tab after a colon": "hydra:\n  job:\n    num:\t5\n",
    "tab in a choice": "hydra:\n  job:\n    num: 5\n  runtime:\n    choices:\n      db:\tmysql\n",
    "tab before a comment": "hydra:\n  job:\n    num: 5\t# c\n",
    "byte-order mark at a line start": "hydra:\n  job:\n\ufeff    num: 5\n",
    "byte-order mark at the start": "\ufeffhydra:\n  job:\n    num: 5\n",
    "empty tag on a choice": "hydra:\n  job:\n    num: 5\n  runtime:\n    choices:\n      db: !\n",
    "empty tag on the job number": "hydra:\n  job:\n    num: !\n",
    "empty tag on the choices": "hydra:\n  job:\n    num: 5\n  runtime:\n    choices: !\n",
    "comment right after a literal header": "hydra:\n  job:\n    num: 5\n  run: |#c\n    text\n",
    "comment right after a folded header": (
        "hydra:\n  job:\n    num: 5\n  runtime:\n    choices:\n      db: >#c\n        mysql\n"
    ),
    "comment right after a keep header": "hydra:\n  job:\n    num: 5\n  run: |+#c\n    text\n",
    "? in a flow mapping": "hydra:\n  job:\n    num: 5\n  run: {dir: a?b}\n",
    "? in a flow sequence": "hydra:\n  job:\n    num: 5\n  overrides: [a=1?]\n",
    "? in a flow choice": "hydra:\n  job:\n    num: 5\n  runtime:\n    choices: {db: my?sql}\n",
    # Canonical apart from an edit a person might make: OmegaConf decides.
    "a comment added": "# edited\nhydra:\n  job:\n    num: 5\n",
    "a flow job": "hydra:\n  job: {num: 5}\n",
}


def test_matches_the_old_read(tmp_path):
    """Every file reads to the same result, or fails with the same error, as
    the full OmegaConf load it replaces."""
    # Numbered files: some case names hold characters Windows refuses (`?`).
    for i, (name, text) in enumerate(_SAME_AS_BEFORE.items()):
        p = tmp_path / f"case{i}.yaml"
        p.write_text(text, encoding="utf-8")
        assert _outcome(read_hydra_job, p) == _outcome(_read_as_before, p), name


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
    "a?b",
    "#c",
    "!",
    "a\tb",
    "adam",
]


# Any text: every Unicode character but lone surrogates, which no file holds.
_ANY_TEXT = st.text(st.characters(exclude_categories=("Cs",)), max_size=12)
# Long text with spaces, line breaks and quotes, which emitters fold and quote.
_LONG_TEXT = st.text(" ab-#:'\"\n\u2028\x85\xe9\U0001f600", min_size=40, max_size=120)


def _hydra_docs():
    """Random hydra.yaml documents: the two fields a reload reads, plus other
    fields, built from keys and values that OmegaConf and YAML treat
    specially, and from arbitrary text."""
    keys = st.one_of(
        st.sampled_from(["a", "b", "job", "1e3", "yes", "", "a?b"]),
        _ANY_TEXT,
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
        _ANY_TEXT,
        _LONG_TEXT,
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
            st.one_of(
                st.none(), st.sampled_from(_EDGE_STRINGS), _ANY_TEXT, st.integers(0, 2)
            ),
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


def _omegaconf_dump(doc) -> str:
    """`doc` through OmegaConf's dumper, as `OmegaConf.to_yaml` writes it, but
    without OmegaConf.create first (which rejects some of these trees)."""
    from omegaconf._utils import get_omega_conf_dumper

    return yaml.dump(
        doc,
        Dumper=get_omega_conf_dumper(),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def test_matches_the_old_read_on_random_files():
    """Whatever a hydra.yaml holds, written as Hydra writes it (so the fast
    path is in play) or by plain `yaml.safe_dump`, the reload returns what the
    full OmegaConf load returned, or fails with the same error type."""

    @settings(max_examples=_PARITY_EXAMPLES, deadline=None)
    @given(_hydra_docs())
    def check(doc):
        with tempfile.TemporaryDirectory() as d:
            for name, text in [
                ("omegaconf", _omegaconf_dump(doc)),
                ("safe", yaml.safe_dump(doc, sort_keys=False)),
            ]:
                p = Path(d) / f"{name}.yaml"
                p.write_text(text, encoding="utf-8")
                assert _outcome(read_hydra_job, p) == _outcome(_read_as_before, p), name

    check()


def test_emitted_yaml_parses_alike_in_libyaml_and_python():
    """What the canonical-text check rests on: YAML that either emitter the
    check uses writes (libyaml's, or OmegaConf's own in pure Python, with
    OmegaConf's representers) parses to the same tree in the plain loader
    (libyaml) and in OmegaConf's loader (pure Python), whole tree compared.
    It also fails if OmegaConf's loader starts treating mappings differently,
    since its construct_mapping is not copied."""
    from omegaconf._utils import get_yaml_loader

    if _hydra_yaml.CParser is None:
        pytest.skip("PyYAML built without libyaml: reloads take OmegaConf's path")
    loader, *dumpers = _hydra_yaml._plain_codecs()

    @settings(max_examples=_PARITY_EXAMPLES, deadline=None)
    @given(_hydra_docs())
    def check(doc):
        for dumper in dumpers:
            text = yaml.dump(
                doc,
                Dumper=dumper,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
            plain = yaml.load(text, Loader=loader)
            python = yaml.load(text, Loader=get_yaml_loader())
            assert repr(plain) == repr(python), dumper

    check()


# Characters and short sequences YAML gives meaning to, including each kind
# libyaml and PyYAML's Python scanner were found to disagree on.
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
    "! ",
    "!\n",
    "&a ",
    "*a",
    "|",
    "|#c",
    ">-#c",
    '"',
    "'",
    "#",
    "@",
    "`",
    ":",
    ": ",
    "- ",
    "? ",
    "?",
    "{a?b}",
    "[a?]",
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
    "!!null ",
    "<<: ",
    "~",
    "${",
    "}",
]


def test_matches_the_old_read_on_mutated_hydra_files(tmp_path):
    """A real hydra.yaml with a few characters inserted or replaced reads the
    same as through OmegaConf, or fails with the same error type."""
    from mushin import multirun
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

    @settings(max_examples=_PARITY_EXAMPLES, deadline=None)
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


def test_an_aliased_string_is_not_written_back(tmp_path, monkeypatch):
    """Writing back a tree whose string is aliased N times would emit it N
    times: a 220 KB file took 2.2 GB. The loader refuses the alias, so the
    file goes to OmegaConf without anything being written back."""
    p = tmp_path / "hydra.yaml"
    p.write_text(
        "hydra:\n  job:\n    num: 3\na: &a "
        + "x" * 10_000
        + "\nb: ["
        + "*a, " * 500
        + "*a]\n",
        encoding="utf-8",
    )
    expected = _outcome(_read_as_before, p)
    # Count calls: an exception raised here would be swallowed by the fast
    # path's own fallback.
    dumps = []
    real_dump = yaml.dump

    def counting_dump(*args, **kwargs):
        dumps.append(1)
        return real_dump(*args, **kwargs)

    monkeypatch.setattr(yaml, "dump", counting_dump)
    assert _hydra_yaml._parse_plain(p) is None
    assert dumps == []
    assert _outcome(read_hydra_job, p) == expected


def test_a_patched_safe_loader_sends_files_to_omegaconf(tmp_path, monkeypatch):
    """OmegaConf's loader subclasses yaml.SafeLoader and the plain loader does
    not, so a library that patches SafeLoader changes OmegaConf.load alone;
    the fast read must then step aside."""
    p = tmp_path / "hydra.yaml"
    tree = {"hydra": {"job": {"num": 3}, "runtime": {"choices": {"db": "mysql"}}}}
    p.write_text(_hydra_text(tree), encoding="utf-8")
    assert read_hydra_job(p) == (3, {"db": "mysql"})  # builds the codecs

    def upper_keys(self, node, deep=False):
        mapping = yaml.constructor.SafeConstructor.construct_mapping(self, node, deep)
        return {k.upper() if isinstance(k, str) else k: v for k, v in mapping.items()}

    monkeypatch.setattr(yaml.SafeLoader, "construct_mapping", upper_keys, raising=False)
    assert _hydra_yaml._plain_codecs() is None
    assert _outcome(read_hydra_job, p) == _outcome(_read_as_before, p)


def test_the_second_write_back_runs_only_where_the_emitters_differ(
    tmp_path, monkeypatch
):
    """OmegaConf's Python dumper (1.8 ms) runs only for text libyaml's emitter
    writes differently (escapes, complex keys), so a hand-edited file does not
    pay for it before going to OmegaConf."""
    calls = []
    real = _hydra_yaml._write_back

    def counting(doc, dumper):
        calls.append(dumper)
        return real(doc, dumper)

    monkeypatch.setattr(_hydra_yaml, "_write_back", counting)
    edited = tmp_path / "edited.yaml"
    edited.write_text(
        "# edited\n" + _hydra_text({"hydra": {"job": {"num": 5}}}), encoding="utf-8"
    )
    assert _hydra_yaml._parse_plain(edited) is None
    assert len(calls) == 1

    calls.clear()
    emoji = tmp_path / "emoji.yaml"
    emoji.write_text(
        _hydra_text({"hydra": {"job": {"num": 5, "name": "rocket\U0001f680"}}}),
        encoding="utf-8",
    )
    assert _hydra_yaml._parse_plain(emoji) is not None
    assert len(calls) == 2
