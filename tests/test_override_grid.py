"""Regression tests for the override -> grid pipeline.

These pin fixes for the silent-wrong-results cluster: fixed (non-swept) string
values must not become accidental sweeps or change type; a duplicated sweep
axis value must be rejected rather than silently collapsed; `combo_key` must be
injective across delimiter-containing values; a dotted sweep on a config field
that already exists must not crash; and a batching sweeper's nested job list
must be flattened.
"""

import re

import numpy as np
import pytest

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


def _run(tmp_path, cls, **overrides):
    wf = cls()
    wf.run(working_dir=str(tmp_path / "s"), **overrides)
    return wf


class _Echo(MultiRunMetricsWorkflow):
    @staticmethod
    def task(x=0, tag="", flag=False):
        return dict(
            x_is=float(x), tag_type=type(tag).__name__, flag_type=type(flag).__name__
        )


def test_fixed_string_with_comma_is_not_an_accidental_sweep(tmp_path):
    # A fixed (non-multirun) string containing a comma must stay a single
    # string value, not be re-split by Hydra into a two-cell sweep.
    wf = _run(tmp_path, _Echo, x=multirun([1, 2]), tag="a,b")
    ds = wf.to_xarray()
    assert set(ds.dims) == {"x"}  # only x is a sweep axis, not tag
    assert (ds["tag_type"] == "str").all()


def test_fixed_string_true_stays_a_string(tmp_path):
    # The string "true" must not be coerced to bool True.
    wf = _run(tmp_path, _Echo, x=multirun([1]), tag="true")
    ds = wf.to_xarray()
    assert str(ds["tag_type"].values.item()) == "str"


def test_actual_bool_stays_bool(tmp_path):
    wf = _run(tmp_path, _Echo, x=multirun([1]), flag=True)
    ds = wf.to_xarray()
    assert str(ds["flag_type"].values.item()) == "bool"


def test_duplicate_sweep_axis_value_is_rejected(tmp_path):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a))

    with pytest.raises(ValueError, match="duplicate"):
        W().run(working_dir=str(tmp_path / "s"), a=multirun([0.0, 1.0, 0.0]))


@pytest.mark.parametrize(
    "values, shown",
    [
        ([1, 2, 1], "1"),
        ([np.float64(0.5), 0.5], "0.5, np.float64(0.5)"),
        (["a,b", "a,b"], "'a,b'"),
        # One text, two types: the cells would share a key (#308 review).
        (["1", 1], "'1', 1"),
    ],
)
def test_duplicate_sweep_value_is_rejected_before_a_dry_run(values, shown):
    """Two values that would share a cell are refused before the dry-run
    preview returns, whatever their spelling (a numpy scalar and the float it
    holds are one value; a string with a comma stays one value)."""

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=0.0)

    with pytest.raises(
        ValueError, match=rf"`a` has duplicate sweep values {re.escape(shown)};"
    ):
        W().run(a=multirun(values), dry_run=True)


@pytest.mark.parametrize(
    "sweep, coords",
    [
        (dict(a=multirun([["x"], ["yy"]])), ["['x']", "['yy']"]),
        (dict(a=multirun([{"k": "x"}, {"k": "yy"}])), ["{'k': 'x'}", "{'k': 'yy'}"]),
        (dict(overrides=["+a='x','yy'"]), ["x", "yy"]),
    ],
)
def test_strings_inside_sweep_values_key_their_cells(tmp_path, sweep, coords):
    """Hydra keeps a quoted string inside a list or dict, or in a raw comma
    sweep, as a ``QuotedString``. The grid must hold the plain string, as each
    cell's config does, or no cell matches and every result is NaN (#308)."""

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            s = a if isinstance(a, str) else (a["k"] if "k" in a else a[0])
            return dict(v=float(len(s)))

    wd = tmp_path / "s"
    wf = W()
    wf.run(working_dir=str(wd), **sweep)
    loaded = W().load_from_dir(wd, "mushin_metrics.json")
    for w in (wf, loaded):
        ds = w.to_xarray()
        assert [str(c) for c in ds["a"].values] == coords
        assert ds["v"].values.tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "raw", ["+a='x',\"x\"", "+a=choice(['x'],[\"x\"])", "+a='1',1"]
)
def test_a_raw_sweep_repeating_a_quoted_value_is_rejected(raw):
    """Values that would share a cell are refused before launch (#308): two
    spellings of one string, or a quoted string and a number of one text."""

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=0.0)

    with pytest.raises(ValueError, match=r"`a` has duplicate sweep values"):
        W().run(overrides=[raw], dry_run=True)


@pytest.mark.parametrize(
    "values, v", [([1, 1.0], [0.0, 1.0]), ([["1"], [1]], [1.0, 0.0])]
)
def test_values_that_key_apart_still_run(tmp_path, values, v):
    """The duplicate check refuses only values that share a cell key: 1 and
    1.0, or ['1'] and [1], are two cells, each with its own result."""

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            x = a if isinstance(a, int | float) else a[0]
            return dict(v=float(isinstance(x, str | float)))

    wd = tmp_path / "s"
    wf = W()
    wf.run(a=multirun(values), working_dir=str(wd))
    loaded = W().load_from_dir(wd, "mushin_metrics.json")
    for w in (wf, loaded):
        assert w.to_xarray()["v"].values.tolist() == v


def test_a_base_workflow_sweep_passes_the_duplicate_check(tmp_path):
    """The check runs in ``BaseWorkflow.run``, which has no
    ``_sanitize_coordinate_for_xarray``: a subclass of it still sweeps, and
    still has its duplicates refused."""
    from mushin.workflows import BaseWorkflow

    class W(BaseWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a))

        def jobs_post_process(self):
            return

    W().run(a=multirun([1, 2]), working_dir=str(tmp_path / "s"))
    with pytest.raises(ValueError, match=r"`a` has duplicate sweep values '1', 1;"):
        W().run(a=multirun(["1", 1]), dry_run=True)


class _Seen(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a, b=""):
        return dict(a_repr=repr(a), b_repr=repr(b))


def test_backslashes_reach_the_task_unchanged(tmp_path):
    """A backslash in a string, swept or fixed, is not doubled on the way to
    the task (#319)."""
    wf = _run(tmp_path, _Seen, a=multirun(["a\\b", "C:\\dir\\"]), b="C:\\data")
    ds = wf.to_xarray()
    assert [str(c) for c in ds["a"].values] == ["a\\b", "C:\\dir\\"]
    assert [str(r) for r in ds["a_repr"].values] == [repr("a\\b"), repr("C:\\dir\\")]
    assert {str(r) for r in ds["b_repr"].values} == {repr("C:\\data")}


@pytest.mark.parametrize(
    "text",
    [
        "out\\${now:%Y}",
        "a\\\\${x}",
        "${oc.env:X,\\\\fs\\share}",
        "${x}\\y",
        "${x}'s",
    ],
)
def test_an_interpolation_string_is_written_as_before(text):
    """A string holding ``${`` is parsed again by OmegaConf, with backslash
    escapes of its own that doubling every backslash undid, so it keeps
    exactly the form it always had; only other strings changed (#319)."""
    from mushin.workflows import _to_override_element

    before = "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"
    assert _to_override_element(text) == before


def test_interpolations_with_backslashes_resolve_as_written(tmp_path, monkeypatch):
    """A Windows working_dir ``runs\\${now:%Y}`` keeps its separator and its
    interpolation (one backslash there would escape it, naming a directory
    ``${now:%Y}``), a UNC default in ``${oc.env:...}`` keeps both leading
    backslashes, and a path value that is not a str is quoted like one."""
    from pathlib import PureWindowsPath

    from mushin.workflows import _to_override_element

    assert _to_override_element(PureWindowsPath("C:/data")) == "'C:\\data'"

    monkeypatch.delenv("MUSHIN_TEST_UNSET", raising=False)
    wf = _Seen()
    wf.run(
        a=multirun(["x", "y"]),
        b="${oc.env:MUSHIN_TEST_UNSET,\\\\fileserver\\datasets}",
        working_dir=str(tmp_path / "runs") + "\\${now:%Y}",
    )
    ds = wf.to_xarray()
    assert [str(r) for r in ds["a_repr"].values] == ["'x'", "'y'"]
    assert {str(r) for r in ds["b_repr"].values} == {repr("\\\\fileserver\\datasets")}
    assert not [p for p in tmp_path.rglob("*") if "${" in p.name]


class _SeenList(MultiRunMetricsWorkflow):
    @staticmethod
    def task(a="", b=None):
        from omegaconf import OmegaConf

        plain = OmegaConf.to_container(b) if OmegaConf.is_config(b) else b
        return dict(b_repr=repr(plain))


def test_a_hydra_list_reaches_the_task_as_written(tmp_path):
    """A `hydra_list` is written as a list of override elements: a backslash
    is not doubled, a quote and a comma stay, and None is null (#321)."""
    from mushin import hydra_list

    values = ["a\\b", "it's", "x,y", None, 1]
    wf = _run(tmp_path, _SeenList, a=multirun(["x"]), b=hydra_list(values))
    ds = wf.to_xarray()
    assert {str(r) for r in ds["b_repr"].values} == {repr(values)}


def test_a_hydra_list_keeps_what_main_read_as_a_list(tmp_path):
    """Main wrote a `hydra_list` through Python's repr, which read back as a
    list for a nested `hydra_list` or an OmegaConf ListConfig: each still
    reaches the task as a list, as does a tuple or a range (#321)."""
    from omegaconf import OmegaConf

    from mushin import hydra_list

    values = [hydra_list([1, 2]), OmegaConf.create(["a"]), (1, "a"), range(2)]
    wf = _run(tmp_path, _SeenList, a=multirun(["x"]), b=hydra_list(values))
    ds = wf.to_xarray()
    expected = [[1, 2], ["a"], [1, "a"], [0, 1]]
    assert {str(r) for r in ds["b_repr"].values} == {repr(expected)}


class _SeenDict(MultiRunMetricsWorkflow):
    @staticmethod
    def task(d=None):
        from omegaconf import OmegaConf

        plain = OmegaConf.to_container(d) if OmegaConf.is_config(d) else d
        return dict(d_repr=repr(plain))


def test_dict_keys_hydra_can_write_reach_the_task(tmp_path):
    # A float and a bool key read back as themselves, not as their text, and
    # a NumPy integer key as the int it equals, as before.
    value = {"a": 1, 2: "y", "a b": 3, 1.5: "f", True: "t", np.int64(7): "n"}
    wf = _run(tmp_path, _SeenDict, d=multirun([value]))
    ds = wf.to_xarray()
    expected = {"a": 1, 2: "y", "a b": 3, 1.5: "f", True: "t", 7: "n"}
    assert {str(r) for r in ds["d_repr"].values} == {repr(expected)}


@pytest.mark.parametrize("key", ["1", "true", "null", "1.5", "a ", "a=b", "a:b"])
def test_a_dict_key_hydra_cannot_write_is_refused(tmp_path, key):
    """Hydra's dict grammar has no quoted key: a str key that reads as another
    type would reach the task changed ('1' as 1, 'a ' as 'a'), and one with
    '=' or ':' does not parse. Refused before launch, naming the key (#321)."""
    with pytest.raises(ValueError, match=re.escape(repr(key))):
        _run(tmp_path, _SeenDict, d=multirun([{key: "x"}]))
    assert not (tmp_path / "s").exists()


def test_a_key_omegaconf_does_not_take_is_refused(tmp_path):
    """Hydra reads `{null:0}` back, but OmegaConf's merge takes no None key:
    refused before launch, not inside it (#321)."""
    with pytest.raises(ValueError, match="OmegaConf"):
        _run(tmp_path, _SeenDict, d=multirun([{None: "x"}]))
    assert not (tmp_path / "s").exists()


def test_a_nan_dict_key_reaches_the_task(tmp_path):
    """NaN never equals itself, but it is the key it was, as on main."""
    wf = _run(tmp_path, _SeenDict, d=multirun([{float("nan"): "x"}]))
    ds = wf.to_xarray()
    assert {str(r) for r in ds["d_repr"].values} == {"{nan: 'x'}"}


def test_a_tuple_dict_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match=re.escape(repr(("a", 1)))):
        _run(tmp_path, _SeenDict, d=multirun([{("a", 1): "x"}]))


def test_a_glob_sweep_is_refused_by_name():
    """A glob sweep picks config-group options; the error says glob is not
    supported and names the param without its `+`."""
    with pytest.raises(ValueError, match=r"for 'a': Glob\..*`glob\(\.\.\.\)`"):
        MultiRunMetricsWorkflow._parse_overrides(["+a=glob(*)"])


def test_combo_key_is_injective_across_delimiters():
    from mushin._sweep_io import combo_key

    # A string value containing the key/value delimiters must not collide with
    # a genuinely different two-parameter combination.
    assert combo_key({"a": "1,b=2"}) != combo_key({"a": 1, "b": 2})
    # round-trippable distinctness for embedded '=' and ','
    assert combo_key({"a": "x=y"}) != combo_key({"a": "x", "y": ""})


def test_dotted_sweep_on_existing_config_field(tmp_path):
    # Sweeping model.width when the config already defines model.width must work
    # (not raise "item already at 'model.width'").
    from hydra_zen import make_config

    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(model=None):
            width = model["width"] if isinstance(model, dict) else model.width
            return dict(w=float(width))

    wf = W(make_config(model=make_config(width=4)))
    wf.run(working_dir=str(tmp_path / "s"), **{"model.width": multirun([4, 8])})
    ds = wf.to_xarray()
    assert sorted(ds["model.width"].values.tolist()) == [4, 8]
    assert float(ds["w"].sel({"model.width": 8})) == 8.0


def test_batching_sweeper_jobs_are_flattened(tmp_path):
    # A sweeper that returns jobs in multiple batches must not AssertionError
    # after all jobs have already run.
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(a):
            return dict(v=float(a))

    wf = W()
    wf.run(
        working_dir=str(tmp_path / "s"),
        overrides=["hydra.sweeper.max_batch_size=1"],
        a=multirun([1.0, 2.0, 3.0]),
    )
    ds = wf.to_xarray()
    assert sorted(ds["v"].values.tolist()) == [1.0, 2.0, 3.0]
