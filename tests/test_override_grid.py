"""Regression tests for the override -> grid pipeline.

These pin fixes for the silent-wrong-results cluster: fixed (non-swept) string
values must not become accidental sweeps or change type; a duplicated sweep
axis value must be rejected rather than silently collapsed; `combo_key` must be
injective across delimiter-containing values; a dotted sweep on a config field
that already exists must not crash; and a batching sweeper's nested job list
must be flattened.
"""

import re
import string
from pathlib import Path
from typing import Optional

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings

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


@pytest.mark.parametrize(
    "values, shown",
    [
        (["true", "x"], "'true' would reach the task as True"),
        (["null"], "'null' would reach the task as None"),
        (["1", "2"], "'1' would reach the task as 1, '2' would reach the task as 2"),
        (["1.5"], "'1.5' would reach the task as 1.5"),
        (["x", "1e3"], "'1e3' would reach the task as 1000.0"),
        (["''"], "\"''\" would reach the task as ''"),
        ([np.str_("1")], f"{np.str_('1')!r} would reach the task as 1"),
        (["it's"], '"it\'s" would not parse in its job'),
        (["r\u00e9sum\u00e9"], "'r\u00e9sum\u00e9' would not parse in its job"),
        # An interpolation string is checked too, in the form it is written.
        (["x", "${now:%Y}'s"], '"${now:%Y}\'s" would not parse in its job'),
    ],
)
def test_a_swept_string_hydra_would_not_keep_is_refused(tmp_path, values, shown):
    """``multirun`` sweeps through Hydra's ``choice()``, which writes each
    job's value without its quotes: a string that reads as another value
    reached the task as that value (its cell NaN, or keyed as the string), and
    one Hydra cannot write unquoted failed inside the launch. Each is refused
    before launch, a dry run included, naming it (#319)."""
    with pytest.raises(ValueError, match=re.escape(f"`a`: {shown}")):
        _run(tmp_path, _Seen, a=multirun(values))
    assert not (tmp_path / "s").exists()
    with pytest.raises(ValueError, match=re.escape(shown)):
        _Seen().run(a=multirun(values), dry_run=True)


def test_the_refusal_names_a_quoted_sweep_that_keeps_the_strings(tmp_path):
    """The refusal's raw comma sweep keeps each value's quotes, so the strings
    it names reach the task as written, each in its own cell (#319)."""
    with pytest.raises(ValueError) as err:
        _run(tmp_path, _Seen, a=multirun(["true", "it's", 2]))
    raw = "+a='true','it\\'s',2"
    assert f"overrides={[raw]!r}" in str(err.value)
    wf = _run(tmp_path, _Seen, overrides=[raw])
    ds = wf.to_xarray()
    assert [str(r) for r in ds["a_repr"].values] == [repr("true"), repr("it's"), "2"]


def test_the_refusal_names_every_axis_and_shortens_a_long_list(tmp_path):
    """Every refused axis is named, a long one with its first values, and the
    raw sweep is written out whole: one cut short would not parse (#319)."""
    values = [str(i) for i in range(8)]
    with pytest.raises(ValueError) as err:
        _run(tmp_path, _Seen, a=multirun(values), b=multirun(["x", "null"]))
    message = str(err.value)
    assert "`a`: '0' would reach the task as 0" in message
    assert "'4' would reach the task as 4, and 3 more." in message
    assert "`b`: 'null' would reach the task as None." in message
    raws = ["+a=" + ",".join(f"'{v}'" for v in values), "+b='x','null'"]
    assert f"overrides={raws!r}" in message
    assert "passed as that type" in message


def test_a_one_value_sweep_is_told_to_pass_the_value_fixed(tmp_path):
    """A one-value comma "sweep" is a fixed value, which would drop the axis,
    so a one-value axis is offered the fixed value and the dimension it can
    keep; a value that does not parse is not offered a type (#319)."""
    with pytest.raises(ValueError) as err:
        _run(tmp_path, _Seen, a=multirun(["it's"]))
    message = str(err.value)
    assert "A one-value sweep (`a`) has no such form: pass the value fixed" in message
    assert "non_multirun_params_as_singleton_dims=True" in message
    assert "overrides=" not in message
    assert "passed as that type" not in message


@pytest.mark.parametrize(
    "values",
    [
        ["x", "a b", "x,y", "a=b", "None", "007", "1a", "C:\\dir\\"],
        [1, 2.5, True],
        [["true"], {"k": "null"}],
    ],
)
def test_a_sweep_hydra_keeps_runs_as_before(tmp_path, values):
    """What Hydra hands its job as written still runs, each job's override in
    the form it always had: the check refuses, it never rewrites (#319)."""
    wf = _run(tmp_path, _Seen, a=multirun(values))
    ds = wf.to_xarray()
    assert [str(r) for r in ds["a_repr"].values] == [repr(v) for v in values]
    assert _job_overrides(tmp_path, values) == _written_by_choice("+a", values)


def _job_overrides(tmp_path, values):
    return [
        (tmp_path / "s" / str(n) / ".hydra" / "overrides.yaml").read_text()
        for n in range(len(values))
    ]


def _written_by_choice(key, values):
    """Each job's overrides.yaml as Hydra's basic sweeper writes a
    `choice(...)` sweep of these values."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    from mushin.workflows import _to_override_element

    sweep = OverridesParser.create().parse_overrides(
        [f"{key}=choice(" + ",".join(_to_override_element(v) for v in values) + ")"]
    )[0]
    return [f"- {key}={job}\n" for job in sweep.sweep_string_iterator()]


# `Optional[str]`, not `str | None`: hydra-zen types a field annotated
# `str | None` as Any, which keeps what Hydra reads.
OPTIONAL_STR = Optional[str]  # noqa: UP045


def _typed(type_, default):
    from hydra_zen import ZenField, make_config

    class W(_Seen):
        pass

    return lambda: W(
        make_config(a=ZenField(type_, default), b=ZenField(type_, default))
    )


@pytest.mark.parametrize(
    "type_, default, values, task_got",
    [
        (str, "x", ["1", "True", "inf"], ["'1'", "'True'", "'inf'"]),
        (OPTIONAL_STR, None, ["1", "007"], ["'1'", "'007'"]),
        (int, 0, ["1", "2"], ["1", "2"]),
        # A NaN is the same value swept and fixed, though never equal.
        (float, 0.0, ["nan", "1.5"], ["nan", "1.5"]),
    ],
)
def test_a_string_the_typed_field_turns_back_runs_as_before(
    tmp_path, type_, default, values, task_got
):
    """The job's config is the second reader: a `str` field turns Hydra's 1
    back into '1', and an `int` field takes '1' as 1 whether swept or fixed,
    so these reach the task as on main, each override as it was (#319)."""
    # The dry run's preview reads the grid's fixed values after the check.
    summary = _typed(type_, default)().run(a=multirun(values), dry_run=True)
    assert summary["axes"]["a"] == values
    assert summary["fixed"] == {}
    wf = _typed(type_, default)()
    wf.run(a=multirun(values), working_dir=str(tmp_path / "s"))
    assert [str(r) for r in wf.to_xarray()["a_repr"].values] == task_got
    prefix = "+" if default is None else ""
    assert _job_overrides(tmp_path, values) == _written_by_choice(prefix + "a", values)


@pytest.mark.parametrize(
    "type_, default, values, shown",
    [
        (str, "x", ["true", "x"], "'true' would reach the task as 'True'"),
        (str, "x", ["x", "1e3"], "'1e3' would reach the task as '1000.0'"),
        (str, "x", ["null", "x"], "'null' would fail in its job"),
        (OPTIONAL_STR, None, ["null", "x"], "'null' would reach the task as None"),
        (str, "x", ["it's", "x"], '"it\'s" would not parse in its job'),
    ],
)
def test_a_string_the_typed_field_still_changes_is_refused(
    tmp_path, type_, default, values, shown
):
    """What the field still changes is refused, as it arrives (#319)."""
    with pytest.raises(ValueError, match=re.escape(f"`a`: {shown}")):
        _typed(type_, default)().run(
            a=multirun(values), working_dir=str(tmp_path / "s")
        )
    assert not (tmp_path / "s").exists()


def test_a_value_only_the_sweep_gives_the_field_is_not_refused():
    """An Enum field takes Hydra's 1 as its member, but not '1' passed fixed:
    the sweep is not what fails there, so it is not refused (#319)."""
    import enum

    class Color(enum.Enum):
        RED = 1
        BLUE = 2

    summary = _typed(Color, Color.RED)().run(a=multirun(["1", "2"]), dry_run=True)
    assert summary["axes"]["a"] == ["1", "2"]


@pytest.mark.parametrize(
    "how", [dict(sweeper="local_test"), dict(overrides=["hydra/sweeper=local_test"])]
)
def test_another_sweeper_is_left_to_write_its_jobs(how):
    """The check models Hydra's basic sweeper; another sweeper writes its
    jobs its own way, so its sweep is not refused (#319)."""
    summary = _Seen().run(a=multirun(["true", "x"]), dry_run=True, **how)
    assert summary["axes"]["a"] == ["true", "x"]
    with pytest.raises(ValueError, match="would reach the task as True"):
        _Seen().run(a=multirun(["true", "x"]), dry_run=True, sweeper="basic")


def test_only_values_written_as_strings_are_read_back(monkeypatch):
    """The check reads back only the values written quoted, one call per
    axis: an axis of numbers costs it nothing (#319)."""
    import mushin.workflows as w

    seen = []

    def read(written):
        seen.append(written)
        return [(text[1:-1], text[1:-1]) for text in written]

    monkeypatch.setattr(w, "_swept_strings_read", read)
    _Seen().run(a=multirun(list(range(1000))), b=multirun([True, 1.5]), dry_run=True)
    assert seen == []
    _Seen().run(a=multirun(["x", 1, "y"]), dry_run=True)
    assert seen == [["'x'", "'y'"]]


@pytest.mark.parametrize("typed", [False, True])
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(text=st.text(st.sampled_from(string.printable.replace("$", "")), max_size=6))
@example(text="true")
@example(text="1")
@example(text="it's")
@example(text="a b")
def test_a_swept_string_reaches_the_task_as_a_fixed_one_or_is_refused(
    tmp_path, typed, text
):
    """Any string swept alone, into an untyped key or a `str` field, is
    refused before launch, or reaches the task as the same string passed as a
    fixed value does (#319)."""
    import tempfile

    make = _typed(str, "") if typed else _Seen
    base = Path(tempfile.mkdtemp(dir=tmp_path))
    try:
        wf = make()
        wf.run(a=multirun([text]), working_dir=str(base / "swept"))
    except ValueError:
        assert not (base / "swept").exists()
        return
    fixed = make()
    fixed.run(a=multirun(["anchor"]), b=text, working_dir=str(base / "fixed"))
    swept = wf.to_xarray()["a_repr"].values.tolist()
    assert swept == fixed.to_xarray()["b_repr"].values.tolist()


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


def test_a_list_config_interpolation_is_written_unresolved():
    """A ListConfig's `${a}` is written as written, for the job to resolve,
    not resolved against the list itself before launch (#321)."""
    from omegaconf import OmegaConf

    from mushin.workflows import _to_override_element

    assert _to_override_element(OmegaConf.create(["${a}"])) == "['${a}']"


def test_an_array_in_a_hydra_list_reaches_the_task_as_lists(tmp_path):
    from mushin import hydra_list

    wf = _run(tmp_path, _SeenList, a=multirun(["x"]), b=hydra_list([np.eye(2)]))
    ds = wf.to_xarray()
    assert {str(r) for r in ds["b_repr"].values} == {repr([[[1.0, 0.0], [0.0, 1.0]]])}


def test_a_user_string_keeps_the_text_main_wrote(tmp_path):
    """Only the sequences main read back as a list are written as one: a
    UserString, a Sequence too, keeps its text instead of recursing (#321)."""
    from collections import UserString

    wf = _run(tmp_path, _Seen, a=multirun([UserString("ab")]))
    ds = wf.to_xarray()
    assert [str(r) for r in ds["a_repr"].values] == [repr("ab")]


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


@pytest.mark.parametrize(
    "key", ["1", "true", "null", "1.5", "a ", "a=b", "a:b", "a:0,b", "a:0},{b"]
)
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


@pytest.mark.parametrize("nan", [float("nan"), np.float32("nan")])
def test_a_nan_dict_key_reaches_the_task(tmp_path, nan):
    """NaN never equals itself, but it is the key it was, as on main."""
    wf = _run(tmp_path, _SeenDict, d=multirun([{nan: "x"}]))
    ds = wf.to_xarray()
    assert {str(r) for r in ds["d_repr"].values} == {"{nan: 'x'}"}


def test_a_str_enum_key_is_written_as_its_value(tmp_path):
    """A key mixing in str is written as its value, as such a value is."""
    import enum

    class Mode(str, enum.Enum):  # noqa: UP042 - str() is 'Mode.TRAIN', the case
        TRAIN = "train"

    wf = _run(tmp_path, _SeenDict, d=multirun([{Mode.TRAIN: 1}]))
    ds = wf.to_xarray()
    assert {str(r) for r in ds["d_repr"].values} == {repr({"train": 1})}


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
