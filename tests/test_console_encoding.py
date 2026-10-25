# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""mushin's console output and files survive an encoding that is not UTF-8.

A Windows pipe writes in the ANSI code page (cp1252 on Western systems), and
``PYTHONIOENCODING=ascii`` or an ASCII locale makes stdout ASCII; ``print()``
of a character outside the encoding then raises ``UnicodeEncodeError`` and the
whole command fails. mushin's own text is ASCII, so it cannot hit that, and the
user data it prints (metric names, swept values) is written as backslash
escapes instead. The text files mushin writes are UTF-8 whatever the locale."""

from __future__ import annotations

import ast
import codecs
import io
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import mushin
import mushin.__main__ as cli
from mushin import multirun
from mushin._console import echo, for_console
from mushin.workflows import MultiRunMetricsWorkflow

ALPHA = "\u03b1"  # Greek alpha: in neither ASCII nor cp1252
METRIC = f"{ALPHA}_score"
# How the escape reads in the output: a backslash, then the code point.
ESCAPED = "\\u03b1"
# A Latin-1 letter: in cp1252, but its cp1252 byte is not UTF-8.
ZURICH = "Z\u00fcrich"
ZURICH_JSON = '{"Z\u00fcrich": 1.0}'
# The consoles of the bug: an ASCII locale or PYTHONIOENCODING, and the code
# page a Western Windows pipe writes in.
ENCODINGS = ["ascii", "cp1252"]


def _sweep(wd: Path, *, bump: float = 0.0, fail=None) -> None:
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr, seed):
            if (lr, seed) == fail:
                raise ValueError("this cell fails")
            # A metric name outside ASCII and cp1252 is user data mushin prints.
            return {"acc": float(lr) + seed + bump, METRIC: float(seed) + bump}

    W().run(
        lr=multirun([0.1, 0.2]),
        seed=multirun([0, 1]),
        working_dir=str(wd),
        on_error="nan",
    )


@pytest.fixture(scope="module")
def sweeps(tmp_path_factory) -> tuple[Path, Path]:
    """Two sweeps of one grid whose metrics differ, so ``diff`` prints deltas."""
    root = tmp_path_factory.mktemp("console")
    a, b = root / "a", root / "b"
    _sweep(a)
    _sweep(b, bump=0.5)
    return a, b


def _same_codec(a: str, b: str) -> bool:
    return codecs.lookup(a).name == codecs.lookup(b).name


def _stream(
    encoding: str, errors: str = "strict"
) -> tuple[io.BytesIO, io.TextIOWrapper]:
    """A text stream over bytes the test can read. ``newline=""`` keeps "\\n"
    as written: the default would write os.linesep, "\\r\\n" on Windows."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding, errors=errors, newline="")
    return raw, stream


def _on_stdout(monkeypatch, stream, fn, *args, **kwargs):
    """Call ``fn`` with ``sys.stdout`` set to ``stream``, then put the real
    one back. The swap happens in the test body and is undone before the test
    ends: pytest's capture sets ``sys.stdout`` again once fixture setup is
    over, and a swap undone only at teardown could restore a closed capture
    stream when capture is off (``pytest -s``)."""
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", stream)
        result = fn(*args, **kwargs)
    stream.flush()
    return result


def _on_ascii_stdout(monkeypatch, fn, *args, **kwargs) -> tuple[object, bytes]:
    """``fn``'s result and the bytes it wrote to a strict ASCII stdout, as on
    a console that cannot encode anything else."""
    raw, stream = _stream("ascii")
    return _on_stdout(monkeypatch, stream, fn, *args, **kwargs), raw.getvalue()


def _subprocess_env(encoding: str, **extra: str) -> dict[str, str]:
    env = {**os.environ, "PYTHONIOENCODING": f"{encoding}:strict", **extra}
    # UTF-8 mode is the workaround this test must not lean on.
    env.pop("PYTHONUTF8", None)
    return env


def _non_ascii_guide(tmp_path: Path) -> Path:
    guide = tmp_path / f"site-{ALPHA}" / "agent_guide.md"
    guide.parent.mkdir()
    guide.write_text("# guide\n", encoding="utf-8")
    return guide


def _assert_columns_line_up(lines: list[str]) -> None:
    """Every line of the table puts each column at the offset of its run of
    dashes, and keeps the two-space gaps between columns blank: what a cell
    measured on text other than what was printed breaks. ``_render`` pads
    every cell, the last one included, so the lines are of one length too.
    A blank cell (a missing value) is aligned like any other."""
    header, sep, *body = lines
    assert {len(line) for line in lines} == {len(sep)}, "\n".join(lines)
    runs = [(m.start(), m.end()) for m in re.finditer(r"-+", sep)]
    for line in (header, *body):
        for (_, end), (start, _) in zip(runs, runs[1:]):
            assert line[end:start] == "  ", line
    # Header names are never blank: each starts its column.
    assert all(header[start] != " " for start, _ in runs), header


# --- In process: each printing function on a strict ASCII stdout ----------


def test_show_prints_user_data_it_cannot_encode_as_escapes(sweeps, monkeypatch):
    a, _ = sweeps
    result, raw = _on_ascii_stdout(monkeypatch, mushin.show, a)
    out = raw.decode("ascii")
    assert f"{ESCAPED}_score" in out
    assert "completed" in out
    # Only the console copy is escaped; the returned table keeps the name.
    assert METRIC in result.table


def _hand_made_sweep(root: Path) -> None:
    """Two cells written as a run writes their sidecars: one whose swept
    value is outside ASCII (a launched sweep cannot have one: Hydra's
    override parser rejects it), and one that failed, whose metrics are
    blank."""
    from mushin._resume import write_cell_status
    from mushin._sweep_io import write_metrics_sidecar

    cells = [(f"{ALPHA}lpha", "completed"), ("b", "failed")]
    for i, (name, status) in enumerate(cells):
        d = root / str(i)
        d.mkdir(parents=True)
        write_cell_status(d, status=status, combo={"name": name}, attempt=1)
        if status == "completed":
            write_metrics_sidecar(d, {METRIC: 1.0, "acc": 0.5})


@pytest.mark.parametrize("errors", ["strict", "backslashreplace"])
def test_show_aligns_the_columns_of_an_escaped_table(tmp_path, monkeypatch, errors):
    # The escape is longer than its character: widths measured on the text
    # rather than on what is printed leave the column too narrow. The escaped
    # header comes before another column, which a misfit would shift; a body
    # cell is escaped too; the failed cell leaves blanks. A backslashreplace
    # stdout (a common Windows reconfigure) never fails, so it must be judged
    # strictly, or it expands the cells after they were measured.
    _hand_made_sweep(tmp_path / "s")
    raw, stream = _stream("ascii", errors=errors)
    result = _on_stdout(
        monkeypatch, stream, mushin.show, tmp_path / "s", metrics=[METRIC, "acc"]
    )
    lines = raw.getvalue().decode("ascii").splitlines()
    assert lines[0].split() == ["name", "status", f"{ESCAPED}_score", "acc"]
    # Rows sort by the swept value: "b" before the alpha.
    assert lines[2].split() == ["b", "failed"]  # blanks
    assert lines[3].split()[:2] == [f"{ESCAPED}lpha", "completed"]
    _assert_columns_line_up(lines)
    _assert_columns_line_up(result.table.splitlines())


def test_show_lays_the_table_out_again_only_when_it_needs_escapes(
    sweeps, monkeypatch, capsys
):
    # Rendering costs show() most of its time: a console that can print the
    # table (pytest's capture is UTF-8) gets it from the one rendering.
    import mushin._show as show_module

    a, _ = sweeps
    renders = []
    real = show_module._render

    def counting(*args, **kwargs):
        renders.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(show_module, "_render", counting)
    mushin.show(a)
    assert len(renders) == 1
    assert METRIC in capsys.readouterr().out
    renders.clear()
    _on_ascii_stdout(monkeypatch, mushin.show, a)
    assert len(renders) == 2


def test_diff_prints_ascii_headers_and_escapes_user_data(sweeps, monkeypatch):
    a, b = sweeps
    result, raw = _on_ascii_stdout(monkeypatch, mushin.diff, a, b)
    out = raw.decode("ascii")
    # One token per column, so the header splits on whitespace.
    header = out.splitlines()[0].split()
    assert header == ["lr", "seed", "acc(b-a)", f"{ESCAPED}_score(b-a)"]
    assert result.table.splitlines()[0].split()[-1] == f"{METRIC}(b-a)"
    row = next(r for r in result.rows if r["lr"] == 0.1 and r["seed"] == 0)
    assert row["deltas"][METRIC][2] == pytest.approx(0.5)
    # The table part (header, dashes, one line per shared cell) lines up.
    _assert_columns_line_up(out.splitlines()[: 2 + len(result.rows)])


def test_dry_run_preview_escapes_a_swept_value(tmp_path, monkeypatch):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(name, seed):
            return {"v": 1.0}

    summary, raw = _on_ascii_stdout(
        monkeypatch,
        W().run,
        name=multirun([f"{ALPHA}lpha", "b"]),
        seed=multirun([0, 1]),
        working_dir=str(tmp_path / "dry"),
        dry_run=True,
    )
    out = raw.decode("ascii")
    assert out.startswith("Dry run: 4 cells, no jobs launched.")
    assert f"'{ESCAPED}lpha'" in out
    assert summary["axes"]["name"] == [f"{ALPHA}lpha", "b"]


# --- guide --path: the real path or an error, never an escaped one ---------

# A path the file system decoded from a byte that is not valid in its
# encoding (surrogateescape on POSIX, surrogatepass on Windows): no stdout
# encoding takes it strictly, but os.fsencode gives its bytes back.
UNDECODABLE = "site-\udcff"


def test_guide_path_writes_bytes_when_stdout_shares_the_file_system_encoding(
    monkeypatch,
):
    raw, stream = _stream(sys.getfilesystemencoding())
    assert _on_stdout(monkeypatch, stream, cli._print_path, UNDECODABLE) == 0
    assert raw.getvalue() == os.fsencode(UNDECODABLE) + b"\n"


@pytest.mark.parametrize("errors", ["replace", "backslashreplace"])
def test_guide_path_ignores_a_lossy_error_handler(monkeypatch, errors):
    # The handler never fails, so asking it whether the path fits would
    # always say yes, and the console would get "site-?" or "site-\udcff".
    raw, stream = _stream(sys.getfilesystemencoding(), errors=errors)
    assert _on_stdout(monkeypatch, stream, cli._print_path, UNDECODABLE) == 0
    assert raw.getvalue() == os.fsencode(UNDECODABLE) + b"\n"


def test_guide_path_fails_when_stdout_would_misread_the_bytes(tmp_path, monkeypatch):
    # A reader decodes stdout in its declared encoding (a Windows pipe in the
    # code page): the path's file-system bytes would read as another path.
    if _same_codec(sys.getfilesystemencoding(), "ascii"):
        pytest.skip("the file system encoding here is ASCII")
    monkeypatch.setattr(cli, "_guide_path", lambda: _non_ascii_guide(tmp_path))
    raw, stream = _stream("ascii")
    stderr = io.StringIO()
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", stream)
        m.setattr(sys, "stderr", stderr)
        code = cli.main(["guide", "--path"])
    stream.flush()
    assert code == 1
    assert raw.getvalue() == b""
    # PYTHONUTF8=1 alone does nothing while PYTHONIOENCODING narrows stdout,
    # so the message names both.
    assert "PYTHONUTF8=1" in stderr.getvalue()
    assert "PYTHONIOENCODING" in stderr.getvalue()


class _NoCodecStdout:
    """Names an encoding the codecs registry does not know, as a wrapper's
    plain ``encoding`` attribute can, and takes any text."""

    encoding = "no-such-codec"
    errors = "strict"

    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, s: str) -> int:
        self.parts.append(s)
        return len(s)

    def flush(self) -> None:
        pass


def test_guide_path_writes_through_a_stream_naming_no_codec(monkeypatch):
    # As for every console write: no known encoding, nothing to judge by, so
    # the path goes out as it is rather than as an error.
    stream = _NoCodecStdout()
    assert _on_stdout(monkeypatch, stream, cli._print_path, f"site-{ALPHA}") == 0
    assert "".join(stream.parts) == f"site-{ALPHA}\n"


class _TeeWithBytes:
    """Reports UTF-8 and has a byte layer, but writes to a terminal and then
    raises from a narrower sink, as a tee-style stdout can."""

    encoding = "utf-8"
    errors = "strict"

    def __init__(self) -> None:
        self.terminal: list[str] = []
        self.buffer = io.BytesIO()

    def write(self, s: str) -> int:
        self.terminal.append(s)
        raise UnicodeEncodeError("ascii", s, 0, len(s), "the log is ASCII")

    def flush(self) -> None:
        pass


def test_guide_path_is_written_at_most_once_to_a_tee(monkeypatch):
    # The path fits the reported encoding, so it is written as text, once. A
    # sink that then raises must not get it a second time as bytes, and the
    # command ends with the message and status 1, not a traceback.
    stream, stderr = _TeeWithBytes(), io.StringIO()
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", stream)
        m.setattr(sys, "stderr", stderr)
        code = cli._print_path(f"site-{ALPHA}")
    assert code == 1
    assert stream.terminal == [f"site-{ALPHA}\n"]
    assert stream.buffer.getvalue() == b""
    assert "PYTHONUTF8=1" in stderr.getvalue()
    assert "PYTHONIOENCODING" in stderr.getvalue()


def test_main_module_runs_by_path(tmp_path):
    # `python path/to/mushin/__main__.py` runs the file as a script, where a
    # relative import has no package.
    result = subprocess.run(
        [sys.executable, cli.__file__, "guide"],
        capture_output=True,
        cwd=tmp_path,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-3000:]
    assert result.stdout.startswith(b"<!-- mushin ")


class _TextOnly:
    """A strict ASCII text stream with no byte layer (no ``.buffer``)."""

    encoding = "ascii"
    errors = "strict"

    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, s: str) -> int:
        s.encode(self.encoding)
        self.parts.append(s)
        return len(s)

    def flush(self) -> None:
        pass


def test_guide_path_fails_rather_than_print_an_escaped_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_guide_path", lambda: _non_ascii_guide(tmp_path))
    stdout, stderr = _TextOnly(), io.StringIO()
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", stdout)
        m.setattr(sys, "stderr", stderr)
        code = cli.main(["guide", "--path"])
    assert code == 1
    assert stdout.parts == []
    assert "cannot represent" in stderr.getvalue()


def test_a_missing_stderr_drops_the_message_instead_of_moving_it_to_stdout(
    tmp_path, monkeypatch
):
    # With fd 2 closed, sys.stderr is None. print(msg, file=sys.stderr) would
    # then write to stdout, where `$(... --path)` would capture the message
    # as the path; echo(file=None) writes nothing, on purpose.
    monkeypatch.setattr(cli, "_guide_path", lambda: tmp_path / "missing.md")
    raw, stdout = _stream("utf-8")
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", stdout)
        m.setattr(sys, "stderr", None)
        code = cli.main(["guide", "--path"])
    stdout.flush()
    assert code == 1
    assert raw.getvalue() == b""


# --- Subprocess: the real interpreter on a console of each encoding --------

_OFFLINE_VIEWS = """
import sys

import mushin
from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow

a, b, dry = sys.argv[1:4]
metric = "\\u03b1_score"
mushin.show(a)
mushin.diff(a, b)
print(mushin.best(a, metric).value, mushin.best(a, metric, over="seed").value)


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(name, seed):
        return {"v": 1.0}


W().run(
    name=multirun(["\\u03b1lpha", "b"]),
    seed=multirun([0, 1]),
    working_dir=dry,
    dry_run=True,
)
"""


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_offline_views_and_dry_run_survive_the_console(sweeps, tmp_path, encoding):
    a, b = sweeps
    result = subprocess.run(
        [sys.executable, "-c", _OFFLINE_VIEWS, str(a), str(b), str(tmp_path / "d")],
        capture_output=True,
        env=_subprocess_env(encoding),
        timeout=300,
    )
    stderr = result.stderr.decode(encoding, errors="replace")
    assert result.returncode == 0, stderr[-3000:]
    assert "UnicodeEncodeError" not in stderr
    out = result.stdout.decode(encoding)
    assert f"{ESCAPED}_score" in out  # show
    assert f"{ESCAPED}_score(b-a)" in out  # diff
    assert "Dry run: 4 cells" in out
    assert f"'{ESCAPED}lpha'" in out


@pytest.fixture(scope="module")
def mushin_on_a_non_ascii_path(tmp_path_factory) -> Path:
    """A copy of the imported mushin package under a directory whose name the
    test consoles cannot encode, so ``guide --path`` has a real path to print."""
    root = tmp_path_factory.mktemp("guide") / f"site-{ALPHA}"
    shutil.copytree(
        Path(mushin.__file__).parent,
        root / "mushin",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return root


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_guide_command_survives_the_console(
    mushin_on_a_non_ascii_path, tmp_path, encoding
):
    path_entries = [str(mushin_on_a_non_ascii_path), os.environ.get("PYTHONPATH")]
    env = _subprocess_env(
        encoding, PYTHONPATH=os.pathsep.join(p for p in path_entries if p)
    )

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, *args],
            capture_output=True,
            env=env,
            # -m puts the cwd first on sys.path; an empty one cannot shadow
            # the copy.
            cwd=tmp_path,
            timeout=300,
        )

    guide = run("-m", "mushin", "guide")
    assert guide.returncode == 0, guide.stderr.decode(errors="replace")[-3000:]
    assert guide.stdout.decode(encoding).startswith("<!-- mushin ")

    # The real path or an error: `test -s "$(... --path)"` must never get a
    # path that names no file with exit status 0.
    found = run("-m", "mushin", "guide", "--path")
    assert b"UnicodeEncodeError" not in found.stderr
    fs = run("-c", "import sys; print(sys.getfilesystemencoding())")
    if _same_codec(fs.stdout.decode().strip(), encoding):
        assert found.returncode == 0, found.stderr.decode(errors="replace")
        path = Path(os.fsdecode(found.stdout.strip()))
        assert path.samefile(mushin_on_a_non_ascii_path / "mushin" / "agent_guide.md")
    else:
        assert found.returncode == 1
        assert found.stdout == b""
        assert b"PYTHONUTF8=1" in found.stderr
        assert b"PYTHONIOENCODING" in found.stderr


# --- Files: UTF-8 whatever the locale --------------------------------------

_UNDER_AN_ASCII_LOCALE = """
import json
import locale
import subprocess
import sys

# The C locale stands in for a Windows code page: open() without an encoding
# then writes in one without the alpha. Where the fallback locale is UTF-8
# anyway there is nothing to test.
try:
    "\\u03b1".encode(locale.getpreferredencoding(False))
except UnicodeEncodeError:
    pass
else:
    sys.exit(77)

# uv's output with a character the locale lacks (a path in it, say), without
# depending on uv or on what it would print here. uv writes UTF-8 bytes; they
# are decoded as subprocess.run would decode them for the arguments it is
# given, so a decode in the locale's encoding fails here as it would there.
_run = subprocess.run


def fake_run(cmd, *args, **kwargs):
    if list(cmd)[:1] != ["uv"]:
        return _run(cmd, *args, **kwargs)
    out = "# /home/\\u03b1/project\\nfoo==1.0\\n".encode("utf-8")
    encoding = kwargs.get("encoding")
    if encoding is None and (kwargs.get("text") or kwargs.get("universal_newlines")):
        encoding = locale.getpreferredencoding(False)
    if encoding is not None:
        out = out.decode(encoding, kwargs.get("errors") or "strict")
    return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr=out[:0])


subprocess.run = fake_run

from mushin import export, multirun
from mushin.workflows import MultiRunMetricsWorkflow

wd, csv = sys.argv[1:3]


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(seed):
        if seed == 1:
            raise ValueError("bad \\u03b1")
        return {"\\u03b1_score": 1.0}


def sweep():
    W().run(seed=multirun([0, 1]), working_dir=wd, on_error="nan", capture_env=True)


def failures():
    from mushin.mcp.server import _get_failures

    return _get_failures(wd)["failures"][0].get("traceback")


steps = {}
for name, step in [
    ("sweep", sweep),
    ("export", lambda: str(export.table(wd, path=csv))),
    ("mcp", failures),
]:
    try:
        steps[name] = step()
    except ImportError as exc:  # mushin.mcp needs the mcp extra
        steps[name] = {"skip": repr(exc)}
    except Exception as exc:
        steps[name] = {"error": repr(exc)}
print(json.dumps(steps))
"""


@pytest.fixture(scope="module")
def ascii_locale_run(tmp_path_factory) -> tuple[Path, Path, dict]:
    """One fail-soft sweep with ``capture_env=True``, its CSV export and its
    MCP failure listing, all under an ASCII locale: the sweep's directory, the
    CSV path and what each step returned (or raised)."""
    root = tmp_path_factory.mktemp("locale")
    wd, csv = root / "s", root / "cells.csv"
    # LC_ALL=C with UTF-8 mode and locale coercion off gives an ASCII locale on
    # Linux and macOS; Windows ignores LC_ALL and keeps its ANSI code page.
    env = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
    }
    env.pop("PYTHONIOENCODING", None)
    result = subprocess.run(
        [sys.executable, "-c", _UNDER_AN_ASCII_LOCALE, str(wd), str(csv)],
        capture_output=True,
        env=env,
        timeout=300,
    )
    if result.returncode == 77:
        pytest.skip("this platform's C locale encodes the test character")
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-3000:]
    steps = json.loads(result.stdout.decode("ascii").splitlines()[-1])
    return wd, csv, steps


def test_fail_soft_survives_an_error_message_the_locale_cannot_encode(
    ascii_locale_run,
):
    wd, _, steps = ascii_locale_run
    assert steps["sweep"] is None, steps["sweep"]
    (error_file,) = wd.glob("*/mushin_error.txt")
    assert f"ValueError: bad {ALPHA}" in error_file.read_text(encoding="utf-8")


def test_env_snapshot_keeps_uv_output_the_locale_cannot_decode(ascii_locale_run):
    # Decoded in an ASCII locale's encoding, uv's UTF-8 output raised, and the
    # snapshot silently fell back to the bare package list.
    wd, _, _ = ascii_locale_run
    snapshot = (wd / "mushin_env.txt").read_text(encoding="utf-8")
    assert f"/home/{ALPHA}/project" in snapshot


def test_export_table_writes_utf8_under_a_locale_that_lacks_a_metric_name(
    ascii_locale_run,
):
    _, csv, steps = ascii_locale_run
    assert steps["export"] == str(csv), steps["export"]
    assert METRIC in csv.read_text(encoding="utf-8").splitlines()[0]


def test_mcp_get_failures_reads_the_traceback_under_that_locale(ascii_locale_run):
    _, _, steps = ascii_locale_run
    if isinstance(steps["mcp"], dict) and "skip" in steps["mcp"]:
        pytest.skip(steps["mcp"]["skip"])
    assert isinstance(steps["mcp"], str), steps["mcp"]
    assert f"ValueError: bad {ALPHA}" in steps["mcp"]


@pytest.mark.parametrize(
    "data",
    [
        # A file in the locale's encoding, as Windows writes by default and as
        # an earlier release read every JSON metrics file: not UTF-8.
        ZURICH_JSON.encode("cp1252"),
        ZURICH_JSON.encode("utf-8"),
        # With a byte-order mark, or in UTF-16/32 (as Windows tools write): the
        # first byte is not "{", so the sniff must decode before it looks.
        ZURICH_JSON.encode("utf-8-sig"),
        ZURICH_JSON.encode("utf-16"),
        ZURICH_JSON.encode("utf-16-le"),
        ZURICH_JSON.encode("utf-16-be"),
        ZURICH_JSON.encode("utf-32"),
        ("\n" + ZURICH_JSON).encode("utf-16-le"),
    ],
    ids=[
        "cp1252-locale",
        "utf-8",
        "utf-8-sig",
        "utf-16-bom",
        "utf-16-le",
        "utf-16-be",
        "utf-32-bom",
        "utf-16-le-leading-newline",
    ],
)
def test_metric_load_fn_reads_a_users_json_in_any_of_its_encodings(
    tmp_path, monkeypatch, data
):
    import locale

    path = tmp_path / "my_metrics.json"
    path.write_bytes(data)
    # UTF-8 mode (PYTHONUTF8=1, which the agent guide recommends on Windows):
    # getpreferredencoding() says utf-8, while the locale's code page, the one
    # the file was written in, is cp1252.
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a: "utf-8")
    monkeypatch.setattr(locale, "getencoding", lambda: "cp1252")
    assert MultiRunMetricsWorkflow.metric_load_fn(path) == {ZURICH: 1.0}


_GIT_UNDER_AN_ASCII_LOCALE = """
import json
import locale
import sys

try:
    "\\u03b1".encode(locale.getpreferredencoding(False))
except UnicodeEncodeError:
    pass
else:
    sys.exit(77)

from mushin._provenance import _git

print(json.dumps(_git()))
"""


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_provenance_records_a_non_ascii_branch_under_any_locale(tmp_path):
    # git writes ref names in UTF-8; decoded in an ASCII locale (or cp1252)
    # the branch was recorded as None (or garbled).
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "start")
    git("checkout", "-q", "-b", f"feat-{ALPHA}")
    env = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
    }
    env.pop("PYTHONIOENCODING", None)
    result = subprocess.run(
        [sys.executable, "-c", _GIT_UNDER_AN_ASCII_LOCALE],
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=300,
    )
    if result.returncode == 77:
        pytest.skip("this platform's C locale encodes the test character")
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-3000:]
    assert json.loads(result.stdout)["branch"] == f"feat-{ALPHA}"


# --- The helper ------------------------------------------------------------


def test_echo_escapes_only_what_the_encoding_lacks():
    # cp1252 has the u-umlaut, so it prints as itself; only the alpha escapes.
    raw, stream = _stream("cp1252")
    echo(f"Z\u00fcrich {ALPHA}", file=stream)
    stream.flush()
    assert raw.getvalue() == b"Z\xfcrich \\u03b1\n"


def test_echo_writes_text_the_stream_can_encode_unchanged():
    raw, stream = _stream("utf-8")
    echo(METRIC, end="", file=stream)
    stream.flush()
    assert raw.getvalue() == METRIC.encode("utf-8")


def test_echo_leaves_a_lenient_error_handler_in_charge():
    # A POSIX locale gives stdout surrogateescape, which writes an undecodable
    # path's lone surrogate back as its original byte; escaping it would not.
    raw, stream = _stream("ascii", errors="surrogateescape")
    echo("\udcff", file=stream)
    stream.flush()
    assert raw.getvalue() == b"\xff\n"


@pytest.mark.parametrize(
    "text, expected",
    [
        (f"\udcff {ALPHA}", b"\xff \\u03b1\n"),
        # One run for the codec: the surrogate right after the alpha must
        # still reach the handler on its own.
        (f"{ALPHA}\udcff", b"\\u03b1\xff\n"),
    ],
)
def test_echo_escapes_only_what_a_lenient_handler_cannot_take(text, expected):
    raw, stream = _stream("ascii", errors="surrogateescape")
    echo(text, file=stream)
    stream.flush()
    assert raw.getvalue() == expected


def test_for_console_escapes_a_long_text_in_one_pass():
    # Re-encoding the rest of the text at each escape is quadratic (20 s for a
    # 50k-row table). A codec that is ASCII and tallies what it is handed
    # counts the work without timing it: one check, then one escaping pass.
    tally: list[int] = []

    def search(name):
        if name != "mushin_tally_ascii":
            return None

        def encode(text, errors="strict"):
            tally.append(len(text))
            return codecs.ascii_encode(text, errors)

        return codecs.CodecInfo(encode, codecs.ascii_decode, name=name)

    text = "\n".join(f"row {i} {ALPHA}" for i in range(1000))
    codecs.register(search)
    try:
        stream = types.SimpleNamespace(encoding="mushin_tally_ascii", errors="strict")
        out = for_console(text, stream)
    finally:
        codecs.unregister(search)
    assert out.count(ESCAPED) == 1000 and out.isascii()
    assert sum(tally) <= 2 * len(text), f"{len(tally)} encodes"


def _scanning_codec(base: str, budget: int) -> tuple[str, object]:
    """A codec that encodes as ``base`` does, written out in Python the way
    CPython's codecs handle errors: find the end of the run of characters it
    cannot encode, hand the run to the error handler, resume where the
    handler says. Looking at more than ``budget`` characters in all fails,
    so a handler that returns one character at a time (and so has the codec
    scan the rest of the run again at every call) fails fast."""
    name = f"mushin_scan_{base}"
    seen = [0]

    def fits(ch: str) -> bool:
        seen[0] += 1
        if seen[0] > budget:
            raise AssertionError(f"looked at more than {budget} characters")
        try:
            ch.encode(base)
        except UnicodeEncodeError:
            return False
        return True

    def encode(text: str, errors: str = "strict"):
        out, i = [], 0
        while i < len(text):
            if fits(text[i]):
                out.append(text[i].encode(base))
                i += 1
                continue
            end = i + 1
            while end < len(text) and not fits(text[end]):
                end += 1
            exc = UnicodeEncodeError(base, text, i, end, "not in the code page")
            replacement, i = codecs.lookup_error(errors)(exc)
            out.append(
                replacement
                if isinstance(replacement, bytes)
                else replacement.encode(base)
            )
        return b"".join(out), len(text)

    def decode(data, errors: str = "strict"):
        return bytes(data).decode(base, errors), len(data)

    def search(n: str):
        return codecs.CodecInfo(encode, decode, name=name) if n == name else None

    return name, search


@pytest.mark.parametrize("errors", ["strict", "surrogateescape"])
@pytest.mark.parametrize("base", ["ascii", "cp1252"])
def test_a_long_unencodable_run_is_escaped_in_one_pass(base, errors):
    # One run of 20k CJK characters, in neither code page: a handler that
    # returned one character at a time took 28 s on cp437.
    text = chr(0x4E2D) * 20_000
    name, search = _scanning_codec(base, budget=5 * len(text))
    codecs.register(search)
    try:
        stream = types.SimpleNamespace(encoding=name, errors=errors)
        out = for_console(text, stream)
    finally:
        codecs.unregister(search)
    assert out == "\\u4e2d" * 20_000


def test_a_stream_naming_no_codec_gets_the_text_as_it_is():
    stream = _NoCodecStdout()
    assert for_console(METRIC, stream) == METRIC
    echo(METRIC, file=stream)
    assert "".join(stream.parts) == f"{METRIC}\n"


class _Tee:
    """Writes to a terminal, then to a cp1252 log, as a tee-style
    ``sys.stdout`` does: the log raises after the terminal has the text."""

    encoding = "cp1252"
    errors = "strict"

    def __init__(self, terminal: io.StringIO, log: io.TextIOWrapper) -> None:
        self.terminal, self.log = terminal, log

    def write(self, s: str) -> int:
        self.terminal.write(s)
        return self.log.write(s)


def test_echo_writes_a_tee_once():
    terminal = io.StringIO()
    raw, log = _stream("cp1252")
    echo(f"{ALPHA} done", file=_Tee(terminal, log))
    log.flush()
    assert terminal.getvalue() == f"{ESCAPED} done\n"
    assert raw.getvalue() == b"\\u03b1 done\n"


class _Forward:
    """Reports UTF-8 but forwards to a narrower sink, as Jupyter's output
    stream (copying to a cp1252 echo stream) and pytest's ``tee-sys`` do."""

    encoding = "utf-8"
    errors = "strict"

    def __init__(self, sink: io.TextIOWrapper) -> None:
        self.sink = sink

    def write(self, s: str) -> int:
        return self.sink.write(s)


def test_echo_survives_a_stream_whose_encoding_is_wider_than_its_sink():
    raw, sink = _stream("ascii")
    echo(METRIC, file=_Forward(sink))
    sink.flush()
    assert raw.getvalue() == b"\\u03b1_score\n"


def test_echo_without_stdout_is_a_no_op(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", None)  # pythonw.exe has no console
        echo("nothing to write to")


def test_a_utf8_console_shows_user_data_as_itself(sweeps, capsys):
    # The escape is a fallback, not a filter: a console that can show the
    # metric name (pytest's capture is UTF-8) gets the name itself.
    a, _ = sweeps
    mushin.show(a)
    assert METRIC in capsys.readouterr().out


# --- Guards: mushin's runtime strings are ASCII, its file I/O UTF-8 --------

SRC = Path(mushin.__file__).parent
# (path relative to the package, the literal) -> why it must stay non-ASCII.
# Empty: an entry needs a reason a plain-ASCII rewrite cannot meet.
ALLOWED: dict[tuple[str, str], str] = {}


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        yield path.relative_to(SRC).as_posix(), tree


def _non_ascii_runtime_strings(tree: ast.AST) -> list[tuple[int, str]]:
    # A string that is a statement of its own (a docstring, or a bare string
    # documenting an attribute) is never written to a console by mushin:
    # help() escapes for the console itself (pydoc), and nothing prints it.
    statements = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in statements
        and not node.value.isascii()
    ]


def test_runtime_string_literals_are_ascii():
    found = [
        f"{rel}:{line}: {value!r}"
        for rel, tree in _modules()
        for line, value in _non_ascii_runtime_strings(tree)
        if (rel, value) not in ALLOWED
    ]
    assert not found, (
        "runtime strings with non-ASCII characters (a console that cannot "
        "encode them escapes or fails them; rewrite in ASCII):\n" + "\n".join(found)
    )


# Modules whose open() takes the file first and then the mode, like the
# builtin; the compressed ones default to binary and mean text only with "t".
_OPEN_FILE_THEN_MODE = {"io", "codecs", "gzip", "bz2", "lzma"}
_OPEN_BINARY_BY_DEFAULT = {"gzip", "bz2", "lzma"}


def _opens_text(node: ast.Call) -> bool | None:
    """Whether an ``open`` call opens a text stream: None for ``os.open``
    (flags and a file descriptor, no text layer), and True when the mode is
    not a literal, since nothing then shows it is binary."""
    func = node.func
    owner = (
        func.value.id
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
        else None
    )
    if owner == "os":
        return None
    # open(file, mode) and module.open(file, mode), but path.open(mode).
    file_first = isinstance(func, ast.Name) or owner in _OPEN_FILE_THEN_MODE
    args = node.args[1:] if file_first else node.args
    modes = [*args[:1], *(k.value for k in node.keywords if k.arg == "mode")]
    if not modes:
        return owner not in _OPEN_BINARY_BY_DEFAULT
    if not isinstance(modes[0], ast.Constant):
        return True
    mode = str(modes[0].value)
    if owner in _OPEN_BINARY_BY_DEFAULT:
        return "t" in mode
    return "b" not in mode


def _text_io_without_encoding(tree: ast.AST) -> list[int]:
    """Lines of ``read_text``/``write_text`` calls, and of ``open`` calls in
    a text mode, that leave the encoding to the locale."""
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in ("read_text", "write_text", "open"):
            continue
        if any(k.arg == "encoding" for k in node.keywords):
            continue
        if name == "open" and not _opens_text(node):
            continue
        lines.append(node.lineno)
    return lines


@pytest.mark.parametrize(
    "call, flagged",
    [
        ("open(p)", True),
        ("open(p, 'w')", True),
        ("open(p, mode)", True),  # a mode nothing shows to be binary
        ("io.open(p, 'r')", True),
        ("codecs.open(p, 'w')", True),
        ("gzip.open(p, 'rt')", True),
        ("p.open()", True),
        ("p.open('a')", True),
        ("p.read_text()", True),
        ("p.write_text(s)", True),
        ("open(p, 'rb')", False),
        ("open(p, encoding='utf-8')", False),
        ("io.open(p, 'rb')", False),
        ("codecs.open(p, 'r', encoding='utf-8')", False),
        ("gzip.open(p)", False),
        ("gzip.open(p, 'rb')", False),
        ("lzma.open(p, mode='wb')", False),
        ("os.open(p, os.O_RDONLY)", False),
        ("p.open('rb')", False),
        ("p.read_text(encoding='utf-8')", False),
    ],
)
def test_the_encoding_guard_tells_text_io_from_binary(call, flagged):
    assert bool(_text_io_without_encoding(ast.parse(call))) is flagged


def test_mushin_reads_and_writes_text_files_as_utf8():
    # A locale-encoded file breaks under cp1252 (a write that raises, a file
    # one machine writes and another misreads), so every text file mushin
    # writes or reads names its encoding.
    found = [
        f"{rel}:{line}"
        for rel, tree in _modules()
        for line in _text_io_without_encoding(tree)
    ]
    assert not found, "text file I/O without encoding=:\n" + "\n".join(found)
