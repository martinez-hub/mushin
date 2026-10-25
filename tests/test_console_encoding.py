# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""mushin's console output survives a console that cannot encode it.

A Windows pipe writes in the ANSI code page (cp1252 on Western systems), and
``PYTHONIOENCODING=ascii`` or an ASCII locale makes stdout ASCII; ``print()``
of a character outside the encoding then raises ``UnicodeEncodeError`` and the
whole command fails. mushin's own text is ASCII, so it cannot hit that, and the
user data it prints (metric names, swept values) is written as backslash
escapes instead."""

from __future__ import annotations

import ast
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import mushin
import mushin.__main__ as cli
from mushin import multirun
from mushin._console import echo
from mushin.workflows import MultiRunMetricsWorkflow

ALPHA = "\u03b1"  # Greek alpha: in neither ASCII nor cp1252
METRIC = f"{ALPHA}_score"
# How the escape reads in the output: a backslash, then the code point.
ESCAPED = "\\u03b1"
# The consoles of the bug: an ASCII locale or PYTHONIOENCODING, and the code
# page a Western Windows pipe writes in.
ENCODINGS = ["ascii", "cp1252"]


def _sweep(wd: Path, *, bump: float = 0.0) -> None:
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(lr, seed):
            # A metric name outside ASCII and cp1252 is user data mushin prints.
            return {"acc": float(lr) + seed + bump, METRIC: float(seed) + bump}

    W().run(lr=multirun([0.1, 0.2]), seed=multirun([0, 1]), working_dir=str(wd))


@pytest.fixture(scope="module")
def sweeps(tmp_path_factory) -> tuple[Path, Path]:
    """Two sweeps of one grid whose metrics differ, so ``diff`` prints deltas."""
    root = tmp_path_factory.mktemp("console")
    a, b = root / "a", root / "b"
    _sweep(a)
    _sweep(b, bump=0.5)
    return a, b


def _stream(
    encoding: str, errors: str = "strict"
) -> tuple[io.BytesIO, io.TextIOWrapper]:
    """A text stream over bytes the test can read. ``newline=""`` keeps "\\n"
    as written: the default would write os.linesep, "\\r\\n" on Windows."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding, errors=errors, newline="")
    return raw, stream


def _on_ascii_stdout(monkeypatch, fn, *args, **kwargs) -> tuple[object, bytes]:
    """Call ``fn`` with ``sys.stdout`` a strict ASCII stream, as on a console
    that cannot encode anything else, and return its result and the bytes it
    wrote.

    The swap happens in the test body, not a fixture: pytest's capture sets
    ``sys.stdout`` again once fixture setup is over."""
    raw, stream = _stream("ascii")
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", stream)
        result = fn(*args, **kwargs)
    stream.flush()
    return result, raw.getvalue()


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
    """``_render`` pads every cell, the last one included, so the lines of a
    table whose widths were measured on the text it prints are of one length,
    and every header and row cell starts where its run of dashes does."""
    header, sep, *body = lines
    assert {len(line) for line in lines} == {len(sep)}, "\n".join(lines)
    starts = [m.start() for m in re.finditer(r"-+", sep)]
    for line in (header, *body):
        for start in starts[1:]:
            assert line[start - 2 : start] == "  " and line[start] != " ", line


# --- In process: each printing function on a strict ASCII stdout ----------


def test_show_prints_user_data_it_cannot_encode_as_escapes(sweeps, monkeypatch):
    a, _ = sweeps
    result, raw = _on_ascii_stdout(monkeypatch, mushin.show, a)
    out = raw.decode("ascii")
    assert f"{ESCAPED}_score" in out
    assert "completed" in out
    # Only the console copy is escaped; the returned table keeps the name.
    assert METRIC in result.table


def test_show_aligns_the_columns_of_an_escaped_table(sweeps, monkeypatch):
    # The escape is longer than the name: widths measured on the name would
    # leave the header wider than its column.
    # The escaped column comes before another, which a misfit would shift.
    a, _ = sweeps
    result, raw = _on_ascii_stdout(monkeypatch, mushin.show, a, metrics=[METRIC, "acc"])
    lines = raw.decode("ascii").splitlines()
    assert lines[0].split() == ["lr", "seed", "status", f"{ESCAPED}_score", "acc"]
    _assert_columns_line_up(lines)
    _assert_columns_line_up(result.table.splitlines())


def test_diff_prints_ascii_headers_and_escapes_user_data(sweeps, monkeypatch):
    a, b = sweeps
    result, raw = _on_ascii_stdout(monkeypatch, mushin.diff, a, b)
    out = raw.decode("ascii")
    assert "acc (b-a)" in out
    assert f"{ESCAPED}_score (b-a)" in out
    assert f"{METRIC} (b-a)" in result.table
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


def test_guide_path_writes_the_bytes_of_a_path_the_console_cannot_encode(
    tmp_path, monkeypatch
):
    # Callers read --path by machine (`test -s "$(... --path)"`), so the output
    # must be the real path: its bytes, not an escape that names no file.
    guide = _non_ascii_guide(tmp_path)
    monkeypatch.setattr(cli, "_guide_path", lambda: guide)
    code, raw = _on_ascii_stdout(monkeypatch, cli.main, ["guide", "--path"])
    assert code == 0
    assert raw == os.fsencode(str(guide)) + b"\n"


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


def test_guide_path_fails_rather_than_print_an_escaped_path(
    tmp_path, monkeypatch, capsys
):
    guide = _non_ascii_guide(tmp_path)
    monkeypatch.setattr(cli, "_guide_path", lambda: guide)
    stdout = _TextOnly()
    monkeypatch.setattr(sys, "stdout", stdout)
    assert cli.main(["guide", "--path"]) == 1
    assert stdout.parts == []
    assert "cannot encode" in capsys.readouterr().err


def test_a_missing_stderr_drops_the_message_instead_of_moving_it_to_stdout(
    tmp_path, monkeypatch, capsys
):
    # With fd 2 closed, sys.stderr is None. The guide's error then goes
    # nowhere, as print(file=None) does; on stdout `$(... --path)` would
    # capture it as the path.
    monkeypatch.setattr(cli, "_guide_path", lambda: tmp_path / "missing.md")
    monkeypatch.setattr(sys, "stderr", None)
    assert cli.main(["guide", "--path"]) == 1
    assert capsys.readouterr().out == ""


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
    assert f"{ESCAPED}_score (b-a)" in out  # diff
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

    def run(*args: str) -> bytes:
        result = subprocess.run(
            [sys.executable, "-m", "mushin", "guide", *args],
            capture_output=True,
            env=env,
            # -m puts the cwd first on sys.path; an empty one cannot shadow
            # the copy.
            cwd=tmp_path,
            timeout=300,
        )
        stderr = result.stderr.decode(encoding, errors="replace")
        assert result.returncode == 0, stderr[-3000:]
        assert "UnicodeEncodeError" not in stderr
        return result.stdout

    assert run().decode(encoding).startswith("<!-- mushin ")
    # The output names the real file, as `test -s "$(... --path)"` needs.
    path = Path(os.fsdecode(run("--path").strip()))
    assert path.is_file()
    assert path.samefile(mushin_on_a_non_ascii_path / "mushin" / "agent_guide.md")


_FAIL_SOFT = """
import locale
import sys

# The C locale stands in for a Windows code page: open() without an encoding
# then writes in an encoding without the alpha. Where the fallback locale is
# UTF-8 anyway there is nothing to test.
try:
    "\\u03b1".encode(locale.getpreferredencoding(False))
except UnicodeEncodeError:
    pass
else:
    sys.exit(77)

from mushin import multirun
from mushin.workflows import MultiRunMetricsWorkflow


class W(MultiRunMetricsWorkflow):
    @staticmethod
    def task(seed):
        if seed == 1:
            raise ValueError("bad \\u03b1")
        return {"v": 1.0}


W().run(seed=multirun([0, 1]), working_dir=sys.argv[1], on_error="nan")
"""


def test_fail_soft_survives_an_error_message_the_locale_cannot_encode(tmp_path):
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
    wd = tmp_path / "s"
    result = subprocess.run(
        [sys.executable, "-c", _FAIL_SOFT, str(wd)],
        capture_output=True,
        env=env,
        timeout=300,
    )
    if result.returncode == 77:
        pytest.skip("this platform's C locale encodes the test character")
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-3000:]
    (error_file,) = wd.glob("*/mushin_error.txt")
    assert f"ValueError: bad {ALPHA}" in error_file.read_text(encoding="utf-8")


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


def test_echo_escapes_only_what_a_lenient_handler_cannot_take():
    # The handler takes the surrogate but not the alpha: the surrogate must
    # still come out as its byte, with only the alpha escaped.
    raw, stream = _stream("ascii", errors="surrogateescape")
    echo(f"\udcff {ALPHA}", file=stream)
    stream.flush()
    assert raw.getvalue() == b"\xff \\u03b1\n"


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
    monkeypatch.setattr(sys, "stdout", None)  # pythonw.exe has no console
    echo("nothing to write to")


def test_a_utf8_console_shows_user_data_as_itself(sweeps, capsys):
    # The escape is a fallback, not a filter: a console that can show the
    # metric name (pytest's capture is UTF-8) gets the name itself.
    a, _ = sweeps
    mushin.show(a)
    assert METRIC in capsys.readouterr().out


# --- Guard: mushin's own runtime strings are ASCII -------------------------

SRC = Path(mushin.__file__).parent
# (path relative to the package, the literal) -> why it must stay non-ASCII.
# Empty: an entry needs a reason a plain-ASCII rewrite cannot meet.
ALLOWED: dict[tuple[str, str], str] = {}


def _non_ascii_runtime_strings(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
    found = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for line, value in _non_ascii_runtime_strings(path):
            if (rel, value) not in ALLOWED:
                found.append(f"{rel}:{line}: {value!r}")
    assert not found, (
        "runtime strings with non-ASCII characters (a console that cannot "
        "encode them escapes or fails them; rewrite in ASCII):\n" + "\n".join(found)
    )
