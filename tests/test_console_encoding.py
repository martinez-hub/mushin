# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""mushin's console output survives a console that cannot encode it.

A Windows pipe writes in the ANSI code page (cp1252 on Western systems), and
``PYTHONIOENCODING=ascii`` or an ASCII locale makes stdout ASCII; ``print()``
of a character outside the encoding then raises ``UnicodeEncodeError`` and the
whole command fails. mushin's own text is ASCII, so it cannot hit that, and the
user data it prints (swept values, metric names, paths) is written as
backslash escapes instead."""

from __future__ import annotations

import ast
import io
import os
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


def _on_ascii_stdout(monkeypatch, fn, *args, **kwargs):
    """Call ``fn`` with ``sys.stdout`` a strict ASCII stream, as on a console
    that cannot encode anything else, and return its result and what it wrote.

    The swap happens in the test body, not a fixture: pytest's capture sets
    ``sys.stdout`` again once fixture setup is over."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    with monkeypatch.context() as m:
        m.setattr(sys, "stdout", stream)
        result = fn(*args, **kwargs)
    stream.flush()
    return result, raw.getvalue().decode("ascii")


def _subprocess_env(encoding: str, **extra: str) -> dict[str, str]:
    env = {**os.environ, "PYTHONIOENCODING": f"{encoding}:strict", **extra}
    # UTF-8 mode is the workaround this test must not lean on.
    env.pop("PYTHONUTF8", None)
    return env


# --- In process: each printing function on a strict ASCII stdout ----------


def test_show_prints_user_data_it_cannot_encode_as_escapes(sweeps, monkeypatch):
    a, _ = sweeps
    result, out = _on_ascii_stdout(monkeypatch, mushin.show, a)
    assert f"{ESCAPED}_score" in out
    assert "completed" in out
    # Only the console copy is escaped; the returned table keeps the name.
    assert METRIC in result.table


def test_diff_prints_ascii_headers_and_escapes_user_data(sweeps, monkeypatch):
    a, b = sweeps
    result, out = _on_ascii_stdout(monkeypatch, mushin.diff, a, b)
    assert "acc (b-a)" in out
    assert f"{ESCAPED}_score (b-a)" in out
    assert f"{METRIC} (b-a)" in result.table
    row = next(r for r in result.rows if r["lr"] == 0.1 and r["seed"] == 0)
    assert row["deltas"][METRIC][2] == pytest.approx(0.5)


def test_dry_run_preview_escapes_a_swept_value(tmp_path, monkeypatch):
    class W(MultiRunMetricsWorkflow):
        @staticmethod
        def task(name, seed):
            return {"v": 1.0}

    summary, out = _on_ascii_stdout(
        monkeypatch,
        W().run,
        name=multirun([f"{ALPHA}lpha", "b"]),
        seed=multirun([0, 1]),
        working_dir=str(tmp_path / "dry"),
        dry_run=True,
    )
    assert out.startswith("Dry run: 4 cells, no jobs launched.")
    assert f"'{ESCAPED}lpha'" in out
    assert summary["axes"]["name"] == [f"{ALPHA}lpha", "b"]


def test_guide_path_with_a_character_the_console_lacks(tmp_path, monkeypatch):
    guide = tmp_path / f"site-{ALPHA}" / "agent_guide.md"
    guide.parent.mkdir()
    guide.write_text("# guide\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_guide_path", lambda: guide)
    code, out = _on_ascii_stdout(monkeypatch, cli.main, ["guide", "--path"])
    assert code == 0
    assert f"site-{ESCAPED}" in out


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

    def run(*args: str) -> str:
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
        return result.stdout.decode(encoding)

    assert run().startswith("<!-- mushin ")
    path = run("--path").strip()
    assert path.endswith("agent_guide.md")
    assert f"site-{ESCAPED}" in path


# --- The helper ------------------------------------------------------------


def _stream(
    encoding: str, errors: str = "strict"
) -> tuple[io.BytesIO, io.TextIOWrapper]:
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding=encoding, errors=errors)


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
