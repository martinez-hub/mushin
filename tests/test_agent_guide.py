# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Keep the agent guide (`src/mushin/agent_guide.md`) honest.

Agents copy the guide's code and follow its links, and the guide ships inside
the package, so a stale keyword, a broken example or a dead link misleads every
agent that reads it. CI runs this file in the docs job too, because the `test`
job skips the docs-only PRs most likely to break it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

GUIDE = resources.files("mushin") / "agent_guide.md"


def _text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_the_guide_ships_as_ascii_markdown():
    text = _text()
    assert text.isascii(), sorted({c for c in text if not c.isascii()})
    first, second = text.splitlines()[:2]
    # docs/for-agents.md includes the guide from line 3, below its own title.
    assert first.startswith("# ") and second == ""


def test_python_m_mushin_guide(capsys):
    from mushin.__main__ import _version, main

    assert main(["guide"]) == 0
    assert capsys.readouterr().out == f"<!-- mushin {_version()} -->\n" + _text()

    assert main(["guide", "--path"]) == 0
    assert Path(capsys.readouterr().out.strip()).is_file()

    assert main([]) == 0
    assert "guide" in capsys.readouterr().out


def test_the_guide_prints_on_a_cp1252_pipe():
    """A Windows pipe writes in the ANSI code page; the guide must survive it."""
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    env.pop("PYTHONUTF8", None)
    result = subprocess.run(
        [sys.executable, "-m", "mushin", "guide"],
        capture_output=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-2000:]
    assert result.stdout.decode("cp1252").startswith("<!-- mushin ")
