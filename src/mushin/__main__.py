# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""``python -m mushin``: print the usage guide for AI coding agents."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import metadata, resources
from pathlib import Path

from ._console import echo, for_console

GUIDE = "agent_guide.md"


def _version() -> str:
    try:
        return metadata.version("mushin-py")
    except metadata.PackageNotFoundError:  # importable but not installed
        return "unknown"


def _guide_path():
    return resources.files("mushin") / GUIDE


def guide_text() -> str:
    """The guide as ``python -m mushin guide`` prints it: a version comment
    (valid Markdown, invisible when rendered), then the shipped file. The
    MCP server's ``guide`` prompt and ``mushin://guide`` resource return the
    same string, so every route to the guide shows the installed version."""
    return f"<!-- mushin {_version()} -->\n" + _guide_path().read_text(encoding="utf-8")


def _print_path(path: str) -> int:
    """Print ``path`` for a caller that reads it by machine (CI runs
    ``test -s "$(python -m mushin guide --path)"``), so the output is the real
    path or nothing: never the escaped text ``echo`` would write.

    The install path can hold a character the console cannot encode (an
    accented Windows user name). Then the path's own bytes go to stdout's
    byte layer, which is what a shell's ``$(...)`` captures on POSIX; on
    Windows ``os.fsencode`` gives UTF-8, which a caller decoding in the code
    page misreads, so ``PYTHONUTF8=1`` is still the fix there. Without a byte
    layer the command fails with a message instead."""
    stream = sys.stdout
    if stream is None:  # pythonw.exe: print() would write nowhere too
        return 0
    if for_console(path, stream) == path:
        try:
            stream.write(path + "\n")
            return 0
        except UnicodeEncodeError:
            pass  # its .encoding did not describe it: try the bytes
    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        echo(
            "python -m mushin guide --path: the guide's path holds a character "
            "this console cannot encode, and stdout takes no bytes; run with "
            "PYTHONUTF8=1, or print the guide itself with `python -m mushin guide`",
            file=sys.stderr,
        )
        return 1
    stream.flush()  # keep the order of anything already written as text
    buffer.write(os.fsencode(path))
    buffer.flush()
    stream.write("\n")  # through the text layer, translated as print() would
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mushin",
        description="mushin's command line. `guide` prints the usage guide for "
        "AI coding agents that matches this version.",
    )
    commands = parser.add_subparsers(dest="command")
    guide = commands.add_parser(
        "guide", help="print the usage guide for AI coding agents"
    )
    guide.add_argument(
        "--path", action="store_true", help="print the guide's file path instead"
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    path = _guide_path()
    if args.path:
        if not Path(str(path)).is_file():  # e.g. mushin imported from a zip
            echo(
                "the guide is not a file on disk here (mushin is imported from "
                "an archive); print it with `python -m mushin guide`",
                file=sys.stderr,
            )
            return 1
        return _print_path(str(path))
    # The guide is ASCII (tests/test_agent_guide.py checks); echo is here so
    # every console route of mushin shares one write path.
    echo(guide_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
