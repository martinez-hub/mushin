# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""``python -m mushin``: print the usage guide for AI coding agents."""

from __future__ import annotations

import argparse
import codecs
import os
import sys
from importlib import metadata, resources
from pathlib import Path

from ._console import echo

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


def _same_codec(a: str, b: str) -> bool:
    try:
        return codecs.lookup(a).name == codecs.lookup(b).name
    except LookupError:
        return False


def _print_path(path: str) -> int:
    """Print ``path`` for a caller that reads it by machine (CI runs
    ``test -s "$(python -m mushin guide --path)"``), so the output is the
    real path or an error, never a path with escapes or replacements.

    The install path can hold a character the console cannot encode (an
    accented Windows user name). Encodability is judged strictly, not with
    the stream's own handler: ``backslashreplace`` or ``replace`` never fail,
    and would write a path that names no file. ``surrogateescape`` is the
    exception, because it writes an undecodable path back as its bytes. A
    path that fails the strict test goes out as its own bytes
    (``os.fsencode``) only when stdout's encoding is the file system's, which
    is how the reader will decode them; otherwise, on a Windows pipe in the
    code page for one, the command fails and says to set ``PYTHONUTF8=1``."""
    stream = sys.stdout
    if stream is None:  # pythonw.exe: print() would write nowhere too
        return 0
    encoding = getattr(stream, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        stream.write(path + "\n")  # a stream that takes any str (io.StringIO)
        return 0
    errors = getattr(stream, "errors", None)
    lossless = "surrogateescape" if errors == "surrogateescape" else "strict"
    try:
        path.encode(encoding, lossless)
        stream.write(path + "\n")
        return 0
    except (UnicodeEncodeError, LookupError):
        pass  # not encodable, or not verifiably: the bytes, or the error
    buffer = getattr(stream, "buffer", None)
    if buffer is not None and _same_codec(encoding, sys.getfilesystemencoding()):
        stream.flush()  # keep the order of anything already written as text
        buffer.write(os.fsencode(path))
        buffer.flush()
        stream.write("\n")  # through the text layer, translated as print() would
        return 0
    echo(
        "python -m mushin guide --path: the guide's path holds a character "
        f"this console's encoding ({encoding}) cannot represent; run with "
        "PYTHONUTF8=1, or print the guide itself with `python -m mushin guide`",
        file=sys.stderr,
    )
    return 1


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
    # The guide is ASCII (tests/test_agent_guide.py checks), so every console
    # encoding takes it as it is: no check, no copy.
    if sys.stdout is not None:  # pythonw.exe has no stdout
        sys.stdout.write(guide_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
