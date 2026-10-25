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

# Absolute, not relative: `python path/to/mushin/__main__.py` runs this file
# as a top-level script, where a relative import has no package to resolve.
from mushin._console import codec, echo

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
    accented Windows user name). Whether it fits is decided as for every
    console write (``mushin._console.codec``): strictly, so that a handler
    that never fails cannot pass a path it would rewrite, and with the path
    as it is for a stream that reports no known encoding. A path that does
    not fit goes out as its own bytes (``os.fsencode``) only when stdout's
    encoding is the file system's, which is how the reader will decode them;
    otherwise, on a Windows pipe in the code page for one, the command fails
    and says how to widen stdout."""
    stream = sys.stdout
    if stream is None:  # pythonw.exe: print() would write nowhere too
        return 0
    judged = codec(stream)
    if judged is not None:
        encoding, errors = judged
        try:
            path.encode(encoding, errors)
        except UnicodeEncodeError:
            return _print_path_bytes(path, stream, encoding)
    try:
        stream.write(path + "\n")
    except UnicodeEncodeError:
        # The stream reported a wider encoding than the sink it forwards to
        # (a tee, pytest's tee-sys, Jupyter echoing to cp1252). Part of the
        # line may be out already; it is not written again, as bytes or text,
        # and the exit status says it is no path to use.
        return _cannot_print_path(getattr(stream, "encoding", None))
    return 0


def _print_path_bytes(path: str, stream, encoding: str) -> int:
    buffer = getattr(stream, "buffer", None)
    if buffer is not None and _same_codec(encoding, sys.getfilesystemencoding()):
        stream.flush()  # keep the order of anything already written as text
        buffer.write(os.fsencode(path))
        buffer.flush()
        stream.write("\n")  # through the text layer, translated as print() would
        return 0
    return _cannot_print_path(encoding)


def _cannot_print_path(encoding: str | None) -> int:
    # PYTHONUTF8=1 alone does not help when PYTHONIOENCODING is what narrowed
    # stdout, since that variable wins: name both.
    echo(
        "python -m mushin guide --path: the guide's path holds a character "
        f"this console cannot represent (stdout reports {encoding}); run with "
        "PYTHONUTF8=1 and with PYTHONIOENCODING unset (or set to utf-8), or "
        "print the guide itself with `python -m mushin guide`",
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
