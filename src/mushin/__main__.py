# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""``python -m mushin``: print the usage guide for AI coding agents."""

from __future__ import annotations

import argparse
import sys
from importlib import metadata, resources
from pathlib import Path

GUIDE = "agent_guide.md"


def _version() -> str:
    try:
        return metadata.version("mushin-py")
    except metadata.PackageNotFoundError:  # importable but not installed
        return "unknown"


def guide_text() -> str:
    """The guide as ``python -m mushin guide`` prints it: a version comment
    (valid Markdown, invisible when rendered), then the shipped file. The
    MCP server's ``guide`` prompt and ``mushin://guide`` resource return the
    same string, so every route to the guide shows the installed version."""
    path = resources.files("mushin") / GUIDE
    return f"<!-- mushin {_version()} -->\n" + path.read_text(encoding="utf-8")


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
    path = resources.files("mushin") / GUIDE
    if args.path:
        if not Path(str(path)).is_file():  # e.g. mushin imported from a zip
            print(
                "the guide is not a file on disk here (mushin is imported from "
                "an archive); print it with `python -m mushin guide`",
                file=sys.stderr,
            )
            return 1
        print(path)
        return 0
    sys.stdout.write(guide_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
