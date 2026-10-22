# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""``python -m mushin``: print the usage guide for AI coding agents."""

from __future__ import annotations

import argparse
import sys
from importlib import metadata, resources

GUIDE = "agent_guide.md"


def _version() -> str:
    try:
        return metadata.version("mushin-py")
    except metadata.PackageNotFoundError:  # importable but not installed
        return "unknown"


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
        print(path)
        return 0
    # An HTML comment: valid Markdown, invisible when rendered.
    sys.stdout.write(f"<!-- mushin {_version()} -->\n")
    sys.stdout.write(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
