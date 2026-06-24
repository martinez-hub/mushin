# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""Console entry point: ``mushin-mcp`` runs the stdio MCP server."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from mushin.mcp.server import create_server


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mushin-mcp",
        description="Read-only MCP server for analyzing mushin experiments.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Restrict experiment access to this directory (recommended).",
    )
    args = parser.parse_args(argv)
    server = create_server(root=args.root)
    server.run()  # FastMCP defaults to stdio transport


if __name__ == "__main__":
    main()
