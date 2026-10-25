# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Console output that survives a console whose encoding is not UTF-8.

``print()`` raises ``UnicodeEncodeError`` when ``sys.stdout`` cannot encode a
character: a Windows pipe writes in the ANSI code page (cp1252, cp932), an
old console in cp437, and ``PYTHONIOENCODING=ascii`` or an ASCII locale makes
stdout ASCII. mushin's own text is ASCII (``tests/test_console_encoding.py``
checks every string literal), but its tables and previews also print user
data (swept values, metric names, paths) that need not be, and failing a whole
``show()`` over one accented metric name is worse than printing it escaped."""

from __future__ import annotations

import sys
from typing import TextIO


def echo(text: str, *, end: str = "\n", file: TextIO | None = None) -> None:
    """Write ``text + end`` to ``file`` (default: ``sys.stdout``, looked up at
    call time so a redirect or pytest's capture applies), as ``print()`` does,
    except that a character the stream's encoding cannot represent is written
    as a backslash escape (``\\u03b1``) instead of raising.

    Escaping is what Python's own ``sys.stderr`` and the interactive prompt's
    display of a result do on such a console. The stream's own error handler
    is tried first, so text it can write, including lone surrogates under the
    ``surrogateescape`` handler a POSIX locale gives stdout (an undecodable
    path round-trips to its bytes), is written unchanged."""
    stream = sys.stdout if file is None else file
    if stream is None:  # pythonw.exe has no console; print() is a no-op there
        return
    line = text + end
    try:
        stream.write(line)
    except UnicodeEncodeError:
        # A text stream encodes the whole string before buffering any of it,
        # so the failed write left nothing behind to duplicate.
        encoding = getattr(stream, "encoding", None) or "ascii"
        try:
            escaped = line.encode(encoding, "backslashreplace").decode(encoding)
        except LookupError:  # an encoding name the codecs registry lacks
            escaped = line.encode("ascii", "backslashreplace").decode("ascii")
        stream.write(escaped)
