# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Console output that survives a console whose encoding is not UTF-8.

``print()`` raises ``UnicodeEncodeError`` when ``sys.stdout`` cannot encode a
character: a Windows pipe writes in the ANSI code page (cp1252, cp932), an
old console in cp437, and ``PYTHONIOENCODING=ascii`` or an ASCII locale makes
stdout ASCII. mushin's own text is ASCII (``tests/test_console_encoding.py``
checks every string literal), but its tables and previews also print user
data (metric names, swept values, paths) that need not be, and failing a
whole ``show()`` over one accented metric name is worse than printing it
escaped."""

from __future__ import annotations

import sys
from typing import Any

# The default of ``file=``: "sys.stdout as it is at call time". A sentinel
# rather than None, so that ``file=sys.stderr`` with no stderr (fd 2 closed,
# pythonw.exe) drops the text, as print() does, instead of moving it to stdout.
STDOUT: Any = object()


def _resolve(file: Any) -> Any:
    return sys.stdout if file is STDOUT else file


def _escape(text: str, encoding: str, errors: str) -> str:
    """``text`` with each run of characters that ``encoding`` cannot take
    under the ``errors`` handler written as backslash escapes (``\\u03b1``).

    Only those characters change: what the handler itself takes, such as a
    lone surrogate under ``surrogateescape`` (an undecodable path, which the
    stream writes back as its original byte), is left for the stream."""
    parts = []
    rest = text
    while True:
        try:
            rest.encode(encoding, errors)
        except UnicodeEncodeError as exc:
            start, end = exc.start, exc.end
            if not 0 <= start < end <= len(rest):  # a codec reporting no range
                start, end = 0, len(rest)
            bad = rest[start:end]
            parts += [rest[:start], bad.encode("ascii", "backslashreplace").decode()]
            rest = rest[end:]
        else:
            parts.append(rest)
            return "".join(parts)


def for_console(text: str, file: Any = STDOUT) -> str:
    """``text`` as ``echo`` writes it to ``file``: unchanged when the stream's
    ``encoding`` and ``errors`` take it, else with escapes for what they do not.

    Callers that lay text out in columns escape each cell with this before
    measuring it, so the escapes do not push the columns out of line. A
    stream that reports no usable encoding gets ``text`` unchanged here."""
    stream = _resolve(file)
    encoding = getattr(stream, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        return text  # io.StringIO and the like: nothing to check against
    errors = getattr(stream, "errors", None)
    try:
        return _escape(text, encoding, errors if isinstance(errors, str) else "strict")
    except LookupError:
        # `encoding` and `errors` are plain attributes, so a wrapper can name
        # something the codecs registry lacks: let the stream decide (echo).
        return text


def echo(text: str, *, end: str = "\n", file: Any = STDOUT) -> None:
    """Write ``text + end`` to ``file`` (default: ``sys.stdout``, looked up at
    call time so a redirect or pytest's capture applies), as ``print()`` does,
    except that a character the stream cannot encode is written as a
    backslash escape (``\\u03b1``) instead of raising.

    Escaping is what Python's own ``sys.stderr`` and the interactive prompt's
    display of a result do on such a console. ``file=None`` writes nothing,
    like ``print()`` with no stdout (pythonw.exe) or a closed stderr.

    The line is checked against the stream's ``encoding`` and ``errors``
    before anything is written, then written once. Only when the stream
    raises anyway (it reports no encoding, or it forwards to a narrower sink
    than its ``encoding`` says, as Jupyter's output stream and pytest's
    ``--capture=tee-sys`` do) is the line written again, escaped to ASCII; a
    stream that wrote part of the line before raising then shows it twice."""
    stream = _resolve(file)
    if stream is None:
        return
    line = for_console(text + end, stream)
    try:
        stream.write(line)
    except UnicodeEncodeError:
        # The stream's encoding did not describe what it accepts, so no codec
        # name can be trusted here: ASCII escapes are the one form that
        # every console encoding mushin can meet takes.
        stream.write(line.encode("ascii", "backslashreplace").decode("ascii"))
