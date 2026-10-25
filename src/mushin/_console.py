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

import codecs
import sys
from typing import Any

# The default of ``file=``: "sys.stdout as it is at call time". A sentinel
# rather than None, so that ``file=None`` can mean "nowhere". That differs
# from print() on purpose: print(file=None) writes to sys.stdout, so
# print(msg, file=sys.stderr) with no stderr (fd 2 closed, pythonw.exe) moves
# an error message onto stdout, where `$(python -m mushin guide --path)`
# would capture it as the path.
STDOUT: Any = object()


def _resolve(file: Any) -> Any:
    return sys.stdout if file is STDOUT else file


def _escaping(errors: str) -> str:
    """The name of a codec error handler that gives each character the
    stream's own ``errors`` handler first and writes it as a backslash escape
    only if that handler rejects it. Registered on first use, once per
    handler name, in the process-wide codecs registry (hence the prefix).

    One character at a time, so that in a run the codec reports as a whole
    (a lone surrogate beside an alpha) the handler still takes what it can:
    under ``surrogateescape`` the surrogate goes out as its original byte."""
    name = f"mushin.console.{errors}"
    try:
        codecs.lookup_error(name)
    except LookupError:
        own = codecs.lookup_error(errors)  # LookupError for an unknown name

        def handle(exc: UnicodeError):
            if not isinstance(exc, UnicodeEncodeError):
                raise exc
            one = UnicodeEncodeError(
                exc.encoding, exc.object, exc.start, exc.start + 1, exc.reason
            )
            try:
                return own(one)
            except UnicodeEncodeError:
                return codecs.backslashreplace_errors(one)

        codecs.register_error(name, handle)
    return name


def for_console(text: str, file: Any = STDOUT) -> str:
    """``text`` as ``echo`` writes it to ``file``: ``text`` itself when the
    stream's ``encoding`` and ``errors`` take it, else a copy with a
    backslash escape (``\\u03b1``) for each character they do not.

    One encode decides; only text that needs escapes is encoded again, in
    one pass, and decoded back for the stream. Callers that lay text out in
    columns escape each cell with this before measuring it, so the escapes
    do not push the columns out of line. A stream that reports no usable
    encoding gets ``text`` unchanged here."""
    stream = _resolve(file)
    encoding = getattr(stream, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        return text  # io.StringIO and the like: nothing to check against
    errors = getattr(stream, "errors", None)
    if not isinstance(errors, str) or not errors:
        errors = "strict"
    try:
        text.encode(encoding, errors)
        return text
    except UnicodeEncodeError:
        data = text.encode(encoding, _escaping(errors))
        # The stream's own handler reverses what it wrote (surrogateescape's
        # bytes come back as the surrogates the stream will write again).
        return data.decode(encoding, errors)
    except LookupError:
        # `encoding` and `errors` are plain attributes, so a wrapper can name
        # something the codecs registry lacks: let the stream decide (echo).
        return text


def echo(text: str, *, end: str = "\n", file: Any = STDOUT) -> None:
    """Write ``text`` and then ``end`` to ``file`` (default: ``sys.stdout``,
    looked up at call time so a redirect or pytest's capture applies), as
    ``print()`` does, except that a character the stream cannot encode is
    written as a backslash escape (``\\u03b1``) instead of raising. ``end`` is
    mushin's own text, ASCII.

    Escaping is what Python's own ``sys.stderr`` and the interactive prompt's
    display of a result do on such a console. ``file=None`` writes nothing
    (``print(file=None)`` would write to stdout; see ``STDOUT``), so
    ``file=sys.stderr`` with stderr closed drops the text.

    ``text`` is checked against the stream's ``encoding`` and ``errors``
    before anything is written. Only when the stream raises anyway (it
    reports no encoding, or it forwards to a narrower sink than its
    ``encoding`` says, as Jupyter's output stream and pytest's
    ``--capture=tee-sys`` do) is the text written again, escaped to ASCII;
    a stream that wrote part of it before raising then shows that part
    twice, and a table escaped this way is not realigned."""
    stream = _resolve(file)
    if stream is None:
        return
    text = for_console(text, stream)
    try:
        stream.write(text)
    except UnicodeEncodeError:
        # The stream's encoding did not describe what it accepts, so no codec
        # name can be trusted here: ASCII escapes are the one form that
        # every console encoding mushin can meet takes.
        stream.write(text.encode("ascii", "backslashreplace").decode("ascii"))
    stream.write(end)
