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
import functools
import sys
from collections.abc import Callable
from typing import Any

# The default of ``file=``: "sys.stdout as it is at call time". A sentinel
# rather than None, so that ``file=None`` can mean "nowhere". That differs
# from print() on purpose: print(file=None) writes to sys.stdout, so
# print(msg, file=sys.stderr) with no stderr (fd 2 closed, pythonw.exe) moves
# an error message onto stdout, where `$(python -m mushin guide --path)`
# would capture it as the path.
STDOUT: Any = object()


def _resolve(file: Any) -> Any:
    """The stream ``file`` names: ``sys.stdout`` for ``STDOUT``."""
    return sys.stdout if file is STDOUT else file


def codec(stream: Any) -> tuple[str, str] | None:
    """The encoding and error handler to judge text for ``stream`` by, or
    None when the stream reports no encoding the codecs registry knows
    (``io.StringIO``, a wrapper naming something else): such a stream gets
    text as it is. ``for_console`` and ``guide --path`` both decide by this,
    so their rules cannot drift apart.

    The handler is ``"strict"`` whatever the stream's own is, so that a
    handler that never fails (``backslashreplace``, ``replace``,
    ``xmlcharrefreplace``, which a Windows ``reconfigure`` commonly sets)
    cannot say text fits when the stream would rewrite it. The one exception
    is ``surrogateescape``, the handler a POSIX locale gives stdout: it
    writes an undecodable path back as its original bytes, losing nothing."""
    encoding = getattr(stream, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        return None
    try:
        codecs.lookup(encoding)
    except LookupError:
        return None
    errors = getattr(stream, "errors", None)
    return encoding, ("surrogateescape" if errors == "surrogateescape" else "strict")


def _escaping(errors: str) -> str:
    """The name of a codec error handler that gives the characters of a run
    the codec cannot encode to the stream's own ``errors`` handler first, and
    writes the ones it rejects as backslash escapes. Registered once per
    handler name, in the process-wide codecs registry (hence the prefix);
    only a non-strict handler needs it, since for "strict" every character is
    rejected and ``backslashreplace`` itself does the work.

    One call returns the longest leading stretch of the run that is all
    taken or all rejected, with its end. Returning one character at a time
    instead would make the codec scan the rest of the run again at each
    call, which on a long run (a CJK table on cp437) is quadratic."""
    name = f"mushin.console.{errors}"
    try:
        codecs.lookup_error(name)
    except LookupError:
        own = codecs.lookup_error(errors)

        def part(exc: UnicodeEncodeError, start: int, end: int) -> UnicodeEncodeError:
            return UnicodeEncodeError(exc.encoding, exc.object, start, end, exc.reason)

        def takes(exc: UnicodeEncodeError, i: int) -> bool:
            try:
                own(part(exc, i, i + 1))
            except UnicodeEncodeError:
                return False
            return True

        def handle(exc: UnicodeError):
            if not isinstance(exc, UnicodeEncodeError):
                raise exc
            kind = takes(exc, exc.start)
            end = exc.start + 1
            while end < exc.end and takes(exc, end) == kind:
                end += 1
            stretch = part(exc, exc.start, end)
            return own(stretch) if kind else codecs.backslashreplace_errors(stretch)

        codecs.register_error(name, handle)
    return name


def for_console(text: str, file: Any = STDOUT) -> str:
    """``text`` as ``echo`` writes it to ``file``: ``text`` itself (the same
    object) when it fits the stream (see ``codec``), else a copy with a
    backslash escape (``\\u03b1``) for each character that does not.

    One encode decides; only text that needs escapes is encoded again, in
    one pass, and decoded back for the stream."""
    judged = codec(_resolve(file))
    if judged is None:
        return text
    encoding, errors = judged
    try:
        text.encode(encoding, errors)
    except UnicodeEncodeError:
        handler = "backslashreplace" if errors == "strict" else _escaping(errors)
        # Decoding with the same handler gives back what it wrote:
        # surrogateescape's bytes return as the surrogates the stream will
        # write as those bytes again.
        return text.encode(encoding, handler).decode(encoding, errors)
    return text


def echo(
    text: str,
    *,
    end: str = "\n",
    file: Any = STDOUT,
    relayout: Callable[[Callable[[str], str]], str] | None = None,
) -> None:
    """Write ``text`` and then ``end`` to ``file`` (default: ``sys.stdout``,
    looked up at call time so a redirect or pytest's capture applies), as
    ``print()`` does, except that a character the stream cannot encode is
    written as a backslash escape (``\\u03b1``) instead of raising. ``end`` is
    mushin's own text, ASCII.

    ``relayout``, for text laid out in columns, renders the text again with
    each cell passed through the function it is given. It is called only
    when ``text`` needs escapes, so that the columns are measured on the
    escaped cells (an escape is longer than its character); text that fits
    is checked once and written as it is.

    Escaping is what Python's own ``sys.stderr`` and the interactive prompt's
    display of a result do on such a console. ``file=None`` writes nothing
    (``print(file=None)`` would write to stdout; see ``STDOUT``), so
    ``file=sys.stderr`` with stderr closed drops the text.

    ``text`` is checked against the stream before anything is written. Only
    when the stream raises anyway (it reports no encoding, or it forwards to
    a narrower sink than its ``encoding`` says, as Jupyter's output stream
    and pytest's ``--capture=tee-sys`` do) is the text written again, escaped
    to ASCII; a stream that wrote part of it before raising then shows that
    part twice, and a table escaped this way is not realigned."""
    stream = _resolve(file)
    if stream is None:
        return
    shown = for_console(text, stream)
    if shown is not text and relayout is not None:
        cell = functools.partial(for_console, file=stream)
        # The lines outside the columns (diff's provenance) are escaped whole.
        shown = for_console(relayout(cell), stream)
    try:
        stream.write(shown)
    except UnicodeEncodeError:
        # The stream's encoding did not describe what it accepts, so no codec
        # name can be trusted here: ASCII escapes are the one form that
        # every console encoding mushin can meet takes.
        stream.write(shown.encode("ascii", "backslashreplace").decode("ascii"))
    stream.write(end)
