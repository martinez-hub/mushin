# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""A fast read of a Hydra job's ``.hydra/hydra.yaml`` that gives what
``OmegaConf.load`` gives.

``load_from_dir`` and the MCP server need two fields from each job's
hydra.yaml: the job number and the config-group choices. ``OmegaConf.load``
parses the file with PyYAML's pure-Python loader and then builds a config from
it, which was about 90% of a 1000-cell reload. Here libyaml parses the file,
the tree is checked as ``OmegaConf.create`` would check it, and the parse is
used only for text that a YAML emitter wrote (see ``_parse_plain``). Any other
file goes to ``OmegaConf.load``, which gives the same result or raises the same
error as before.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from yaml.composer import Composer, ComposerError
from yaml.constructor import SafeConstructor
from yaml.events import AliasEvent
from yaml.resolver import Resolver

try:  # libyaml, which PyYAML's wheels include
    from yaml.cyaml import CDumper, CParser
except ImportError:  # a PyYAML built without it: every file goes to OmegaConf
    CDumper = CParser = None


if CParser is not None:

    class _LibyamlLoader(Composer, CParser, SafeConstructor, Resolver):
        """libyaml's parser under PyYAML's Python composer.

        CSafeLoader also composes in C, recursing on the C stack with no depth
        limit, so a deeply nested file (about 20,000 levels, 60 KB) kills the
        process past any ``except``. The Python composer raises RecursionError
        instead, as the pure-Python loader of ``OmegaConf.load`` does. On
        Hydra's 3 KB hydra.yaml it takes 0.4 ms, against 0.3 ms all in C and
        3 ms all in Python.

        It refuses any alias (``*a``), which Hydra never writes, so the file
        goes to OmegaConf. Writing an aliased tree back repeats each aliased
        string once per alias, which lets a small crafted file take memory
        that grows with the square of its size, and a cyclic alias would make
        OmegaConf.create recurse until RecursionError.

        OmegaConf's loader also refuses a duplicate key. This one keeps the
        last value, as PyYAML does, and the round trip in ``_parse_plain``
        refuses the file instead: a mapping parsed with one entry for two keys
        cannot be written back as the same text.
        """

        def __init__(self, stream: str) -> None:
            CParser.__init__(self, stream)
            Composer.__init__(self)
            SafeConstructor.__init__(self)
            Resolver.__init__(self)

        def compose_node(self, parent: Any, index: Any) -> Any:
            if self.check_event(AliasEvent):
                raise ComposerError(
                    None, None, "found an alias", self.peek_event().start_mark
                )
            return super().compose_node(parent, index)


# Names Python itself puts in a class __dict__ (3.13 adds the last two).
_CLASS_METADATA = {
    "__module__",
    "__doc__",
    "__qualname__",
    "__dict__",
    "__weakref__",
    "__firstlineno__",
    "__static_attributes__",
}
# What OmegaConf's loader and dumper define over yaml.SafeLoader and
# yaml.Dumper, in omegaconf 2.2.3 to 2.3.1. The plain codecs copy the loader's
# tables and the dumper's representers; the loader's construct_mapping (its
# duplicate-key check) is enforced by the round trip instead.
_OMEGACONF_LOADER_ATTRS = {
    "construct_mapping",
    "yaml_constructors",
    "yaml_implicit_resolvers",
}
_OMEGACONF_DUMPER_ATTRS = {
    "str_representer",
    "str_representer_added",
    "yaml_representers",
}

# The PyYAML tables a class may hold in its own __dict__ once something adds
# to them.
_YAML_TABLES = {
    "yaml_constructors",
    "yaml_multi_constructors",
    "yaml_implicit_resolvers",
}

# Hydra's hydra.yaml nests about 6 deep. OmegaConf.create recurses per level
# and overflows somewhere past 100, at a depth that depends on the recursion
# limit, so a deeper file is left to it.
_MAX_PLAIN_DEPTH = 16

# Strings holding `${` up to this length have their syntax check cached.
_CACHED_INTERPOLATION_LEN = 200

# (inputs, codecs) from the last build; see _plain_codecs.
_codecs_cache: tuple[tuple, tuple[type, type, type] | None] | None = None


def read_hydra_job(hydra_yaml: Path | str) -> tuple[Any, dict | None]:
    """The job number and the config-group choices in a job's ``hydra.yaml``,
    as ``OmegaConf.load`` reads them, or the error it raises."""
    return hydra_job_fields(load_hydra_record(hydra_yaml))


def load_hydra_record(hydra_yaml: Path | str) -> Any:
    """A job's hydra.yaml, loaded: the parsed tree (a dict) when the fast read
    can take the file, else ``OmegaConf.load``'s config. Raises what
    ``OmegaConf.load`` raises. ``load_from_dir`` loads every job's record
    before reading any field, as it did with ``OmegaConf.load``, so a sweep
    with several broken files reports the same error."""
    hydra_yaml = Path(hydra_yaml)
    doc = _parse_plain(hydra_yaml)
    if doc is not None:
        return doc
    from hydra_zen import load_from_yaml

    return load_from_yaml(hydra_yaml)


def hydra_job_fields(record: Any) -> tuple[Any, dict | None]:
    """The job number and choices of a ``load_hydra_record`` result.

    From a parsed tree they are read directly when they are what Hydra writes
    (an int; option names that are strings or None). Otherwise (``???``,
    ``${...}``, a missing field) they come from ``OmegaConf.create`` on that
    tree, which is the config ``OmegaConf.load`` would have built from it.
    """
    if isinstance(record, dict):
        fields = _plain_hydra_fields(record)
        if fields is not None:
            return fields
        from omegaconf import OmegaConf

        record = OmegaConf.create(record)
    hydra = record.hydra
    return hydra.job.num, hydra_choices(hydra)


def hydra_choices(hydra: Any) -> dict | None:
    """``runtime.choices`` of a ``hydra`` node, or None where it has none."""
    try:
        ch = hydra["runtime"]["choices"]
        return None if ch is None else dict(ch)
    except Exception:  # noqa: BLE001 - older dirs without choices
        return None


def _parse_plain(hydra_yaml: Path) -> dict | None:
    """The file parsed as ``OmegaConf.load`` would parse it, without building
    a config, or None to leave the file to ``OmegaConf.load``: on any error, so
    it raises its own, and on anything the two could treat differently.

    libyaml is more lenient than PyYAML's pure-Python scanner, which
    ``OmegaConf.load`` uses: it accepts tabs as separators, a byte-order mark
    at any line start, an empty ``!`` tag, a comment right after a block-scalar
    header and a ``?`` inside a flow scalar, among others. Rather than chase
    each one, the parsed tree is written back with OmegaConf's representers and
    ``OmegaConf.to_yaml``'s settings, and the parse is used only if that
    reproduces the file byte for byte: text that a YAML emitter wrote, which
    both parsers read alike (``tests/test_hydra_yaml_read.py`` tests exactly
    that). libyaml's emitter is tried first (0.4 ms); OmegaConf's own, the one
    Hydra writes with (1.8 ms), only where the two differ, as for a character
    outside the BMP or a long key. Anything hand-edited goes to OmegaConf.
    """
    try:
        codecs = _plain_codecs()
        if codecs is None:
            return None
        loader, *dumpers = codecs
        text = hydra_yaml.read_text(encoding="utf-8")
        doc = yaml.load(text, Loader=loader)
        if not isinstance(doc, dict) or not _omegaconf_would_create(doc):
            return None
        libyaml_dumper, omegaconf_dumper = dumpers
        written = _write_back(doc, libyaml_dumper)
        if written == text:
            return doc
        # Over 6,000 random documents, the two emitters wrote different text
        # only where one of them wrote a complex key (`? `) or an escape, and
        # escapes only occur in double-quoted scalars. Elsewhere this second
        # write-back is skipped, so a hand-edited file does not pay for it (a
        # backslash in a plain scalar, such as a Windows path, does not count).
        # A skip costs at most the fast path, never the result.
        if _emitters_may_differ(written) or _emitters_may_differ(text):
            if _write_back(doc, omegaconf_dumper) == text:
                return doc
    except Exception:  # noqa: BLE001 - OmegaConf decides, and raises its own error
        return None
    return None


def _write_back(doc: dict, dumper: type) -> str:
    """``doc`` as ``OmegaConf.to_yaml`` writes a config, on ``dumper``."""
    return yaml.dump(
        doc,
        Dumper=dumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def _emitters_may_differ(text: str) -> bool:
    return "? " in text or ("\\" in text and '"' in text)


def _plain_codecs() -> tuple[type, type, type] | None:
    """(loader, libyaml dumper, OmegaConf's dumper) that read and write YAML
    as OmegaConf does, or None (no libyaml, or OmegaConf's private helpers
    moved or changed shape), which sends every file to OmegaConf.

    OmegaConf builds its loader afresh on every load. These are rebuilt
    whenever anything they are built from changes: OmegaConf's two helpers, or
    any entry of the PyYAML tables those start from, as when a library adds or
    replaces a resolver, constructor or representer."""
    global _codecs_cache
    if CParser is None:
        return None
    inputs = _omegaconf_inputs()
    cached = _codecs_cache
    if cached is not None and cached[0] == inputs:
        return cached[1]
    codecs = _build_plain_codecs(inputs)
    _codecs_cache = (inputs, codecs)
    return codecs


def _omegaconf_inputs() -> tuple:
    """Everything OmegaConf's loader and dumper are built from, by value,
    including what ``yaml.SafeLoader`` itself defines, so a later patch of it
    is seen."""
    from omegaconf import _utils

    dumper = _utils.get_omega_conf_dumper()
    return (
        _utils.get_yaml_loader,
        _utils.get_omega_conf_dumper,
        dumper,
        tuple(
            (k, v) for k, v in vars(yaml.SafeLoader).items() if k not in _YAML_TABLES
        ),
        *(
            tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in t.items())
            for t in (
                yaml.SafeLoader.yaml_implicit_resolvers,
                yaml.SafeLoader.yaml_constructors,
                yaml.SafeLoader.yaml_multi_constructors,
                dumper.yaml_representers,
                dumper.yaml_multi_representers,
                dumper.yaml_implicit_resolvers,
            )
        ),
    )


def _build_plain_codecs(inputs: tuple) -> tuple[type, type, type] | None:
    get_yaml_loader, _, omegaconf_dumper = inputs[:3]
    omegaconf_loader = get_yaml_loader()
    if not (
        _defines(omegaconf_loader, yaml.SafeLoader, _OMEGACONF_LOADER_ATTRS)
        and _defines(omegaconf_dumper, yaml.Dumper, _OMEGACONF_DUMPER_ATTRS)
        and _safe_loader_is_pyyamls()
    ):
        return None
    loader = type(
        "_HydraYamlLoader",
        (_LibyamlLoader,),
        {
            # OmegaConf reads `1e3` as a float, leaves dates as strings and
            # builds pathlib tags; its tables carry all of that.
            "yaml_implicit_resolvers": omegaconf_loader.yaml_implicit_resolvers,
            "yaml_constructors": omegaconf_loader.yaml_constructors,
            "yaml_multi_constructors": omegaconf_loader.yaml_multi_constructors,
        },
    )
    libyaml_dumper = type(
        "_HydraYamlDumper",
        (CDumper,),
        {
            "yaml_representers": omegaconf_dumper.yaml_representers,
            "yaml_multi_representers": omegaconf_dumper.yaml_multi_representers,
            "yaml_implicit_resolvers": omegaconf_dumper.yaml_implicit_resolvers,
        },
    )
    return loader, libyaml_dumper, omegaconf_dumper


def _safe_loader_is_pyyamls() -> bool:
    """Whether ``yaml.SafeLoader``, which OmegaConf's loader subclasses, is
    PyYAML's own: its usual bases, and nothing defined on it but PyYAML's
    ``__init__`` and the tables. The plain loader does not subclass it, so a
    library that patches it would change OmegaConf.load and not this."""
    from yaml.parser import Parser
    from yaml.reader import Reader
    from yaml.scanner import Scanner

    if yaml.SafeLoader.__bases__ != (
        Reader,
        Scanner,
        Parser,
        Composer,
        SafeConstructor,
        Resolver,
    ):
        return False
    own = set(vars(yaml.SafeLoader)) - _CLASS_METADATA - _YAML_TABLES
    init = vars(yaml.SafeLoader).get("__init__")
    return own == {"__init__"} and (
        getattr(getattr(init, "__code__", None), "co_filename", None)
        == yaml.loader.__file__
    )


def _defines(cls: Any, base: type, attrs: set[str]) -> bool:
    """Whether ``cls`` is a direct subclass of ``base`` defining exactly
    ``attrs``, past the names Python itself adds."""
    return (
        isinstance(cls, type)
        and cls.__bases__ == (base,)
        and set(vars(cls)) - _CLASS_METADATA == attrs
    )


def _plain_hydra_fields(doc: dict) -> tuple[int, dict | None] | None:
    """The job number and choices from the parsed tree, when they are what
    Hydra writes: an int, and a mapping of group names to option names (or
    None). None otherwise (``???``, ``${...}``, an interpolated node, a missing
    field), for OmegaConf to resolve or reject."""
    try:
        hydra = doc["hydra"]
        num = hydra["job"]["num"]
        runtime = hydra.get("runtime")
        choices = None if runtime is None else runtime.get("choices")
    except Exception:  # noqa: BLE001 - OmegaConf decides, and raises its own error
        return None
    if type(num) is not int:
        return None
    if choices is None:
        return num, None
    if not isinstance(choices, dict) or not all(
        isinstance(k, str) and (v is None or _is_option_name(v))
        for k, v in choices.items()
    ):
        return None
    return num, choices


def _is_option_name(v: Any) -> bool:
    return isinstance(v, str) and v != "???" and "${" not in v


def _omegaconf_would_create(tree: Any) -> bool:
    """Whether ``OmegaConf.create`` accepts this parsed YAML, by its own rules:
    every key of a type a DictConfig takes, every leaf of a type its nodes take,
    and valid syntax in every string holding ``${`` (it parses each one).

    False, which sends the file to OmegaConf to decide, also for nesting
    deeper than ``_MAX_PLAIN_DEPTH`` and if those private helpers move. (No
    container is reached twice: the loader refuses aliases.) Walks with a
    stack, since a recursion here would undo the composer's RecursionError
    guard."""
    try:
        from omegaconf import DictKeyType
        from omegaconf._utils import is_primitive_type_annotation

        key_types = DictKeyType.__args__  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - only the fast path is lost
        return False
    stack = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict) or type(node) in (list, tuple):
            if depth > _MAX_PLAIN_DEPTH:
                return False
            if isinstance(node, dict):
                if not all(isinstance(k, key_types) for k in node):
                    return False
                stack.extend((v, depth + 1) for v in node.values())
            else:
                stack.extend((v, depth + 1) for v in node)
        elif isinstance(node, str):
            if "${" in node and not _valid_interpolation(node):
                return False
        elif not is_primitive_type_annotation(node):
            return False
    return True


def _valid_interpolation(text: str) -> bool:
    """OmegaConf's syntax check on a string holding ``${``. Short strings are
    cached: the cells of a sweep repeat the same few, and a full grammar parse
    is slow. Long ones are not, since the MCP server lives long."""
    if len(text) <= _CACHED_INTERPOLATION_LEN:
        return _valid_interpolation_cached(text)
    return _check_interpolation(text)


@functools.lru_cache(maxsize=256)
def _valid_interpolation_cached(text: str) -> bool:
    return _check_interpolation(text)


def _check_interpolation(text: str) -> bool:
    try:
        from omegaconf._utils import get_value_kind

        get_value_kind(text, strict_interpolation_validation=True)
    except Exception:  # noqa: BLE001 - a syntax error, or the helper moved
        return False
    return True
