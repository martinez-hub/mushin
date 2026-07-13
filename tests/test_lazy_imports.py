"""The top-level import stays light; peripheral subsystems load on first use."""
import subprocess
import sys

import pytest


def _fresh_import_probe(code: str) -> str:
    """Run `code` in a fresh interpreter and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_import_mushin_does_not_load_benchmark_or_llm():
    out = _fresh_import_probe(
        "import sys, mushin;"
        "print('mushin.benchmark' in sys.modules, 'mushin.llm' in sys.modules)"
    )
    assert out == "False False"


def test_benchmark_names_resolve_lazily():
    out = _fresh_import_probe(
        "import sys, mushin;"
        "obj = mushin.compare;"
        "print(callable(obj), 'mushin.benchmark' in sys.modules)"
    )
    assert out == "True True"


def test_llm_submodule_resolves_lazily():
    out = _fresh_import_probe(
        "import sys, mushin;"
        "mod = mushin.llm;"
        "print(mod.__name__, 'mushin.llm' in sys.modules)"
    )
    assert out == "mushin.llm True"


def test_unknown_attribute_raises_attribute_error():
    import mushin

    try:
        mushin.does_not_exist  # noqa: B018
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError")


@pytest.mark.parametrize("name", ["BaseWorkflow", "RobustnessCurve"])
def test_deprecated_names_warn_but_resolve(name):
    import mushin

    with pytest.warns(DeprecationWarning, match=name):
        obj = getattr(mushin, name)
    from mushin import workflows

    assert obj is getattr(workflows, name)


@pytest.mark.parametrize("name", ["BaseWorkflow", "RobustnessCurve"])
def test_deprecated_names_absent_from_all(name):
    import mushin

    assert name not in mushin.__all__
