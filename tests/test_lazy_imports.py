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


def test_import_mushin_does_not_load_pytorch_lightning():
    # The Lightning integration (and pytorch_lightning, ~1.1s of a cold import)
    # must load only on first use, not on a bare `import mushin`.
    out = _fresh_import_probe(
        "import sys, mushin;"
        "print('pytorch_lightning' in sys.modules, 'mushin.lightning' in sys.modules)"
    )
    assert out == "False False"


def test_import_mushin_does_not_load_eval_dependencies():
    # scipy comes only from the `eval` extra, so a core install does not have
    # it: an eager import anywhere on the `import mushin` path turns the core
    # install's first line into a ModuleNotFoundError. torchmetrics does reach a
    # core install (pytorch-lightning depends on it), so for it this guards
    # import time, like the Lightning check above.
    out = _fresh_import_probe(
        "import sys, mushin;"
        "print('torchmetrics' in sys.modules, 'scipy' in sys.modules)"
    )
    assert out == "False False"


def test_import_mushin_and_a_sweep_record_do_not_load_ipython():
    # The project-code record reads a notebook's cells only when IPython is
    # loaded already (a kernel); it never imports it.
    out = _fresh_import_probe(
        "import sys, mushin\n"
        "from mushin._resume import project_fingerprints\n"
        "def task(a):\n    return a\n"
        "project_fingerprints(task)\n"
        "print('IPython' in sys.modules)"
    )
    assert out == "False"


def test_lightning_names_resolve_lazily():
    out = _fresh_import_probe(
        "import sys, mushin;"
        "obj = mushin.HydraDDP;"
        "print(obj.__name__, 'pytorch_lightning' in sys.modules)"
    )
    assert out == "HydraDDP True"


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


def test_import_mushin_does_not_load_torch():
    # torch is ~80% of a cold `import mushin` (~490ms of ~610ms) and the core
    # sweep -> dataset path only needs it at runtime (torch.load); it must not
    # load at import time.
    out = _fresh_import_probe("import sys, mushin; print('torch' in sys.modules)")
    assert out == "False"


def test_eval_available_reflects_the_eval_modules(monkeypatch):
    import importlib.util

    from mushin import _extras

    assert _extras.EVAL_MODULES == ("scipy", "torchmetrics")
    assert _extras.eval_available() is True  # the dev environment has the extra

    real_find_spec = importlib.util.find_spec

    def without_scipy(name, *args, **kwargs):
        return None if name == "scipy" else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_scipy)
    assert _extras.eval_available() is False


def test_eval_available_does_not_import_the_eval_modules():
    """`import mushin` calls eval_available, so it must not import scipy or
    torchmetrics (and with it torch)."""
    code = (
        "import sys; from mushin._extras import eval_available; eval_available(); "
        "print('torchmetrics' in sys.modules, 'scipy' in sys.modules)"
    )
    assert _fresh_import_probe(code) == "False False"


def test_module_docstring_points_agents_to_the_guide():
    import mushin

    assert mushin.__doc__ and "python -m mushin guide" in mushin.__doc__


def test_dir_lists_no_deprecated_aliases():
    """help(mushin) touches every name dir() lists; a deprecated alias then
    warns on every render (and raises under -W error)."""
    import warnings

    import mushin

    assert "BaseWorkflow" not in dir(mushin)
    assert "RobustnessCurve" not in dir(mushin)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mushin.BaseWorkflow  # still importable  # noqa: B018
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_all_lists_the_eval_names_when_the_extra_is_installed():
    import mushin

    assert {"compare", "llm", "classification_battery"} <= set(mushin.__all__)
    assert sorted(mushin.__all__) == dir(mushin)


def test_require_eval_raises_the_install_hint(monkeypatch):
    import importlib

    from mushin import _extras

    real_import = importlib.import_module

    def without_scipy(name, *args, **kwargs):
        if name == "scipy":
            raise ModuleNotFoundError("No module named 'scipy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without_scipy)
    with pytest.raises(ImportError, match="the install hint") as info:
        _extras.require_eval("the install hint")
    assert isinstance(info.value.__cause__, ModuleNotFoundError)
