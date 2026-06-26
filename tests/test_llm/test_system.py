class _Gen:
    """Module-level class so hydra_zen.builds can resolve its import path."""

    constructions = 0

    def __init__(self, tag):
        _Gen.constructions += 1
        self.tag = tag

    def __call__(self, inputs, seed):
        return [self.tag] * len(inputs)


def test_callable_passthrough():
    from mushin.llm._system import as_system

    def sys(inputs, seed):
        return list(inputs)

    assert as_system(sys) is sys


def test_hydra_zen_config_instantiated_once():
    from hydra_zen import builds

    from mushin.llm._system import as_system

    _Gen.constructions = 0
    system = as_system(builds(_Gen, tag="x"))
    assert _Gen.constructions == 1  # built exactly once
    assert system(["a", "b"], 0) == ["x", "x"]


def test_non_callable_rejected():
    import pytest

    from mushin.llm._system import as_system

    with pytest.raises(TypeError, match="callable"):
        as_system(42)
