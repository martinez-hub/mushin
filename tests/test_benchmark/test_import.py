def test_benchmark_package_imports():
    import mushin.benchmark  # noqa: F401


def test_third_party_deps_available():
    import pandas  # noqa: F401
    import scipy.stats  # noqa: F401
    import torchmetrics  # noqa: F401
