def test_benchmark_package_imports():
    import mushin.benchmark  # noqa: F401


def test_third_party_deps_available():
    import pandas  # noqa: F401
    import scipy.stats  # noqa: F401
    import torchmetrics  # noqa: F401


def test_public_task_api_exports():
    import mushin
    from mushin import (  # noqa: F401
        Task,
        benchmark,
        classification_battery,
        detection_battery,
        get_task,
        list_tasks,
        register_task,
        segmentation_battery,
    )

    for name in [
        "Task",
        "register_task",
        "get_task",
        "list_tasks",
        "classification_battery",
        "segmentation_battery",
        "detection_battery",
    ]:
        assert name in mushin.__all__, f"{name} missing from mushin.__all__"
        assert name in benchmark.__all__, f"{name} missing from benchmark.__all__"
    assert callable(classification_battery)
    assert "classification" in list_tasks()
