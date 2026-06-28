import pytest

from mushin.benchmark._tasks import get_task_spec


def test_known_tasks():
    assert get_task_spec("classification").prob_metrics == frozenset({"auroc", "ece"})
    assert get_task_spec("segmentation").prob_metrics == frozenset()
    assert callable(get_task_spec("segmentation").battery)
    assert callable(get_task_spec("segmentation").predict_fn)


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="not a registered task"):
        get_task_spec("bogus_task")


def test_detection_task_registered_and_optional_num_classes():
    from mushin.benchmark._tasks import get_task_spec

    spec = get_task_spec("detection")
    assert spec.requires_num_classes is False
    assert spec.prob_metrics == frozenset()
    # classification still requires num_classes
    assert get_task_spec("classification").requires_num_classes is True


def test_task_is_public_and_frozen():
    from dataclasses import FrozenInstanceError

    from mushin.benchmark._tasks import Task

    t = Task(
        battery=lambda num_classes, ignore_index=None: {},
        predict_fn=lambda model, x: (x, x),
    )
    # prob_metrics and requires_num_classes have defaults; description too
    assert t.prob_metrics == frozenset()
    assert t.requires_num_classes is True
    assert t.description == ""
    with pytest.raises(FrozenInstanceError):
        t.description = "nope"


def test_taskspec_alias_still_works():
    from mushin.benchmark._tasks import Task, TaskSpec

    assert TaskSpec is Task


def test_builtins_have_descriptions():
    from mushin.benchmark._tasks import _TASKS

    for name, spec in _TASKS.items():
        assert spec.description, f"{name} should have a non-empty description"


def _toy_task():
    from mushin.benchmark._tasks import Task

    return Task(
        battery=lambda num_classes, ignore_index=None: {},
        predict_fn=lambda model, x: (x, x),
        description="toy",
    )


def test_list_tasks_returns_builtins_with_descriptions():
    from mushin.benchmark._tasks import list_tasks

    tasks = list_tasks()
    assert set(tasks) >= {"classification", "segmentation", "detection"}
    assert all(desc for desc in tasks.values())


def test_register_and_get_task():
    from mushin.benchmark._tasks import get_task, register_task

    register_task("toy_reg", _toy_task())
    assert get_task("toy_reg").description == "toy"


def test_register_duplicate_requires_overwrite():
    from mushin.benchmark._tasks import register_task

    register_task("toy_dup", _toy_task())
    with pytest.raises(ValueError, match="already registered"):
        register_task("toy_dup", _toy_task())
    register_task("toy_dup", _toy_task(), overwrite=True)


def test_register_validates_inputs():
    from mushin.benchmark._tasks import register_task

    with pytest.raises(ValueError, match="non-empty"):
        register_task("", _toy_task())
    with pytest.raises(TypeError, match="Task"):
        register_task("bad", object())


def test_get_task_unknown_raises():
    from mushin.benchmark._tasks import get_task

    with pytest.raises(ValueError, match="not a registered task"):
        get_task("nope_not_a_task")
