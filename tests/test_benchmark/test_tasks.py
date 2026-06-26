import pytest

from mushin.benchmark._tasks import get_task_spec


def test_known_tasks():
    assert get_task_spec("classification").prob_metrics == frozenset({"auroc", "ece"})
    assert get_task_spec("segmentation").prob_metrics == frozenset()
    assert callable(get_task_spec("segmentation").battery)
    assert callable(get_task_spec("segmentation").predict_fn)


def test_unknown_task_raises():
    with pytest.raises(NotImplementedError, match="not supported"):
        get_task_spec("bogus_task")


def test_detection_task_registered_and_optional_num_classes():
    from mushin.benchmark._tasks import get_task_spec

    spec = get_task_spec("detection")
    assert spec.requires_num_classes is False
    assert spec.prob_metrics == frozenset()
    # classification still requires num_classes
    assert get_task_spec("classification").requires_num_classes is True
