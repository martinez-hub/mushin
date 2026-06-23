import pytest

from mushin.benchmark._tasks import get_task_spec


def test_known_tasks():
    assert get_task_spec("classification").prob_metrics == frozenset({"auroc", "ece"})
    assert get_task_spec("segmentation").prob_metrics == frozenset()
    assert callable(get_task_spec("segmentation").battery)
    assert callable(get_task_spec("segmentation").predict_fn)


def test_unknown_task_raises():
    with pytest.raises(NotImplementedError, match="not supported"):
        get_task_spec("detection")
