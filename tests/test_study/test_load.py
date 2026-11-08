import pytest
import torch

from _study_checkpoints import loader as _loader
from _study_checkpoints import save_models as _save_models
from mushin.benchmark import BenchmarkResult
from mushin.study._load import evaluate_checkpoints


def test_evaluate_checkpoints_returns_benchmark_result(tmp_path):
    checkpoints = _save_models(tmp_path, ["m1", "m2"])
    result = evaluate_checkpoints(
        checkpoints,
        load_fn=lambda p: torch.load(p, weights_only=False),
        data=_loader(),
        task="classification",
        num_classes=3,
        test="welch",
    )
    assert isinstance(result, BenchmarkResult)
    assert set(result.data.dims) == {"method", "seed"}
    assert result.data.sizes == {"method": 2, "seed": 2}
    assert "accuracy" in result.data.data_vars


def test_evaluate_checkpoints_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_checkpoints(
            {}, load_fn=lambda p: p, data=None, task="classification", num_classes=2
        )
