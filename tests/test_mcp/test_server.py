# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mushin.mcp.server import RootError, _resolve, _to_jsonable


def test_to_jsonable_scalar_tensor():
    assert _to_jsonable(torch.tensor(0.5)) == 0.5


def test_to_jsonable_tensor_array():
    assert _to_jsonable(torch.tensor([1.0, 2.0])) == [1.0, 2.0]


def test_to_jsonable_numpy_and_nested():
    out = _to_jsonable({"a": np.float32(1.5), "b": [np.int64(2)]})
    assert out == {"a": 1.5, "b": [2]}


def test_to_jsonable_omegaconf():
    cfg = OmegaConf.create({"lr": 0.1, "nested": {"seed": 0}})
    assert _to_jsonable(cfg) == {"lr": 0.1, "nested": {"seed": 0}}


def test_to_jsonable_non_finite_float_becomes_string():
    assert _to_jsonable(float("inf")) == "inf"


def test_resolve_no_root_returns_absolute(tmp_path):
    target = tmp_path / "exp"
    target.mkdir()
    assert _resolve(target, None) == target.resolve()


def test_resolve_inside_root_ok(tmp_path):
    root = tmp_path
    target = tmp_path / "exp"
    target.mkdir()
    assert _resolve(target, root) == target.resolve()


def test_resolve_outside_root_raises(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    with pytest.raises(RootError):
        _resolve(outside, root)


def _make_experiment(base: Path, lrs=(0.1, 0.2)) -> Path:
    """Build a minimal 2-run Hydra multirun layout under ``base``."""
    for i, lr in enumerate(lrs):
        run = base / str(i)
        (run / ".hydra").mkdir(parents=True)
        OmegaConf.save(
            OmegaConf.create({"lr": lr, "seed": 0}),
            run / ".hydra" / "config.yaml",
        )
        torch.save({"accuracy": torch.tensor(0.8 + 0.1 * i)}, run / "metrics.pt")
    return base


def test_list_experiments_finds_runs(tmp_path):
    from mushin.mcp.server import _list_experiments

    base = _make_experiment(tmp_path / "exp")
    out = _list_experiments(base)
    assert out["count"] == 2
    assert sorted(Path(r).name for r in out["runs"]) == ["0", "1"]
