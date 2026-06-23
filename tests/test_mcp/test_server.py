# SPDX-License-Identifier: MIT
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
