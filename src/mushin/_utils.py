# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hydra_zen import load_from_yaml
from omegaconf import DictConfig, ListConfig
from torch import nn

log = logging.getLogger(__name__)


def original_cwd() -> Path:
    """Placeholder — real implementation added in Task 3."""
    return Path.cwd()


def load_from_checkpoint(
    model: nn.Module,
    *,
    ckpt: str | Path | None = None,
    weights_key: str | None = None,
    weights_key_strip: str | None = None,
    model_attr: str | None = None,
) -> nn.Module:
    """Load model weights.

    Parameters
    ----------
    model : Module
        The PyTorch Module

    ckpt : Optional[Union[str, Path]]
        The path to the file containing the model weights. If no path is provided
        the model will not be updated.

    weights_key : Optional[str] (default: "state_dict")
        (load_module=False) The key from the checkpoint file containing the model
        weights.

    weights_key_strip : Optional[str] (default: "model")
        (load_module=False) The prefix to remove from each weight's key prior
        to loading the module.

    model_attr : Optional[str] (default: "model")
        (load_module=False) The attribute of the module containing the `torch.nn.Module`

    Returns
    -------
    module : LightningModule
    """
    if ckpt is None:
        return model

    ckpt = Path(str(ckpt))
    if not ckpt.exists():
        ckpt = Path.home() / ".torch" / "models" / ckpt
    log.info(f"Loading model checkpoint from {ckpt}")

    # weights_only=False: these are trusted, self-produced checkpoints that may
    # hold more than tensors. torch 2.6 flipped this default to True.
    ckpt_data: dict[str, Any] = torch.load(ckpt, map_location="cpu", weights_only=False)

    if weights_key is not None:
        assert weights_key in ckpt_data
        ckpt_data = ckpt_data[weights_key]

    if weights_key_strip:
        if not weights_key_strip.endswith("."):
            weights_key_strip = weights_key_strip + "."

        ckpt_data = {
            k[len(weights_key_strip) :]: v
            for k, v in ckpt_data.items()
            if k.startswith(weights_key_strip)
        }

    if model_attr is None:
        # The weights can be loaded in directly
        model.load_state_dict(ckpt_data)

    else:
        assert hasattr(model, model_attr)
        getattr(model, model_attr).load_state_dict(ckpt_data)

    return model


@dataclass
class Experiment:
    working_dir: str
    cfg: dict | ListConfig | DictConfig | None
    ckpts: list[str]
    metrics: dict


def load_experiment(
    exp_path: str | Path, search_path: str | Path | None = None
) -> Experiment | list[Experiment]:
    """Loads all configuration and metrics outputs in an experiment directory.

    Parameters
    ----------
    exp_path: Union[str, Path]
        The directory to search for data. Directory must include the
        ".hydra/config.yaml" file.

    Returns
    ----------
    exps: Union[Experiment, List[Experiment]]

    """
    assert Path(exp_path).exists(), f"{exp_path} not found"

    # first find all .hydra directories
    if search_path is None:
        search_path = ".hydra"
    hydra_dirs = sorted(Path(exp_path).absolute().glob(f"**/{str(search_path)}"))

    # For each file load metrics data
    exps = []
    for path in hydra_dirs:
        # Save experiment configuration — load directly from the canonical
        # .hydra/config.yaml so that DDP rank configs (e.g. .pl_hydra_rank_*/
        # config.yaml) do not cause a silent cfg=None.
        config_path = path / "config.yaml"
        cfg = load_from_yaml(config_path) if config_path.exists() else None

        # Load metrics files
        files = path.parent.glob("*.pt")
        metrics = dict()
        for f in files:
            name = f.name
            metrics[name[:-3]] = torch.load(f, weights_only=False)

        # Load path to checkpoints
        ckpts = [str(ckpt.resolve()) for ckpt in path.parent.glob("**/*.ckpt")]

        # Append experiment to list; working_dir is the per-job directory
        # (path.parent), not its parent (path.parent.parent) which collapses
        # every multirun job to the shared root.
        exps.append(Experiment(str(path.parent), cfg, ckpts, metrics))

    if len(exps) == 1:
        return exps[0]

    return exps
