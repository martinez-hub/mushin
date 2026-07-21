# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra_zen import load_from_yaml
from omegaconf import DictConfig, ListConfig

if TYPE_CHECKING:
    from torch import nn

log = logging.getLogger(__name__)


def _hydra_original_cwd() -> str | None:
    """Return Hydra's launch cwd if a run is active, else None.

    Isolated for monkeypatching in tests. Hydra raises if no run is active, so we
    swallow that and let the caller fall back to the process cwd.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        from hydra.utils import get_original_cwd

        if HydraConfig.initialized():
            return get_original_cwd()
    except Exception:  # hydra not initialised / not installed in this context
        return None
    return None


def original_cwd() -> Path:
    """Directory the experiment was launched from.

    Inside a Hydra job the process cwd is the per-job output directory, so relative
    paths in a ``task()`` silently resolve against the wrong place. Use this to
    anchor paths against the launch directory instead::

        data = load(mushin.original_cwd() / "data" / "train.csv")

    Outside a Hydra run this is just the current working directory.
    """
    launch = _hydra_original_cwd()
    return Path(launch) if launch is not None else Path.cwd()


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

    weights_key : Optional[str] (default: None)
        The key from the checkpoint file containing the model weights. When
        None, the checkpoint dict is loaded into ``model`` directly.

    weights_key_strip : Optional[str] (default: None)
        A prefix to remove from each weight's key prior to loading. When None,
        keys are used as-is.

    model_attr : Optional[str] (default: None)
        The attribute of ``model`` to load the weights into. When None, they
        are loaded into ``model`` itself.

    Returns
    -------
    model : Module
        The same ``model`` instance, with its weights loaded.
    """
    if ckpt is None:
        return model

    import torch

    ckpt = Path(str(ckpt))
    if not ckpt.exists():
        ckpt = Path.home() / ".torch" / "models" / ckpt
    log.info(f"Loading model checkpoint from {ckpt}")

    # weights_only=False: these are trusted, self-produced checkpoints that may
    # hold more than tensors. torch 2.6 flipped this default to True.
    ckpt_data: dict[str, Any] = torch.load(ckpt, map_location="cpu", weights_only=False)

    if weights_key is not None:
        if weights_key not in ckpt_data:
            raise KeyError(
                f"weights_key {weights_key!r} not in checkpoint {ckpt}; "
                f"available keys: {sorted(map(str, ckpt_data))}"
            )
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
        if not hasattr(model, model_attr):
            raise AttributeError(
                f"model {type(model).__name__} has no attribute "
                f"{model_attr!r} to load the checkpoint weights into"
            )
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
    if not Path(exp_path).exists():
        raise FileNotFoundError(f"{exp_path} not found")

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
        import torch

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
