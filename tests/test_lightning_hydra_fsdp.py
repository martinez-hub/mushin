# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT

import pytest


def test_hydra_fsdp_is_fsdp_strategy_with_mixin():
    from pytorch_lightning.strategies import FSDPStrategy

    from mushin.lightning.launchers import HydraFSDP, _HydraReattachMixin

    assert issubclass(HydraFSDP, FSDPStrategy)
    assert issubclass(HydraFSDP, _HydraReattachMixin)


def test_hydra_fsdp_configures_reattach_launcher_single_node():
    import torch

    from mushin.lightning.launchers import HydraFSDP, _HydraReattachLauncher

    strat = HydraFSDP.__new__(HydraFSDP)  # avoid full __init__/accelerator setup

    class _Env:
        creates_processes_externally = False

    strat.cluster_environment = _Env()
    # num_processes is derived from parallel_devices on FSDPStrategy
    strat.parallel_devices = [torch.device("cpu")] * 2
    strat.num_nodes = 1
    strat._configure_launcher()
    assert isinstance(strat._launcher, _HydraReattachLauncher)
    # The launcher must receive the correct topology: a swapped num_processes/
    # num_nodes wiring bug would still pass an isinstance check but break the
    # WORLD_SIZE computation and the rank-spawn loop bound.
    assert strat._launcher.num_processes == 2
    assert strat._launcher.num_nodes == 1
    assert strat._rank_0_will_call_children_scripts is True


def test_hydra_fsdp_steps_aside_under_external_launcher():
    import torch

    from mushin.lightning.launchers import HydraFSDP

    strat = HydraFSDP.__new__(HydraFSDP)

    class _Env:
        creates_processes_externally = True

    strat.cluster_environment = _Env()
    # num_processes is derived from parallel_devices on FSDPStrategy
    strat.parallel_devices = [torch.device("cpu")] * 4
    strat.num_nodes = 2
    strat._launcher = None
    strat._configure_launcher()
    assert strat._launcher is None


def test_hydra_fsdp_exported():
    import mushin
    from mushin import HydraFSDP as A
    from mushin.lightning import HydraFSDP as B

    assert A is B
    assert "HydraFSDP" in mushin.__all__
    assert "HydraFSDP" in mushin.lightning.__all__


@pytest.mark.cluster
def test_hydra_fsdp_multirun_end_to_end():
    # Human-run gate: a real FSDP run under Hydra multirun needs 2+ GPUs, so it
    # cannot run in CI. Deselected by default (addopts excludes `cluster`); run
    # with `pytest -m cluster` on a multi-GPU host. See
    # examples/sharding_fsdp_multirun.py for the actual runnable demo.
    pytest.skip("requires >=2 GPUs; run the example on a multi-GPU host")
