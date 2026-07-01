# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT


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
