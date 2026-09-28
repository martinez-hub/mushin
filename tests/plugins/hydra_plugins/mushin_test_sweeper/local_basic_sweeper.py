# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""A minimal sweeper plugin for ``tests/test_workflows.py``.

Hydra's plugin registry admits a sweeper only if it is defined under the
``hydra_plugins`` namespace package. hydra-zen's ``launch`` is moving onto that
registry (mit-ll-responsible-ai/hydra-zen#885), after which a sweeper defined
inside a test module is refused. Living here keeps the test valid on both the
current and the upcoming hydra-zen. ``pyproject.toml`` puts ``tests/plugins``
on ``pythonpath`` so Hydra's scanner finds it before any launch.
"""

from hydra.core.config_store import ConfigStore
from hydra.plugins.sweeper import Sweeper
from hydra_zen import builds


class LocalBasicSweeper(Sweeper):
    def setup(self, *, hydra_context, task_function, config):
        pass

    def sweep(self, arguments):
        return dict(hi=1)


ConfigStore.instance().store(
    group="hydra/sweeper", name="local_test", node=builds(LocalBasicSweeper)
)
