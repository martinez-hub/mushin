# Adapted from MASSACHUSETTS INSTITUTE OF TECHNOLOGY's responsible-ai-toolbox
# (tests/conftest.py). Copyright 2023 MIT. SPDX-License-Identifier: MIT
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless backend for all tests; set before pyplot import

import pytest
from hypothesis import Verbosity, settings

# usage:
#   pytest tests --hypothesis-profile <profile-name>
settings.register_profile("ci", deadline=None)
# deadline=None: CI runners are noisy-neighbor machines; Hypothesis's default
# 200ms per-example deadline turns load spikes into flaky DeadlineExceeded
# failures for any @given test that forgets a per-test override.
settings.register_profile("fast", max_examples=10, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)


@pytest.fixture()
def cleandir(tmp_path):
    """Run function in a temporary directory."""
    old_dir = os.getcwd()  # get current working directory (cwd)
    os.chdir(tmp_path)  # change cwd to the temp-directory
    yield tmp_path  # yields control to the test to be run
    os.chdir(old_dir)
    logging.shutdown()


@pytest.fixture(autouse=True)
def _restore_task_registry():
    """Snapshot and restore the global task registry around every test so that
    tests which call register_task() don't leak entries into other tests."""
    from mushin.benchmark._tasks import _TASKS

    snapshot = dict(_TASKS)
    yield
    _TASKS.clear()
    _TASKS.update(snapshot)


@pytest.fixture
def restore_config_group(request):
    """Remove a user-stored top-level ConfigStore group after the test so it
    does not leak into the session. A blanket snapshot/restore is unsafe —
    Hydra lazily registers its built-in ``sweeper``/``launcher`` groups during
    a sweep, and restoring an earlier snapshot would drop them. Pass the group
    name(s) via ``@pytest.mark.config_group("model")``."""
    from hydra.core.config_store import ConfigStore

    groups = [g for m in request.node.iter_markers("config_group") for g in m.args]
    yield
    repo = ConfigStore.instance().repo
    for g in groups:
        repo.pop(g, None)
