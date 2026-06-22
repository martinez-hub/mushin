# Adapted from MASSACHUSETTS INSTITUTE OF TECHNOLOGY's responsible-ai-toolbox
# (tests/conftest.py). Copyright 2023 MIT. SPDX-License-Identifier: MIT
import logging
import os

import pytest
from hypothesis import Verbosity, settings

# usage:
#   pytest tests --hypothesis-profile <profile-name>
settings.register_profile("ci", deadline=None)
settings.register_profile("fast", max_examples=10)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)


@pytest.fixture()
def cleandir(tmp_path):
    """Run function in a temporary directory."""
    old_dir = os.getcwd()  # get current working directory (cwd)
    os.chdir(tmp_path)  # change cwd to the temp-directory
    yield tmp_path  # yields control to the test to be run
    os.chdir(old_dir)
    logging.shutdown()
