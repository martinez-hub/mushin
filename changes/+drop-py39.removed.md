Dropped support for Python 3.9 (end-of-life October 2025); `mushin` now requires
Python >= 3.10. This refreshes the dependency lockfile to patched versions of
pillow, urllib3, aiohttp, filelock, requests, pytest, and pytorch-lightning,
clearing the Dependabot security alerts anchored on the old Python-3.9 dependency
branch. The `scipy` (>= 1.13) and `matplotlib` (>= 3.9) floors are raised to their
first NumPy-2-compatible releases, and the `mcp` extra no longer needs a Python
version gate.
