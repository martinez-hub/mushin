Dropped support for Python 3.9 (end-of-life October 2025); `mushin` now requires
Python >= 3.10. This refreshes the dependency lockfile to patched versions of
pillow, urllib3, aiohttp, filelock, requests, pytest, and pytorch-lightning,
clearing the Dependabot security alerts anchored on the old Python-3.9 dependency
branch. The `matplotlib` floor is raised to >= 3.6 (the first release with Python
3.10 wheels), and the `mcp` extra no longer needs a Python version gate.
