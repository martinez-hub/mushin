---
name: guide
description: Use before writing or changing an experiment that uses mushin (mushin-py; sweeps, multirun, compare_methods, Study). Loads mushin's usage guide for AI coding agents, matching the installed version, with the rules that prevent wrong results and wasted compute.
allowed-tools: Bash(python -m mushin guide:*), Bash(python3 -m mushin guide:*), Bash(echo:*)
---

# mushin usage guide

The guide below is printed by the installed mushin, so it matches the
version this project uses. Follow it before writing or changing an
experiment.

!`python -m mushin guide 2>/dev/null || python3 -m mushin guide 2>/dev/null || echo "The mushin guide is not available here; see the note below."`

If the guide is not above, mushin is missing, older than 0.16, or not on
this shell's `python` or `python3` (activate the project's environment, or run
Claude Code from it). Install it with `pip install "mushin-py[eval]"` into the
project's environment (not `pip install mushin`, a PyPI name that is not this
project), or on mushin 0.10 to 0.15 read
https://raw.githubusercontent.com/martinez-hub/mushin/main/src/mushin/agent_guide.md
instead.
