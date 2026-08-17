"""Execute each example notebook with ONLY the packages it tells the reader to install.

The regular `notebooks` CI job runs nbmake with every extra installed
(`--extra eval --extra viz --extra netcdf`), so it proves the cells execute but
is structurally blind to a notebook whose stated prerequisites are wrong: a
notebook that needs `eval` while its callout names only `viz` still passes
there. That exact defect shipped once and was caught only in review.

This script closes that gap. For each notebook it parses the `%pip install ...`
line out of the notebook's own requirements callout, builds a throwaway venv
containing exactly that (plus the test runner), and executes the notebook in it.
A notebook whose callout under-specifies its dependencies fails here.

Usage: python scripts/check_notebook_prereqs.py [notebook ...]
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "docs" / "notebooks"
# The callout is markdown of the form: `%pip install scikit-learn "mushin-py[viz]"`
PIP_RE = re.compile(r"%pip install ([^\n`]+)")
# Packages the runner needs that are not part of what a reader installs.
RUNNER = ["pytest", "nbmake", "ipykernel"]


def declared_install(path: Path) -> list[str]:
    """Return the pip arguments the notebook tells the reader to run."""
    nb = json.loads(path.read_text())
    md = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"
    )
    matches = PIP_RE.findall(md)
    if not matches:
        raise SystemExit(
            f"{path.name}: no '%pip install ...' found. Every notebook must state "
            "its prerequisites in a requirements callout so this check can verify them."
        )
    # A notebook may repeat the line; they must agree.
    specs = {" ".join(shlex.split(m.strip())) for m in matches}
    if len(specs) > 1:
        raise SystemExit(f"{path.name}: conflicting install lines: {sorted(specs)}")
    return shlex.split(matches[0].strip())


def as_local(args: list[str]) -> list[str]:
    """Point any mushin-py requirement at this checkout instead of PyPI."""
    out = []
    for a in args:
        m = re.fullmatch(r"mushin-py(\[[a-z,]+\])?", a)
        out.append(f"mushin-py{m.group(1) or ''} @ {REPO}" if m else a)
    return out


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, **kw)


def main(argv: list[str]) -> int:
    paths = (
        [Path(a).resolve() for a in argv] if argv else sorted(NOTEBOOKS.glob("*.ipynb"))
    )
    if not paths:
        raise SystemExit("no notebooks found")

    groups: dict[tuple[str, ...], list[Path]] = defaultdict(list)
    for p in paths:
        groups[tuple(declared_install(p))].append(p)

    print(f"{len(paths)} notebook(s) in {len(groups)} distinct prerequisite group(s)\n")
    failures: list[str] = []

    for spec, members in sorted(groups.items(), key=lambda kv: kv[1][0].name):
        names = ", ".join(p.name for p in members)
        print(f"=== {' '.join(spec)}\n    {names}")
        venv = Path(tempfile.mkdtemp(prefix="nbprereq-"))
        try:
            run(["uv", "venv", "--python", "3.11", str(venv), "-q"], check=True)
            install = run(
                ["uv", "pip", "install", "-q", *as_local(list(spec)), *RUNNER],
                env={**os.environ, "VIRTUAL_ENV": str(venv)},
            )
            if install.returncode != 0:
                failures.append(f"{names}: install failed for `{' '.join(spec)}`")
                print("    INSTALL FAILED\n")
                continue
            proc = run(
                [
                    str(venv / "bin" / "python"),
                    "-m",
                    "pytest",
                    "--nbmake",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    *[str(p) for p in members],
                ]
            )
            if proc.returncode != 0:
                failures.append(
                    f"{names}: failed with only `{' '.join(spec)}` installed — "
                    "the notebook's stated prerequisites are incomplete"
                )
                print("    FAILED\n")
            else:
                print("    ok\n")
        finally:
            shutil.rmtree(venv, ignore_errors=True)

    if failures:
        print("\nNotebook prerequisite check FAILED:")
        for f in failures:
            print(f"  - {f}")
        print(
            "\nFix the notebook's requirements callout so it names everything the "
            "notebook imports (or drop the unused import)."
        )
        return 1
    print("All notebooks run with exactly the prerequisites they declare.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
