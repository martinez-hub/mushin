# Copyright 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# SPDX-License-Identifier: MIT
"""Keep the agent guide (`src/mushin/agent_guide.md`) honest.

Agents copy the guide's code and follow its links, and the guide ships inside
the package, so a stale keyword, a broken example or a dead link misleads every
agent that reads it. CI runs this file in the docs job too, because the `test`
job skips the docs-only PRs most likely to break it.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

import mushin
from _docs_site import (
    ROOT,
    _nav_urls,
    _site,
    markdown_links,
    python_blocks,
    stray_python_fences,
)

GUIDE = resources.files("mushin") / "agent_guide.md"


def _text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_the_guide_ships_as_ascii_markdown():
    text = _text()
    assert text.isascii(), sorted({c for c in text if not c.isascii()})
    first, second = text.splitlines()[:2]
    # docs/for-agents.md includes the guide from line 3, below its own title.
    assert first.startswith("# ") and second == ""


def test_python_m_mushin_guide(capsys):
    from mushin.__main__ import _version, main

    assert main(["guide"]) == 0
    assert capsys.readouterr().out == f"<!-- mushin {_version()} -->\n" + _text()

    assert main(["guide", "--path"]) == 0
    assert Path(capsys.readouterr().out.strip()).is_file()

    assert main([]) == 0
    assert "guide" in capsys.readouterr().out


def test_guide_text_is_what_the_command_prints(capsys):
    """The MCP prompt and resource return guide_text(); it must be exactly
    what `python -m mushin guide` prints, so the two can never drift."""
    from mushin.__main__ import guide_text, main

    assert main(["guide"]) == 0
    assert capsys.readouterr().out == guide_text()
    assert guide_text().startswith("<!-- mushin ")
    assert guide_text().endswith(_text())


def test_the_guide_prints_on_a_cp1252_pipe():
    """A Windows pipe writes in the ANSI code page; the guide must survive it."""
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    env.pop("PYTHONUTF8", None)
    result = subprocess.run(
        [sys.executable, "-m", "mushin", "guide"],
        capture_output=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-2000:]
    assert result.stdout.decode("cp1252").startswith("<!-- mushin ")


RAW_PREFIX = "https://raw.githubusercontent.com/martinez-hub/mushin/main/"
# The files that point agents on mushin 0.10 to 0.15 at the raw guide.
RAW_LINK_FILES = ["docs/for-agents.md", "README.md", "docs/llms.txt"]
# Keywords the guide's prose names that are not parameters of run() or
# compare_methods: xarray's, and the example knob of the per-method rule.
PROSE_KEYWORDS_ELSEWHERE = {"keep_attrs", "level"}


def _python_blocks() -> list[str]:
    return python_blocks(_text())


def test_every_python_example_is_tagged_python():
    """Only fences tagged exactly ```python at the start of a line run below.
    Any other Python fence (indented, `~~~`, `Python`, `python3`, `pycon`, or
    with attributes) would show code that never runs, so it fails here."""
    stray = stray_python_fences(_text())
    assert not stray, stray


def test_the_guide_code_runs(tmp_path):
    """Every python block, in order, as one script, the way an agent runs it.
    A real script, so `inspect.getsource` works for mushin's fingerprints, and
    Hydra's global state stays out of this process."""
    pytest.importorskip("scipy")  # the template calls compare_methods
    blocks = _python_blocks()
    assert blocks, "the guide has no python example"
    (tmp_path / "guide_example.py").write_text("\n".join(blocks), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "guide_example.py"],
        cwd=tmp_path,
        # As the guide tells agents to run mushin on Windows.
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]


def _run_parameters() -> set[str]:
    from mushin.workflows import BaseWorkflow, MultiRunMetricsWorkflow

    names: set[str] = set()
    for fn in (MultiRunMetricsWorkflow.run, BaseWorkflow.run):
        for p in inspect.signature(fn).parameters.values():
            if p.kind is not p.VAR_KEYWORD and p.name != "self":
                names.add(p.name)
    return names


def _is_mushin_sweep(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "sweep"
        and isinstance(node.value, ast.Name)
        and node.value.id == "mushin"
    )


def _parameters(fn: ast.FunctionDef) -> set[str]:
    return {a.arg for a in [*fn.args.args, *fn.args.kwonlyargs]}


def test_the_guide_run_keywords_exist():
    """Each keyword of a mushin `run()` call in the guide is a parameter of the
    task or of `run()`. `run()` accepts arbitrary task overrides, so a renamed
    safety keyword (`dry_run=`, `working_dir=`) would otherwise pass silently.
    `Sweep.run` is `(self, **kwargs)`, so its signature proves nothing."""
    tree = ast.parse("\n".join(_python_blocks()))
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    # The class API: MultiRunMetricsWorkflow or a subclass, with the parameters
    # of the subclass's `task` method.
    classes: dict[str, set[str]] = {"MultiRunMetricsWorkflow": set()}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
            ast.unparse(base).endswith("MultiRunMetricsWorkflow") for base in node.bases
        ):
            task = next(
                (
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "task"
                ),
                None,
            )
            classes[node.name] = _parameters(task) if task else set()

    def workflow_task(node: ast.expr) -> set[str] | None:
        """The task's parameters if `node` instantiates a workflow class."""
        if not isinstance(node, ast.Call):
            return None
        return classes.get(ast.unparse(node.func).rsplit(".", 1)[-1])

    handles: dict[str, set[str]] = {}  # name -> its task's parameters
    dicts: dict[str, list[str]] = {}  # name bound to a dict literal or dict(...)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            _is_mushin_sweep(d) for d in node.decorator_list
        ):
            handles[node.name] = _parameters(node)
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        target, value = node.targets[0].id, node.value
        if isinstance(value, ast.Call) and _is_mushin_sweep(value.func):
            first = value.args[0] if value.args else None
            fn = functions.get(first.id) if isinstance(first, ast.Name) else None
            handles[target] = _parameters(fn) if fn else set()
        elif workflow_task(value) is not None:
            handles[target] = workflow_task(value)
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "dict"
            and not value.args
            and all(k.arg for k in value.keywords)  # no dict(**other)
        ):
            dicts[target] = [k.arg for k in value.keywords]
        elif isinstance(value, ast.Dict) and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str)
            for k in value.keys  # a `{**other}` key is None, so it is not bound
        ):
            dicts[target] = [k.value for k in value.keys]

    allowed_run = _run_parameters()
    bad, checked = [], 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
        ):
            continue
        owner = node.func.value
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "workflow"
            and isinstance(owner.value, ast.Name)
            and owner.value.id in handles
        ):
            owner = owner.value  # experiment.workflow.run(...) is the same run
        if isinstance(owner, ast.Name) and owner.id in handles:
            task_params = handles[owner.id]
        elif workflow_task(owner) is not None:  # e.g. MyWorkflow().run(...)
            task_params = workflow_task(owner)
        else:
            continue  # not a mushin run(), e.g. subprocess.run
        checked += 1
        keywords: list[str] = []
        for kw in node.keywords:
            if kw.arg is not None:
                keywords.append(kw.arg)
            elif isinstance(kw.value, ast.Name) and kw.value.id in dicts:
                keywords += dicts[kw.value.id]
            else:
                bad.append(f"example line {node.lineno}: `**{ast.unparse(kw.value)}`")
        allowed = allowed_run | task_params
        bad += [
            f"example line {node.lineno}: `{k}=` is not a run() parameter"
            for k in keywords
            if k not in allowed
        ]
    assert checked, "found no mushin run() call in the guide"
    assert not bad, bad


def test_the_guide_links_are_nav_pages():
    links = [url.split("#", 1)[0] for url in markdown_links(_text())]
    site_links = [url for url in links if url.startswith(_site())]
    # If site_url moves, every old link stops matching it: fail, not pass.
    assert site_links, f"the guide has no link into the docs site {_site()}"
    stale = sorted(set(site_links) - _nav_urls())
    assert not stale, f"the guide links to pages not in mkdocs.yml's nav: {stale}"
    others = [url for url in links if not url.startswith(_site())]
    assert not others, f"the guide links outside the docs site: {others}"


def test_raw_guide_links_name_a_file_in_the_repo():
    """The raw-GitHub URL is the only fallback D2 gives agents on mushin 0.10 to
    0.15, so each file that tells agents to run the command must carry it, on
    this repo's main branch and naming a file that exists."""
    for rel in RAW_LINK_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        urls = re.findall(r"https://raw\.githubusercontent\.com/[^\s)>`\"']+", text)
        assert urls, f"{rel} does not link the raw guide"
        for url in urls:
            assert url.startswith(RAW_PREFIX), f"{rel}: {url} is not on {RAW_PREFIX}"
            assert (ROOT / url.removeprefix(RAW_PREFIX)).is_file(), f"{rel}: {url}"


def test_the_guide_prose_keywords_exist():
    """The rules name keywords in prose (`confirm_above=N`, `sample=N`,
    `resume=True`, `capture_env=True`...). A renamed one would leave the guide
    telling agents to pass a keyword that `run()` takes as a task override."""
    pytest.importorskip("scipy")
    from mushin.benchmark import compare_methods

    tree = ast.parse("\n".join(_python_blocks()))
    task_params = {
        a.arg
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        for a in n.args.args
    }
    prose = re.sub(r"^```.*?^```$", "", _text(), flags=re.DOTALL | re.MULTILINE)
    named = set(re.findall(r"`([a-z_]+)=", prose))
    allowed = (
        _run_parameters()
        | set(inspect.signature(compare_methods).parameters)
        | task_params
        | set(inspect.signature(mushin.best).parameters)  # best(over=)
        | PROSE_KEYWORDS_ELSEWHERE
    )
    assert named, "found no keyword in the guide's prose"
    assert not named - allowed, sorted(named - allowed)


def test_the_guide_install_facts_match_pyproject():
    """The Python range and the extras the guide states match pyproject.toml,
    as tests/test_llms_txt.py checks for llms.txt."""
    import tomllib

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = tomllib.loads(pyproject)["project"]
    minors = sorted(
        int(c.rsplit(".", 1)[1])
        for c in project["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    )
    match = re.search(r"Python 3\.(\d+) to 3\.(\d+) is supported", _text())
    assert match, "could not find the guide's 'Python 3.X to 3.Y is supported'"
    assert (int(match[1]), int(match[2])) == (minors[0], minors[-1])

    extras = set(project["optional-dependencies"])
    install = _text().split("## Install", 1)[1].split("\n## ", 1)[0]
    missing = sorted(
        e for e in extras if f"[{e}]" not in install and f"`{e}`" not in install
    )
    assert not missing, f"the guide's install section omits extras {missing}"
    unknown = set(re.findall(r"mushin-py\[(\w+)\]", _text())) - extras
    assert not unknown, f"the guide names extras pyproject.toml lacks: {unknown}"


def test_the_guide_renders_as_lists_on_the_docs_site():
    """The docs site renders with Python-Markdown, which needs 4-space indents
    for blocks inside a list item. A fence inside a bullet, or a list item it
    does not continue, turns the rest of the list into one paragraph of literal
    `- ` text; a `- [ ]` checklist shows literal brackets (no tasklist
    extension)."""
    markdown = pytest.importorskip("markdown")
    pytest.importorskip("pymdownx")
    html = markdown.markdown(
        _text(), extensions=["tables", "admonition", "pymdownx.superfences"]
    )
    paragraphs = re.findall(r"<p>(.*?)</p>", html, re.DOTALL)
    broken = [p for p in paragraphs if re.search(r"(^|\n)\s*(- |\d+\. )", p)]
    assert not broken, broken
    assert "[ ]" not in html
