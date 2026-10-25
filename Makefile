# mushin developer shortcuts. All targets run through uv.
# Usage: `make <target>` (e.g. `make check`). Run `make help` to list targets.

HYPOTHESIS_PROFILE ?= fast
PYTHON ?= 3.11

.DEFAULT_GOAL := help
.PHONY: help sync test test-fast test-py test-lowest lint format format-check spell check all changelog changelog-draft docs docs-build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync: ## Create/update the dev environment (incl. the eval extra) from uv.lock
	uv sync --extra eval

test: ## Run the full test suite (eval extra on, so benchmark/LLM/Study tests run)
	uv run --extra eval pytest tests/ --hypothesis-profile $(HYPOTHESIS_PROFILE) -p no:cacheprovider

test-fast: ## Run tests with the fast hypothesis profile (alias of default)
	$(MAKE) test HYPOTHESIS_PROFILE=fast

test-py: ## Run tests on a specific Python version, e.g. `make test-py PYTHON=3.12`
	uv run --extra eval --python $(PYTHON) pytest tests/ --hypothesis-profile $(HYPOTHESIS_PROFILE) -p no:cacheprovider

test-lowest: ## Run the suite against the LOWEST declared dep versions (x86_64 Linux via Docker; mirrors the min-versions CI job)
	docker run --rm --platform linux/amd64 -v "$(PWD)":/src:ro -e UV_LINK_MODE=copy \
		ghcr.io/astral-sh/uv:python3.11-bookworm-slim \
		bash -c "apt-get update -qq && apt-get install -y -qq git >/dev/null \
			&& mkdir /app && tar -C /src --exclude=./.venv --exclude=./.git -cf - . | tar -C /app --no-same-owner -xf - \
			&& cd /app && git init -q && git add -A && git -c user.name=test-lowest -c user.email=test-lowest@localhost commit -qm snapshot \
			&& uv sync --resolution lowest-direct --extra eval \
			&& uv pip install 'setuptools<81' \
			&& uv run --no-sync pytest tests/ --hypothesis-profile fast -p no:cacheprovider"

lint: ## Lint with ruff
	uv run ruff check .

format: ## Auto-format with ruff
	uv run ruff format .

format-check: ## Check formatting without modifying files
	uv run ruff format --check .

# The spell-check scope. CI runs this target; the codespell hook in
# .pre-commit-config.yaml repeats the scope as a path regex, so change both.
spell: ## Spell-check with codespell
	uv run codespell --check-hidden src tests examples plugins .claude-plugin README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md AGENTS.md CLAUDE.md docs/llms.txt pyproject.toml changes

check: lint format-check spell test ## Run all checks (what CI runs)

all: check ## Alias for `check`

changelog:  ## Assemble news fragments into CHANGELOG.md (VERSION=X.Y.Z)
	uv run towncrier build --version $(VERSION)

changelog-draft:  ## Preview the next changelog section without consuming fragments (VERSION=X.Y.Z)
	uv run towncrier build --draft --version $(VERSION)

docs: ## Serve the docs site locally with live reload
	uv run --group docs mkdocs serve

docs-build: ## Build the docs site with strict checking (what CI runs)
	uv run --group docs mkdocs build --strict
