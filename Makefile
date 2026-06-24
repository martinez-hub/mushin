# mushin developer shortcuts. All targets run through uv.
# Usage: `make <target>` (e.g. `make check`). Run `make help` to list targets.

HYPOTHESIS_PROFILE ?= fast
PYTHON ?= 3.11

.DEFAULT_GOAL := help
.PHONY: help sync test test-fast test-py test-lowest lint format format-check spell check all changelog changelog-draft docs docs-build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync: ## Create/update the dev environment from uv.lock
	uv sync

test: ## Run the full test suite
	uv run pytest tests/ --hypothesis-profile $(HYPOTHESIS_PROFILE) -p no:cacheprovider

test-fast: ## Run tests with the fast hypothesis profile (alias of default)
	$(MAKE) test HYPOTHESIS_PROFILE=fast

test-py: ## Run tests on a specific Python version, e.g. `make test-py PYTHON=3.12`
	uv run --python $(PYTHON) pytest tests/ --hypothesis-profile $(HYPOTHESIS_PROFILE) -p no:cacheprovider

test-lowest: ## Run the suite against the LOWEST declared dep versions (x86_64 Linux via Docker; mirrors the min-versions CI job)
	docker run --rm --platform linux/amd64 -v "$(PWD)":/app -w /app \
		-e UV_PROJECT_ENVIRONMENT=/opt/venv -e UV_LINK_MODE=copy \
		ghcr.io/astral-sh/uv:python3.9-bookworm-slim \
		bash -c "uv sync --resolution lowest-direct && uv run pytest tests/ --hypothesis-profile fast"

lint: ## Lint with ruff
	uv run ruff check .

format: ## Auto-format with ruff
	uv run ruff format .

format-check: ## Check formatting without modifying files
	uv run ruff format --check .

spell: ## Spell-check with codespell
	uv run codespell src tests README.md CHANGELOG.md CONTRIBUTING.md RELEASING.md pyproject.toml changes

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
