##@ Linters

SOURCE_DIRS ?= ./src
TEST_DIRS ?= ./tests
PYTHONPATH ?= ./src

.PHONY: check-ruff
check-ruff:  ## Check ruff linter rules, but don't apply fixers. Used by CI.
	uv run $(UV_RUN_OPTS) ruff check $(SOURCE_DIRS)
	uv run $(UV_RUN_OPTS) ruff check $(TEST_DIRS)

.PHONY: check-ruff-format
check-ruff-format:  ## Check formatting with ruff.
	uv run $(UV_RUN_OPTS) ruff format --check $(SOURCE_DIRS)
	uv run $(UV_RUN_OPTS) ruff format --check $(TEST_DIRS)

.PHONY: fix-ruff
fix-ruff:  ## Run ruff linter, applying fixers. Used by pre-commit.
	uv run $(UV_RUN_OPTS) ruff check --fix $(SOURCE_DIRS)
	uv run $(UV_RUN_OPTS) ruff check --fix $(TEST_DIRS)

.PHONY: fix-ruff-format
fix-ruff-format:  ## Run ruff linter, applying formatting fixes.
	uv run $(UV_RUN_OPTS) ruff format $(SOURCE_DIRS)
	uv run $(UV_RUN_OPTS) ruff format $(TEST_DIRS)

.PHONY: lint
lint: check-ruff-format check-ruff mypy	## Run all linters

.PHONY: delint
delint: fix-ruff-format fix-ruff  ## Try fixing all linting issues automatically.

.PHONY: mypy
mypy: ## Run mypy checks.
	uv run $(UV_RUN_OPTS) mypy --check-untyped-defs --no-site-packages $(SOURCE_DIRS)

