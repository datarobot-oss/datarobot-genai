##@ Tests

RESULTS := test-results.xml
JUNIT_REPORT ?= test_results/$(RESULTS)
JUNIT_FLAGS := --junitxml=$(JUNIT_REPORT)
PYTEST_OPTS ?= $(JUNIT_FLAGS) -vv -s
PYTEST_OPTS_W_COVERAGE ?= $(JUNIT_FLAGS) --cov=datarobot_genai --cov-branch --cov-fail-under=80 --no-cov-on-fail

.PHONY: test
test: ## Run repo tests via pytest
	uv run $(UV_RUN_OPTS) pytest $(PYTEST_OPTS_W_COVERAGE) tests

.PHONY: test-acceptance
test-acceptance: ## Run acceptance tests via pytest
	uv run $(UV_RUN_OPTS) pytest $(PYTEST_OPTS) tests/acceptance
