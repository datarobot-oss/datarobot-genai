##@ Tests

PYTHON_MINOR_VERSION := $(shell uv run $(UV_RUN_OPTS) python -c 'import sys; print(sys.version_info.minor)')
ifeq ($(PYTHON_MINOR_VERSION), 10)
    NAT_IGNORED_TESTS := tests/nat
    COV_FAIL_UNDER := 78
else
    NAT_IGNORED_TESTS := tests/nat/integration
    COV_FAIL_UNDER := 80
endif


RESULTS := test-results.xml
JUNIT_REPORT ?= test_results/$(RESULTS)
JUNIT_FLAGS := --junitxml=$(JUNIT_REPORT)
PYTEST_OPTS ?= $(JUNIT_FLAGS) -vv -s
PYTEST_OPTS_W_COVERAGE ?= $(JUNIT_FLAGS) --cov=datarobot_genai --cov-branch --cov-fail-under=$(COV_FAIL_UNDER) --no-cov-on-fail

.PHONY: test
test: ## Run repo unit tests via pytest
	uv run $(UV_RUN_OPTS) pytest $(PYTEST_OPTS_W_COVERAGE) tests --ignore=tests/drmcp/integration --ignore=tests/drmcp/acceptance --ignore=$(NAT_IGNORED_TESTS)

# Map module names to test directory names (for modules with different naming)
ifeq ($(MODULE),llamaindex)
    TEST_MODULE := llama_index
else
    TEST_MODULE := $(MODULE)
endif

test-module: ## Run repo unit tests via pytest for a specific module
	uv run $(UV_RUN_OPTS) pytest tests/$(TEST_MODULE) --ignore=tests/drmcp/integration --ignore=tests/drmcp/acceptance --ignore=$(NAT_IGNORED_TESTS)
