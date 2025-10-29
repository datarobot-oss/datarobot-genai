##@ Environment

.PHONY: update-env
update-env: ## Update environment with dev + all extras (integration tests)
	uv sync --all-extras --dev

.PHONY: update-env-dev
update-env-dev: ## Update environment with dev only
	uv sync --dev
