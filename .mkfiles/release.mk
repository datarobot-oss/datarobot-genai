.PHONY: tag-and-push
tag-and-push: ## Create annotated tag from pyproject version and push
	@set -euo pipefail; \
	VERSION=$$(grep -E '^[[:space:]]*version[[:space:]]*=' -m1 pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/'); \
	echo "Tagging version: $$VERSION"; \
	git add -A; \
	git commit -m "Release v$$VERSION" || true; \
	git push; \
	git tag -a "v$$VERSION" -m "Release v$$VERSION"; \
	git push origin "v$$VERSION"
