1. Ask user for a dev build name
2. In ~/workspace/recipe-datarobot-agent-application/agent/pyproject.toml add:

```
[tool.uv.sources]
datarobot-genai = { index = "test-pypi" }

[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple/"

[[tool.uv.index]]
name = "test-pypi"
url = "https://test.pypi.org/simple/"
```

3. Replace `datarobot-genai` version to the dev build version.
