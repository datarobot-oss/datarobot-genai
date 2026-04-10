# Test matrix

Agents has variety of frameworks, and each framework can be configured with different LLM options. We aim to cover here all combinations with at least a minimal test set.

# Environment

For all tests you need 4 .env files:
- .env with regular variables
- .env.LLM with variables for individual LLM test-cases

# Running agent and tests

Commands to run tests and run an agent are intentionally separated: this allows you to see the agent logs, and easily stop it after the session. Running agent
in the background would like its process to hang on a part, which is inconvenient.

To run an agent, do:
```shell
AGENT=llamaindex LLM=external task run-dragent
```
This will install depenendencies, and start an agent with a proper configuration.

> Mind that order of env variables is important: for Task to pick up them to load dynamic env files you should put them at the start of the line!

To run a test against an agent, do:

```shell
AGENT=llamaindex LLM=external task test-dragent
```

Tests are agent and LLM specific, some cases might be skipped or interpreted differently based on context.
