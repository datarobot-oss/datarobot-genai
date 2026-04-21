# E2E test matrix

The repository supports several agent frameworks, and each framework can use different LLM options. The tests aim to cover these combinations with at least a minimal set.

## Environment

You need two environment files for all tests:

- `.env`&mdash;regular variables.
- `.env.LLM`&mdash;variables for individual LLM test cases.

## Run the agent and tests

Commands for running tests and running an agent stay separate on purpose. That way you can watch agent logs and stop the process after a session. Running the agent in the background often leaves the process stuck in a noisy state, which is awkward for interactive use.

To start an agent:

```shell
AGENT=llamaindex LLM=external task run-dragent
```

This installs dependencies and starts an agent with the right configuration.

**Important:** Put `AGENT` and `LLM` at the start of the line. Task uses them to load dynamic env files; order matters.

To run tests against a running agent:

```shell
AGENT=llamaindex LLM=external task test-dragent
```

Tests vary by agent and LLM context; some cases are skipped or interpreted differently depending on configuration.
