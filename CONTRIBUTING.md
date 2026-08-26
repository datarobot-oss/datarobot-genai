# Contributing Guidelines

Guidelines for developing and contributing to this project.

## List of project maintainers

- [Cavit Cakir](https://github.com/cavitcakir)
- [James Clemens](https://github.com/jpclemens0)


## Opening new issues

- Before opening a new issue check if there are any existing FAQ entries (if one exists), issues or pull requests that match your case
- Open an issue, and make sure to label the issue accordingly - bug, improvement, feature request, etc...
- Be as specific and detailed as possible

## Did you find a bug?

- Do not open up a GitHub issue if the bug is a security
vulnerability, instead email the maintainers directly or email
oss-community-management@datarobot.com if they do not respond within
seven days
- Ensure the bug was not already reported in the projects Issues section
- Open an issue as described above

## Changelog

All pull requests should include an entry in the [CHANGELOG.md](CHANGELOG.md) file unless the changes are trivial (e.g., fixing typos, minor documentation updates).

If your PR doesn't require a changelog entry (e.g., documentation-only changes, CI configuration), add the `skip-changelog` label to your pull request.

For a stack of dependent PRs, one entry anywhere in the stack covers all of them — see
[Stacks](#stacks) below.

## Versioning and releases

Any PR that changes `src/datarobot_genai/**` must bump `version` in [pyproject.toml](pyproject.toml)
(run `task install` afterwards so `uv.lock` stays in sync). The `version-check` job enforces this by
comparing your branch against `main`.

### Stacks

A [stack](https://docs.github.com/en/pull-requests/get-started/about-stacked-prs) of dependent PRs is
one unit of release, so `version-check` and `changelog-check` evaluate the stack rather than each PR
in it. Both resolve the stack's tip — the PR that lands last — and check that PR against the stack's
base. Because the tip's tree contains every layer below it, one version bump and one CHANGELOG entry
anywhere in the stack satisfy the check for all of its PRs. Put the bump on the tip: mid-stack it only
becomes a `pyproject.toml` conflict to re-resolve on the next rebase.

Nothing needs to be labelled for this — the checks read stack membership from the API on every run.
Two consequences worth knowing:

- Bumping the tip does not automatically re-run the checks on the PRs below it, since pushing to one
  branch is not an event on the others. `gh stack sync` pushes the whole stack and re-runs everything;
  re-running the job from the Actions UI also works, because it re-reads the stack rather than the
  event payload.
- If the tip isn't rebased onto a bump made lower down, the tip's `pyproject.toml` still shows the old
  version and the check fails. `gh stack sync` fixes that too.

### Merging without a bump

When `main` moves, the release workflow tags and publishes whatever version it finds there — and only
if that tag doesn't already exist yet. Merging a PR without a bump is therefore safe: nothing is
released, and the next PR that does bump cuts a single release covering everything merged since the
last one.

## Responding to issues and pull requests

This project's maintainers will make every effort to respond to any
open issues as soon as possible.

If you don't get a response within seven days of creating your issue or
pull request, please send us an email at oss-community-management@datarobot.com
