"""Pin the PR-time vulnerability audit of uv.lock (#214).

The Trivy gate in container.yml runs only after a release tag exists, so a CVE
in uv.lock used to surface as a release with no published image. The `lint`
job in test.yml audits the lock on every PR instead, and `just audit` runs the
same command locally. These tests fail if the CI step is removed, moved out of
the `lint` job, or drifts from the justfile recipe.

The files are read as text: PyYAML is not a dependency of this project.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
JUSTFILE = REPO_ROOT / "justfile"
RENOVATE = REPO_ROOT / "renovate.json"

AUDIT_STEP_NAME = "Audit locked dependencies for known vulnerabilities"

# Each side writes the exported requirements to its own temp location
# ($RUNNER_TEMP in CI, $TMPDIR locally); everything else must match.
_REQUIREMENTS_PATH = re.compile(r'"[^"\s]*requirements-audit\.txt"')


def _normalise(commands: list[str]) -> list[str]:
    return [_REQUIREMENTS_PATH.sub("<requirements>", " ".join(c.split())) for c in commands]


def _job_block(workflow: str, job: str) -> str:
    """Return the text of one top-level job under `jobs:`."""
    match = re.search(rf"^  {re.escape(job)}:\n(.*?)(?=^  \S|\Z)", workflow, re.MULTILINE | re.DOTALL)
    assert match, f"job {job!r} not found in {TEST_WORKFLOW.name}"
    return match.group(1)


def _step_run_commands(job_text: str, step_name: str) -> list[str]:
    """Return the non-empty lines of a step's `run: |` block."""
    match = re.search(
        rf"^(\s*)- name: {re.escape(step_name)}\n\s*run: \|\n((?:\1  .*\n|\s*\n)+)",
        job_text,
        re.MULTILINE,
    )
    assert match, f"step {step_name!r} with a `run: |` block not found"
    return [line.strip() for line in match.group(2).splitlines() if line.strip()]


def _recipe_commands(justfile: str, recipe: str) -> list[str]:
    match = re.search(rf"^{re.escape(recipe)}:\n((?:[ \t]+.*\n)+)", justfile, re.MULTILINE)
    assert match, f"recipe {recipe!r} not found in justfile"
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


def test_lint_job_audits_the_lock():
    commands = _step_run_commands(_job_block(TEST_WORKFLOW.read_text(), "lint"), AUDIT_STEP_NAME)

    assert any(c.startswith("uv export --frozen --no-dev") for c in commands)
    assert any(c.startswith("uvx pip-audit==") for c in commands), "pip-audit must be pinned to an exact version"


def test_ci_audit_matches_just_audit():
    ci = _step_run_commands(_job_block(TEST_WORKFLOW.read_text(), "lint"), AUDIT_STEP_NAME)
    local = _recipe_commands(JUSTFILE.read_text(), "audit")

    assert _normalise(ci) == _normalise(local)


def test_renovate_opens_security_prs_from_osv():
    # Dependabot alerts are disabled on this repository, so OSV is the only
    # source Renovate has for vulnerability PRs (#214 option 2).
    assert json.loads(RENOVATE.read_text()).get("osvVulnerabilityAlerts") is True
