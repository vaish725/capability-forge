"""Tests for ReplayEngine against the real hostile-legacy fixture over a real browser and driver -
no LLM involved anywhere, matching what replay actually is. Artifacts are constructed directly
(not recorded via discovery) so each test can target one specific replay behavior precisely:
plain success, a locator falling back to a later tier (recoverable_then_success), a business
outcome, and each of the ways a run can hard_failure.
"""

import json
from pathlib import Path

import pytest

from capability_forge.guardrails.policy import AllowlistPolicy, Guardrail
from capability_forge.replay.engine import ParamValidationError, ReplayEngine
from capability_forge.schema.artifact import CapabilityArtifact, Checkpoint, InputParam, LocatorTier, OutputField, StepAction, TargetSpec
from capability_forge.surfaces.playwright_driver import PlaywrightDriver
from capability_forge.utils.evidence import EvidenceWriter

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "hostile_legacy_page.html"
FIXTURE_URL = f"file://{FIXTURE_PATH}"


def locator(strategy="role", value="", confidence=0.9):
    return LocatorTier(strategy=strategy, value=value, confidence=confidence)


def step(step_id, action_type, locators, input_value=None, description="a step", risk="safe_reversible"):
    return StepAction(step_id=step_id, action_type=action_type, locators=locators, input_value=input_value, risk=risk, description=description)


LOGIN_STEPS = [
    step("s1", "type", [locator(value='role=textbox[name="User Name:"]')], "jdoe", "Enter username"),
    step("s2", "type", [locator(value='role=textbox[name="Password:"]')], "secret", "Enter password"),
    step("s3", "click", [locator(value='role=button[name="Login"]')], None, "Submit login"),
]


def artifact(target_base_url, steps, checkpoint, expected_outcome_type="success", business_outcome_reason=None, inputs=None, outputs=None, checkpoint_extract=None):
    return CapabilityArtifact(
        artifact_id="test_artifact",
        schema_version="1.0",
        target=TargetSpec(base_url=target_base_url, app_name="legacy_bank"),
        goal_description="Test artifact",
        inputs=inputs or [],
        outputs=outputs or [],
        steps=steps,
        checkpoint=checkpoint,
        risk_summary="safe",
        expected_outcome_type=expected_outcome_type,
        business_outcome_reason=business_outcome_reason,
    )


@pytest.fixture
def guardrail():
    policy = AllowlistPolicy(
        allowed_domains=["127.0.0.1"],
        allowed_action_types={"*": ["click", "type", "navigate", "wait", "select"]},
        risky_keywords=["confirm", "submit", "delete", "transfer"],
        sensitive_fields=["account_number"],
    )
    return Guardrail(policy)


@pytest.fixture
def driver(page):
    return PlaywrightDriver(page, locator_timeout_ms=1500)


@pytest.fixture
def fixture_server():
    from capability_forge.utils.local_server import serve_directory

    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    with serve_directory(fixtures_dir) as base_url:
        yield base_url


def engine(driver, guardrail):
    return ReplayEngine(driver, guardrail)


# --- plain success ---------------------------------------------------------------------------


def test_replay_reaches_checkpoint_and_returns_success(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = LOGIN_STEPS + [
        step("s4", "type", [locator(value='role=textbox[name="Member ID:"]')], "12345", "Enter member id"),
        step("s5", "click", [locator(value='role=button[name="Search"]')], None, "Search"),
    ]
    checkpoint = Checkpoint(description="Balance shown", locator=locator(value='role=cell[name="$4500.00"]'), extract=None)
    art = artifact(target, steps, checkpoint)

    result = engine(driver, guardrail).run(art)

    assert result.status == "success"
    assert result.artifact_id == "test_artifact"
    assert len(result.steps) == 5
    assert all(s.outcome_type == "success" for s in result.steps)
    assert result.failed_step_id is None


def test_replay_extracts_declared_outputs(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = LOGIN_STEPS + [
        step("s4", "type", [locator(value='role=textbox[name="Member ID:"]')], "12345", "Enter member id"),
        step("s5", "click", [locator(value='role=button[name="Search"]')], None, "Search"),
    ]
    checkpoint = Checkpoint(
        description="Balance shown",
        locator=locator(value='role=cell[name="$4500.00"]'),
        extract={"balance": "css=.balance-value"},
    )
    art = artifact(target, steps, checkpoint, outputs=[OutputField(name="balance", type="string", description="x")])

    result = engine(driver, guardrail).run(art)

    assert result.status == "success"
    assert result.outputs == {"balance": "$4500.00"}


# --- templated params -------------------------------------------------------------------------


def test_replay_substitutes_declared_params_into_steps(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = LOGIN_STEPS + [
        step("s4", "type", [locator(value='role=textbox[name="Member ID:"]')], "{{member_id}}", "Enter member id"),
        step("s5", "click", [locator(value='role=button[name="Search"]')], None, "Search"),
    ]
    checkpoint = Checkpoint(description="Balance shown", locator=locator(value='role=cell[name="$250.75"]'), extract=None)
    art = artifact(
        target, steps, checkpoint,
        inputs=[InputParam(name="member_id", type="string", required=True, description="x")],
    )

    result = engine(driver, guardrail).run(art, params={"member_id": "67890"})

    assert result.status == "success"


def test_missing_required_param_rejected_before_touching_the_browser(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    checkpoint = Checkpoint(description="x", locator=locator(value='role=heading[name="First Fidelity Member Services - Internal Portal"]'), extract=None)
    art = artifact(
        target, LOGIN_STEPS, checkpoint,
        inputs=[InputParam(name="member_id", type="string", required=True, description="x")],
    )
    with pytest.raises(ParamValidationError, match="missing required"):
        engine(driver, guardrail).run(art, params={})
    # Confirm the browser was never touched - still on about:blank or wherever it started.
    assert "hostile_legacy_page" not in driver.page.url


def test_undeclared_param_rejected(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    checkpoint = Checkpoint(description="x", locator=locator(value='role=heading[name="First Fidelity Member Services - Internal Portal"]'), extract=None)
    art = artifact(target, LOGIN_STEPS, checkpoint)
    with pytest.raises(ParamValidationError, match="undeclared"):
        engine(driver, guardrail).run(art, params={"typo_param": "x"})


@pytest.mark.parametrize(
    "declared_type,value,expected_substring",
    [
        ("int", 42, "42"),
        ("int", "42", "42"),
        ("float", 4.5, "4.5"),
        ("bool", True, "True"),
        ("bool", "true", "True"),
    ],
)
def test_param_type_coercion_produces_the_right_string(driver, guardrail, fixture_server, declared_type, value, expected_substring):
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = [step("s1", "type", [locator(value='role=textbox[name="User Name:"]')], "{{val}}", "Enter value")]
    checkpoint = Checkpoint(description="x", locator=locator(value='role=heading[name="First Fidelity Member Services - Internal Portal"]'), extract=None)
    art = artifact(target, steps, checkpoint, inputs=[InputParam(name="val", type=declared_type, required=True, description="x")])

    result = engine(driver, guardrail).run(art, params={"val": value})

    assert result.status == "success"
    assert driver.page.locator("#txtUserName").input_value() == expected_substring


def test_bool_param_rejects_a_non_boolean_string(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    checkpoint = Checkpoint(description="x", locator=locator(value='role=heading[name="First Fidelity Member Services - Internal Portal"]'), extract=None)
    art = artifact(target, LOGIN_STEPS, checkpoint, inputs=[InputParam(name="flag", type="bool", required=True, description="x")])
    with pytest.raises(ParamValidationError, match="expects bool"):
        engine(driver, guardrail).run(art, params={"flag": "maybe"})


# --- recoverable_then_success: a step falls back to a later locator tier ------------------------


def test_step_falling_back_to_css_tier_reports_recoverable_then_success(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = [
        step(
            "s3", "click",
            [locator(value='role=button[name="Not The Real Name"]'), locator(strategy="css", value="#btnLogin")],
            None, "Click login (first tier deliberately stale)",
        ),
    ]
    checkpoint = Checkpoint(description="x", locator=locator(value='role=textbox[name="Member ID:"]'), extract=None)
    steps = LOGIN_STEPS[:2] + steps  # fill credentials, then click via the stale-then-fallback step
    art = artifact(target, steps, checkpoint)

    result = engine(driver, guardrail).run(art)

    assert result.status == "recoverable_then_success"
    assert result.steps[-1].outcome_type == "recoverable"
    assert result.steps[-1].locator_tier_used == "css"


# --- business_outcome ---------------------------------------------------------------------------


def test_replay_of_a_business_outcome_artifact_reports_business_outcome(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = LOGIN_STEPS + [
        step("s4", "type", [locator(value='role=textbox[name="Member ID:"]')], "00001", "Enter member id"),
        step("s5", "click", [locator(value='role=button[name="Search"]')], None, "Search"),
    ]
    checkpoint = Checkpoint(description="Member not found", locator=locator(value='role=alert[name="No member found matching that ID."]'), extract=None)
    art = artifact(target, steps, checkpoint, expected_outcome_type="business_outcome", business_outcome_reason="member_not_found")

    result = engine(driver, guardrail).run(art)

    assert result.status == "business_outcome"
    assert result.business_outcome_reason == "member_not_found"


# --- hard_failure: three distinct ways a run can fail -------------------------------------------


def test_hard_failure_when_a_step_locator_never_resolves(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = [step("s1", "click", [locator(value='role=button[name="Does Not Exist"]')], None, "Click nonexistent button")]
    checkpoint = Checkpoint(description="x", locator=locator(value='role=heading[name="First Fidelity Member Services - Internal Portal"]'), extract=None)
    art = artifact(target, steps, checkpoint)

    result = engine(driver, guardrail).run(art)

    assert result.status == "hard_failure"
    assert result.failed_step_id == "s1"
    assert result.observed_state is not None
    assert result.outputs == {}


def test_hard_failure_when_checkpoint_never_resolves(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    checkpoint = Checkpoint(description="x", locator=locator(value='role=cell[name="$999999.99"]'), extract=None)
    art = artifact(target, LOGIN_STEPS, checkpoint)

    result = engine(driver, guardrail).run(art)

    assert result.status == "hard_failure"
    assert result.failed_step_id == "checkpoint"


def test_hard_failure_when_guardrail_blocks_an_action_type(driver, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    restrictive_policy = AllowlistPolicy(
        allowed_domains=["127.0.0.1"],
        allowed_action_types={"*": ["click"]},  # "type" is not allowed
        risky_keywords=[],
        sensitive_fields=[],
    )
    restrictive_guardrail = Guardrail(restrictive_policy)
    checkpoint = Checkpoint(description="x", locator=locator(value='role=heading[name="First Fidelity Member Services - Internal Portal"]'), extract=None)
    art = artifact(target, LOGIN_STEPS, checkpoint)

    result = engine(driver, restrictive_guardrail).run(art)

    assert result.status == "hard_failure"
    assert result.failed_step_id == "s1"
    assert "blocked by policy" in result.observed_state


# --- evidence logging --------------------------------------------------------------------------
# An EvidenceWriter is entirely optional (all tests above pass none); these tests cover what gets
# written to disk when one is supplied, using the exact same log.jsonl schema as discovery.


def test_evidence_writer_logs_one_line_per_step_plus_a_terminal_summary(driver, guardrail, fixture_server, tmp_path):
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = LOGIN_STEPS + [
        step("s4", "type", [locator(value='role=textbox[name="Member ID:"]')], "12345", "Enter member id"),
        step("s5", "click", [locator(value='role=button[name="Search"]')], None, "Search"),
    ]
    checkpoint = Checkpoint(description="Balance shown", locator=locator(value='role=cell[name="$4500.00"]'), extract=None)
    art = artifact(target, steps, checkpoint)
    writer = EvidenceWriter(run_id="test_replay_success", mode="replay", root=tmp_path)

    result = ReplayEngine(driver, guardrail, evidence_writer=writer).run(art)

    assert result.status == "success"
    log_lines = [json.loads(line) for line in (writer.run_dir / "log.jsonl").read_text().splitlines()]
    assert len(log_lines) == len(steps) + 1  # one line per step, plus the terminal summary line
    assert [line["step_id"] for line in log_lines[:-1]] == [s.step_id for s in steps]
    assert log_lines[-1]["step_id"] == "terminal"
    assert log_lines[-1]["outcome_type"] == "success"
    assert all(line["mode"] == "replay" for line in log_lines)
    # One screenshot per step, plus the terminal one.
    assert len(list((writer.run_dir / "screenshots").glob("*.png"))) == len(steps) + 1


def test_evidence_writer_captures_a_hard_failure_with_debuggable_state(driver, guardrail, fixture_server, tmp_path):
    # The evidence bundle's whole point per the design: prove a run that hits a real error is
    # detected and reported, not silently swallowed. Same locator-never-resolves failure as
    # test_hard_failure_when_a_step_locator_never_resolves above, but here asserting on what
    # actually landed on disk rather than just the returned ReplayResult.
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = [step("s1", "click", [locator(value='role=button[name="Does Not Exist"]')], None, "Click nonexistent button")]
    checkpoint = Checkpoint(description="x", locator=locator(value='role=heading[name="First Fidelity Member Services - Internal Portal"]'), extract=None)
    art = artifact(target, steps, checkpoint)
    writer = EvidenceWriter(run_id="test_replay_hard_failure", mode="replay", root=tmp_path)

    result = ReplayEngine(driver, guardrail, evidence_writer=writer).run(art)

    assert result.status == "hard_failure"
    log_lines = [json.loads(line) for line in (writer.run_dir / "log.jsonl").read_text().splitlines()]
    assert len(log_lines) == 1  # only the failed step - the run never reached a terminal line
    failure_line = log_lines[0]
    assert failure_line["step_id"] == "s1"
    assert failure_line["outcome_type"] == "hard_failure"
    assert failure_line["expected_state"] == "Click nonexistent button"
    assert failure_line["observed_state"]  # populated, not empty - enough to debug without re-running
    assert failure_line["evidence_ref"] is not None
    assert (writer.run_dir / failure_line["evidence_ref"]).exists()  # the screenshot it points to is real


def test_evidence_writer_registers_typed_values_as_secrets_before_dispatch(driver, guardrail, fixture_server, tmp_path):
    # Extends the credential-leak defense (see utils/evidence.py's module docstring) to replay: a
    # step's rendered input_value is registered for scrubbing before the step is dispatched,
    # exactly as discovery already does for its own typed/selected values - not because today's
    # replay log lines carry enough raw page text to leak it yet (_log_step never writes a
    # literal input_value or raw DOM text, unlike discovery's observed_state), but so the
    # discipline holds automatically if that ever changes, rather than depending on someone
    # remembering to re-add it later.
    target = f"{fixture_server}/hostile_legacy_page.html"
    checkpoint = Checkpoint(description="x", locator=locator(value='role=heading[name="First Fidelity Member Services - Internal Portal"]'), extract=None)
    art = artifact(target, LOGIN_STEPS, checkpoint)
    writer = EvidenceWriter(run_id="test_replay_secret_registration", mode="replay", root=tmp_path)

    ReplayEngine(driver, guardrail, evidence_writer=writer).run(art)

    assert "jdoe" in writer._secret_values
    assert "secret" in writer._secret_values
