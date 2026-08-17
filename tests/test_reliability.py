"""Tests for the multi-run stability check: aggregation math against controlled fake drivers (so
pass_rate/sample_size are exact and deterministic), plus one real end-to-end run against the
fixture over real browser pages proving the actual wiring (fresh PlaywrightDriver per page, real
ReplayEngine, real CapabilityArtifact.model_copy) works, not just the arithmetic in isolation.
"""

from pathlib import Path

import pytest

from capability_forge.guardrails.policy import AllowlistPolicy, Guardrail
from capability_forge.replay.engine import ParamValidationError
from capability_forge.replay.reliability import DEFAULT_SAMPLE_SIZE, check_stability
from capability_forge.schema.artifact import CapabilityArtifact, Checkpoint, InputParam, LocatorTier, StepAction, TargetSpec
from capability_forge.surfaces.playwright_driver import CheckpointNotReachedError

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "hostile_legacy_page.html"


def locator(strategy="role", value="", confidence=0.9):
    return LocatorTier(strategy=strategy, value=value, confidence=confidence)


def step(step_id, action_type, locators, input_value=None, description="a step", risk="safe_reversible"):
    return StepAction(step_id=step_id, action_type=action_type, locators=locators, input_value=input_value, risk=risk, description=description)


def artifact(target_base_url, steps, checkpoint, inputs=None):
    return CapabilityArtifact(
        artifact_id="test_artifact",
        schema_version="1.0",
        target=TargetSpec(base_url=target_base_url, app_name="legacy_bank"),
        goal_description="Test artifact",
        inputs=inputs or [],
        steps=steps,
        checkpoint=checkpoint,
        risk_summary="safe",
        expected_outcome_type="success",
    )


@pytest.fixture
def guardrail():
    policy = AllowlistPolicy(
        allowed_domains=["127.0.0.1"],
        allowed_action_types={"*": ["click", "type", "navigate", "wait", "select"]},
        risky_keywords=[],
        sensitive_fields=[],
    )
    return Guardrail(policy)


@pytest.fixture
def fixture_server():
    from capability_forge.utils.local_server import serve_directory

    with serve_directory(FIXTURE_PATH.parent) as base_url:
        yield base_url


class _FakePage:
    """Stands in for a Playwright Page in tests that fake the driver entirely - needs goto()
    (ReplayEngine.run()'s own implicit initial navigation calls self.driver.page.goto directly)
    and close(), since check_stability calls close() unconditionally after every run."""

    def __init__(self):
        self.closed = False

    def goto(self, url):
        pass

    def close(self):
        self.closed = True


# --- aggregation math, against a controlled fake driver ------------------------------------------


def _minimal_artifact():
    checkpoint = Checkpoint(description="x", locator=locator(value="role=heading"), extract=None)
    return artifact("http://127.0.0.1:9999", [step("s1", "wait", [locator(value="role=heading")])], checkpoint)


def test_check_stability_computes_pass_rate_from_mixed_outcomes(monkeypatch, guardrail):
    call_count = {"n": 0}

    class _AlternatingDriver:
        """Every other constructed instance fails at the checkpoint - lets pass_rate be pinned
        down exactly, rather than depending on real page flakiness to produce a mixed result."""

        def __init__(self, page):
            self.page = page
            call_count["n"] += 1
            self._fail = call_count["n"] % 2 == 0

        def act(self, step, params=None):
            return None

        def verify_checkpoint(self, checkpoint):
            if self._fail:
                raise CheckpointNotReachedError("simulated failure")
            return {}

    monkeypatch.setattr("capability_forge.replay.reliability.PlaywrightDriver", _AlternatingDriver)

    result = check_stability(_minimal_artifact(), page_factory=_FakePage, guardrail=guardrail, sample_size=4)

    assert result.artifact.reliability.pass_rate == 0.5
    assert result.artifact.reliability.sample_size == 4
    assert len(result.runs) == 4
    assert [r.status for r in result.runs] == ["success", "hard_failure", "success", "hard_failure"]


def test_check_stability_reports_full_pass_rate_when_every_run_succeeds(monkeypatch, guardrail):
    class _AlwaysSucceedsDriver:
        def __init__(self, page):
            self.page = page

        def act(self, step, params=None):
            return None

        def verify_checkpoint(self, checkpoint):
            return {}

    monkeypatch.setattr("capability_forge.replay.reliability.PlaywrightDriver", _AlwaysSucceedsDriver)

    result = check_stability(_minimal_artifact(), page_factory=_FakePage, guardrail=guardrail, sample_size=3)

    assert result.artifact.reliability.pass_rate == 1.0


def test_returned_artifact_is_a_new_object_not_a_mutation(monkeypatch, guardrail):
    class _AlwaysSucceedsDriver:
        def __init__(self, page):
            self.page = page

        def act(self, step, params=None):
            return None

        def verify_checkpoint(self, checkpoint):
            return {}

    monkeypatch.setattr("capability_forge.replay.reliability.PlaywrightDriver", _AlwaysSucceedsDriver)

    original = _minimal_artifact()
    result = check_stability(original, page_factory=_FakePage, guardrail=guardrail, sample_size=1)

    assert original.reliability is None  # the input artifact was never touched
    assert result.artifact is not original
    assert result.artifact.reliability is not None
    # Everything else about the artifact is unchanged - only reliability was added.
    assert result.artifact.model_dump(exclude={"reliability"}) == original.model_dump(exclude={"reliability"})


def test_pages_are_closed_after_each_run(monkeypatch, guardrail):
    class _AlwaysSucceedsDriver:
        def __init__(self, page):
            self.page = page

        def act(self, step, params=None):
            return None

        def verify_checkpoint(self, checkpoint):
            return {}

    monkeypatch.setattr("capability_forge.replay.reliability.PlaywrightDriver", _AlwaysSucceedsDriver)

    created_pages = []

    def factory():
        p = _FakePage()
        created_pages.append(p)
        return p

    check_stability(_minimal_artifact(), page_factory=factory, guardrail=guardrail, sample_size=3)

    assert len(created_pages) == 3
    assert all(p.closed for p in created_pages)


def test_default_sample_size_is_five():
    assert DEFAULT_SAMPLE_SIZE == 5


def test_param_validation_error_propagates_after_the_first_attempt_only(guardrail):
    checkpoint = Checkpoint(description="x", locator=locator(value="role=heading"), extract=None)
    art = artifact(
        "http://127.0.0.1:9999", [step("s1", "wait", [locator(value="role=heading")])], checkpoint,
        inputs=[InputParam(name="required_param", type="string", required=True, description="x")],
    )
    attempts = {"n": 0}

    def factory():
        attempts["n"] += 1
        return _FakePage()

    with pytest.raises(ParamValidationError, match="missing required"):
        check_stability(art, page_factory=factory, guardrail=guardrail, params={}, sample_size=5)

    assert attempts["n"] == 1  # never repeated the same guaranteed-to-fail check 5 times


# --- real end-to-end run, over real browser pages -------------------------------------------------


def test_check_stability_against_the_real_fixture_uses_a_fresh_driver_per_page(browser, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    steps = [
        step("s1", "type", [locator(value='role=textbox[name="User Name:"]')], "jdoe", "Enter username"),
        step("s2", "type", [locator(value='role=textbox[name="Password:"]')], "secret", "Enter password"),
        step("s3", "click", [locator(value='role=button[name="Login"]')], None, "Submit login"),
    ]
    checkpoint = Checkpoint(description="x", locator=locator(value='role=textbox[name="Member ID:"]'), extract=None)
    art = artifact(target, steps, checkpoint)

    created_pages = []

    def factory():
        p = browser.new_page()
        created_pages.append(p)
        return p

    result = check_stability(art, page_factory=factory, guardrail=guardrail, sample_size=2)

    assert result.artifact.reliability.pass_rate == 1.0
    assert result.artifact.reliability.sample_size == 2
    assert result.artifact.reliability.avg_duration_ms > 0
    assert len(result.runs) == 2
    assert all(r.status == "success" for r in result.runs)
    # Two genuinely distinct pages were created and are now both closed - not one page reused.
    assert len(created_pages) == 2
    assert created_pages[0] is not created_pages[1]
    assert all(p.is_closed() for p in created_pages)
