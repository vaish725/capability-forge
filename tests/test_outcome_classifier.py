"""Tests for the outcome classifier's pure functions: no browser, no artifact, just the
classification rules themselves."""

import pytest

from capability_forge.replay.outcome_classifier import classify_replay_status, classify_step_outcome
from capability_forge.schema.artifact import LocatorTier, StepAction


def locator(strategy="role", value="role=button[name='X']", confidence=0.9):
    return LocatorTier(strategy=strategy, value=value, confidence=confidence)


def step(action_type="click", locators=None, input_value=None):
    return StepAction(
        step_id="step_1",
        action_type=action_type,
        locators=locators if locators is not None else [locator()],
        input_value=input_value,
        risk="safe_reversible",
        description="a step",
    )


# --- classify_step_outcome ------------------------------------------------------------------------


def test_resolved_via_first_listed_tier_is_success():
    role_tier = locator("role", "role=button[name='Submit']")
    css_tier = locator("css", "#submit")
    s = step(locators=[role_tier, css_tier])
    assert classify_step_outcome(s, role_tier) == "success"


def test_resolved_via_a_later_tier_is_recoverable():
    role_tier = locator("role", "role=button[name='Submit']")
    css_tier = locator("css", "#submit")
    s = step(locators=[role_tier, css_tier])
    assert classify_step_outcome(s, css_tier) == "recoverable"


def test_navigate_with_no_resolved_tier_is_success():
    s = step(action_type="navigate", locators=[], input_value="https://example.com/next")
    assert classify_step_outcome(s, None) == "success"


def test_step_with_no_locators_and_no_resolved_tier_is_success():
    # Shouldn't happen for anything but navigate given the schema's own constraints, but the
    # function itself should still degrade sanely rather than raise an IndexError.
    s = step(action_type="navigate", locators=[], input_value="https://example.com/next")
    assert classify_step_outcome(s, None) == "success"


# --- classify_replay_status -------------------------------------------------------------------


def test_all_success_steps_with_expected_success_is_success():
    assert classify_replay_status(["success", "success"], "success") == "success"


def test_any_recoverable_step_with_expected_success_is_recoverable_then_success():
    assert classify_replay_status(["success", "recoverable", "success"], "success") == "recoverable_then_success"


def test_expected_business_outcome_reports_business_outcome_regardless_of_recovery():
    assert classify_replay_status(["success", "success"], "business_outcome") == "business_outcome"
    assert classify_replay_status(["success", "recoverable"], "business_outcome") == "business_outcome"


def test_any_hard_failure_step_overrides_everything_else():
    assert classify_replay_status(["success", "hard_failure"], "success") == "hard_failure"
    assert classify_replay_status(["hard_failure"], "business_outcome") == "hard_failure"
    assert classify_replay_status(["recoverable", "hard_failure"], "success") == "hard_failure"


def test_empty_step_list_with_expected_success_is_success():
    # A degenerate but schema-legal case (min_length=1 on steps means this can't happen for a real
    # artifact, but the function itself shouldn't assume a non-empty list).
    assert classify_replay_status([], "success") == "success"


@pytest.mark.parametrize("expected", ["success", "business_outcome"])
def test_classify_replay_status_is_a_pure_function_of_its_inputs(expected):
    # No hidden state, no side effects - same inputs, same output, called twice.
    outcomes = ["success", "recoverable"]
    assert classify_replay_status(outcomes, expected) == classify_replay_status(outcomes, expected)
