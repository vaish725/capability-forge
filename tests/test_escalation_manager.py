"""Tests for EscalationManager: the state machine itself (no browser, no LLM - just transitions
and the HandoffRecord they produce), plus the default CLI operator console in isolation.
"""

import json

import pytest

from capability_forge.escalation.manager import (
    EscalationManager,
    EscalationTrigger,
    IllegalTransitionError,
    OperatorDecision,
    cli_operator_console,
)
from capability_forge.utils.evidence import EvidenceWriter


def trigger(**overrides):
    defaults = dict(
        reason="hard_failure",
        run_id="run_1",
        goal_description="Check the balance",
        step_id="s1",
        description="Locator did not resolve",
        screenshot_ref=None,
    )
    return EscalationTrigger(**{**defaults, **overrides})


def fake_console(decision, notes=""):
    """Returns an OperatorConsole that ignores the trigger and always returns the given decision -
    the standard way tests stand in for a human without blocking on real stdin."""
    return lambda t: OperatorDecision(decision=decision, notes=notes)


# --- the state machine itself --------------------------------------------------------------------


def test_run_handoff_resume_ends_in_agent_active():
    manager = EscalationManager(run_id="run_1", operator_console=fake_console("resume", "fixed it live"))
    record = manager.run_handoff(trigger())

    assert manager.state == "AGENT_ACTIVE"
    assert record.decision == "resume"
    assert record.notes == "fixed it live"
    assert record.run_id == "run_1"
    assert record.reason == "hard_failure"
    assert record.step_id == "s1"


def test_run_handoff_abort_ends_in_done():
    manager = EscalationManager(run_id="run_1", operator_console=fake_console("abort", "unrecoverable"))
    record = manager.run_handoff(trigger())

    assert manager.state == "DONE"
    assert record.decision == "abort"


def test_a_second_escalation_after_resume_works_normally():
    # Nothing about resolving one handoff should prevent starting another - state resets cleanly
    # back to AGENT_ACTIVE, the same starting state as a fresh manager.
    manager = EscalationManager(run_id="run_1", operator_console=fake_console("resume"))
    manager.run_handoff(trigger(step_id="s1"))
    record = manager.run_handoff(trigger(step_id="s2"))

    assert manager.state == "AGENT_ACTIVE"
    assert record.step_id == "s2"


def test_taken_at_is_recorded_at_escalate_not_at_resolve():
    manager = EscalationManager(run_id="run_1")
    manager.escalate(trigger())
    taken_at = manager._taken_at
    manager.hand_to_human()
    record = manager.resolve(OperatorDecision(decision="resume"))

    assert record.taken_at == taken_at
    assert record.released_at >= record.taken_at


# --- illegal transitions: the state machine refuses to guess ------------------------------------


def test_hand_to_human_before_escalate_raises():
    manager = EscalationManager(run_id="run_1")
    with pytest.raises(IllegalTransitionError):
        manager.hand_to_human()


def test_resolve_before_hand_to_human_raises():
    manager = EscalationManager(run_id="run_1")
    manager.escalate(trigger())
    with pytest.raises(IllegalTransitionError):
        manager.resolve(OperatorDecision(decision="resume"))


def test_escalate_twice_in_a_row_without_resolving_raises():
    manager = EscalationManager(run_id="run_1")
    manager.escalate(trigger())
    with pytest.raises(IllegalTransitionError):
        manager.escalate(trigger())


def test_resolve_after_a_run_has_already_ended_raises():
    manager = EscalationManager(run_id="run_1", operator_console=fake_console("abort"))
    manager.run_handoff(trigger())  # ends in DONE
    with pytest.raises(IllegalTransitionError):
        manager.resolve(OperatorDecision(decision="resume"))


# --- evidence integration -------------------------------------------------------------------------


def test_handoff_is_written_to_handoffs_jsonl_when_a_writer_is_supplied(tmp_path):
    writer = EvidenceWriter(run_id="test_escalation_run", mode="replay", root=tmp_path)
    manager = EscalationManager(run_id="test_escalation_run", operator_console=fake_console("resume", "clicked past the interstitial"), evidence_writer=writer)

    manager.run_handoff(trigger(reason="dead_end_detected", step_id=None, description="Stuck on the same page state"))

    lines = [json.loads(line) for line in (writer.run_dir / "handoffs.jsonl").read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["reason"] == "dead_end_detected"
    assert lines[0]["decision"] == "resume"
    assert lines[0]["notes"] == "clicked past the interstitial"
    assert lines[0]["run_id"] == "test_escalation_run"


def test_no_evidence_writer_means_no_disk_write():
    # Mirrors AgentLoop/ReplayEngine's own convention: evidence_writer is optional, and omitting
    # it must not crash the handoff, just skip persisting it.
    manager = EscalationManager(run_id="run_1", operator_console=fake_console("resume"))
    record = manager.run_handoff(trigger())
    assert record.decision == "resume"  # completed normally, nothing written anywhere


# --- the default CLI operator console -------------------------------------------------------------


def test_cli_operator_console_returns_the_typed_decision(monkeypatch, capsys):
    responses = iter(["resume", "fixed the stale locator manually"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    decision = cli_operator_console(trigger(description="Step s1 could not resolve its locator"))

    assert decision.decision == "resume"
    assert decision.notes == "fixed the stale locator manually"
    printed = capsys.readouterr().out
    assert "hard_failure" in printed
    assert "Step s1 could not resolve its locator" in printed


def test_cli_operator_console_reprompts_on_an_invalid_decision(monkeypatch):
    responses = iter(["maybe", "resume", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    decision = cli_operator_console(trigger())

    assert decision.decision == "resume"
