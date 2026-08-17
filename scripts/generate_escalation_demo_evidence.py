"""Generates one real evidence bundle demonstrating human-in-the-loop escalation end to end:
a run that trips the discovery loop's dead-end guard, pauses for a human, and completes the goal
after the human resumes it.

    python -m scripts.generate_escalation_demo_evidence

What's real here, and what's scripted, stated explicitly (the same distinction this project's own
test suite already draws, and the same one README/REPORT.md are expected to call out): the browser,
the Playwright driver, the guardrail, the AgentLoop's dead-end guard, the EscalationManager's full
state machine, and the EvidenceWriter writing real files to evidence/<run_id>/ are all the actual
production code, unmodified. Two things are scripted, both clearly labeled in the printed output:
  - The LLM's decisions (a ScriptedClient stands in for a real Anthropic call, exactly like this
    project's own test suite already does for testing loop mechanics) - reproducing "get stuck
    repeatedly submitting an empty login form" on demand is not something a real Claude call could
    be relied on to do consistently, and this script's purpose is to demonstrate the escalation
    mechanism itself, not the model's own decision-making (which the discovery CLI's real demo path
    already exercises).
  - The operator's decision (a scripted "resume" stands in for a human typing at the CLI prompt) -
    for the same reproducibility reason. cli_operator_console (the real default console) is not
    used here for exactly one call to keep this script non-interactive and runnable in CI/on
    demand; the state machine, HandoffRecord, and evidence writing it drives are untouched by that
    substitution.

The scripted sequence: click Login three times with both fields left empty (the same validation-
error state each time, since the fixture's own handleLogin() re-renders the identical alert) -
which trips the dead-end guard on the third repeat (repeated_state_limit's default). The operator
is asked, resumes, and the script's next three scripted actions actually fill in the login form and
complete it for real, reaching the checkpoint - not a no-op resume, but a genuine "the run picks up
and finishes the goal after a human unblocked it."
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from playwright.sync_api import sync_playwright

from capability_forge.discovery.agent_loop import AgentLoop
from capability_forge.escalation.manager import EscalationManager, OperatorDecision
from capability_forge.guardrails.policy import Guardrail
from capability_forge.surfaces.playwright_driver import PlaywrightDriver
from capability_forge.utils.evidence import EvidenceWriter, new_run_id
from capability_forge.utils.local_server import serve_directory

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _tool_use(name, tool_input, block_id):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


class _ScriptedMessage:
    def __init__(self, content):
        self.content = content


class _ScriptedClient:
    """The same scripted-response technique test_agent_loop.py already uses to test loop
    mechanics without a live LLM call - see the module docstring for why that substitution is
    appropriate here too."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        if not self._responses:
            raise AssertionError("Scripted response sequence ran out - the script's own scenario needs another turn than it planned for.")
        return self._responses.pop(0)


def _scripted_operator_console(trigger):
    """Stands in for cli_operator_console for this one non-interactive run - see the module
    docstring. A real run would block here on an actual human typing at the terminal."""
    print("\n--- ESCALATION (scripted operator, not a live human, for reproducibility) ---")
    print(f"Reason: {trigger.reason}")
    print(f"Context: {trigger.description}")
    print("Decision: resume (scripted)")
    return OperatorDecision(decision="resume", notes="Scripted operator: dismissed the repeated validation error and let the agent continue.")


def main() -> int:
    guardrail = Guardrail.from_yaml()
    run_id = new_run_id("discovery")
    evidence_writer = EvidenceWriter(run_id=run_id, mode="discovery", sensitive_fields=set(guardrail.policy.sensitive_fields))
    escalation_manager = EscalationManager(run_id=run_id, operator_console=_scripted_operator_console, evidence_writer=evidence_writer)

    responses = [
        _ScriptedMessage([_tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "Attempt to submit the login form."}, "t1")]),
        _ScriptedMessage([_tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "Retry submitting the login form."}, "t2")]),
        _ScriptedMessage([_tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "Retry submitting the login form again."}, "t3")]),
        # By this point the dead-end guard has fired, an escalation ran, and the operator resumed.
        # The scripted "model" now takes the actions that were missing all along.
        _ScriptedMessage([_tool_use("type", {"role": "textbox", "name": "User Name:", "value": "jdoe", "risk": "safe_reversible", "reasoning": "Enter the username now that a human confirmed the form was actually stuck, not broken."}, "t4")]),
        _ScriptedMessage([_tool_use("type", {"role": "textbox", "name": "Password:", "value": "demo-fixture-password", "risk": "safe_reversible", "reasoning": "Enter the password."}, "t5")]),
        _ScriptedMessage([_tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "Submit the now-completed login form."}, "t6")]),
        _ScriptedMessage(
            [
                _tool_use(
                    "done",
                    {
                        "outcome_type": "success",
                        "checkpoint_role": "textbox",
                        "checkpoint_name": "Member ID:",
                        "summary": "Reached the accounts screen after a human resumed a login attempt that was stuck submitting an empty form.",
                    },
                    "t7",
                )
            ]
        ),
    ]
    client = _ScriptedClient(responses)

    with serve_directory(FIXTURES_DIR) as base_url, sync_playwright() as p:
        target = f"{base_url}/hostile_legacy_page.html"
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        driver = PlaywrightDriver(page)
        agent = AgentLoop(
            driver,
            guardrail,
            client=client,
            repeated_state_limit=3,
            evidence_writer=evidence_writer,
            escalation_manager=escalation_manager,
        )
        result = agent.run("Log in and look up a member's account", target)
        browser.close()

    print(f"\nEvidence written to: {evidence_writer.run_dir}")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Steps recorded: {len(result.steps)}")
    handoffs_path = evidence_writer.run_dir / "handoffs.jsonl"
    if handoffs_path.exists():
        print(f"Handoffs recorded: {len(handoffs_path.read_text().splitlines())}")
    return 0 if result.stop_reason in ("goal_complete", "business_outcome") else 1


if __name__ == "__main__":
    sys.exit(main())
