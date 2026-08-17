"""Replay mode CLI entry point.

    python -m capability_forge.replay --artifact artifacts/parabank_check_account_balance.json \\
        --params '{"member_id": "12345"}'

Wires together a launched Playwright browser, the guardrail policy loaded from
config/allowlist.yaml, and ReplayEngine into a single runnable command - the replay-mode
counterpart to discover.py, living as this package's __main__.py rather than a sibling module
named replay.py, since capability_forge.replay is already a package (engine.py,
outcome_classifier.py) and Python doesn't allow a module and a package of the same name side by
side. Runs fully offline: no ANTHROPIC_API_KEY needed anywhere in this path, since ReplayEngine
itself involves no LLM call (see replay/engine.py's own module docstring).

Escalation is on by default here, using the real cli_operator_console - a human at the terminal is
asked before a risky_irreversible step runs unconfirmed, and again if a step or the checkpoint
hits a hard failure, matching how the design's own top-level pitch ("when the system can't safely
proceed, it pauses and hands off to a human") is meant to actually work for a real invocation, not
just inside a test. --no-escalation opts out for a scripted/CI context with no human available to
answer a prompt, in which case a hard_failure or an unconfirmed risky step ends the run immediately
instead of pausing - the exact same fail-closed behavior ReplayEngine already has with no
escalation_manager supplied at all.
"""

import argparse
import json
import sys

from playwright.sync_api import sync_playwright

from capability_forge.escalation.manager import EscalationManager
from capability_forge.guardrails.policy import Guardrail
from capability_forge.replay.engine import ParamValidationError, ReplayEngine, ReplayResult
from capability_forge.schema.artifact import CapabilityArtifact
from capability_forge.surfaces.playwright_driver import PlaywrightDriver
from capability_forge.utils.evidence import EvidenceWriter, new_run_id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a saved capability artifact deterministically. No LLM involved.")
    parser.add_argument("--artifact", required=True, help="Path to a saved CapabilityArtifact JSON file.")
    parser.add_argument(
        "--params",
        default="{}",
        help='JSON object of input params the artifact declares, e.g. \'{"member_id": "12345"}\'. Defaults to no params.',
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser headless. Default is headed, so a replay run is visible while it happens.",
    )
    parser.add_argument(
        "--confirm-risky",
        action="store_true",
        help="Authorize every risky_irreversible step in this artifact to run without per-step confirmation.",
    )
    parser.add_argument(
        "--no-escalation",
        action="store_true",
        help="Disable human-in-the-loop escalation. A hard failure or an unconfirmed risky step ends the run immediately instead of pausing for an operator.",
    )
    parser.add_argument("--no-evidence", action="store_true", help="Skip writing an evidence bundle for this run.")
    return parser


def print_result(result: ReplayResult) -> None:
    print(f"Status: {result.status}")
    if result.status == "hard_failure":
        print(f"Failed step: {result.failed_step_id}")
        print(f"Expected: {result.expected_state}")
        print(f"Observed: {result.observed_state}")
        return
    if result.business_outcome_reason:
        print(f"Business outcome: {result.business_outcome_reason}")
    if result.outputs:
        print("Outputs:")
        for name, value in result.outputs.items():
            print(f"  - {name}: {value!r}")
    print(f"Steps executed: {len(result.steps)}")
    for step in result.steps:
        tier_note = f" (via {step.locator_tier_used} tier)" if step.locator_tier_used else ""
        print(f"  - [{step.outcome_type}] {step.step_id}{tier_note}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        artifact = CapabilityArtifact.load(args.artifact)
    except Exception as exc:  # noqa: BLE001 - a bad path or a schema-invalid file should be one clean message, not a raw traceback
        print(f"Could not load artifact {args.artifact!r}: {exc}", file=sys.stderr)
        return 1

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"--params is not valid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(params, dict):
        print('--params must be a JSON object, e.g. \'{"member_id": "12345"}\'', file=sys.stderr)
        return 1

    guardrail = Guardrail.from_yaml()

    evidence_writer = None
    if not args.no_evidence:
        evidence_writer = EvidenceWriter(run_id=new_run_id("replay"), mode="replay", sensitive_fields=set(guardrail.policy.sensitive_fields))

    escalation_manager = None
    if not args.no_escalation:
        # Reuses the evidence writer's own run_id when one exists, so a HandoffRecord and its
        # run's log.jsonl/screenshots always agree on which run they belong to; falls back to a
        # fresh id only when running with --no-evidence, where there's no writer to borrow one from.
        run_id = evidence_writer.run_id if evidence_writer is not None else new_run_id("replay")
        escalation_manager = EscalationManager(run_id=run_id, evidence_writer=evidence_writer)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page()
        driver = PlaywrightDriver(page)
        engine = ReplayEngine(driver, guardrail, evidence_writer=evidence_writer, escalation_manager=escalation_manager)
        try:
            result = engine.run(artifact, params=params, confirm_risky=args.confirm_risky)
        except ParamValidationError as exc:
            print(f"Param validation failed: {exc}", file=sys.stderr)
            return 1
        finally:
            browser.close()

    if evidence_writer is not None:
        print(f"Evidence written to: {evidence_writer.run_dir}")
    print_result(result)
    return 0 if result.status != "hard_failure" else 1


if __name__ == "__main__":
    sys.exit(main())
