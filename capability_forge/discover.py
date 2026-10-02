"""Discovery mode CLI entry point.

    python -m capability_forge.discover --goal "..." --target "https://..."

Wires together a launched Playwright browser, the guardrail policy loaded from
config/allowlist.yaml, and AgentLoop into a single runnable command - this is the actual code
behind the demo path documented in README.md. Requires ANTHROPIC_API_KEY (discovery mode is the
one part of this project that isn't offline-runnable, unlike replay - see README for why).

Escalation is on by default here too (matching replay/__main__.py), using the real
cli_operator_console: if the dead-end guard trips, a human at the terminal is asked before the run
just ends. Without this, the top-level pitch this whole project makes ("when the system can't
safely proceed, it pauses and hands off to a human") would only ever be true inside a test, never
for a real invocation of the actual CLI - found and fixed while building replay's own CLI
alongside this one, not something this file shipped with originally. --no-escalation opts out for
a scripted/CI context with no human available to answer a prompt.

With --artifact-id, a successful run is recorded and saved as artifacts/<artifact-id>.json - the
step that turns discovery's output into something replay can use. --param NAME=VALUE (repeatable)
says which literal values in the goal are inputs, e.g. --param member_id=12345; those become
{{member_id}} in the saved artifact. This path did not exist before: artifact_recorder was only
called from tests and a one-off session, so the README's "a discovery run is recorded into an
artifact" was true of the module but not of any command a reader could run.
"""

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import anthropic
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from capability_forge.discovery.agent_loop import AgentLoop
from capability_forge.discovery.artifact_recorder import UnrecordableRunError, record_artifact
from capability_forge.escalation.manager import EscalationManager
from capability_forge.guardrails.policy import Guardrail
from capability_forge.surfaces.playwright_driver import PlaywrightDriver
from capability_forge.utils.evidence import EvidenceWriter, new_run_id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a discovery loop against a target URL.")
    parser.add_argument("--goal", required=True, help="Natural-language goal for the agent to accomplish.")
    parser.add_argument("--target", required=True, help="Target URL to start the run from.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser headless. Default is headed, so a discovery run is visible while it happens.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override the loop's default max_steps.")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Override the loop's default timeout.")
    parser.add_argument(
        "--no-escalation",
        action="store_true",
        help="Disable human-in-the-loop escalation. A dead-end guard trip ends the run immediately instead of pausing for an operator.",
    )
    parser.add_argument(
        "--artifact-id",
        default=None,
        help="Record a successful run as artifacts/<ARTIFACT_ID>.json (lowercase snake_case). Without it the run is only printed.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Declare that VALUE (as it appears in the goal) is the input NAME. Repeatable.",
    )
    parser.add_argument("--app-name", default=None, help="Target app name recorded in the artifact. Default: the target's hostname.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Where --artifact-id saves to. Default: artifacts/.")
    return parser


def parse_param_map(raw_params: list[str]) -> dict[str, str]:
    """--param NAME=VALUE flags -> the recorder's param_map (literal value -> param name)."""
    param_map: dict[str, str] = {}
    for raw in raw_params:
        name, sep, value = raw.partition("=")
        if not sep or not name or not value:
            # Names only, never the raw flag: a param value can be a password, and an error
            # message is printed straight to the terminal.
            problem = "no '='" if not sep else "an empty name" if not name else f"an empty value for {name!r}"
            raise ValueError(f"--param must look like NAME=VALUE, got one with {problem}")
        param_map[value] = name
    return param_map


def print_result(result) -> None:
    print(f"Stop reason: {result.stop_reason}")
    if result.business_outcome_reason:
        print(f"Business outcome: {result.business_outcome_reason}")
    if result.summary:
        print(f"Summary: {result.summary}")
    print(f"Steps recorded: {len(result.steps)}")
    for step in result.steps:
        print(f"  - [{step.risk}] {step.action_type}: {step.description}")
    if result.checkpoint:
        print(f"Checkpoint: {result.checkpoint.description} (locator: {result.checkpoint.locator.value})")
    if result.extract_log:
        print(f"Values read via extract: {len(result.extract_log)}")
        for entry in result.extract_log:
            print(f"  - {entry['role']} {entry['name']!r}: {entry['value']!r}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        param_map = parse_param_map(args.param)
    except ValueError as exc:
        parser.error(str(exc))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Discovery mode requires a real Anthropic API key - "
            "set it in .env or the environment. Replay mode works fully offline without one.",
            file=sys.stderr,
        )
        return 1

    guardrail = Guardrail.from_yaml()

    loop_kwargs = {}
    if args.max_steps is not None:
        loop_kwargs["max_steps"] = args.max_steps
    if args.timeout_seconds is not None:
        loop_kwargs["timeout_seconds"] = args.timeout_seconds

    run_id = new_run_id("discovery")
    evidence_writer = EvidenceWriter(run_id=run_id, mode="discovery", sensitive_fields=set(guardrail.policy.sensitive_fields))
    escalation_manager = None if args.no_escalation else EscalationManager(run_id=run_id, evidence_writer=evidence_writer)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page()
        driver = PlaywrightDriver(page)
        agent = AgentLoop(driver, guardrail, evidence_writer=evidence_writer, escalation_manager=escalation_manager, **loop_kwargs)
        try:
            result = agent.run(args.goal, args.target)
        except anthropic.AnthropicError as exc:
            # A bad key, rate limit, or network issue talking to the API - a clean message beats
            # a raw SDK traceback for something this likely to happen at the command line.
            print(f"Anthropic API error: {exc}", file=sys.stderr)
            return 1
        finally:
            browser.close()

    print(f"Evidence written to: {evidence_writer.run_dir}")
    print_result(result)
    if args.artifact_id is None:
        return 0
    return save_artifact(result, args, param_map)


def save_artifact(result, args, param_map: dict[str, str]) -> int:
    """Record the run and save it; print what was recorded, or why it couldn't be."""
    try:
        artifact = record_artifact(
            result,
            artifact_id=args.artifact_id,
            app_name=args.app_name or urlparse(args.target).hostname or "app",
            param_map=param_map,
        )
    except (UnrecordableRunError, ValueError) as exc:
        print(f"Not recorded: {exc}", file=sys.stderr)
        return 1
    path = Path(args.artifacts_dir) / f"{artifact.artifact_id}.json"
    replaced = path.exists()
    artifact.save(path)
    print(f"Artifact {'replaced' if replaced else 'saved'}: {path}")
    print(f"  inputs: {[param.name for param in artifact.inputs]}")
    print(f"  outputs: {[f'{o.name} ({o.format})' for o in artifact.outputs]}")
    print(f"  checkpoint: {artifact.checkpoint.locator.value}")
    print(f"  identity: {artifact.checkpoint.identity or 'none (the final page shows no input value)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
