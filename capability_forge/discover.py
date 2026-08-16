"""Discovery mode CLI entry point.

    python -m capability_forge.discover --goal "..." --target "https://..."

Wires together a launched Playwright browser, the guardrail policy loaded from
config/allowlist.yaml, and AgentLoop into a single runnable command - this is the actual code
behind the demo path documented in README.md. Requires ANTHROPIC_API_KEY (discovery mode is the
one part of this project that isn't offline-runnable, unlike replay - see README for why).
"""

import argparse
import os
import sys

import anthropic
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from capability_forge.discovery.agent_loop import AgentLoop
from capability_forge.guardrails.policy import Guardrail
from capability_forge.surfaces.playwright_driver import PlaywrightDriver


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
    return parser


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
    args = build_arg_parser().parse_args(argv)

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

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page()
        driver = PlaywrightDriver(page)
        agent = AgentLoop(driver, guardrail, **loop_kwargs)
        try:
            result = agent.run(args.goal, args.target)
        except anthropic.AnthropicError as exc:
            # A bad key, rate limit, or network issue talking to the API - a clean message beats
            # a raw SDK traceback for something this likely to happen at the command line.
            print(f"Anthropic API error: {exc}", file=sys.stderr)
            return 1
        finally:
            browser.close()

    print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
