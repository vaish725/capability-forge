"""Runs the multi-run stability check against a saved artifact and writes the result back to
disk, populating (or refreshing) its `reliability` field with real, freshly-measured data.

    python -m scripts.run_stability_check --artifact artifacts/fixture_check_account_balance.json \\
        --params '{"member_id": "12345"}' [--params '{"member_id": "67890"}' ...] [--sample-size 5] [--headless]

--params is repeatable: each input set is replayed --sample-size times, and the saved reliability
records how many distinct input sets the pass rate covers.

This is real replay, N times against N fresh pages - not a simulation - so the artifact's own
target needs to actually be reachable (a locally served fixture, or a live site) exactly as it
would for a normal `python -m capability_forge.replay` invocation.
"""

import argparse
import json
import sys

from playwright.sync_api import sync_playwright

from capability_forge.guardrails.policy import Guardrail
from capability_forge.replay.reliability import DEFAULT_SAMPLE_SIZE, check_stability
from capability_forge.schema.artifact import CapabilityArtifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the multi-run stability check against a saved artifact and save the result.")
    parser.add_argument("--artifact", required=True, help="Path to a saved CapabilityArtifact JSON file - overwritten in place with the result.")
    parser.add_argument("--params", action="append", default=None, help="JSON object of input params the artifact declares. Repeat for several input sets.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help=f"How many independent runs to average over (default {DEFAULT_SAMPLE_SIZE}).")
    parser.add_argument("--headless", action="store_true", help="Run the browser headless.")
    args = parser.parse_args(argv)

    artifact = CapabilityArtifact.load(args.artifact)
    param_sets = [json.loads(raw) for raw in (args.params or ["{}"])]
    guardrail = Guardrail.from_yaml()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        result = check_stability(
            artifact,
            page_factory=lambda: browser.new_page(),
            guardrail=guardrail,
            param_sets=param_sets,
            sample_size=args.sample_size,
        )
        browser.close()

    result.artifact.save(args.artifact)

    reliability = result.artifact.reliability
    print(f"Saved to: {args.artifact}")
    print(f"pass_rate: {reliability.pass_rate}")
    print(f"avg_duration_ms: {reliability.avg_duration_ms:.1f}")
    print(f"sample_size: {reliability.sample_size} ({reliability.distinct_param_sets} input set(s) x {args.sample_size})")
    for i, run in enumerate(result.runs, start=1):
        # The input set's number, not its values: params can be credentials (ParaBank's are).
        print(f"  run {i} (input set {(i - 1) // args.sample_size + 1}): {run.status} {run.observed_state or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
