# capability-forge

An LLM discovers how to complete a task inside a legacy, API-less UI once, by driving a real
browser (observe, decide, act). That successful run becomes a typed, versioned, deterministically
replayable **capability artifact** that an AI agent can invoke afterward without any further LLM
involvement. When the system can't safely proceed on its own, it pauses, hands the live session to
a human, and resumes once they're done.

Guardrails (allowlist, risk classification, redaction) apply to every action in both discovery and
replay.

## Status

In progress. Built and tested so far: the capability artifact schema, the guardrail policy
(allowlist, risk classification, redaction), the Playwright surface driver, the discovery loop,
per-run evidence capture (step-by-step log, screenshots, redacted transcript, written to
`evidence/<run_id>/`), the artifact recorder (turns a discovery run into a versioned, parameterized
`CapabilityArtifact`), and the replay engine (executes a saved artifact deterministically, no LLM
involved, and classifies the outcome as success, business_outcome, recoverable_then_success, or
hard_failure). The discovery loop has been run live against both the bundled fixture and
ParaBank's public demo banking site; a real recorded example is at
`artifacts/parabank_check_account_balance.json`. Not yet built: human-in-the-loop escalation and a
replay CLI entry point. The demo path below is real and runnable today for discovery mode; a replay
command will be added here once that CLI exists.

## Setup

- Python 3.11+
- `pip install -r requirements.txt`
- `playwright install chromium`
- Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (required for discovery mode only -
  replay mode is designed to run fully offline, once it exists)

## Demo path

The guardrail allowlist checks a real domain, and `file://` URLs have none - so the bundled
fixture is served locally rather than opened directly. In one terminal:

```
python -m http.server 8000 --directory fixtures
```

In another:

```
python -m capability_forge.discover --goal "Look up the balance for member 12345" --target "http://127.0.0.1:8000/hostile_legacy_page.html"
```

Prints the stop reason, every step the agent took (with its risk classification), and the
verified checkpoint once the run completes. `--headless` runs without a visible browser window;
omit it to watch the run happen. `--max-steps` and `--timeout-seconds` override the loop's
defaults (25 steps, 180 seconds) if needed. No live network dependency other than the Anthropic
API call itself.

A replay command (`python -m capability_forge.replay ...`) will be added here once the replay
engine is implemented.

## Folder structure

- `capability_forge/discovery/` - the observe-decide-act loop that learns a task and records it.
- `capability_forge/replay/` - deterministic execution of a recorded artifact, no LLM involved.
- `capability_forge/schema/` - the typed capability artifact contract (Pydantic models).
- `capability_forge/guardrails/` - allowlist, risk classification, and redaction policy.
- `capability_forge/escalation/` - human-in-the-loop pause/handoff/resume state machine.
- `capability_forge/surfaces/` - the driver that observes/acts on a concrete UI (Playwright today).
- `capability_forge/api/` - optional thin API exposing recorded capabilities to an agent.
- `capability_forge/utils/` - shared helpers, including log/artifact redaction.
- `config/` - guardrail policy (`allowlist.yaml`).
- `artifacts/` - recorded capability artifacts (JSON).
- `evidence/` - per-run logs, screenshots, and transcripts.
- `fixtures/` - a static local page used for offline, no-network replay testing.
- `tests/` - unit tests for the classifier and guardrails, plus one replay integration test.

## License

MIT, see `LICENSE`.
