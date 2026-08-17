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
`CapabilityArtifact`), the replay engine (executes a saved artifact deterministically, no LLM
involved, classifies the outcome as success, business_outcome, recoverable_then_success, or
hard_failure, and writes the same evidence bundle discovery does when given an evidence writer),
and human-in-the-loop escalation (`escalation/manager.py`): a state machine
(`AGENT_ACTIVE -> PAUSED_FOR_HUMAN -> HUMAN_ACTIVE -> RESUMING -> AGENT_ACTIVE | DONE`) that hands
the live browser session to a human via a mock CLI operator console when discovery's dead-end guard
trips, when a replay step or checkpoint hits a hard failure, or before a risky/irreversible replay
step runs unconfirmed - and records what the human decided (`handoffs.jsonl`) either way. The
discovery loop has been run live against both the bundled fixture and ParaBank's public demo
banking site; a real recorded example is at `artifacts/parabank_check_account_balance.json`. A real
evidence bundle showing escalation firing end to end (a run that gets stuck, pauses, a human
resumes it, and it completes the goal) is at `evidence/discovery_1786949371/` - see
`scripts/generate_escalation_demo_evidence.py` for exactly what's real versus scripted about that
run. Both discovery and replay have a real CLI entry point now (`python -m capability_forge.discover`
/ `python -m capability_forge.replay`, see the demo path below) - the only remaining piece is
`REPORT.md`.

## Setup

- Python 3.11+
- `pip install -r requirements.txt`
- `playwright install chromium`
- Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (required for discovery mode only -
  replay mode runs fully offline, no key needed)

## Demo path

The guardrail allowlist checks a real domain, and `file://` URLs have none - so the bundled
fixture is served locally rather than opened directly. In one terminal:

```
python -m http.server 8000 --directory fixtures
```

In another, to run discovery:

```
python -m capability_forge.discover --goal "Look up the balance for member 12345" --target "http://127.0.0.1:8000/hostile_legacy_page.html"
```

Prints the stop reason, every step the agent took (with its risk classification), and the
verified checkpoint once the run completes. `--headless` runs without a visible browser window;
omit it to watch the run happen. `--max-steps` and `--timeout-seconds` override the loop's
defaults (25 steps, 180 seconds) if needed. `--no-escalation` disables the human-in-the-loop pause
(on by default) for a scripted/CI context with no operator available to answer a prompt. No live
network dependency other than the Anthropic API call itself.

Or to replay a saved artifact against the same fixture, fully offline (no API key needed) - this
one's `target.base_url` points at the local fixture server started above, so it's the same
"Look up the balance" capability as the discovery command, just replayed deterministically instead
of re-discovered:

```
python -m capability_forge.replay --artifact artifacts/fixture_check_account_balance.json --params '{"member_id": "12345"}'
```

Prints the run's status (`success`, `business_outcome`, `recoverable_then_success`, or
`hard_failure`), any extracted outputs, and a per-step outcome breakdown. `--confirm-risky`
authorizes any risky_irreversible step in the artifact to run without a separate confirmation
prompt; `--no-escalation` and `--no-evidence` behave the same way they do for discovery. Exits
non-zero on `hard_failure`, so it's usable as a scripted health check. `artifacts/parabank_check_account_balance.json`
is a second real example recorded against ParaBank's live demo site instead of the bundled
fixture - replaying it needs live network access to parabank.parasoft.com, so it isn't part of this
fully-offline demo path.

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
- `evidence/` - per-run logs, screenshots, transcripts, and handoff records.
- `fixtures/` - a static local page used for offline, no-network replay testing.
- `scripts/` - standalone scripts for generating specific evidence examples on demand.
- `tests/` - the project's test suite.

## License

MIT, see `LICENSE`.
