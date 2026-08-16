# capability-forge

An LLM discovers how to complete a task inside a legacy, API-less UI once, by driving a real
browser (observe, decide, act). That successful run becomes a typed, versioned, deterministically
replayable **capability artifact** that an AI agent can invoke afterward without any further LLM
involvement. When the system can't safely proceed on its own, it pauses, hands the live session to
a human, and resumes once they're done.

Guardrails (allowlist, risk classification, redaction) apply to every action in both discovery and
replay.

## Status

Repository scaffolding only. Discovery loop, replay engine, and guardrails are not implemented
yet. Setup instructions and a copy-pasteable demo path will be added here once the core loop is
working end to end.

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
