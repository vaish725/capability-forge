# capability-forge

An LLM discovers how to complete a task inside a legacy, API-less UI once, by driving a real
browser (observe, decide, act). That successful run becomes a typed, versioned, deterministically
replayable **capability artifact** that an AI agent can invoke afterward without any further LLM
involvement. When the system can't safely proceed on its own, it pauses, hands the live session to
a human, and resumes once they're done.

Guardrails (allowlist, risk classification, redaction) apply to every action in both discovery and
replay.

See `REPORT.md` for the schema rationale and safety write-up.

## Architecture

One process, two modes sharing one artifact contract and one core:

```
                       CLI / thin API
     DISCOVERY (LLM in the loop)      REPLAY (no LLM, deterministic)
     goal + target URL                artifact + params
     observe -> decide -> act         step executor (locator tiers, checkpoint)
     artifact recorder                outcome classifier
     -> /artifacts/*.json             -> /evidence/ (log, screenshots)
              \_____________ Escalation Manager (shared) _____________/
                     \___ Guardrails: allowlist + risk + redaction ___/
```

## Status

Feature-complete against the assignment's six core requirements, plus both priority stretch goals.

- **Discovery** - an LLM drives a real browser (Playwright) through an observe-decide-act loop,
  wrapped in the same guardrail policy (allowlist, risk classification, redaction) that wraps
  replay, with a dead-end guard against infinite loops.
- **Capability artifacts** - a discovery run is recorded into a typed, versioned, parameterized
  `CapabilityArtifact` (`artifact_recorder.py`), with per-run evidence (log, screenshots, redacted
  transcript) written to `evidence/<run_id>/`.
- **Replay** - executes a saved artifact deterministically, no LLM involved, classifying the
  outcome as `success` / `business_outcome` / `recoverable_then_success` / `hard_failure`.
- **Escalation** (`escalation/manager.py`) - a state machine
  (`AGENT_ACTIVE -> PAUSED_FOR_HUMAN -> HUMAN_ACTIVE -> RESUMING -> AGENT_ACTIVE | DONE`) hands the
  live browser session to a human via a CLI operator console when discovery's dead-end guard
  trips, a replay step or checkpoint hits a hard failure, or a risky/irreversible replay step
  awaits confirmation - and records what the human decided either way.
- **Both CLI entry points are real**: `python -m capability_forge.discover` and
  `python -m capability_forge.replay` (see the demo path below).
- **Both priority stretch goals are built**: multi-run stability (`replay/reliability.py`, replays
  an artifact N times against N independent fresh pages and writes a real `ReliabilityInfo` back
  onto it) and an agent-facing capability API (`api/main.py` - `GET /capabilities` /
  `POST /capabilities/{id}/invoke`).

Every one of the above has a real, checked-in example backing it - see Evidence below, not just
this list. Discovery has been run live against both the bundled fixture and ParaBank's public demo
banking site (`artifacts/parabank_check_account_balance.json`). Not attempted: two lower-priority
stretch goals (confidence/approval gating, cross-tenant canonicalization) - see `REPORT.md`'s Cuts
section for why they were deprioritized.

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

To see the hard_failure path for real, replay the same artifact with `--params '{"member_id": "00000"}'`
instead - `00000` is the fixture's own deterministic trigger for a simulated backend error
(`SYS-500`), so the checkpoint never resolves and the run reports `hard_failure` with the failed
step, expected/observed state, and a screenshot of the actual error.

To measure an artifact's reliability for real (stretch goal) - runs it N independent times against
N fresh pages and writes the aggregate pass rate/timing back onto the artifact file:

```
python -m scripts.run_stability_check --artifact artifacts/fixture_check_account_balance.json --params '{"member_id": "12345"}'
```

`artifacts/fixture_check_account_balance.json` already carries real reliability data from a run of
this, not a placeholder - `pass_rate: 1.0` over 5 independent runs.

To run the agent-facing capability API (stretch goal) with the fixture server still up:

```
uvicorn capability_forge.api.main:app
```

`GET /capabilities` lists every recorded artifact with its typed input/output contract and
reliability data; `POST /capabilities/{artifact_id}/invoke` (body: `{"params": {...}, "confirm_risky": false}`)
runs a real replay and returns the `ReplayResult` - a hard_failure is returned as a normal 200
response (a well-formed result, not a broken request), an unknown artifact_id 404s, and a param
mismatch 400s.

## Evidence

Every claim above has a real, checked-in example backing it, not just a description:

- **Discovery success against the live target (ParaBank)** - `evidence/discovery_1786935840/`
  (log, screenshots, redacted transcript). This is the assignment's required live discovery run,
  and also the fresh, clean run recorded after the credential-leak fix described in `REPORT.md`'s
  Safety section - it's the artifact proving that fix actually works, not just the writeup's word
  for it.
- **Replay success** - `evidence/replay_1786951099/`, produced by the exact replay command above,
  against the bundled fixture.
- **Replay hard_failure** - `evidence/replay_1786951120/`, produced by the exact `member_id=00000`
  command above - `screenshots/step_06.png` shows the actual SYS-500 error state at the point of
  failure.
- **Escalation firing end to end** (a run that gets stuck, pauses, a human resumes it, and it goes
  on to complete the goal) - `evidence/discovery_1786949371/`, against the bundled fixture,
  including a `handoffs.jsonl` showing the recorded decision. See
  `scripts/generate_escalation_demo_evidence.py` for exactly what's real versus scripted about
  that one.

## Folder structure

- `capability_forge/discovery/` - the observe-decide-act loop that learns a task and records it.
- `capability_forge/replay/` - deterministic execution of a recorded artifact (no LLM involved),
  plus the multi-run stability check that populates `ReliabilityInfo`.
- `capability_forge/schema/` - the typed capability artifact contract (Pydantic models).
- `capability_forge/guardrails/` - allowlist, risk classification, and redaction policy.
- `capability_forge/escalation/` - human-in-the-loop pause/handoff/resume state machine.
- `capability_forge/surfaces/` - the driver that observes/acts on a concrete UI (Playwright today).
- `capability_forge/api/` - thin FastAPI layer exposing recorded capabilities to an agent.
- `capability_forge/utils/` - shared helpers, including log/artifact redaction.
- `config/` - guardrail policy (`allowlist.yaml`).
- `artifacts/` - recorded capability artifacts (JSON).
- `evidence/` - per-run logs, screenshots, transcripts, and handoff records.
- `fixtures/` - a static local page used for offline, no-network replay testing.
- `scripts/` - standalone scripts for generating evidence examples and refreshing reliability data.
- `tests/` - the project's test suite.

## License

MIT, see `LICENSE`.
