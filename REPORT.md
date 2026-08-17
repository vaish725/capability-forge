# REPORT

An LLM discovers how to complete a task inside a legacy, API-less UI once, by driving a real
browser (observe, decide, act). That run becomes a typed, versioned, deterministically replayable
capability artifact an AI agent can invoke afterward with no further LLM involvement. When the
system can't safely proceed, it pauses, hands the live session to a human, and resumes once
they're done. Guardrails (allowlist, risk classification, redaction) wrap every action in both
modes.

Every claim below points at real, checked-in evidence (`/evidence/`, `/artifacts/`) or a specific
test, not just a description of intended behavior - see `README.md`'s own Evidence section for the
direct paths.

## Architecture

One process, two modes sharing one artifact contract and one core:

```
                     CLI / thin API
        DISCOVERY MODE            REPLAY MODE
     (LLM in the loop)        (no LLM, deterministic)
   goal + target URL           artifact + params
        |                            |
        v                            v
   observe -> decide -> act     step executor (locator
   (Claude + Playwright)         tiers, checkpoint)
        |                            |
        v                            v
   artifact recorder            outcome classifier
        |                            |
        v                            v
   /artifacts/*.json          /evidence/ (log, screenshots)
        |____________________________|
                     |
             Escalation Manager (shared by both modes)
                     |
     Guardrail Layer (allowlist + risk + redaction),
          wraps every action in both modes
```

Key decisions:

| Decision | Choice | Trade-off accepted |
|---|---|---|
| Process model | Single process, sync execution | Won't scale to concurrent runs out of the box - design-only extension (Heterogeneity section) |
| Perception | Accessibility tree + role locators primary; CSS fallback; coordinate last resort | More engineering than raw CSS selectors, but survives markup churn the way a human/screen-reader would |
| Session boundary | The Playwright browser context is the unit handed to a human | Requires non-headless (or a live debug endpoint) for any run that might escalate |
| LLM in replay | Never | Core requirement; the entire deterministic-replay claim rests on this |
| Discovery vs. replay | Two entry points, one Surface Driver / Guardrail / Escalation core | More upfront abstraction than two independent scripts |

The critical seam (Surface Driver knows *how* to act on a concrete UI; Capability Artifact
describes *what* to do, surface-agnostically, via strategy-tagged locators; Replay Engine
orchestrates driver + artifact + guardrail + escalation) is what let two entire features -
human-in-the-loop escalation and per-run evidence logging - slot into the existing discovery loop
and replay engine as optional constructor arguments, with no changes to the seam itself. That's a
stronger claim than "the design allows for it": it's what actually happened when those two
features were built.

The `PlaywrightDriver` implementation surfaced real findings the design alone didn't predict: a
locator tier only counts as resolved if it matches exactly one element (an ambiguous match forces
fallback, not a guess - proven by a fixture whose `.btn` class is deliberately reused across every
button); `observe()` must descend into every iframe explicitly, or the model is blind to an entire
flow living inside one; and `role=alert[name=...]` needed a second, looser fallback because ARIA's
accessible-name computation doesn't cover live regions the way it covers buttons and headings.

## Artifact schema

```python
class LocatorTier(BaseModel):
    strategy: Literal["role", "css", "coordinate"]
    value: str
    confidence: float

class StepAction(BaseModel):
    step_id: str
    action_type: Literal["click", "type", "navigate", "wait", "select"]
    locators: list[LocatorTier]       # tried in order; empty only for navigate
    input_value: str | None           # may reference {{param}}
    risk: Literal["safe_reversible", "risky_irreversible"]
    description: str

class Checkpoint(BaseModel):
    description: str
    locator: LocatorTier
    extract: dict[str, str] | None    # named outputs, keys must match declared OutputFields exactly

class CapabilityArtifact(BaseModel):
    artifact_id: str
    schema_version: str
    target: TargetSpec                # base_url, app_name, tenant_overrides
    inputs: list[InputParam]
    outputs: list[OutputField]
    steps: list[StepAction]
    checkpoint: Checkpoint
    risk_summary: Literal["safe", "contains_risky_steps"]
    expected_outcome_type: Literal["success", "business_outcome"]
    reliability: ReliabilityInfo | None
```

Locators are a list, not a single value, so replay can fall back through tiers - the concrete
mechanism behind "self-healing" locator resolution. `input_value` templates on `{{param}}` so one
artifact serves many invocations without re-recording. `risk` lives on the step, not just the
artifact, because a single artifact can mix safe reads and one risky write, and the guardrail
needs per-step granularity. `checkpoint.extract` is where declared outputs are actually read -
"how do I know I succeeded" and "what do I return" are kept as one verification step rather than
two that could drift apart, enforced by a cross-field validator at load time. `expected_outcome_type`
was added mid-build, not designed up front: a `business_outcome` checkpoint (a named, expected
non-error result) resolves mechanically identically to a `success` checkpoint at replay time, so
there was no way to tell them apart without the artifact self-declaring which one it is.

`reliability` is no longer a described-but-empty field. `artifacts/fixture_check_account_balance.json`
carries real measured data - `pass_rate: 1.0`, `sample_size: 5` - from an actual run of the
multi-run stability check (`replay/reliability.py`), which replays an artifact N times against N
independently created and closed pages, not one page reused, because the isolation needs to match
what a real invocation gets for the number to mean anything.

## Determinism & error handling

Per-step outcome: `success | recoverable | hard_failure`. Per-run: `success | business_outcome |
recoverable_then_success | hard_failure`. `recoverable_then_success` is replay's only form of
automatic recovery without an LLM - a step that needed a later locator tier than the one it was
recorded with, but still resolved.

"Recoverable" is worth being precise about, because it names three genuinely different mechanisms
across this codebase, unified only by convenience: discovery's interstitial/retry handling
(prose-level guidance to the model), discovery's own evidence log (a dispatched-but-unrecorded
attempt, retried by the LLM), and replay's locator-tier fallback. Replay only ever produces the
third, because it has no LLM to improvise the other two - stated explicitly in
`outcome_classifier.py`'s own docstring rather than left to imply replay can detect interstitials,
which it structurally cannot.

Real, checked-in evidence for the taxonomy: `evidence/replay_1786951099/` (plain success),
`evidence/replay_1786951120/` (`hard_failure`, via the fixture's own deterministic `SYS-500`
trigger - `screenshots/step_06.png` shows the actual error state, not a synthetic one). The
locator-fallback tier itself is not just a design idea: ParaBank's real login form has two
`<input>` fields with no accessible name at all (no `<label for>` pairing), which role+name alone
cannot disambiguate - the fixture never exposed this, since it was built by the same person who
built the driver and was biased toward clean labels. Fixed by adding an optional positional `nth`
tier (rated at lower confidence, 0.75 vs 0.9, since it's positional rather than semantic), verified
against the real target, not just the fixture.

## Heterogeneity & multi-tenant

Design-only, per the assignment's own scope note - but grounded in two real surfaces now, not
speculation: one live target (ParaBank) and one deliberately hostile designed target (the bundled
fixture: reused CSS classes, an iframe-only account flow, an alert whose accessible name isn't
computed the standard way). The extension story rests on two things already load-bearing in the
schema, not hypothetical: locator strategy is tagged per-tier (`role | css | coordinate`), so a
future `uia_id` tier for a desktop driver slots in without a schema change; and `TargetSpec`
separates routing (`base_url`, `tenant_overrides`) from the flow definition (`steps`), so a base
artifact plus a small per-tenant override map could cover multiple tenants of the same vendor
product without a full re-recording. Neither `tenant_overrides` application nor a second driver
implementation is built - naming the gap plainly rather than implying more than the schema alone
provides.

## Escalation & handoff

State machine, `escalation/manager.py`: `AGENT_ACTIVE -> PAUSED_FOR_HUMAN -> HUMAN_ACTIVE ->
RESUMING -> (AGENT_ACTIVE | DONE)`. `EscalationManager` is the single enforced source of truth for
who controls the session - every transition checks state first and raises rather than guessing on
an out-of-order call, which is what "avoiding split-brain in the handoff logic" means concretely.

Three trigger conditions, one per calling module: discovery's dead-end guard (`agent_loop.py`), a
replay step or checkpoint hard failure, and a `risky_irreversible` replay step awaiting
confirmation. The fail-closed default is real, not aspirational: a risky step with no
`confirm_risky=True` and no `EscalationManager` never executes unconfirmed. A hard-failure "resume"
retries the identical action exactly once, never a loop - if it's still broken after a human had a
chance to intervene, that's a genuine failure, not something worth asking again.

What's mocked versus real, stated plainly: the operator console is a CLI prompt
(`cli_operator_console`), not a live co-browsing UI - the assignment's own scope note permits this.
The mechanism it drives is real: the live browser session (non-headless), the state machine, and
the `HandoffRecord` written to `handoffs.jsonl`. Real, generated evidence at
`evidence/discovery_1786949371/`: a run that trips the dead-end guard three times, pauses, is
resumed, and goes on to actually complete the goal - not a pause-then-give-up. Only the LLM's and
the operator's decisions are scripted there, for reproducibility; everything else in that run is
the same production code path a live invocation uses.

## Safety

Three layers wrap every action in both modes: an allowlist (domain + action type, checked before
every `act()` call), risk classification (the model self-tags `safe_reversible` /
`risky_irreversible`; a keyword heuristic can only escalate that tag, never downgrade it), and
redaction (field-name matching plus a second, value-based layer - `register_secret()` - added
after a real incident, described below).

Known limitations, stated plainly rather than implied to be fully covered:

- **Risk classification trusts the model's self-tag unless a keyword overrides it.** The
  escalate-only direction is a real guarantee (nothing talks a genuinely risky action back down to
  safe), not a complete one - an adversarial prompt, or more likely a risky action phrased outside
  the configured keyword list, silently stays at whatever the model self-tagged.
- **Redaction has two named gaps.** Field-name matching is exact (`acct_num` isn't caught by
  name); neither layer parses a string value that itself embeds structured data - confirmed a
  non-issue for what this schema currently produces, re-examine if that changes.
- **`register_secret()` only catches a value the system already knew about in advance.** It closed
  a real leak (a typed value resurfacing unprompted in a later page observation - see below), but
  it cannot recognize a brand-new secret an operator introduces somewhere it never saw first,
  concretely a password typed fresh into a `HandoffRecord.notes` field describing a live
  intervention. No code fix closes this for a truly novel secret; mitigated by an explicit warning
  in the console's own prompt, not a scrubbing mechanism.
- **Replay's hard-failure retry reissues the identical driver call blind, no separate
  re-observation step.** Provably fine for role/css tiers (Playwright re-resolves them fresh
  against the live DOM every call); not for the coordinate tier, whose value is a fixed pixel
  position - accepted because coordinate is the documented last resort and discovery's own
  screenshot-free design never actually produces one.
- **Escalation is not wired into the capability API's `invoke()`.** An HTTP request has no human
  at a terminal to block on - structurally the wrong mechanism for a programmatic caller, not just
  unbuilt. A risky step needs `confirm_risky: true` up front; a hard_failure returns as one.

A real incident, disclosed rather than omitted: a ParaBank throwaway password was committed and
pushed to this public repo. Root cause: Playwright's accessibility snapshot exposes an `<input>`'s
current value as part of its own description, even for `type="password"` fields - a leak path
neither the screenshot masking nor stripping the tool call's literal value caught, since the value
resurfaced unprompted in a later raw page observation instead. Remediated: credentials rotated,
history rewritten and force-pushed, `register_secret()` built as the generalized second redaction
layer (any registered literal scrubbed from every subsequent write, regardless of field), and
verified by iterating every blob in the rewritten history plus a fresh clone from the remote. The
honest limit on what that force-push guarantees: it scrubs this repository, not every cache or
fork made in the window before the fix, nor does it guarantee every cached view on GitHub's side is
purged (GitHub does support a request process for that, which exists precisely for this scenario).

## Cuts

- **Playwright's CDP remote-debugging protocol, in favor of the browser's existing non-headless
  window.** A deliberate trade-off, not an oversight: CDP is the credible answer for handing a
  session to a remote operator, but this project's actual demo only ever needs a human at the same
  terminal, so building CDP exposure wasn't necessary to demonstrate the real mechanism (pause,
  hand over control, capture a decision, resume). Would build CDP exposure first if this needed to
  support a remote operator.
- **Tier 2 stretch goals, per the design's own explicit instruction to stop after Tier 1 unless
  clearly ahead of schedule.** Confidence & approval gating (a `draft`/`approved` state on
  `CapabilityArtifact`, gated on `reliability.pass_rate` clearing a threshold, `invoke()` refusing
  `draft` artifacts without `force=true`) is a natural, low-effort extension of the reliability
  work already built - would build the gate itself next, since the signal it would gate on already
  exists. Cross-tenant canonicalization (a second fixture variant, a `target.overrides` map, one
  artifact replaying successfully against both) is the most differentiating and most expensive
  Tier 2 item - correctly last in priority, not attempted.
- **The capability API has no auth, rate limiting, concurrency control across simultaneous
  `invoke()` calls, or shared/pooled browser** - each call launches and closes its own browser,
  correct and simple at demo scale, not what production traffic would need.
- **Assisted fallback during replay** (a bounded, single-step LLM assist when a locator can't
  resolve, logged as evidence) was named as a possible design extension early on and not built -
  the escalation-and-retry mechanism that was built instead covers the same "let something rescue
  a failing step" need without reintroducing an LLM into the replay path at all.
