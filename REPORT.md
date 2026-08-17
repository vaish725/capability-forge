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

One process, two modes sharing one artifact contract and one core (see `README.md` for the
component diagram): a Surface Driver that knows *how* to act on a concrete UI; a Capability
Artifact that describes *what* to do, surface-agnostically, via strategy-tagged locators; and a
Replay Engine that orchestrates driver + artifact + guardrail + escalation. Discovery and replay
are two entry points sharing that one core rather than two independent scripts - more upfront
abstraction, paid back when escalation and per-run evidence logging both slotted into the existing
loop and engine as optional constructor arguments, with no change to the seam itself. That's a
stronger claim than "the design allows for it": it's what actually happened when those two
features were built.

Two other decisions worth naming directly: perception uses the accessibility tree and role
locators as the primary strategy, CSS as fallback, and coordinate as last resort (a locator is how
a recorded step identifies which specific on-screen element to act on, not just that one exists;
"tier" means trying progressively less reliable identification methods in that order, falling back
only when an earlier one fails to resolve to exactly one element) - more engineering than raw CSS
selectors, but it survives markup churn closer to how a human or screen-reader would. And the unit
handed to a human on escalation is the Playwright browser
context itself, which requires non-headless (or a live debug endpoint) for any run that might
escalate - a real cost, accepted because it's what makes a mid-run handoff actually work rather
than just be designed for.

The `PlaywrightDriver` implementation confirmed that seam under real use rather than leaving it
theoretical:

- A locator tier only counts as resolved if it matches exactly one element - ambiguous forces
  fallback, not a guess, proven by a fixture whose `.btn` class is deliberately reused across
  every button.
- `observe()` must descend into every iframe explicitly, or the model is blind to a flow living
  inside one.
- `role=alert[name=...]` needed a looser fallback, since ARIA's accessible-name computation
  doesn't cover live regions the way it covers buttons and headings.

## Artifact schema

The three fields that carry the actual design argument (full schema:
`capability_forge/schema/artifact.py`):

```python
locators: list[LocatorTier]       # tried in order, per step; empty only for navigate
risk: Literal["safe_reversible", "risky_irreversible"]   # lives on the step, not the artifact
checkpoint.extract: dict[str, str] | None                # named outputs read here, and only here
```

Locators are a list, not a single value, so replay can fall back through tiers - the concrete
mechanism behind "self-healing" locator resolution. `input_value` templates on `{{param}}` so one
artifact serves many invocations without re-recording. `risk` lives on the step, not just the
artifact, since a single artifact can mix safe reads and one risky write, and the guardrail needs
per-step granularity. `checkpoint.extract` is where declared outputs are actually read - "how do I
know I succeeded" and "what do I return" are kept as one verification step rather than two that
could drift apart, enforced by a cross-field validator at load time. `expected_outcome_type` was
added mid-build: a `business_outcome` checkpoint (a named, expected non-error result) resolves
identically to a `success` checkpoint at replay time, so there was no way to tell them apart
without the artifact self-declaring which one it is.

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

"Recoverable" names three genuinely different mechanisms, unified only by convenience:

- Discovery's handling of unexpected pop-up screens mid-flow (e.g. a session-timeout warning) -
  prose-level guidance to the model.
- Discovery's own evidence log - a dispatched-but-unrecorded attempt, retried by the LLM.
- Replay's locator-tier fallback.

Replay only ever produces the third, since it has no LLM to improvise the other two - stated
explicitly in `outcome_classifier.py`'s own docstring rather than left to imply replay can detect
an unexpected pop-up screen, which it structurally cannot.

Real, checked-in evidence for the taxonomy: `evidence/replay_1786951099/` (plain success),
`evidence/replay_1786951120/` (`hard_failure`, via the fixture's own deliberately unrecoverable
`SYS-500` trigger, built specifically to prove replay detects and reports failure correctly rather
than masking it - `screenshots/step_06.png` shows the actual error state).

Separately, the locator-fallback tier proved itself against a real, unplanned failure too:
ParaBank's real login form has two `<input>` fields with no accessible name at all (no
`<label for>` pairing), which role+name alone cannot disambiguate - the fixture never exposed
this, since it was built by the same person who built the driver and was biased toward clean
labels. Fixed by adding an optional positional `nth` tier (rated at lower confidence, 0.75 vs 0.9,
since it's positional rather than semantic), verified against the real target, not just the
fixture.

## Heterogeneity & multi-tenant

Design-only, per the assignment's own scope note - but grounded in two real surfaces (ParaBank,
plus the fixture's deliberately hostile markup: reused CSS classes, an iframe-only flow, a
non-standard alert). Two things already load-bearing in the schema carry the extension story:
locator strategy is tagged per-tier (`role | css | coordinate`), so a future `uia_id` tier for a
desktop driver slots in without a schema change; and `TargetSpec` separates routing (`base_url`,
`tenant_overrides`) from the flow definition (`steps`), so a base artifact plus a per-tenant
override map could cover multiple tenants of one vendor product without a full re-recording.
Neither `tenant_overrides` application nor a second driver is built - naming the gap plainly.

Drift detection would build on data already collected, not need new instrumentation: every step
already records which `locator_tier_used` it actually resolved through (`ReplayResult.steps`, and
`evidence/*/log.jsonl` when evidence is enabled), so which tier a step resolved through is never
lost per run. What's missing is aggregation of that signal *across* runs - `ReliabilityInfo`
intentionally collapses to one pass/fail number (`reliability.py`'s own docstring explains why),
discarding per-step tier detail rather than trending it. A capability whose tier-1 (role)
resolution rate degrades over successive runs - falling back to tier 2 or 3 more often - is a
concrete drift signal a monitoring layer could threshold and alert on; the raw per-run detail to
compute that trend already exists in `StabilityCheckResult.runs`, but nothing aggregates or
thresholds it today. Building that aggregation is the natural next step, not attempted now, per
the same "don't build scaling infrastructure prematurely" scope call as the rest of this section.

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

What's mocked versus real: the operator console is a CLI prompt (`cli_operator_console`), not a
live co-browsing UI - the assignment's own scope note permits this. What it drives is real: the
live browser session (non-headless), the state machine, and the `HandoffRecord` written to
`handoffs.jsonl`. Two real bundles, for the two escalation-bearing trigger conditions: at
`evidence/discovery_1786949371/`, a run that trips the dead-end guard three times, pauses, is
resumed, and goes on to actually complete the goal - not a pause-then-give-up (only the LLM's and
the operator's decisions are scripted there, for reproducibility; everything else is the same
production code path a live invocation uses). At `evidence/replay_1787003705/`, a `hard_failure`
escalation run with nothing scripted at all - a real terminal session, paused on a real `input()`
call, resumed by a real typed decision; `handoffs.jsonl` carries a genuine typo in the notes field
("clicked on search buttom") that no generator produces.

## Safety

Three layers wrap every action in both modes:

- **Allowlist** - domain + action type, checked before every `act()` call.
- **Risk classification** - the model self-tags `safe_reversible` / `risky_irreversible`; a
  keyword heuristic can only escalate that tag, never downgrade it.
- **Redaction** - field-name matching plus a second, value-based layer (`register_secret()`),
  added after a real incident, described below.

Known limitations, stated plainly rather than implied to be fully covered:

- **Risk classification trusts the model's self-tag unless a keyword overrides it.** Escalate-only
  is a real guarantee (nothing talks a genuinely risky action back down to safe), not a complete
  one - a risky action phrased outside the configured keyword list silently stays self-tagged.
- **Redaction has two named gaps.** Field-name matching is exact (`acct_num` isn't caught by
  name); neither layer parses structured data embedded inside a string value - confirmed a
  non-issue for this schema today, re-examine if that changes.
- **`register_secret()` only catches a value the system already knew about in advance.** It closed
  a real leak (a typed value resurfacing unprompted in a later page observation - see below), but
  can't recognize a brand-new secret introduced somewhere it never saw first, concretely a password
  typed fresh into `HandoffRecord.notes`. No code fix closes this for a truly novel secret;
  mitigated by an explicit warning in the console's own prompt, not a scrubbing mechanism.
- **Replay's hard-failure retry reissues the identical driver call blind, no re-observation step.**
  Fine for role/css tiers (Playwright re-resolves them fresh against the live DOM every call); not
  for coordinate, whose value is a fixed pixel position - accepted since coordinate is the
  documented last resort and discovery's screenshot-free design never actually produces one.
- **Escalation is not wired into the capability API's `invoke()`.** An HTTP request has no human at
  a terminal to block on - structurally the wrong mechanism, not just unbuilt. A risky step needs
  `confirm_risky: true` up front; a hard_failure returns as one.

A real incident, disclosed rather than omitted: a ParaBank throwaway password was committed and
pushed to this public repo. Root cause: Playwright's accessibility snapshot exposes an `<input>`'s
current value as part of its own description, even for `type="password"` fields - neither
screenshot masking nor stripping the tool call's literal value caught this, since the value
resurfaced unprompted in a later raw page observation instead. Remediated: credentials rotated,
history rewritten and force-pushed, `register_secret()` built as the generalized second redaction
layer (any registered literal scrubbed from every subsequent write, regardless of field), and
verified by iterating every blob in the rewritten history plus a fresh clone from the remote. The
honest limit on the force-push: it scrubs this repository, not every cache or fork made in the
window before the fix, nor does it guarantee every cached view on GitHub's side is purged (GitHub
does support a request process for that, which exists precisely for this scenario). The
regenerated run, all three fixes active simultaneously, is checked in at
`evidence/discovery_1786935840/` - the same bundle README's Evidence section lists as the required
live-target discovery run, so the fix is independently verifiable rather than asserted here.

## Cuts

- **Playwright's CDP remote-debugging protocol, in favor of the browser's existing non-headless
  window.** Deliberate, not an oversight: CDP is the credible answer for handing a session to a
  *remote* operator, but this project's demo only ever needs a human at the same terminal. Would
  build CDP exposure first if that changed.
- **Lower-priority stretch goals**, deprioritized per the design's own instruction to stop after
  the priority stretch goals unless clearly ahead of schedule. Confidence & approval gating (a
  `draft`/`approved` state gated on
  `reliability.pass_rate`, `invoke()` refusing `draft` without `force=true`) is a low-effort
  extension of the reliability work already built - would build it next, since the signal it would
  gate on already exists. Cross-tenant canonicalization is the more expensive item - correctly
  last in priority, not attempted.
- **The capability API has no auth, rate limiting, concurrency control, or shared/pooled browser**
  - each call launches and closes its own browser, correct at demo scale, not production scale.
- **Assisted fallback during replay** (a bounded, single-step LLM assist when a locator can't
  resolve) was considered early and not built - the escalation-and-retry mechanism that was built
  instead covers the same need without reintroducing an LLM into the replay path.
