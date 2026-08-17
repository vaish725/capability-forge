"""Human-in-the-loop escalation and handoff.

Implements the design's state machine (Section 4.6):
    AGENT_ACTIVE -> PAUSED_FOR_HUMAN -> HUMAN_ACTIVE -> RESUMING -> (AGENT_ACTIVE | DONE)

EscalationManager is the single source of truth for who is in control of the live session at any
moment - callers (AgentLoop, ReplayEngine) never mutate control state themselves, they only ask
this manager to run a handoff and act on the HandoffRecord it returns. This is deliberate: a
split-brain where the automated loop and some other piece of code disagree about whether a human
is currently driving the browser is exactly the failure mode a single authoritative state machine
rules out. Every transition method checks the current state before doing anything and raises
IllegalTransitionError rather than guessing if a caller asks for a transition out of order.

Three trigger conditions are named in the design, one per calling module:
  - dead_end_detected: AgentLoop's repeated-state guard fires (the page hasn't moved forward after
    several mutating actions).
  - hard_failure: ReplayEngine hits a step it can't execute, or a checkpoint that never resolves.
  - risky_step_confirmation: ReplayEngine is about to perform a risky_irreversible step and the
    caller didn't pass confirm_risky=True.
All three funnel through the same escalate() -> hand_to_human() -> resolve() sequence -
trigger.reason is what varies, not the mechanism. See replay/engine.py and discovery/agent_loop.py
for where each is actually wired in, and what the caller does with the resulting decision (retry
once, proceed with a confirmed step, or finalize the run) - that policy belongs to the caller, not
to this module, since only the caller knows what "resume" should mean for the specific thing that
triggered the escalation.

What "exposing the live session" means here, decided explicitly rather than left to the PRD's
looser "expose via CDP" phrasing:
For this project's actual demo, the browser already runs non-headless (see discover.py's default),
so the literal OS-level window the automated loop was just driving is already visible and clickable
by a human at the same machine the moment control is handed over - no additional exposure step is
needed, and none is implemented here. Playwright also supports exposing a page over its CDP
remote-debugging protocol (chrome://inspect, or a websocket-based remote viewer), which is the
credible answer for a real deployed system handing a session to a remote operator instead of one
sitting at the same terminal - that's a real design, but it's infrastructure this project doesn't
need to stand up just to demonstrate the actual mechanism (pause, hand over control, capture what
was decided, resume), so it's written up as the production answer in REPORT.md rather than built
here.

The "operator console" is a pluggable callable (OperatorConsole), not a UI: given an
EscalationTrigger it returns an OperatorDecision. The default implementation
(cli_operator_console) is a blocking terminal prompt - printed context, typed decision - which
satisfies the assignment's own scope note ("a full real-time co-browsing operator console is out
of scope... mock the operator UI if needed, but make the handoff mechanism and control-transfer
model real") at the lowest cost to build and the highest testability: a fake console is just a
function returning a canned OperatorDecision, so every state transition here is testable without
a terminal, a browser, or a human in the loop at all.
"""

import time
from dataclasses import dataclass
from typing import Callable, Literal

from capability_forge.utils.evidence import EvidenceWriter

EscalationState = Literal["AGENT_ACTIVE", "PAUSED_FOR_HUMAN", "HUMAN_ACTIVE", "RESUMING", "DONE"]
EscalationReason = Literal["hard_failure", "risky_step_confirmation", "dead_end_detected"]
OperatorDecisionType = Literal["resume", "abort"]


class IllegalTransitionError(Exception):
    """Raised when a caller asks the state machine to do something out of order (e.g. resolve()
    before hand_to_human() was ever called). Who's in control of the session must never be
    ambiguous, so an out-of-order call is treated as a caller bug and refused loudly, not papered
    over by guessing what was probably meant."""


@dataclass
class EscalationTrigger:
    """Why control is being handed to a human, and enough context for them to act on it without
    digging through logs first - the fields a real operator console (or the mock CLI one here)
    actually displays."""

    reason: EscalationReason
    run_id: str
    goal_description: str
    step_id: str | None
    description: str  # human-readable: what's happening and why a human is needed
    screenshot_ref: str | None = None


@dataclass
class OperatorDecision:
    """What the human decided, plus a free-text note about what they did. notes is the "what they
    did" half of HandoffRecord's requirement (Section 4.6) for a minimal console that doesn't
    instrument the operator's individual clicks in the live browser - the operator states it
    themselves rather than it being inferred."""

    decision: OperatorDecisionType
    notes: str = ""


@dataclass
class HandoffRecord:
    """One completed handoff, written to the run's evidence bundle. Carries the trigger's own
    context inline rather than just an id referencing it, so a record stands on its own - an
    evidence reviewer reading handoffs.jsonl shouldn't need to cross-reference a separate trigger
    log just to know why control changed hands."""

    run_id: str
    reason: EscalationReason
    step_id: str | None
    description: str
    decision: OperatorDecisionType
    notes: str
    taken_at: str  # iso8601, when PAUSED_FOR_HUMAN began
    released_at: str  # iso8601, when the operator's decision was recorded


OperatorConsole = Callable[[EscalationTrigger], OperatorDecision]


def cli_operator_console(trigger: EscalationTrigger) -> OperatorDecision:
    """Default OperatorConsole: a blocking terminal prompt. See the module docstring for why a
    minimal CLI console, not a local HTML page, is the deliberate choice here."""
    print(f"\n--- ESCALATION: {trigger.reason} ---")
    print(f"Run: {trigger.run_id}")
    print(f"Goal: {trigger.goal_description}")
    if trigger.step_id:
        print(f"Step: {trigger.step_id}")
    print(trigger.description)
    if trigger.screenshot_ref:
        print(f"Screenshot: {trigger.screenshot_ref}")
    print("The live browser window is now yours. Take whatever action is needed, then respond below.")
    while True:
        raw = input("Type 'resume' to hand control back to the agent, or 'abort' to end the run: ").strip().lower()
        if raw in ("resume", "abort"):
            notes = input("Optional note about what you did (blank to skip): ").strip()
            return OperatorDecision(decision=raw, notes=notes)  # type: ignore[arg-type]
        print("Please type exactly 'resume' or 'abort'.")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class EscalationManager:
    """The single source of truth for who currently controls the live session, for one run.
    Constructed fresh per run - state is not meant to survive across runs, matching how AgentLoop
    and ReplayEngine are themselves constructed per run."""

    def __init__(
        self,
        run_id: str,
        operator_console: OperatorConsole = cli_operator_console,
        evidence_writer: EvidenceWriter | None = None,
    ):
        self.run_id = run_id
        self.operator_console = operator_console
        self.evidence_writer = evidence_writer
        self.state: EscalationState = "AGENT_ACTIVE"
        self._pending_trigger: EscalationTrigger | None = None
        self._taken_at: str | None = None

    def escalate(self, trigger: EscalationTrigger) -> None:
        """AGENT_ACTIVE -> PAUSED_FOR_HUMAN. Called by the automated loop the moment a trigger
        condition fires, before any human has actually looked at anything."""
        self._require_state("AGENT_ACTIVE", "escalate")
        self.state = "PAUSED_FOR_HUMAN"
        self._pending_trigger = trigger
        self._taken_at = _now()

    def hand_to_human(self) -> None:
        """PAUSED_FOR_HUMAN -> HUMAN_ACTIVE. The live session is now under human control - nothing
        further happens automatically until resolve() is called."""
        self._require_state("PAUSED_FOR_HUMAN", "hand_to_human")
        self.state = "HUMAN_ACTIVE"

    def resolve(self, decision: OperatorDecision) -> HandoffRecord:
        """HUMAN_ACTIVE -> RESUMING -> (AGENT_ACTIVE | DONE). Writes the HandoffRecord to the
        evidence bundle (if a writer was supplied) and returns it, so the caller can decide what
        "resume" or "abort" actually means for the specific thing that triggered the escalation
        (retry a step, proceed with a confirmed risky action, or finalize the run) without needing
        to re-derive anything from this manager's internal state."""
        self._require_state("HUMAN_ACTIVE", "resolve")
        self.state = "RESUMING"
        trigger = self._pending_trigger
        assert trigger is not None  # guaranteed by escalate() having run first - see _require_state

        record = HandoffRecord(
            run_id=self.run_id,
            reason=trigger.reason,
            step_id=trigger.step_id,
            description=trigger.description,
            decision=decision.decision,
            notes=decision.notes,
            taken_at=self._taken_at or _now(),
            released_at=_now(),
        )
        if self.evidence_writer is not None:
            self.evidence_writer.log_handoff(record)

        self.state = "AGENT_ACTIVE" if decision.decision == "resume" else "DONE"
        self._pending_trigger = None
        self._taken_at = None
        return record

    def run_handoff(self, trigger: EscalationTrigger) -> HandoffRecord:
        """Convenience entry point bundling the whole cycle (escalate -> hand_to_human ->
        operator_console -> resolve) for the common case, where a caller has no reason to drive
        the individual transitions itself. The transitions above stay separately callable (and
        separately tested) for anything that does need finer control."""
        self.escalate(trigger)
        self.hand_to_human()
        decision = self.operator_console(trigger)
        return self.resolve(decision)

    def _require_state(self, expected: EscalationState, method_name: str) -> None:
        if self.state != expected:
            raise IllegalTransitionError(
                f"{method_name}() requires state {expected!r}, but current state is {self.state!r}"
            )
