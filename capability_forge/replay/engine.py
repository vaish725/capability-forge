"""Replay engine.

Executes a CapabilityArtifact deterministically: validates input params against the artifact's
typed schema, drives the Surface Driver through each step in order (falling back across locator
tiers via PlaywrightDriver.act() itself, not any logic here), verifies the checkpoint, and returns
a structured ReplayResult. No LLM is involved anywhere in this module - every decision is either a
direct schema check or a mechanical read of what the driver actually did.

Guardrail checks happen before every step, exactly as in discovery, and mirror discovery's own
convention (agent_loop.py) for what "the domain" means per action_type: a navigate step is checked
against its own destination (the step's input_value once params are substituted in), every other
action_type is checked against the artifact's target.base_url, not the page's current live URL.
This is a deliberate consistency choice, not an oversight - replay should apply the exact same
policy shape discovery already does, so a step that would have been blocked during discovery can
never quietly slip through during replay instead.

Evidence logging (log.jsonl / screenshots), when an EvidenceWriter is supplied, uses the exact same
schema discovery writes (see utils/evidence.py's module docstring), with mode="replay" - one
log line per step plus a "terminal" summary line, matching agent_loop.py's own per-attempt-plus-
finish() pattern. A typed/selected step's rendered value is registered as a secret before the step
is dispatched, mirroring discovery's own discipline (see evidence.py) for the same reason: a value
that isn't scrubbed until after it's already been observed once is scrubbed too late. No
transcript.json is written for replay - that file is specifically the LLM message history, and a
replay run has no LLM turns to record.

Escalation (escalation/manager.py), when an EscalationManager is supplied, covers exactly the two
trigger conditions Section 4.6 assigns to replay:
  - risky_step_confirmation: before a risky_irreversible step is dispatched, unless the caller
    already passed confirm_risky=True, control is handed to a human; "resume" proceeds with the
    step, "abort" (or declining to answer) ends the run as a hard_failure. Without an
    EscalationManager at all, a risky step with confirm_risky=False fails closed rather than
    executing unconfirmed - the design's own stated default (Section 8/9: "risky actions in
    replay mode are, by default, executed only if the calling agent explicitly passes
    confirm_risky=true").
  - hard_failure: if a step's driver.act() call raises, or the checkpoint never resolves, control
    is handed to a human before the run is finalized. A "resume" decision means the human may have
    fixed something live in the browser, so the exact same action is retried - exactly once, never
    looped - and the run continues normally if it now succeeds. "abort" (or no EscalationManager)
    finalizes as hard_failure exactly as before this pass.
A guardrail PolicyViolation and an unresolved {{param}} in a navigate destination are deliberately
NOT escalation triggers, even though both also produce a hard_failure - a policy block is a
safety decision, not a transient technical fault a human fixing the live page would resolve, and
escalating past one would undermine what the allowlist is for; an unresolved param is an artifact-
authoring bug (a schema-level guarantee that was supposed to prevent this was skipped upstream),
not something waiting in the browser to be fixed by hand. Both keep failing closed with no human
in the loop, exactly as before this pass.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from capability_forge.escalation.manager import EscalationManager, EscalationTrigger
from capability_forge.guardrails.policy import Guardrail, PolicyViolation
from capability_forge.replay.outcome_classifier import ReplayStatus, StepOutcomeType, classify_replay_status, classify_step_outcome
from capability_forge.schema.artifact import CapabilityArtifact, LocatorTier, StepAction, render_template
from capability_forge.surfaces.playwright_driver import CheckpointNotReachedError, PlaywrightDriver
from capability_forge.utils.evidence import EvidenceWriter


class ParamValidationError(Exception):
    """Raised when the params passed to ReplayEngine.run() don't match the artifact's declared
    input schema - a missing required param, an undeclared one, or a value that doesn't fit its
    declared type. Raised before the browser is touched at all."""


@dataclass
class StepReplayRecord:
    """What happened for one step during replay - the per-step detail behind a ReplayResult."""

    step_id: str
    outcome_type: StepOutcomeType
    locator_tier_used: str | None  # "role" | "css" | "coordinate", None for navigate
    duration_ms: int


@dataclass
class ReplayResult:
    """The structured result of one replay run (Section 4.3 / 7.1 of the design).

    `status` is the rolled-up run-level verdict; `steps` is where the per-step detail behind it
    lives. In particular, a `status` of `recoverable_then_success` only says "at least one step
    fell back to a non-primary locator tier" - it collapses "one step, once" and "every step, on
    this run" into the same value. A caller that cares about the difference (a future reliability
    scorer or escalation threshold deciding whether an artifact's primary locators are degrading)
    should count `outcome_type == "recoverable"` across `steps` directly rather than branching on
    `status` alone. See outcome_classifier.py's module docstring for why "recoverable" here means
    specifically locator-tier fallback, and nothing broader."""

    artifact_id: str
    status: ReplayStatus
    outputs: dict[str, str]
    steps: list[StepReplayRecord] = field(default_factory=list)
    business_outcome_reason: str | None = None
    # Populated only when status == "hard_failure" - enough detail to debug without re-running,
    # per Section 7.1's own requirement that a hard_failure log line carry step_id/expected/
    # observed state, not just "it failed".
    failed_step_id: str | None = None
    expected_state: str | None = None
    observed_state: str | None = None


class ReplayEngine:
    """Runs one CapabilityArtifact against a driver-controlled page. Stateless across calls -
    holds only the driver, guardrail, and (optionally) evidence writer / escalation manager it was
    constructed with. evidence_writer and escalation_manager are both optional and off by default:
    a caller that wants neither (e.g. most of this module's own tests, which assert on the
    returned ReplayResult directly) simply doesn't pass them, matching how AgentLoop already
    treats evidence_writer."""

    def __init__(
        self,
        driver: PlaywrightDriver,
        guardrail: Guardrail,
        evidence_writer: EvidenceWriter | None = None,
        escalation_manager: EscalationManager | None = None,
    ):
        self.driver = driver
        self.guardrail = guardrail
        self.evidence_writer = evidence_writer
        self.escalation_manager = escalation_manager

    def run(self, artifact: CapabilityArtifact, params: dict[str, Any] | None = None, confirm_risky: bool = False) -> ReplayResult:
        """confirm_risky: the calling agent's explicit, up-front confirmation that any
        risky_irreversible step in this artifact is authorized to run. Defaults to False, matching
        the design's own stated default (Section 8/9) - a risky step is never executed just
        because it's in the artifact, only because the caller (or, failing that, a human via
        escalation) explicitly said so for this specific run."""
        normalized_params = _validate_and_normalize_params(artifact, params or {})
        run_start = time.monotonic()

        # Implicit initial navigation to the target, matching discovery's own convention
        # (target.base_url is where a run starts, before step 1, and is not itself a StepAction).
        self.driver.page.goto(artifact.target.base_url)

        step_records: list[StepReplayRecord] = []
        for step in artifact.steps:
            destination = artifact.target.base_url
            if step.action_type == "navigate":
                try:
                    destination = render_template(step.input_value, normalized_params) or artifact.target.base_url
                except KeyError as exc:
                    return self._hard_failure(artifact, step_records, run_start, step.step_id, step.description, f"unresolved param: {exc}")

            try:
                self.guardrail.check_action(destination, step.action_type)
            except PolicyViolation as exc:
                return self._hard_failure(artifact, step_records, run_start, step.step_id, step.description, f"blocked by policy: {exc}")

            if step.risk == "risky_irreversible" and not confirm_risky:
                confirmed, failure = self._confirm_risky_step(artifact, step_records, run_start, step)
                if not confirmed:
                    return failure  # either declined by the operator, or no way to ask - fails closed either way

            # Register a typed/selected value as a secret to scrub *before* dispatching the step,
            # not after - the whole point of register_secret() is that it must be known before the
            # value has a chance to resurface unprompted in a later page observation (see the
            # credential leak this defends against, documented in utils/evidence.py). Best-effort:
            # an unresolved param here just means driver.act() is about to raise the same KeyError
            # a moment later, which the except block below already turns into a hard_failure.
            if self.evidence_writer is not None and step.action_type in ("type", "select"):
                try:
                    rendered_value = render_template(step.input_value, normalized_params)
                except KeyError:
                    rendered_value = None
                if rendered_value:
                    self.evidence_writer.register_secret(rendered_value)

            step_start = time.monotonic()
            try:
                resolved_tier = self._act_with_escalation(artifact, step, normalized_params)
            except Exception as exc:  # noqa: BLE001 - any failure to execute this step is a hard_failure, whatever its type
                return self._hard_failure(artifact, step_records, run_start, step.step_id, step.description, str(exc))
            duration_ms = int((time.monotonic() - step_start) * 1000)

            outcome_type = classify_step_outcome(step, resolved_tier)
            step_records.append(
                StepReplayRecord(
                    step_id=step.step_id,
                    outcome_type=outcome_type,
                    locator_tier_used=resolved_tier.strategy if resolved_tier else None,
                    duration_ms=duration_ms,
                )
            )
            self._log_step(step, resolved_tier, destination, outcome_type, duration_ms)

        try:
            outputs = self._verify_checkpoint_with_escalation(artifact)
        except CheckpointNotReachedError as exc:
            return self._hard_failure(artifact, step_records, run_start, "checkpoint", artifact.checkpoint.description, str(exc))

        status = classify_replay_status([r.outcome_type for r in step_records], artifact.expected_outcome_type)
        business_outcome_reason = artifact.business_outcome_reason if status == "business_outcome" else None
        if self.evidence_writer is not None:
            # Terminal summary line, mirroring agent_loop.py's finish(). Kept within the per-line
            # outcome_type enum (success | business_outcome | recoverable | hard_failure, per the
            # PRD's Section 7.1 schema) rather than emitting the run-level "recoverable_then_success"
            # value here - that refinement already lives on every step's own log line above, so the
            # terminal line only needs to say whether the run overall ended well or not.
            self.evidence_writer.save_screenshot(self.driver.page)
            self.evidence_writer.log_step(
                step_id="terminal",
                action_attempted=f"replay_complete(status={status})",
                locator_tier_used=None,
                expected_state=artifact.checkpoint.description,
                observed_state=business_outcome_reason or "checkpoint verified",
                outcome_type="business_outcome" if status == "business_outcome" else "success",
                duration_ms=int((time.monotonic() - run_start) * 1000),
            )
        return ReplayResult(
            artifact_id=artifact.artifact_id,
            status=status,
            outputs=outputs,
            steps=step_records,
            business_outcome_reason=business_outcome_reason,
        )

    def _log_step(
        self,
        step: StepAction,
        resolved_tier: LocatorTier | None,
        destination: str,
        outcome_type: StepOutcomeType,
        duration_ms: int,
    ) -> None:
        """Write one log.jsonl line plus its screenshot for an already-executed step. No-op if no
        evidence_writer was supplied."""
        if self.evidence_writer is None:
            return
        if step.action_type == "navigate":
            action_attempted = f"navigate({destination})"
            observed_state = "navigated"
        elif resolved_tier is not None:
            # A role-tier value is already a full "role=..." selector string (self-describing);
            # css and coordinate values are bare ("button.submit", "120,340") and need the
            # strategy prefixed on to stay readable. Checking rather than assuming which is which
            # avoids "role=role=..." showing up in a real log line the moment a role-tier value's
            # own convention is what it already is - found by reading a real generated evidence
            # bundle, not by inspecting this code in isolation.
            prefix = f"{resolved_tier.strategy}="
            value_str = resolved_tier.value if resolved_tier.value.startswith(prefix) else f"{prefix}{resolved_tier.value}"
            action_attempted = f"{step.action_type}({value_str})"
            observed_state = f"resolved via {resolved_tier.strategy} tier"
        else:
            action_attempted = f"{step.action_type}()"
            observed_state = "resolved"
        evidence_ref = self.evidence_writer.save_screenshot(self.driver.page)
        self.evidence_writer.log_step(
            step_id=step.step_id,
            action_attempted=action_attempted,
            locator_tier_used=resolved_tier.strategy if resolved_tier else None,
            expected_state=step.description,
            observed_state=observed_state,
            outcome_type=outcome_type,
            duration_ms=duration_ms,
            evidence_ref=evidence_ref,
        )

    def _confirm_risky_step(
        self,
        artifact: CapabilityArtifact,
        step_records: list[StepReplayRecord],
        run_start: float,
        step: StepAction,
    ) -> tuple[bool, "ReplayResult | None"]:
        """Gate a risky_irreversible step behind explicit confirmation - either the caller already
        gave it (confirm_risky=True, checked by the caller before this is ever reached) or a human
        grants it live via escalation. Returns (True, None) if the step is cleared to proceed, or
        (False, <the finished hard_failure ReplayResult>) if not. Fails closed with no
        EscalationManager available - an unconfirmed risky step never executes just because
        nobody was around to ask, per the design's own stated default (see the module docstring)."""
        if self.escalation_manager is None:
            return False, self._hard_failure(
                artifact, step_records, run_start, step.step_id, step.description,
                "risky_irreversible step requires confirm_risky=True or an EscalationManager; neither was provided",
            )
        trigger = EscalationTrigger(
            reason="risky_step_confirmation",
            run_id=self.escalation_manager.run_id,
            goal_description=artifact.goal_description,
            step_id=step.step_id,
            description=f"About to perform a risky, irreversible action: {step.description}",
            screenshot_ref=self._maybe_screenshot(),
        )
        record = self.escalation_manager.run_handoff(trigger)
        if record.decision != "resume":
            return False, self._hard_failure(
                artifact, step_records, run_start, step.step_id, step.description,
                f"risky step not confirmed by operator (decision={record.decision!r}): {record.notes or 'no note given'}",
            )
        return True, None

    def _act_with_escalation(self, artifact: CapabilityArtifact, step: StepAction, normalized_params: dict[str, str]) -> LocatorTier | None:
        """Execute driver.act() once; if it raises and an escalation_manager is available, hand
        the live session to a human and retry the exact same call once if they resume. No retry
        loop beyond that single extra attempt - if the step still fails after a human had a
        chance to intervene live, it's a genuine hard_failure, not something worth asking again."""
        try:
            return self.driver.act(step, normalized_params)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see the try/except this wraps in run()
            if self.escalation_manager is None:
                raise
            trigger = EscalationTrigger(
                reason="hard_failure",
                run_id=self.escalation_manager.run_id,
                goal_description=artifact.goal_description,
                step_id=step.step_id,
                description=f"Step {step.step_id!r} failed: {exc}",
                screenshot_ref=self._maybe_screenshot(),
            )
            record = self.escalation_manager.run_handoff(trigger)
            if record.decision != "resume":
                raise
            return self.driver.act(step, normalized_params)

    def _verify_checkpoint_with_escalation(self, artifact: CapabilityArtifact) -> dict[str, str]:
        """Same one-retry-after-a-human-looked-at-it pattern as _act_with_escalation, for the
        checkpoint itself never resolving."""
        try:
            return self.driver.verify_checkpoint(artifact.checkpoint)
        except CheckpointNotReachedError as exc:
            if self.escalation_manager is None:
                raise
            trigger = EscalationTrigger(
                reason="hard_failure",
                run_id=self.escalation_manager.run_id,
                goal_description=artifact.goal_description,
                step_id="checkpoint",
                description=f"Checkpoint not reached: {exc}",
                screenshot_ref=self._maybe_screenshot(),
            )
            record = self.escalation_manager.run_handoff(trigger)
            if record.decision != "resume":
                raise
            return self.driver.verify_checkpoint(artifact.checkpoint)

    def _maybe_screenshot(self) -> str | None:
        if self.evidence_writer is None:
            return None
        return self.evidence_writer.save_screenshot(self.driver.page)

    def _hard_failure(
        self,
        artifact: CapabilityArtifact,
        step_records: list[StepReplayRecord],
        run_start: float,
        failed_step_id: str,
        expected_state: str,
        observed_state: str,
    ) -> ReplayResult:
        if self.evidence_writer is not None:
            evidence_ref = self.evidence_writer.save_screenshot(self.driver.page)
            self.evidence_writer.log_step(
                step_id=failed_step_id,
                action_attempted=f"hard_failure(step={failed_step_id})",
                locator_tier_used=None,
                expected_state=expected_state,
                observed_state=observed_state,
                outcome_type="hard_failure",
                duration_ms=int((time.monotonic() - run_start) * 1000),
                evidence_ref=evidence_ref,
            )
        return ReplayResult(
            artifact_id=artifact.artifact_id,
            status="hard_failure",
            outputs={},
            steps=step_records,
            failed_step_id=failed_step_id,
            expected_state=expected_state,
            observed_state=observed_state,
        )


def _validate_and_normalize_params(artifact: CapabilityArtifact, params: dict[str, Any]) -> dict[str, str]:
    """Check params against artifact.inputs before touching the browser at all: every required
    param present, no undeclared param silently ignored (a typo should fail loudly, not be
    dropped), and every value fits its declared type. Returns a dict[str, str] - render_template
    (called inside driver.act()) only ever substitutes strings, whatever native type a caller
    passed in (a JSON request body, for instance, naturally produces int/float/bool already
    parsed)."""
    declared = {param.name: param for param in artifact.inputs}

    undeclared = set(params) - set(declared)
    if undeclared:
        raise ParamValidationError(f"undeclared param(s): {sorted(undeclared)}")

    missing = {name for name, param in declared.items() if param.required and name not in params}
    if missing:
        raise ParamValidationError(f"missing required param(s): {sorted(missing)}")

    return {name: _coerce_param(name, value, declared[name].type) for name, value in params.items()}


def _coerce_param(name: str, value: Any, declared_type: str) -> str:
    if declared_type == "string":
        return str(value)
    if declared_type == "int":
        if isinstance(value, bool):  # bool is a subclass of int in Python - reject explicitly
            raise ParamValidationError(f"param {name!r} expects int, got bool")
        try:
            return str(int(value))
        except (TypeError, ValueError):
            raise ParamValidationError(f"param {name!r} expects int, got {value!r}") from None
    if declared_type == "float":
        if isinstance(value, bool):
            raise ParamValidationError(f"param {name!r} expects float, got bool")
        try:
            return str(float(value))
        except (TypeError, ValueError):
            raise ParamValidationError(f"param {name!r} expects float, got {value!r}") from None
    if declared_type == "bool":
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return str(value.lower() == "true")
        raise ParamValidationError(f"param {name!r} expects bool, got {value!r}")
    raise ParamValidationError(f"param {name!r} has unrecognized declared type {declared_type!r}")
