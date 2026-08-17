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

Not yet wired up here, left as explicit scope for a later pass: per-step evidence logging
(log.jsonl / screenshots / transcript) for replay runs, and the escalation hand-off that a
hard_failure or a risky_irreversible step is meant to trigger (escalation/manager.py doesn't exist
yet). This module's own job, per its stub docstring, stops at producing a correct ReplayResult.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from capability_forge.guardrails.policy import Guardrail, PolicyViolation
from capability_forge.replay.outcome_classifier import ReplayStatus, StepOutcomeType, classify_replay_status, classify_step_outcome
from capability_forge.schema.artifact import CapabilityArtifact, InputParam, render_template
from capability_forge.surfaces.playwright_driver import CheckpointNotReachedError, PlaywrightDriver


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
    """The structured result of one replay run (Section 4.3 / 7.1 of the design)."""

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
    holds only the driver and guardrail it was constructed with."""

    def __init__(self, driver: PlaywrightDriver, guardrail: Guardrail):
        self.driver = driver
        self.guardrail = guardrail

    def run(self, artifact: CapabilityArtifact, params: dict[str, Any] | None = None) -> ReplayResult:
        normalized_params = _validate_and_normalize_params(artifact, params or {})

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
                    return self._hard_failure(artifact, step_records, step.step_id, step.description, f"unresolved param: {exc}")

            try:
                self.guardrail.check_action(destination, step.action_type)
            except PolicyViolation as exc:
                return self._hard_failure(artifact, step_records, step.step_id, step.description, f"blocked by policy: {exc}")

            step_start = time.monotonic()
            try:
                resolved_tier = self.driver.act(step, normalized_params)
            except Exception as exc:  # noqa: BLE001 - any failure to execute this step is a hard_failure, whatever its type
                return self._hard_failure(artifact, step_records, step.step_id, step.description, str(exc))
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

        try:
            outputs = self.driver.verify_checkpoint(artifact.checkpoint)
        except CheckpointNotReachedError as exc:
            return self._hard_failure(artifact, step_records, "checkpoint", artifact.checkpoint.description, str(exc))

        status = classify_replay_status([r.outcome_type for r in step_records], artifact.expected_outcome_type)
        return ReplayResult(
            artifact_id=artifact.artifact_id,
            status=status,
            outputs=outputs,
            steps=step_records,
            business_outcome_reason=artifact.business_outcome_reason if status == "business_outcome" else None,
        )

    def _hard_failure(
        self,
        artifact: CapabilityArtifact,
        step_records: list[StepReplayRecord],
        failed_step_id: str,
        expected_state: str,
        observed_state: str,
    ) -> ReplayResult:
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
