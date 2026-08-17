"""Outcome classifier.

Pure functions mapping what actually happened during a replay to the closed outcome taxonomy
(Section 7 of the design: success | business_outcome | recoverable | hard_failure for a single
step, success | business_outcome | recoverable_then_success | hard_failure for a whole run). Kept
separate from replay/engine.py so the classification rules themselves - not the orchestration
around them (driving the browser, catching exceptions, assembling a result) - are directly
testable without a real page.

Two closely related but distinct questions, answered by the two functions here:

1. classify_step_outcome: did this one step, having already executed without raising, need a
   fallback locator tier to succeed? Replay has no LLM to improvise a recovery the way discovery
   does - the only form of "automatic recovery" it can offer is the driver's own already-built
   locator-tier fallback chain (role -> css -> coordinate). A step that resolved via its
   first-listed tier is a clean success; a step that only resolved via a later tier succeeded, but
   not the way it was expected to (a role/name likely drifted since the artifact was recorded) -
   "recoverable" names that honestly rather than silently calling it plain success. A step whose
   action_type has no locator at all (navigate) is always success once it's executed without
   raising - there was nothing to fall back from.

2. classify_replay_status: given every step's outcome and what the artifact itself expected
   (CapabilityArtifact.expected_outcome_type - added specifically because a business_outcome
   checkpoint resolves exactly the same way a success one does, so there is no way to tell them
   apart without the artifact self-declaring which one it is), what is the run's overall status?
   Any hard_failure among the steps means the whole run is hard_failure, full stop. Otherwise the
   run's status is business_outcome or success according to what the artifact declared it would
   be - business_outcome is never further split into a "recoverable" variant, since
   ReplayResult.status's enum only defines recoverable_then_success as a refinement of success (a
   run reproducing a named non-error result is either that or a hard failure, not something in
   between). A success run that needed at least one step to fall back to a later tier is reported
   as recoverable_then_success rather than plain success, so that signal isn't silently lost.
"""

from typing import Literal

from capability_forge.schema.artifact import LocatorTier, StepAction

StepOutcomeType = Literal["success", "recoverable", "hard_failure"]
ReplayStatus = Literal["success", "business_outcome", "recoverable_then_success", "hard_failure"]


def classify_step_outcome(step: StepAction, resolved_tier: LocatorTier | None) -> StepOutcomeType:
    """Classify one already-executed step (driver.act() did not raise). resolved_tier is what
    driver.act() returned - None for navigate (no locator involved), otherwise the LocatorTier
    that was actually used. hard_failure is never returned here: a step that couldn't be resolved
    or executed at all raises out of driver.act() instead of returning, and the caller (the replay
    engine) is responsible for catching that and classifying the run as hard_failure directly -
    this function only ever sees a step that succeeded, and only decides how cleanly it did."""
    if resolved_tier is None or not step.locators:
        return "success"
    if resolved_tier == step.locators[0]:
        return "success"
    return "recoverable"


def classify_replay_status(
    step_outcomes: list[StepOutcomeType],
    expected_outcome_type: Literal["success", "business_outcome"],
) -> ReplayStatus:
    """Classify the whole run from its steps' outcomes plus the artifact's own declared
    expectation. Assumes the checkpoint was already confirmed to verify - a caller whose
    checkpoint never resolved should classify as hard_failure directly, without calling this at
    all, since no combination of step outcomes changes that."""
    if "hard_failure" in step_outcomes:
        return "hard_failure"
    if expected_outcome_type == "business_outcome":
        return "business_outcome"
    if "recoverable" in step_outcomes:
        return "recoverable_then_success"
    return "success"
