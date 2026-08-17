"""Artifact recorder.

Turns a DiscoveryRun's raw step history into a validated CapabilityArtifact: templates literal
input values into {{param}} placeholders, declares the InputParam/OutputField contract, and
assembles everything the schema requires. Only DiscoveryRun.steps and DiscoveryRun.checkpoint are
transformed; the schema's own cross-field validators (Section 6 of the design) do the actual
correctness enforcement, this module's job is just building a well-formed candidate to hand them.

Two design decisions, made explicit here rather than guessed at implementation time:

1. Which stop_reason values are recordable. Both "goal_complete" and "business_outcome" are - both
   go through the exact same checkpoint verification in agent_loop.py's _handle_done before the
   run is allowed to stop, so a business_outcome run (e.g. "this member doesn't exist") is just as
   deterministically replayable and just as legitimate a capability as a success run ("member not
   found" is itself a useful thing to be able to check for, repeatably, without an LLM). Anything
   else (give_up, max_steps_exceeded, timeout_exceeded, dead_end_detected) has no verified
   checkpoint at all and is structurally unrecordable.

2. Which literal values get templated, and where. The caller supplies an explicit param_map
   (literal value -> param name) rather than the recorder guessing - deliberately conservative,
   since auto-detecting "this string looks like an input" is exactly the kind of silent heuristic
   this project has avoided elsewhere (risk classification, redaction). Substitution is applied to
   every free-text field a saved artifact carries: each step's input_value and description,
   goal_description, and checkpoint.description - never a locator's value. The first real
   ParaBank run is why this covers more than input_value: goal_description and checkpoint.description
   are both built from the model's own prose (the run's original goal string, and its own summary
   of what it did), and that prose repeated the literal username in plain text even though the
   corresponding step.input_value had already been templated - a login credential parameterized
   out of the steps but still sitting in the clear elsewhere in the same saved file defeats the
   entire point of parameterizing it. Only StepAction.input_value is checked by the schema's own
   {{param}}-must-be-declared validator; templating the others is this module's own responsibility,
   not something the schema enforces, since they're display text, not executed at replay time.

   One thing this still cannot help with: checkpoint.locator.value and checkpoint.extract selector
   values are plain text captured at discovery time, with no templating support in the schema at
   all (the cross-field validator that resolves {{param}} references only ever looks at
   step.input_value). Parameterizing a value that also appears inside the checkpoint's own locator
   text would produce an artifact whose checkpoint can never verify for any input other than the
   one it was recorded with - so the caller is still responsible for not mapping a value that leaks
   into the checkpoint's locator, and this module makes no attempt to detect that automatically.
"""

from capability_forge.discovery.agent_loop import DiscoveryRun
from capability_forge.schema.artifact import (
    CapabilityArtifact,
    Checkpoint,
    InputParam,
    OutputField,
    StepAction,
    TargetSpec,
    extract_template_params,
)

# stop_reason values with a verified checkpoint behind them - the only ones a run can be recorded
# from. Anything else means the run never reached a confirmed terminal state.
_RECORDABLE_STOP_REASONS = {"goal_complete", "business_outcome"}


class UnrecordableRunError(Exception):
    """Raised when a DiscoveryRun has no verified checkpoint to record from (give_up, a stopping
    guard, or a timeout) - there is nothing deterministic to replay."""


def record_artifact(
    run: DiscoveryRun,
    artifact_id: str,
    app_name: str,
    param_map: dict[str, str] | None = None,
    input_types: dict[str, str] | None = None,
    input_descriptions: dict[str, str] | None = None,
    outputs: list[OutputField] | None = None,
    checkpoint_extract: dict[str, str] | None = None,
    schema_version: str = "1.0",
) -> CapabilityArtifact:
    """Build a validated CapabilityArtifact from a completed DiscoveryRun.

    param_map: literal value -> param name. Substring-substituted (longest literal first, so a
    short literal that happens to be a substring of a longer one doesn't get replaced first and
    corrupt the longer match) against every free-text field the artifact carries - each step's
    input_value and description, goal_description, and checkpoint.description - and an InputParam
    is declared for each name actually used in a step's input_value. Values not in the map stay
    literal - most commonly structural navigation URLs and anything that also appears in the
    checkpoint's own locator text (see the module docstring's second design decision).

    outputs / checkpoint_extract: declared together, by the caller, from whatever the run actually
    read (extract_log) or from re-inspecting the run's evidence - not inferred here. Both empty is
    a legitimate, honest result for a run that never named a value it extracted; the schema's own
    validator (checkpoint.extract keys must exactly match declared output names) is what actually
    enforces consistency between the two, this function just passes them through.
    """
    if run.stop_reason not in _RECORDABLE_STOP_REASONS:
        raise UnrecordableRunError(
            f"stop_reason={run.stop_reason!r} has no verified checkpoint to record from "
            f"(only {sorted(_RECORDABLE_STOP_REASONS)} do)"
        )
    if run.checkpoint is None:
        # Should be unreachable given the stop_reason check above (both recordable stop reasons
        # always carry a checkpoint per agent_loop.py's _handle_done), but asserted explicitly
        # rather than silently trusted, since a None checkpoint reaching CapabilityArtifact's
        # required `checkpoint` field would otherwise fail with a much less informative error.
        raise UnrecordableRunError(f"stop_reason={run.stop_reason!r} but run.checkpoint is None")

    param_map = param_map or {}
    input_types = input_types or {}
    input_descriptions = input_descriptions or {}

    steps = [_template_step(step, param_map) for step in run.steps]
    used_param_names = _params_used(steps)

    inputs = [
        InputParam(
            name=name,
            type=input_types.get(name, "string"),
            required=True,
            description=input_descriptions.get(name, f"Value substituted for {{{{{name}}}}}."),
        )
        for name in sorted(used_param_names)
    ]

    checkpoint = Checkpoint(
        description=_apply_param_map(run.checkpoint.description, param_map),
        locator=run.checkpoint.locator,
        extract=checkpoint_extract,
    )

    has_risky_step = any(step.risk == "risky_irreversible" for step in steps)

    return CapabilityArtifact(
        artifact_id=artifact_id,
        schema_version=schema_version,
        target=TargetSpec(base_url=run.target_url, app_name=app_name),
        goal_description=_apply_param_map(run.goal_description, param_map),
        inputs=inputs,
        outputs=outputs or [],
        steps=steps,
        checkpoint=checkpoint,
        risk_summary="contains_risky_steps" if has_risky_step else "safe",
    )


def _apply_param_map(text: str, param_map: dict[str, str]) -> str:
    """Replace every param_map literal found in text with {{name}}, longest literal first (a
    short literal that happens to be a substring of a longer one, e.g. "16" inside "16896", must
    not get replaced first and corrupt the longer match). Shared by every free-text field this
    module templates - see the module docstring for why that's more than just input_value."""
    if not param_map:
        return text
    templated = text
    for literal in sorted(param_map, key=len, reverse=True):
        templated = templated.replace(literal, f"{{{{{param_map[literal]}}}}}")
    return templated


def _template_step(step: StepAction, param_map: dict[str, str]) -> StepAction:
    """Return a copy of step with any param_map literal replaced by {{name}} in both input_value
    and description. Locators are left untouched - templating never applies to a locator's value,
    matching exactly what the schema's own cross-field validator checks for input_value, and
    deliberately extended here (beyond what the schema enforces) to the step's own description
    text too, for the same reason goal_description and checkpoint.description are templated."""
    if not param_map:
        return step
    updates: dict[str, str] = {"description": _apply_param_map(step.description, param_map)}
    if step.input_value is not None:
        updates["input_value"] = _apply_param_map(step.input_value, param_map)
    return step.model_copy(update=updates)


def _params_used(steps: list[StepAction]) -> set[str]:
    """Every {{name}} placeholder actually present across all steps' input_value, after
    templating - reuses the schema's own extractor so this can never disagree with what the
    schema itself considers a template reference."""
    used: set[str] = set()
    for step in steps:
        used |= extract_template_params(step.input_value)
    return used
