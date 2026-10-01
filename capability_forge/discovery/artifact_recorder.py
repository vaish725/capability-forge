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

3. Locators never contain a value that varies between runs, and this module refuses to record one
   that does (ValueBoundLocatorError) rather than leaving that to the caller. An earlier version
   left it to the caller, and the fixture artifact recorded that way had a checkpoint of
   role=cell[name="$4500.00"] - the balance it read - so it hard-failed for every member except
   the one it was recorded with. A value varies if it is an input (a param_map literal) or an
   output (a value extracted with an output_name). The check is an exact match against those known
   values, not a guess about what "looks like" data. checkpoint.identity is the one exception: it
   is supposed to contain an input value, and is templated with it ({{param}}); if it contains no
   input value it says nothing about which record this is, so it is dropped.

4. Outputs come from the run itself when the caller doesn't declare them: every extract_log entry
   with an output_name becomes an OutputField (with its format and the value read, as `example`)
   plus a checkpoint.extract entry using the value_locator derived during the run. An output with
   no value_locator can't be found again for a different value, so the run is refused instead of
   being recorded with a value-bound selector.
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
    selector_contains_literal,
)

CURRENT_SCHEMA_VERSION = "1.1"  # 1.1 added checkpoint.identity, OutputField.format/example

# stop_reason values with a verified checkpoint behind them - the only ones a run can be recorded
# from. Anything else means the run never reached a confirmed terminal state.
_RECORDABLE_STOP_REASONS = {"goal_complete", "business_outcome"}


class UnrecordableRunError(Exception):
    """Raised when a DiscoveryRun has no verified checkpoint to record from (give_up, a stopping
    guard, or a timeout) - there is nothing deterministic to replay."""


class ValueBoundLocatorError(UnrecordableRunError):
    """Raised when a locator the artifact would carry contains an input or output value (see the
    module docstring's third design decision), or an output has no value-independent locator."""


def record_artifact(
    run: DiscoveryRun,
    artifact_id: str,
    app_name: str,
    param_map: dict[str, str] | None = None,
    input_types: dict[str, str] | None = None,
    input_descriptions: dict[str, str] | None = None,
    outputs: list[OutputField] | None = None,
    checkpoint_extract: dict[str, str] | None = None,
    schema_version: str = CURRENT_SCHEMA_VERSION,
) -> CapabilityArtifact:
    """Build a validated CapabilityArtifact from a completed DiscoveryRun.

    param_map: literal value -> param name. Substring-substituted (longest literal first, so a
    short literal that happens to be a substring of a longer one doesn't get replaced first and
    corrupt the longer match) against every free-text field the artifact carries - each step's
    input_value and description, goal_description, and checkpoint.description - and an InputParam
    is declared for each name actually used in a step's input_value. Values not in the map stay
    literal - most commonly structural navigation URLs and anything that also appears in the
    checkpoint's own locator text (see the module docstring's second design decision).

    outputs / checkpoint_extract: built from the run's own extract_log when both are omitted (see
    the module docstring's fourth design decision); a caller can still declare them explicitly.
    Either way, the schema's own validator (checkpoint.extract keys must exactly match declared
    output names) enforces consistency between the two. No outputs at all is a legitimate result
    for a run that never named a value it extracted (e.g. a business_outcome).

    Raises ValueBoundLocatorError rather than return an artifact that only replays for the inputs
    and outputs it was recorded with.
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

    if outputs is None and checkpoint_extract is None:
        outputs, checkpoint_extract = _outputs_from_extract_log(run)

    # Everything that varies between runs, checked against every locator the artifact will carry.
    varying = {literal: f"input {name!r}" for literal, name in param_map.items()}
    for output in outputs or []:
        if output.example:
            varying[output.example] = f"output {output.name!r}"
    locators_to_check = [("checkpoint locator", run.checkpoint.locator.value)]
    locators_to_check += [(f"output {name!r} locator", selector) for name, selector in (checkpoint_extract or {}).items()]
    locators_to_check += [(f"step {step.step_id!r} locator", tier.value) for step in steps for tier in step.locators]
    for label, selector in locators_to_check:
        for literal, source in varying.items():
            if selector_contains_literal(selector, literal):
                raise ValueBoundLocatorError(
                    f"{label} {selector!r} contains the value of {source} ({literal!r}), so the artifact "
                    "would only replay while that value stays the same"
                )

    # The description is replay's expected-state text for every input, so an output value left
    # in it (the model's own prose) would misstate what another input's page should show.
    description = _apply_param_map(run.checkpoint.description, param_map)
    for output in outputs or []:
        if output.example:
            description = description.replace(output.example, f"<{output.name}>")

    checkpoint = Checkpoint(
        description=description,
        locator=run.checkpoint.locator,
        identity=_template_identity(run.checkpoint.identity, param_map),
        extract=checkpoint_extract,
    )

    has_risky_step = any(step.risk == "risky_irreversible" for step in steps)

    # stop_reason -> expected_outcome_type: "goal_complete" produced a success checkpoint,
    # "business_outcome" produced a business_outcome one (the only two recordable stop reasons -
    # see _RECORDABLE_STOP_REASONS above). business_outcome_reason is templated for the same
    # reason goal_description and checkpoint.description are: it's free text the model wrote, and
    # nothing rules out it having echoed a literal value that was just parameterized elsewhere.
    expected_outcome_type = "success" if run.stop_reason == "goal_complete" else "business_outcome"
    business_outcome_reason = (
        _apply_param_map(run.business_outcome_reason, param_map)
        if expected_outcome_type == "business_outcome" and run.business_outcome_reason
        else None
    )

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
        expected_outcome_type=expected_outcome_type,
        business_outcome_reason=business_outcome_reason,
    )


def _outputs_from_extract_log(run: DiscoveryRun) -> tuple[list[OutputField], dict[str, str] | None]:
    """One OutputField + checkpoint.extract entry per output_name in the run's extract_log (the
    last read wins if a name was extracted more than once)."""
    latest: dict[str, dict] = {}
    for entry in run.extract_log:
        if entry.get("output_name"):
            latest[entry["output_name"]] = entry
    outputs: list[OutputField] = []
    extract: dict[str, str] = {}
    for name, entry in latest.items():
        if not entry.get("value_locator"):
            raise ValueBoundLocatorError(
                f"output {name!r} ({entry['value']!r}) was read from an element that can't be located "
                "without using its own value, so it can't be found again when the value differs"
            )
        outputs.append(
            OutputField(
                name=name,
                type="string",
                description=f"Read at the checkpoint ({entry['output_format']}).",
                format=entry["output_format"],
                example=entry["value"],
            )
        )
        extract[name] = entry["value_locator"]
    return outputs, extract or None  # None, not {}: no outputs means no checkpoint.extract at all


def _template_identity(identity: str | None, param_map: dict[str, str]) -> str | None:
    """Replace input literals inside the identity selector's quoted text with {{name}}. Returns
    None if there were none: an identity element that shows no input value doesn't tie the result
    to an input, which is its only job."""
    if identity is None:
        return None
    templated = identity
    for literal in sorted(param_map, key=len, reverse=True):
        if selector_contains_literal(templated, literal):
            templated = templated.replace(literal, f"{{{{{param_map[literal]}}}}}")
    return templated if extract_template_params(templated) else None


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
