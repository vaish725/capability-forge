"""CapabilityArtifact schema.

The typed, versioned contract produced by discovery and consumed by replay. Every model here is
frozen to its declared shape (`extra="forbid"`) because a malformed artifact should fail loudly at
load time, not silently mis-execute against a live UI.

Beyond the field shapes, a handful of cross-field invariants are enforced so the artifact can never
be saved (or loaded) in a state that contradicts its own design rationale:
  - `risk_summary` must agree with whether any step is actually `risky_irreversible` - it is a
    rollup of the per-step `risk` field, never an independent claim.
  - `{{param}}` placeholders inside a step's `input_value` must reference a declared input, so a
    replay never discovers a missing param mid-run instead of at artifact-load time.
  - `checkpoint.extract` is where declared outputs are actually read from the page, so its keys
    must exactly match the declared `outputs` names - otherwise "what the checkpoint returns" and
    "what the artifact promises to return" could silently drift apart. This is also why
    StepAction.action_type has no "extract" option: a mid-flow extract would have nowhere to put
    its result, so all extraction is pushed to this one place.
  - a `navigate` step must carry a non-empty `input_value` (its destination). The implicit
    navigation to `target.base_url` happens once, before step 1, and is not itself a StepAction -
    every StepAction with action_type="navigate" is an explicit, additional hop and needs to say
    where it's going. `navigate` is also the one action type allowed to omit `locators` entirely -
    it targets a URL, not a page element, so there is nothing to find a locator for.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Matches "{{param_name}}" style placeholders used inside StepAction.input_value.
# Shared as a module-level constant so the replay engine substitutes against the exact same
# pattern this schema validates against, instead of maintaining a second copy.
TEMPLATE_PARAM_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def extract_template_params(value: str | None) -> set[str]:
    """Return the set of {{param}} names referenced inside a step's input_value, if any."""
    if not value:
        return set()
    return set(TEMPLATE_PARAM_PATTERN.findall(value))


def render_template(value: str | None, params: dict[str, str]) -> str | None:
    """Substitute {{param}} placeholders in value with the given params. The counterpart to
    extract_template_params: that finds the names, this fills them in. Used by a Surface Driver
    at replay time to turn a step's raw input_value into the literal text/URL to act on. Raises
    KeyError if value references a param not present in params - callers should already have
    validated referenced params against the artifact's declared inputs (the schema's own
    cross-field check does this at artifact-load time), so a KeyError here means that check was
    skipped, not that the param is legitimately optional.
    """
    if value is None:
        return None

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return params[name]

    return TEMPLATE_PARAM_PATTERN.sub(_replace, value)


class LocatorTier(BaseModel):
    """One way to find an element, tagged with the strategy used and a discovery-time confidence
    score. Replay tries a step's locators in list order, falling through tiers on failure."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["role", "css", "coordinate"]
    value: str = Field(min_length=1)  # e.g. "role=button[name='Submit']" or "button.submit"
    confidence: float = Field(ge=0.0, le=1.0)  # heuristic score assigned at discovery time


class StepAction(BaseModel):
    """A single recorded action: what to do, where to find the element (in fallback order), and
    the risk tier it was frozen at during discovery.

    No "extract" action_type: data extraction only happens at the checkpoint (see
    Checkpoint.extract), never mid-flow. A mid-flow extract step would have nowhere to put its
    result today (StepAction has no output_key), so keeping extraction to a single place means
    success-verification and output-extraction can never drift apart into two mechanisms."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    action_type: Literal["click", "type", "navigate", "wait", "select"]
    # No Field(min_length=1) here: "navigate" has no element to locate (it's a direct URL
    # navigation via input_value) and is the one action type allowed an empty list - enforced
    # below instead of at the field level, since the rule depends on action_type.
    locators: list[LocatorTier] = Field(default_factory=list)
    input_value: str | None = None  # e.g. text to type; may reference a param via "{{member_id}}"
    risk: Literal["safe_reversible", "risky_irreversible"]
    description: str = Field(min_length=1)  # human-readable, for review/debugging

    @model_validator(mode="after")
    def _navigation_and_input_actions_require_a_value(self) -> "StepAction":
        # "type"/"select" are meaningless without something to type or select. "navigate" is
        # meaningless without a destination: the replay engine only navigates to target.base_url
        # implicitly, once, before step 1 - any explicit navigate step in the flow is going
        # somewhere else and must say where via input_value (a path/URL, optionally templated).
        needs_value = {"type", "select", "navigate"}
        if self.action_type in needs_value and not self.input_value:
            raise ValueError(f"action_type={self.action_type!r} requires a non-empty input_value")
        return self

    @model_validator(mode="after")
    def _only_navigate_may_omit_locators(self) -> "StepAction":
        # click/type/select/wait all target a specific element and are meaningless with nothing
        # to find it by; navigate targets a URL, not an element, so it's the one action type that
        # may have an empty locators list.
        if self.action_type != "navigate" and len(self.locators) == 0:
            raise ValueError(f"action_type={self.action_type!r} requires at least one locator")
        return self


# The set of valid action types, derived from StepAction's own field rather than duplicated as a
# separate literal list. Anything that needs to validate an action type against this taxonomy
# (e.g. the guardrail allowlist config) imports this constant instead of hardcoding the list, so
# the two can never drift out of sync the way action_type and the checkpoint invariant briefly did.
ACTION_TYPES: tuple[str, ...] = get_args(StepAction.model_fields["action_type"].annotation)


class Checkpoint(BaseModel):
    """The element/state that must be present for a run to count as having reached its goal, plus
    the named outputs read from that same state."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    locator: LocatorTier
    extract: dict[str, str] | None = None  # named outputs, e.g. {"balance": "css=.balance-value"}


class InputParam(BaseModel):
    """A typed input the artifact accepts at replay time, referenced by steps via {{name}}."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: Literal["string", "int", "float", "bool"]
    required: bool
    description: str = Field(min_length=1)


class OutputField(BaseModel):
    """A typed output the artifact promises to return, sourced from checkpoint.extract."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: Literal["string", "int", "float", "bool"]
    description: str = Field(min_length=1)


class TargetSpec(BaseModel):
    """Where this artifact runs. Separates routing (which app/tenant instance) from the flow
    definition (the steps), so a base artifact plus a small per-tenant override map can cover
    multiple tenants of the same underlying vendor product without a full re-recording."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    app_name: str = Field(min_length=1)
    tenant_overrides: dict[str, str] | None = None

    @field_validator("base_url")
    @classmethod
    def _base_url_must_have_scheme(cls, value: str) -> str:
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {value!r}")
        return value


class ReliabilityInfo(BaseModel):
    """Aggregate stability signal from replaying the same artifact + params N times. Absent until
    the multi-run stability check has been run at least once."""

    model_config = ConfigDict(extra="forbid")

    pass_rate: float = Field(ge=0.0, le=1.0)
    avg_duration_ms: float = Field(ge=0.0)
    sample_size: int = Field(ge=1)  # how many runs pass_rate/avg_duration_ms were computed over
    last_checked: datetime


class CapabilityArtifact(BaseModel):
    """A surface-agnostic, replayable description of how to accomplish one goal: the steps, the
    inputs/outputs contract, and the checkpoint that proves success."""

    model_config = ConfigDict(extra="forbid")

    # Lowercase snake_case identifier: used as both a lookup key and a filename on disk.
    artifact_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    schema_version: str = Field(pattern=r"^\d+\.\d+$")  # e.g. "1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target: TargetSpec
    goal_description: str = Field(min_length=1)  # the original NL goal this was recorded from
    inputs: list[InputParam] = Field(default_factory=list)
    outputs: list[OutputField] = Field(default_factory=list)
    steps: list[StepAction] = Field(min_length=1)
    checkpoint: Checkpoint
    risk_summary: Literal["safe", "contains_risky_steps"]
    reliability: ReliabilityInfo | None = None  # populated by the multi-run stability check

    @model_validator(mode="after")
    def _step_ids_are_unique(self) -> "CapabilityArtifact":
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            duplicates = {sid for sid in step_ids if step_ids.count(sid) > 1}
            raise ValueError(f"duplicate step_id(s): {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def _input_and_output_names_are_unique(self) -> "CapabilityArtifact":
        input_names = [param.name for param in self.inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError(f"duplicate input param name(s): {sorted(set(n for n in input_names if input_names.count(n) > 1))}")
        output_names = [field.name for field in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"duplicate output field name(s): {sorted(set(n for n in output_names if output_names.count(n) > 1))}")
        return self

    @model_validator(mode="after")
    def _risk_summary_matches_steps(self) -> "CapabilityArtifact":
        # risk_summary is a rollup of per-step risk, never an independent claim - it must never
        # drift out of sync with what the steps themselves say.
        has_risky_step = any(step.risk == "risky_irreversible" for step in self.steps)
        expected = "contains_risky_steps" if has_risky_step else "safe"
        if self.risk_summary != expected:
            raise ValueError(
                f"risk_summary={self.risk_summary!r} does not match steps "
                f"(expected {expected!r} given the recorded per-step risk tiers)"
            )
        return self

    @model_validator(mode="after")
    def _template_params_reference_declared_inputs(self) -> "CapabilityArtifact":
        declared = {param.name for param in self.inputs}
        for step in self.steps:
            referenced = extract_template_params(step.input_value)
            undeclared = referenced - declared
            if undeclared:
                raise ValueError(
                    f"step {step.step_id!r} references undeclared input param(s): {sorted(undeclared)}"
                )
        return self

    @model_validator(mode="after")
    def _checkpoint_extract_matches_declared_outputs(self) -> "CapabilityArtifact":
        declared_outputs = {field.name for field in self.outputs}
        extract_keys = set(self.checkpoint.extract or {})
        if declared_outputs != extract_keys:
            raise ValueError(
                "checkpoint.extract keys must exactly match declared outputs: "
                f"declared={sorted(declared_outputs)}, extract={sorted(extract_keys)}"
            )
        return self

    def save(self, path: str | Path) -> None:
        """Write this artifact to disk as pretty-printed, schema-validated JSON."""
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "CapabilityArtifact":
        """Read and validate an artifact from disk."""
        return cls.model_validate_json(Path(path).read_text())
