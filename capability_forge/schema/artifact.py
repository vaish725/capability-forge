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
    "what the artifact promises to return" could silently drift apart.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

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


class LocatorTier(BaseModel):
    """One way to find an element, tagged with the strategy used and a discovery-time confidence
    score. Replay tries a step's locators in list order, falling through tiers on failure."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["role", "css", "coordinate"]
    value: str = Field(min_length=1)  # e.g. "role=button[name='Submit']" or "button.submit"
    confidence: float = Field(ge=0.0, le=1.0)  # heuristic score assigned at discovery time


class StepAction(BaseModel):
    """A single recorded action: what to do, where to find the element (in fallback order), and
    the risk tier it was frozen at during discovery."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    action_type: Literal["click", "type", "navigate", "wait", "extract", "select"]
    locators: list[LocatorTier] = Field(min_length=1)  # ordered, tried in sequence at replay time
    input_value: str | None = None  # e.g. text to type; may reference a param via "{{member_id}}"
    risk: Literal["safe_reversible", "risky_irreversible"]
    description: str = Field(min_length=1)  # human-readable, for review/debugging

    @model_validator(mode="after")
    def _typed_or_selected_value_must_be_present(self) -> "StepAction":
        # "type" and "select" are meaningless without something to type or select.
        if self.action_type in ("type", "select") and not self.input_value:
            raise ValueError(f"action_type={self.action_type!r} requires a non-empty input_value")
        return self


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
