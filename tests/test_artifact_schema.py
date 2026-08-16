"""Tests for the CapabilityArtifact schema: field-level constraints and the cross-field
invariants (risk_summary consistency, template param references, checkpoint/output alignment)
that keep a saved artifact from silently drifting out of sync with itself.
"""

import copy

import pytest
from pydantic import ValidationError

from capability_forge.schema.artifact import (
    CapabilityArtifact,
    Checkpoint,
    InputParam,
    LocatorTier,
    OutputField,
    StepAction,
    TargetSpec,
    extract_template_params,
)


def make_locator(**overrides) -> dict:
    base = {"strategy": "role", "value": "role=button[name='Submit']", "confidence": 0.9}
    base.update(overrides)
    return base


def make_step(**overrides) -> dict:
    base = {
        "step_id": "step_1",
        "action_type": "click",
        "locators": [make_locator()],
        "input_value": None,
        "risk": "safe_reversible",
        "description": "Click the submit button.",
    }
    base.update(overrides)
    return base


def make_artifact_dict(**overrides) -> dict:
    """A minimal, valid artifact payload. Tests mutate a deep copy of this to isolate one field
    of interest per test rather than re-declaring the whole shape each time."""
    base = {
        "artifact_id": "lookup_member_balance",
        "schema_version": "1.0",
        "target": {"base_url": "https://example.com", "app_name": "legacy_bank"},
        "goal_description": "Look up a member's account balance.",
        "inputs": [],
        "outputs": [],
        "steps": [make_step()],
        "checkpoint": {
            "description": "Balance page loaded.",
            "locator": make_locator(strategy="css", value=".balance-value"),
            "extract": None,
        },
        "risk_summary": "safe",
    }
    base.update(overrides)
    return base


def build(**overrides) -> CapabilityArtifact:
    return CapabilityArtifact.model_validate(make_artifact_dict(**overrides))


# --- happy path -------------------------------------------------------------------------------


def test_minimal_valid_artifact_constructs():
    artifact = build()
    assert artifact.artifact_id == "lookup_member_balance"
    assert artifact.risk_summary == "safe"
    assert artifact.reliability is None
    # created_at is defaulted, not required by the caller.
    assert artifact.created_at is not None


def test_artifact_with_inputs_outputs_and_templating_constructs():
    payload = make_artifact_dict(
        inputs=[{"name": "member_id", "type": "string", "required": True, "description": "Member id."}],
        outputs=[{"name": "balance", "type": "float", "description": "Current balance."}],
        steps=[
            make_step(
                action_type="type",
                input_value="{{member_id}}",
            )
        ],
        checkpoint={
            "description": "Balance page loaded.",
            "locator": make_locator(strategy="css", value=".balance-value"),
            "extract": {"balance": "css=.balance-value"},
        },
    )
    artifact = CapabilityArtifact.model_validate(payload)
    assert artifact.inputs[0].name == "member_id"
    assert artifact.outputs[0].name == "balance"


# --- LocatorTier --------------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_locator_confidence_out_of_range_rejected(confidence):
    with pytest.raises(ValidationError):
        LocatorTier.model_validate(make_locator(confidence=confidence))


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_locator_confidence_boundary_values_accepted(confidence):
    LocatorTier.model_validate(make_locator(confidence=confidence))


def test_locator_empty_value_rejected():
    with pytest.raises(ValidationError):
        LocatorTier.model_validate(make_locator(value=""))


def test_locator_invalid_strategy_rejected():
    with pytest.raises(ValidationError):
        LocatorTier.model_validate(make_locator(strategy="xpath"))


# --- StepAction ----------------------------------------------------------------------------------


def test_step_requires_at_least_one_locator():
    with pytest.raises(ValidationError):
        StepAction.model_validate(make_step(locators=[]))


@pytest.mark.parametrize("action_type", ["type", "select", "navigate"])
def test_type_select_and_navigate_require_input_value(action_type):
    with pytest.raises(ValidationError):
        StepAction.model_validate(make_step(action_type=action_type, input_value=None))


@pytest.mark.parametrize("action_type", ["type", "select", "navigate"])
def test_type_select_and_navigate_reject_empty_string_input_value(action_type):
    with pytest.raises(ValidationError):
        StepAction.model_validate(make_step(action_type=action_type, input_value=""))


@pytest.mark.parametrize("action_type", ["click", "wait"])
def test_other_action_types_allow_missing_input_value(action_type):
    step = StepAction.model_validate(make_step(action_type=action_type, input_value=None))
    assert step.input_value is None


def test_extract_is_not_a_valid_action_type():
    # Extraction only happens at the checkpoint; a mid-flow "extract" step has nowhere to put its
    # result (no output_key field), so it was dropped from the action_type literal entirely.
    with pytest.raises(ValidationError):
        StepAction.model_validate(make_step(action_type="extract"))


def test_navigate_with_destination_accepted():
    step = StepAction.model_validate(
        make_step(action_type="navigate", input_value="/accounts/{{member_id}}")
    )
    assert step.input_value == "/accounts/{{member_id}}"


def test_step_locators_are_tried_in_declared_order():
    payload = make_step(
        locators=[
            make_locator(strategy="role", confidence=0.9),
            make_locator(strategy="css", value=".submit", confidence=0.5),
        ]
    )
    step = StepAction.model_validate(payload)
    assert [loc.strategy for loc in step.locators] == ["role", "css"]


# --- TargetSpec -----------------------------------------------------------------------------------


@pytest.mark.parametrize("base_url", ["ftp://example.com", "example.com", "www.example.com"])
def test_target_spec_rejects_url_without_http_scheme(base_url):
    with pytest.raises(ValidationError):
        TargetSpec.model_validate({"base_url": base_url, "app_name": "legacy_bank"})


@pytest.mark.parametrize("base_url", ["http://example.com", "https://example.com"])
def test_target_spec_accepts_http_and_https(base_url):
    TargetSpec.model_validate({"base_url": base_url, "app_name": "legacy_bank"})


def test_target_spec_tenant_overrides_defaults_to_none():
    spec = TargetSpec.model_validate({"base_url": "https://example.com", "app_name": "legacy_bank"})
    assert spec.tenant_overrides is None


def test_target_spec_accepts_tenant_overrides():
    spec = TargetSpec.model_validate(
        {
            "base_url": "https://example.com",
            "app_name": "legacy_bank",
            "tenant_overrides": {"login_route": "/tenant2/login"},
        }
    )
    assert spec.tenant_overrides == {"login_route": "/tenant2/login"}


# --- artifact_id / schema_version format -------------------------------------------------------


@pytest.mark.parametrize("artifact_id", ["Lookup_Balance", "1lookup", "lookup-balance", "lookup balance", ""])
def test_invalid_artifact_id_rejected(artifact_id):
    with pytest.raises(ValidationError):
        build(artifact_id=artifact_id)


@pytest.mark.parametrize("artifact_id", ["lookup_member_balance", "a", "a1", "lookup_2"])
def test_valid_artifact_id_accepted(artifact_id):
    artifact = build(artifact_id=artifact_id)
    assert artifact.artifact_id == artifact_id


@pytest.mark.parametrize("schema_version", ["1", "v1.0", "1.0.0", ""])
def test_invalid_schema_version_rejected(schema_version):
    with pytest.raises(ValidationError):
        build(schema_version=schema_version)


def test_valid_schema_version_accepted():
    artifact = build(schema_version="2.3")
    assert artifact.schema_version == "2.3"


# --- cross-field: step_id / param / output name uniqueness -------------------------------------


def test_duplicate_step_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate step_id"):
        build(steps=[make_step(step_id="step_1"), make_step(step_id="step_1")])


def test_duplicate_input_names_rejected():
    dup_input = {"name": "member_id", "type": "string", "required": True, "description": "id"}
    with pytest.raises(ValidationError, match="duplicate input param name"):
        build(inputs=[dup_input, copy.deepcopy(dup_input)])


def test_duplicate_output_names_rejected():
    dup_output = {"name": "balance", "type": "float", "description": "balance"}
    with pytest.raises(ValidationError, match="duplicate output field name"):
        build(
            outputs=[dup_output, copy.deepcopy(dup_output)],
            checkpoint={
                "description": "loaded",
                "locator": make_locator(),
                "extract": {"balance": "css=.balance-value"},
            },
        )


# --- cross-field: risk_summary consistency ------------------------------------------------------


def test_risk_summary_safe_with_risky_step_rejected():
    with pytest.raises(ValidationError, match="risk_summary"):
        build(
            steps=[make_step(risk="risky_irreversible")],
            risk_summary="safe",
        )


def test_risk_summary_contains_risky_with_no_risky_step_rejected():
    with pytest.raises(ValidationError, match="risk_summary"):
        build(
            steps=[make_step(risk="safe_reversible")],
            risk_summary="contains_risky_steps",
        )


def test_risk_summary_contains_risky_matches_risky_step():
    artifact = build(
        steps=[make_step(risk="risky_irreversible")],
        risk_summary="contains_risky_steps",
    )
    assert artifact.risk_summary == "contains_risky_steps"


def test_risk_summary_safe_with_mixed_steps_one_risky_rejected():
    # even one risky step among several safe ones must flip the rollup.
    with pytest.raises(ValidationError, match="risk_summary"):
        build(
            steps=[
                make_step(step_id="step_1", risk="safe_reversible"),
                make_step(step_id="step_2", risk="risky_irreversible"),
            ],
            risk_summary="safe",
        )


# --- cross-field: template params must reference declared inputs -------------------------------


def test_template_param_referencing_undeclared_input_rejected():
    with pytest.raises(ValidationError, match="undeclared input param"):
        build(
            steps=[make_step(action_type="type", input_value="{{member_id}}")],
            inputs=[],
        )


def test_template_param_referencing_declared_input_accepted():
    artifact = build(
        steps=[make_step(action_type="type", input_value="{{member_id}}")],
        inputs=[{"name": "member_id", "type": "string", "required": True, "description": "id"}],
    )
    assert artifact.inputs[0].name == "member_id"


def test_step_with_multiple_template_params_all_must_be_declared():
    with pytest.raises(ValidationError, match="undeclared input param"):
        build(
            steps=[make_step(action_type="type", input_value="{{first_name}} {{last_name}}")],
            inputs=[{"name": "first_name", "type": "string", "required": True, "description": "x"}],
        )


def test_extract_template_params_helper():
    assert extract_template_params("{{member_id}}") == {"member_id"}
    assert extract_template_params("{{first}} {{last}}") == {"first", "last"}
    assert extract_template_params("no params here") == set()
    assert extract_template_params(None) == set()
    assert extract_template_params("") == set()


# --- cross-field: checkpoint.extract must match declared outputs exactly -----------------------


def test_checkpoint_extract_missing_declared_output_rejected():
    with pytest.raises(ValidationError, match="checkpoint.extract"):
        build(
            outputs=[{"name": "balance", "type": "float", "description": "balance"}],
            checkpoint={"description": "loaded", "locator": make_locator(), "extract": None},
        )


def test_checkpoint_extract_extra_undeclared_key_rejected():
    with pytest.raises(ValidationError, match="checkpoint.extract"):
        build(
            outputs=[],
            checkpoint={
                "description": "loaded",
                "locator": make_locator(),
                "extract": {"balance": "css=.balance-value"},
            },
        )


def test_checkpoint_extract_exact_match_accepted():
    artifact = build(
        outputs=[{"name": "balance", "type": "float", "description": "balance"}],
        checkpoint={
            "description": "loaded",
            "locator": make_locator(),
            "extract": {"balance": "css=.balance-value"},
        },
    )
    assert artifact.checkpoint.extract == {"balance": "css=.balance-value"}


def test_no_outputs_and_no_extract_accepted():
    artifact = build(outputs=[], checkpoint={"description": "loaded", "locator": make_locator(), "extract": None})
    assert artifact.outputs == []
    assert artifact.checkpoint.extract is None


# --- structural strictness: unknown fields rejected everywhere ---------------------------------


def test_unknown_top_level_field_rejected():
    payload = make_artifact_dict()
    payload["unexpected_field"] = "surprise"
    with pytest.raises(ValidationError):
        CapabilityArtifact.model_validate(payload)


def test_unknown_nested_field_rejected():
    payload = make_artifact_dict()
    payload["steps"][0]["unexpected_field"] = "surprise"
    with pytest.raises(ValidationError):
        CapabilityArtifact.model_validate(payload)


def test_steps_cannot_be_empty():
    with pytest.raises(ValidationError):
        build(steps=[])


# --- reliability info ------------------------------------------------------------------------


def test_reliability_info_optional_and_defaults_to_none():
    artifact = build()
    assert artifact.reliability is None


def test_reliability_info_accepts_valid_payload():
    artifact = build(
        reliability={
            "pass_rate": 0.8,
            "avg_duration_ms": 1234.5,
            "sample_size": 5,
            "last_checked": "2026-08-15T00:00:00Z",
        }
    )
    assert artifact.reliability.pass_rate == 0.8
    assert artifact.reliability.sample_size == 5


@pytest.mark.parametrize("pass_rate", [-0.1, 1.1])
def test_reliability_info_pass_rate_out_of_range_rejected(pass_rate):
    with pytest.raises(ValidationError):
        build(
            reliability={
                "pass_rate": pass_rate,
                "avg_duration_ms": 100.0,
                "sample_size": 5,
                "last_checked": "2026-08-15T00:00:00Z",
            }
        )


# --- save/load round-trip via disk -------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    original = build(
        inputs=[{"name": "member_id", "type": "string", "required": True, "description": "id"}],
        outputs=[{"name": "balance", "type": "float", "description": "balance"}],
        steps=[make_step(action_type="type", input_value="{{member_id}}")],
        checkpoint={
            "description": "loaded",
            "locator": make_locator(),
            "extract": {"balance": "css=.balance-value"},
        },
    )
    path = tmp_path / "lookup_member_balance.json"
    original.save(path)

    loaded = CapabilityArtifact.load(path)

    assert loaded == original


def test_saved_file_is_valid_json_with_expected_keys(tmp_path):
    artifact = build()
    path = tmp_path / "artifact.json"
    artifact.save(path)

    import json

    data = json.loads(path.read_text())
    assert data["artifact_id"] == "lookup_member_balance"
    assert data["risk_summary"] == "safe"
