"""Tests for record_artifact: turning a DiscoveryRun into a validated CapabilityArtifact.

Includes a test built directly from the shape of the real ParaBank discovery run
(evidence/discovery_1786929361/) - login with two literal credentials, a checkpoint that's a whole
table row (role=cell never resolved on ParaBank's real markup, role=row did), and an empty
extract_log - so the recorder is proven against what discovery actually produced live, not just an
idealized fixture-shaped run.
"""

import pytest
from pydantic import ValidationError

from capability_forge.discovery.agent_loop import DiscoveryRun
from capability_forge.discovery.artifact_recorder import UnrecordableRunError, record_artifact
from capability_forge.schema.artifact import CapabilityArtifact, Checkpoint, LocatorTier, OutputField, StepAction


def locator(strategy="role", value='role=button[name="Login"]', confidence=0.9):
    return LocatorTier(strategy=strategy, value=value, confidence=confidence)


def step(step_id="step_1", action_type="click", locators=None, input_value=None, risk="safe_reversible", description="a step"):
    return StepAction(
        step_id=step_id,
        action_type=action_type,
        locators=locators if locators is not None else [locator()],
        input_value=input_value,
        risk=risk,
        description=description,
    )


def make_run(stop_reason="goal_complete", steps=None, checkpoint="default", goal_description="Look up a balance", **overrides):
    if checkpoint == "default":
        checkpoint = Checkpoint(description="Reached the goal", locator=locator(value='role=cell[name="$100.00"]'), extract=None)
    return DiscoveryRun(
        goal_description=goal_description,
        target_url="https://example.com/app",
        steps=steps if steps is not None else [step()],
        stop_reason=stop_reason,
        checkpoint=checkpoint,
        **overrides,
    )


# --- recordable stop reasons ---------------------------------------------------------------------


def test_goal_complete_run_is_recordable():
    artifact = record_artifact(make_run(stop_reason="goal_complete"), artifact_id="lookup_balance", app_name="parabank")
    assert artifact.artifact_id == "lookup_balance"
    assert artifact.goal_description == "Look up a balance"
    assert artifact.target.base_url == "https://example.com/app"
    assert artifact.target.app_name == "parabank"
    assert artifact.risk_summary == "safe"


def test_business_outcome_run_is_recordable():
    # Just as legitimate and just as checkpoint-verified as a success run - see the module
    # docstring's first design decision.
    artifact = record_artifact(make_run(stop_reason="business_outcome"), artifact_id="check_member", app_name="parabank")
    assert artifact.artifact_id == "check_member"


@pytest.mark.parametrize("stop_reason", ["give_up", "max_steps_exceeded", "timeout_exceeded", "dead_end_detected"])
def test_unverified_stop_reasons_are_not_recordable(stop_reason):
    run = make_run(stop_reason=stop_reason, checkpoint=None)
    with pytest.raises(UnrecordableRunError):
        record_artifact(run, artifact_id="x", app_name="app")


# --- templating literal values into {{param}} placeholders ----------------------------------------


def test_param_map_templates_matching_step_input_values():
    steps = [
        step(step_id="s1", action_type="type", input_value="jdoe", locators=[locator(value='role=textbox[name="User Name:"]')]),
        step(step_id="s2", action_type="type", input_value="hunter2", locators=[locator(value='role=textbox[name="Password:"]')]),
        step(step_id="s3", action_type="click"),
    ]
    run = make_run(steps=steps)
    artifact = record_artifact(run, artifact_id="a", app_name="app", param_map={"jdoe": "username", "hunter2": "password"})

    assert artifact.steps[0].input_value == "{{username}}"
    assert artifact.steps[1].input_value == "{{password}}"
    assert artifact.steps[2].input_value is None  # click has no input_value to touch

    input_names = {p.name for p in artifact.inputs}
    assert input_names == {"username", "password"}
    assert all(p.type == "string" and p.required for p in artifact.inputs)


def test_param_map_only_templates_input_value_not_locators():
    # A literal that happens to also appear inside a locator's value must not be touched -
    # templating never applies to a locator's value, matching the schema's own validator scope.
    steps = [step(step_id="s1", action_type="type", input_value="16896", locators=[locator(value='css=#acct-16896')])]
    run = make_run(steps=steps)
    artifact = record_artifact(run, artifact_id="a", app_name="app", param_map={"16896": "account_id"})
    assert artifact.steps[0].input_value == "{{account_id}}"
    assert artifact.steps[0].locators[0].value == "css=#acct-16896"


def test_param_map_also_templates_goal_description_and_checkpoint_description():
    # Found against the real ParaBank run: a literal credential parameterized out of
    # step.input_value was still sitting in the clear in goal_description and
    # checkpoint.description, both built from the model's own prose. A login credential
    # "parameterized" out of the steps but still readable elsewhere in the same saved file
    # defeats the point of parameterizing it at all.
    run = make_run(
        steps=[step(step_id="s1", action_type="type", input_value="jdoe")],
        goal_description="Log in as jdoe and check the balance.",
        checkpoint=Checkpoint(description="Logged in as jdoe and found the balance.", locator=locator(), extract=None),
    )
    artifact = record_artifact(run, artifact_id="a", app_name="app", param_map={"jdoe": "username"})
    assert artifact.goal_description == "Log in as {{username}} and check the balance."
    assert artifact.checkpoint.description == "Logged in as {{username}} and found the balance."


def test_param_map_also_templates_step_description():
    steps = [step(step_id="s1", action_type="click", description="Click login for jdoe's account")]
    run = make_run(steps=steps)
    artifact = record_artifact(run, artifact_id="a", app_name="app", param_map={"jdoe": "username"})
    assert artifact.steps[0].description == "Click login for {{username}}'s account"


def test_param_map_prefers_longest_literal_match():
    # "168" is a substring of "16896" - the longer literal must win so "16896" doesn't get
    # partially replaced into "{{short}}96".
    steps = [step(step_id="s1", action_type="type", input_value="16896")]
    run = make_run(steps=steps)
    artifact = record_artifact(run, artifact_id="a", app_name="app", param_map={"168": "short", "16896": "full"})
    assert artifact.steps[0].input_value == "{{full}}"


def test_no_param_map_leaves_all_steps_literal():
    steps = [step(step_id="s1", action_type="type", input_value="jdoe")]
    run = make_run(steps=steps)
    artifact = record_artifact(run, artifact_id="a", app_name="app")
    assert artifact.steps[0].input_value == "jdoe"
    assert artifact.inputs == []


def test_input_descriptions_and_types_are_applied_per_param():
    steps = [step(step_id="s1", action_type="type", input_value="42")]
    run = make_run(steps=steps)
    artifact = record_artifact(
        run,
        artifact_id="a",
        app_name="app",
        param_map={"42": "amount"},
        input_types={"amount": "int"},
        input_descriptions={"amount": "The transfer amount."},
    )
    param = artifact.inputs[0]
    assert param.type == "int"
    assert param.description == "The transfer amount."


def test_unmapped_param_gets_a_default_description():
    steps = [step(step_id="s1", action_type="type", input_value="jdoe")]
    run = make_run(steps=steps)
    artifact = record_artifact(run, artifact_id="a", app_name="app", param_map={"jdoe": "username"})
    assert artifact.inputs[0].description == "Value substituted for {{username}}."


# --- outputs / checkpoint.extract are passed through, not inferred --------------------------------


def test_outputs_and_checkpoint_extract_are_attached_when_supplied():
    run = make_run(checkpoint=Checkpoint(description="Found it", locator=locator(value='role=cell[name="$4500.00"]'), extract=None))
    artifact = record_artifact(
        run,
        artifact_id="a",
        app_name="app",
        outputs=[OutputField(name="balance", type="string", description="The account balance.")],
        checkpoint_extract={"balance": "css=.balance-value"},
    )
    assert artifact.outputs[0].name == "balance"
    assert artifact.checkpoint.extract == {"balance": "css=.balance-value"}


def test_mismatched_outputs_and_checkpoint_extract_fail_schema_validation():
    # The recorder doesn't reconcile these itself - the schema's own cross-field validator is what
    # actually enforces consistency, and it must still fire here, not be silently bypassed.
    run = make_run()
    with pytest.raises(ValidationError):
        record_artifact(
            run,
            artifact_id="a",
            app_name="app",
            outputs=[OutputField(name="balance", type="string", description="x")],
            checkpoint_extract=None,
        )


def test_no_outputs_supplied_is_a_valid_empty_result():
    # Honest, not a bug: a run whose extract_log is empty (nothing was ever named during
    # discovery) legitimately produces zero declared outputs - see the real ParaBank test below.
    artifact = record_artifact(make_run(), artifact_id="a", app_name="app")
    assert artifact.outputs == []
    assert artifact.checkpoint.extract is None


# --- risk_summary rollup --------------------------------------------------------------------------


def test_risky_step_rolls_up_to_contains_risky_steps():
    steps = [step(step_id="s1", risk="risky_irreversible")]
    artifact = record_artifact(make_run(steps=steps), artifact_id="a", app_name="app")
    assert artifact.risk_summary == "contains_risky_steps"


# --- built directly from the shape of the real, live ParaBank run --------------------------------


def test_records_the_real_parabank_run_shape():
    # Mirrors evidence/discovery_1786929361/ exactly: two typed credentials, a click, two
    # navigates, a wait, and a checkpoint that's a whole table row (role=cell never resolved on
    # ParaBank's actual markup - role=row did) with an empty extract_log, so this is what the
    # recorder actually has to work with, not an idealized shape.
    steps = [
        step(
            step_id="step_1", action_type="type", input_value="demo_user_16896",
            locators=[locator(value='role=textbox[name=""] >> nth=0', confidence=0.75)],
            description="Enter username into login form",
        ),
        step(
            step_id="step_2", action_type="type", input_value="demo_pw_16896x",
            locators=[locator(value='role=textbox[name=""] >> nth=1', confidence=0.75)],
            description="Enter password into login form",
        ),
        step(step_id="step_3", action_type="click", locators=[locator(value='role=button[name="Log In"]')], description="Submit login form"),
        step(
            step_id="step_4", action_type="navigate", locators=[],
            input_value="https://parabank.parasoft.com/parabank/activity.htm?id=16896",
            description="Navigate directly to account activity/details page for account 16896",
        ),
        step(
            step_id="step_5", action_type="navigate", locators=[],
            input_value="https://parabank.parasoft.com/parabank/overview.htm",
            description="Check accounts overview list",
        ),
        step(step_id="step_6", action_type="wait", locators=[locator(value='role=table[name=""]')], description="Wait for accounts table"),
    ]
    # goal_description and checkpoint.description are the model's own real prose from this run,
    # word for word - both genuinely repeated the literal username in plain text, exactly the leak
    # test_param_map_also_templates_goal_description_and_checkpoint_description guards against.
    checkpoint = Checkpoint(
        description="Logged in as demo_user_16896 and found that account 16896 has a current balance of $423.50 (available amount also $423.50).",
        locator=locator(value='role=row[name="16896 $423.50 $423.50"]', confidence=0.9),
        extract=None,
    )
    run = DiscoveryRun(
        goal_description="Log in to ParaBank with username 'demo_user_16896' and password 'demo_pw_16896x'. Then find and report the current balance of account 16896.",
        target_url="https://parabank.parasoft.com/parabank/index.htm",
        steps=steps,
        stop_reason="goal_complete",
        checkpoint=checkpoint,
        summary="Logged in as demo_user_16896 and found that account 16896 has a current balance of $423.50 (available amount also $423.50).",
        extract_log=[],
    )

    artifact = record_artifact(
        run,
        artifact_id="parabank_check_balance",
        app_name="parabank",
        param_map={"demo_user_16896": "username", "demo_pw_16896x": "password"},
        # 16896 deliberately left unmapped: it also appears inside the checkpoint's own recorded
        # text, which the schema has no templating support for - see the module docstring.
    )

    assert artifact.steps[0].input_value == "{{username}}"
    assert artifact.steps[1].input_value == "{{password}}"
    assert artifact.steps[3].input_value == "https://parabank.parasoft.com/parabank/activity.htm?id=16896"
    assert {p.name for p in artifact.inputs} == {"username", "password"}
    assert artifact.outputs == []
    assert artifact.checkpoint.extract is None
    assert artifact.checkpoint.locator.value == 'role=row[name="16896 $423.50 $423.50"]'
    assert artifact.risk_summary == "safe"
    assert artifact.goal_description == "Log in to ParaBank with username '{{username}}' and password '{{password}}'. Then find and report the current balance of account 16896."
    assert artifact.checkpoint.description == "Logged in as {{username}} and found that account 16896 has a current balance of $423.50 (available amount also $423.50)."

    # The whole point: no credential anywhere in the saved artifact, not just in the one field
    # that was checked by hand originally.
    dumped_text = artifact.model_dump_json()
    assert "demo_user_16896" not in dumped_text
    assert "demo_pw_16896x" not in dumped_text

    # Full round trip, exactly what save()/load() does - the actual artifact this run would
    # produce on disk.
    dumped = artifact.model_dump_json()
    restored = CapabilityArtifact.model_validate_json(dumped)
    assert restored == artifact
