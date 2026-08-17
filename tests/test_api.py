"""Tests for the agent-facing capability API: route behavior against a controlled artifact
registry and a stubbed browser (so no real browser/network is needed for most cases), plus one
real end-to-end invoke against the fixture proving the actual wiring works.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from capability_forge.api.main import create_app, load_artifacts
from capability_forge.replay.engine import ParamValidationError, ReplayResult
from capability_forge.schema.artifact import CapabilityArtifact, Checkpoint, InputParam, LocatorTier, OutputField, ReliabilityInfo, StepAction, TargetSpec


def make_artifact(artifact_id="test_capability", inputs=None, with_output=False, reliability=None):
    outputs = [OutputField(name="balance", type="string", description="x")] if with_output else []
    extract = {"balance": "css=.balance-value"} if with_output else None
    return CapabilityArtifact(
        artifact_id=artifact_id,
        schema_version="1.0",
        target=TargetSpec(base_url="http://127.0.0.1:9999", app_name="x"),
        goal_description="A test capability",
        inputs=inputs if inputs is not None else [InputParam(name="member_id", type="string", required=True, description="x")],
        outputs=outputs,
        steps=[StepAction(step_id="s1", action_type="wait", locators=[LocatorTier(strategy="role", value="role=heading", confidence=0.9)], risk="safe_reversible", description="wait")],
        checkpoint=Checkpoint(description="x", locator=LocatorTier(strategy="role", value="role=heading", confidence=0.9), extract=extract),
        risk_summary="safe",
        expected_outcome_type="success",
        reliability=reliability,
    )


# --- GET /capabilities -----------------------------------------------------------------------


def test_list_capabilities_returns_the_typed_contract_not_internal_steps():
    art = make_artifact(with_output=True)
    client = TestClient(create_app({art.artifact_id: art}))

    response = client.get("/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["artifact_id"] == "test_capability"
    assert body[0]["inputs"][0]["name"] == "member_id"
    assert body[0]["outputs"][0]["name"] == "balance"
    assert "steps" not in body[0]  # internal detail, not part of the agent-facing contract
    assert "checkpoint" not in body[0]


def test_list_capabilities_includes_reliability_when_present():
    from datetime import datetime, timezone

    reliability = ReliabilityInfo(pass_rate=1.0, avg_duration_ms=1234.5, sample_size=5, last_checked=datetime.now(timezone.utc))
    art = make_artifact(reliability=reliability)
    client = TestClient(create_app({art.artifact_id: art}))

    body = client.get("/capabilities").json()

    assert body[0]["reliability"]["pass_rate"] == 1.0
    assert body[0]["reliability"]["sample_size"] == 5


def test_list_capabilities_reliability_is_null_when_never_checked():
    art = make_artifact(reliability=None)
    client = TestClient(create_app({art.artifact_id: art}))

    body = client.get("/capabilities").json()

    assert body[0]["reliability"] is None


def test_list_capabilities_returns_empty_list_for_an_empty_registry():
    client = TestClient(create_app({}))
    assert client.get("/capabilities").json() == []


# --- POST /capabilities/{id}/invoke, stubbed browser ---------------------------------------------


def _stub_browser(monkeypatch, run_side_effect):
    """Stubs sync_playwright/PlaywrightDriver/ReplayEngine at the points api/main.py actually
    imports them, mirroring the same technique test_replay_cli.py already uses - lets invoke's
    own request/response handling be tested without a real browser."""

    class _FakePage:
        pass

    class _FakeBrowser:
        def new_page(self):
            return _FakePage()

        def close(self):
            pass

    class _FakeChromium:
        def launch(self, headless):
            return _FakeBrowser()

    class _FakePlaywrightContext:
        def __enter__(self):
            return SimpleNamespace(chromium=_FakeChromium())

        def __exit__(self, *args):
            return False

    class _FakeReplayEngine:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, artifact, params=None, confirm_risky=False):
            if isinstance(run_side_effect, Exception):
                raise run_side_effect
            return run_side_effect

    monkeypatch.setattr("capability_forge.api.main.sync_playwright", lambda: _FakePlaywrightContext())
    monkeypatch.setattr("capability_forge.api.main.PlaywrightDriver", lambda page: page)
    monkeypatch.setattr("capability_forge.api.main.ReplayEngine", _FakeReplayEngine)


def test_invoke_returns_404_for_an_unknown_artifact_id():
    client = TestClient(create_app({}))
    response = client.post("/capabilities/does_not_exist/invoke", json={"params": {}})
    assert response.status_code == 404


def test_invoke_returns_the_replay_result_on_success(monkeypatch):
    art = make_artifact()
    client = TestClient(create_app({art.artifact_id: art}))
    _stub_browser(monkeypatch, ReplayResult(artifact_id=art.artifact_id, status="success", outputs={"balance": "$4500.00"}))

    response = client.post(f"/capabilities/{art.artifact_id}/invoke", json={"params": {"member_id": "12345"}})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["outputs"] == {"balance": "$4500.00"}


def test_invoke_returns_200_with_a_hard_failure_result_not_an_http_error(monkeypatch):
    # A hard_failure is a well-formed result, not a broken request - the HTTP call itself
    # succeeded in reporting what happened, so it must not surface as a 4xx/5xx.
    art = make_artifact()
    client = TestClient(create_app({art.artifact_id: art}))
    _stub_browser(
        monkeypatch,
        ReplayResult(artifact_id=art.artifact_id, status="hard_failure", outputs={}, failed_step_id="s1", expected_state="x", observed_state="y"),
    )

    response = client.post(f"/capabilities/{art.artifact_id}/invoke", json={"params": {"member_id": "12345"}})

    assert response.status_code == 200
    assert response.json()["status"] == "hard_failure"
    assert response.json()["failed_step_id"] == "s1"


def test_invoke_returns_400_for_a_param_validation_error(monkeypatch):
    art = make_artifact()
    client = TestClient(create_app({art.artifact_id: art}))
    _stub_browser(monkeypatch, ParamValidationError("missing required param(s): ['member_id']"))

    response = client.post(f"/capabilities/{art.artifact_id}/invoke", json={"params": {}})

    assert response.status_code == 400
    assert "member_id" in response.json()["detail"]


def test_invoke_defaults_params_and_confirm_risky_when_omitted(monkeypatch):
    art = make_artifact(inputs=[])  # no required inputs, so an empty body is legitimate
    client = TestClient(create_app({art.artifact_id: art}))
    _stub_browser(monkeypatch, ReplayResult(artifact_id=art.artifact_id, status="success", outputs={}))

    response = client.post(f"/capabilities/{art.artifact_id}/invoke", json={})

    assert response.status_code == 200


# --- load_artifacts() -------------------------------------------------------------------------


def test_load_artifacts_skips_a_schema_invalid_file_rather_than_crashing(tmp_path, capsys):
    (tmp_path / "bad.json").write_text("{}")  # valid JSON, missing every required field
    good = make_artifact()
    good.save(tmp_path / "good.json")

    registry = load_artifacts(tmp_path)

    assert list(registry) == ["test_capability"]
    assert "bad.json" in capsys.readouterr().out


def test_load_artifacts_ignores_non_json_files(tmp_path):
    (tmp_path / ".gitkeep").write_text("")
    registry = load_artifacts(tmp_path)
    assert registry == {}


# --- real end-to-end invoke, over a real browser --------------------------------------------------


@pytest.fixture
def fixture_server():
    from pathlib import Path

    from capability_forge.utils.local_server import serve_directory

    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    with serve_directory(fixtures_dir) as base_url:
        yield base_url


def test_invoke_against_the_real_fixture_over_a_real_browser(fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    art = CapabilityArtifact(
        artifact_id="fixture_capability",
        schema_version="1.0",
        target=TargetSpec(base_url=target, app_name="legacy_bank"),
        goal_description="Look up a member's balance",
        inputs=[InputParam(name="member_id", type="string", required=True, description="x")],
        outputs=[OutputField(name="balance", type="string", description="x")],
        steps=[
            StepAction(step_id="s1", action_type="type", locators=[LocatorTier(strategy="role", value='role=textbox[name="User Name:"]', confidence=0.9)], input_value="jdoe", risk="safe_reversible", description="Enter username"),
            StepAction(step_id="s2", action_type="type", locators=[LocatorTier(strategy="role", value='role=textbox[name="Password:"]', confidence=0.9)], input_value="secret", risk="safe_reversible", description="Enter password"),
            StepAction(step_id="s3", action_type="click", locators=[LocatorTier(strategy="role", value='role=button[name="Login"]', confidence=0.9)], risk="safe_reversible", description="Submit login"),
            StepAction(step_id="s4", action_type="type", locators=[LocatorTier(strategy="role", value='role=textbox[name="Member ID:"]', confidence=0.9)], input_value="{{member_id}}", risk="safe_reversible", description="Enter member id"),
            StepAction(step_id="s5", action_type="click", locators=[LocatorTier(strategy="role", value='role=button[name="Search"]', confidence=0.9)], risk="safe_reversible", description="Search"),
        ],
        checkpoint=Checkpoint(description="Balance shown", locator=LocatorTier(strategy="role", value='role=cell[name="$4500.00"]', confidence=0.9), extract={"balance": "css=.balance-value"}),
        risk_summary="safe",
        expected_outcome_type="success",
    )
    client = TestClient(create_app({art.artifact_id: art}))

    response = client.post(f"/capabilities/{art.artifact_id}/invoke", json={"params": {"member_id": "12345"}})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["outputs"] == {"balance": "$4500.00"}

    listing = client.get("/capabilities").json()
    assert listing[0]["artifact_id"] == "fixture_capability"
