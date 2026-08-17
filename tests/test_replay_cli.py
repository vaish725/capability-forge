"""Tests for the replay CLI's argument parsing and its offline-checkable guards (a missing/invalid
artifact path, malformed --params JSON). The actual replay run this CLI triggers needs a real
browser, so it isn't exercised here beyond a stubbed-browser integration test - see
ReplayEngine's own tests for the execution mechanics, run against the real fixture.
"""

from types import SimpleNamespace

import pytest

from capability_forge.replay.__main__ import build_arg_parser, main
from capability_forge.replay.engine import ReplayResult
from capability_forge.schema.artifact import CapabilityArtifact, Checkpoint, InputParam, LocatorTier, StepAction, TargetSpec


def save_minimal_artifact(path, inputs=None):
    """A schema-valid artifact with exactly the one step CapabilityArtifact requires - just
    enough shape for the CLI's own argument/error handling to be tested without needing a real
    recorded run."""
    CapabilityArtifact(
        artifact_id="a", schema_version="1.0",
        target=TargetSpec(base_url="http://127.0.0.1:9999", app_name="x"),
        goal_description="g", inputs=inputs or [], outputs=[],
        steps=[StepAction(step_id="s1", action_type="wait", locators=[LocatorTier(strategy="role", value="role=heading", confidence=0.9)], risk="safe_reversible", description="wait for the page")],
        checkpoint=Checkpoint(description="x", locator=LocatorTier(strategy="role", value="role=heading", confidence=0.9), extract=None),
        risk_summary="safe", expected_outcome_type="success",
    ).save(path)


def test_artifact_is_required():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parses_artifact_and_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(["--artifact", "artifacts/x.json"])
    assert args.artifact == "artifacts/x.json"
    assert args.params == "{}"
    assert args.headless is False
    assert args.confirm_risky is False
    assert args.no_escalation is False
    assert args.no_evidence is False


def test_parses_optional_flags():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--artifact", "a.json",
            "--params", '{"member_id": "12345"}',
            "--headless", "--confirm-risky", "--no-escalation", "--no-evidence",
        ]
    )
    assert args.params == '{"member_id": "12345"}'
    assert args.headless is True
    assert args.confirm_risky is True
    assert args.no_escalation is True
    assert args.no_evidence is True


def test_main_reports_a_missing_artifact_file_cleanly(capsys):
    exit_code = main(["--artifact", "artifacts/does_not_exist.json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Could not load artifact" in captured.err


def test_main_reports_a_schema_invalid_artifact_file_cleanly(tmp_path, capsys):
    bad_artifact = tmp_path / "bad.json"
    bad_artifact.write_text("{}")  # valid JSON, but missing every required field
    exit_code = main(["--artifact", str(bad_artifact)])
    assert exit_code == 1
    assert "Could not load artifact" in capsys.readouterr().err


def test_main_reports_malformed_params_json_cleanly(tmp_path, capsys):
    artifact_path = tmp_path / "a.json"
    save_minimal_artifact(artifact_path)

    exit_code = main(["--artifact", str(artifact_path), "--params", "not json"])
    assert exit_code == 1
    assert "--params is not valid JSON" in capsys.readouterr().err


def test_main_reports_non_object_params_cleanly(tmp_path, capsys):
    artifact_path = tmp_path / "a.json"
    save_minimal_artifact(artifact_path)

    exit_code = main(["--artifact", str(artifact_path), "--params", "[1, 2, 3]"])
    assert exit_code == 1
    assert "--params must be a JSON object" in capsys.readouterr().err


def test_main_reports_param_validation_errors_cleanly(monkeypatch, tmp_path, capsys):
    # Stubs sync_playwright (browser launch) at the point replay/__main__.py actually imports it,
    # and ReplayEngine.run() to raise the exact error a real artifact/params mismatch would - lets
    # this test cover the CLI's own error-to-exit-code handling without a real browser or a real
    # artifact recorded against the fixture.
    artifact_path = tmp_path / "a.json"
    save_minimal_artifact(artifact_path, inputs=[InputParam(name="member_id", type="string", required=True, description="x")])

    monkeypatch.setattr("capability_forge.utils.evidence.DEFAULT_EVIDENCE_ROOT", tmp_path)

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

    from capability_forge.replay.engine import ParamValidationError

    class _FakeReplayEngine:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, artifact, params=None, confirm_risky=False):
            raise ParamValidationError("missing required param(s): ['member_id']")

    monkeypatch.setattr("capability_forge.replay.__main__.sync_playwright", lambda: _FakePlaywrightContext())
    monkeypatch.setattr("capability_forge.replay.__main__.PlaywrightDriver", lambda page: page)
    monkeypatch.setattr("capability_forge.replay.__main__.ReplayEngine", _FakeReplayEngine)

    exit_code = main(["--artifact", str(artifact_path)])  # no --params, but member_id is required

    assert exit_code == 1
    assert "Param validation failed" in capsys.readouterr().err


def test_main_exits_nonzero_on_hard_failure_and_writes_evidence(monkeypatch, tmp_path, capsys):
    artifact_path = tmp_path / "a.json"
    save_minimal_artifact(artifact_path)

    monkeypatch.setattr("capability_forge.utils.evidence.DEFAULT_EVIDENCE_ROOT", tmp_path)

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
        def __init__(self, driver, guardrail, evidence_writer=None, escalation_manager=None):
            self.evidence_writer = evidence_writer
            self.escalation_manager = escalation_manager

        def run(self, artifact, params=None, confirm_risky=False):
            return ReplayResult(artifact_id="a", status="hard_failure", outputs={}, failed_step_id="s1", expected_state="x", observed_state="y")

    monkeypatch.setattr("capability_forge.replay.__main__.sync_playwright", lambda: _FakePlaywrightContext())
    monkeypatch.setattr("capability_forge.replay.__main__.PlaywrightDriver", lambda page: page)
    monkeypatch.setattr("capability_forge.replay.__main__.ReplayEngine", _FakeReplayEngine)

    exit_code = main(["--artifact", str(artifact_path)])

    assert exit_code == 1
    printed = capsys.readouterr().out
    assert "Status: hard_failure" in printed
    assert "Failed step: s1" in printed
    assert "Evidence written to" in printed  # evidence is on by default


def test_no_evidence_and_no_escalation_flags_are_honored(monkeypatch, tmp_path, capsys):
    artifact_path = tmp_path / "a.json"
    save_minimal_artifact(artifact_path)

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

    captured_kwargs = {}

    class _FakeReplayEngine:
        def __init__(self, driver, guardrail, evidence_writer=None, escalation_manager=None):
            captured_kwargs["evidence_writer"] = evidence_writer
            captured_kwargs["escalation_manager"] = escalation_manager

        def run(self, artifact, params=None, confirm_risky=False):
            return ReplayResult(artifact_id="a", status="success", outputs={})

    monkeypatch.setattr("capability_forge.replay.__main__.sync_playwright", lambda: _FakePlaywrightContext())
    monkeypatch.setattr("capability_forge.replay.__main__.PlaywrightDriver", lambda page: page)
    monkeypatch.setattr("capability_forge.replay.__main__.ReplayEngine", _FakeReplayEngine)

    exit_code = main(["--artifact", str(artifact_path), "--no-evidence", "--no-escalation"])

    assert exit_code == 0
    assert captured_kwargs["evidence_writer"] is None
    assert captured_kwargs["escalation_manager"] is None
    assert "Evidence written to" not in capsys.readouterr().out
