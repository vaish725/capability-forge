"""Tests for the discover CLI's argument parsing and its offline-checkable guard (missing
ANTHROPIC_API_KEY). The actual discovery run this CLI triggers needs a real API key and a real
browser, so it isn't exercised here - see AgentLoop's own tests for the loop mechanics, tested
with a scripted client instead.
"""

from types import SimpleNamespace

import anthropic
import pytest

from capability_forge.discover import build_arg_parser, main, parse_param_map, save_artifact
from capability_forge.discovery.agent_loop import DiscoveryRun
from capability_forge.schema.artifact import CapabilityArtifact, Checkpoint, LocatorTier, StepAction


def test_goal_and_target_are_required():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parses_goal_and_target():
    parser = build_arg_parser()
    args = parser.parse_args(["--goal", "Find the balance", "--target", "https://example.com"])
    assert args.goal == "Find the balance"
    assert args.target == "https://example.com"
    assert args.headless is False
    assert args.max_steps is None
    assert args.timeout_seconds is None


def test_parses_optional_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--goal", "g", "--target", "t",
            "--headless", "--max-steps", "5", "--timeout-seconds", "30", "--no-escalation",
        ]
    )
    assert args.headless is True
    assert args.max_steps == 5
    assert args.timeout_seconds == 30.0
    assert args.no_escalation is True


def test_escalation_defaults_to_enabled():
    parser = build_arg_parser()
    args = parser.parse_args(["--goal", "g", "--target", "t"])
    assert args.no_escalation is False


def test_main_exits_cleanly_without_an_api_key(monkeypatch, capsys):
    # Also stub load_dotenv() to a no-op: if a real .env file with a real key exists (which it
    # will once discovery mode is actually used), main()'s own load_dotenv() call would otherwise
    # re-populate ANTHROPIC_API_KEY right after this deletes it, making the test's pass/fail
    # depend on whether a .env happens to exist rather than on the guard logic being tested.
    monkeypatch.setattr("capability_forge.discover.load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exit_code = main(["--goal", "g", "--target", "https://example.com"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY is not set" in captured.err


def test_main_reports_anthropic_api_errors_cleanly(monkeypatch, capsys, tmp_path):
    # Confirms the API-error path exits cleanly (message on stderr, exit code 1) rather than
    # letting a raw SDK traceback surface - without needing a real browser or a real API call.
    # Stubs sync_playwright (browser launch) and AgentLoop (the thing that would raise) at the
    # points discover.py actually imports them.
    monkeypatch.setattr("capability_forge.discover.load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-this-test")
    # main() also constructs a real EvidenceWriter before touching the browser at all - redirect
    # its root to a pytest tmp_path so this test doesn't leave a directory under the repo's real
    # evidence/ folder behind every time it runs.
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

    class _FakeAgentLoop:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, goal, target):
            raise anthropic.AuthenticationError(
                message="invalid key",
                response=SimpleNamespace(status_code=401, headers={}, request=SimpleNamespace()),
                body=None,
            )

    monkeypatch.setattr("capability_forge.discover.sync_playwright", lambda: _FakePlaywrightContext())
    monkeypatch.setattr("capability_forge.discover.PlaywrightDriver", lambda page: page)
    monkeypatch.setattr("capability_forge.discover.AgentLoop", _FakeAgentLoop)

    exit_code = main(["--goal", "g", "--target", "http://127.0.0.1:9999/x"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Anthropic API error" in captured.err



# --- recording: --artifact-id / --param ------------------------------------------------------------


def test_parse_param_map_maps_literal_values_to_param_names():
    assert parse_param_map(["member_id=12345", "note=a=b"]) == {"12345": "member_id", "a=b": "note"}


@pytest.mark.parametrize("bad", ["member_id", "=12345", "member_id="])
def test_parse_param_map_rejects_malformed_flags(bad):
    with pytest.raises(ValueError, match="NAME=VALUE"):
        parse_param_map([bad])


def test_artifact_id_and_params_default_to_not_recording():
    args = build_arg_parser().parse_args(["--goal", "g", "--target", "t"])
    assert args.artifact_id is None
    assert args.param == []


def _completed_run(checkpoint_name="Current Balance:"):
    return DiscoveryRun(
        goal_description="Look up the balance for member 12345",
        target_url="http://127.0.0.1:8000/hostile_legacy_page.html",
        steps=[
            StepAction(
                step_id="step_1", action_type="type", input_value="12345", risk="safe_reversible", description="Enter 12345",
                locators=[LocatorTier(strategy="role", value='role=textbox[name="Member ID:"]', confidence=0.9)],
            )
        ],
        stop_reason="goal_complete",
        checkpoint=Checkpoint(description="Found it", locator=LocatorTier(strategy="role", value=f'role=cell[name="{checkpoint_name}"]', confidence=0.9)),
        extract_log=[{
            "role": "cell", "name": "$4500.00", "value": "$4500.00", "output_name": "balance", "output_format": "currency",
            "value_locator": "role=row[name=/^Current Balance:/] >> role=cell >> nth=1",
        }],
    )


def _record_args(tmp_path, artifact_id="my_balance"):
    return build_arg_parser().parse_args(
        ["--goal", "g", "--target", "http://127.0.0.1:8000/hostile_legacy_page.html", "--artifact-id", artifact_id, "--artifacts-dir", str(tmp_path)]
    )


def test_save_artifact_writes_a_loadable_parameterized_artifact(tmp_path, capsys):
    assert save_artifact(_completed_run(), _record_args(tmp_path), {"12345": "member_id"}) == 0

    artifact = CapabilityArtifact.load(tmp_path / "my_balance.json")
    assert artifact.steps[0].input_value == "{{member_id}}"
    assert artifact.goal_description == "Look up the balance for member {{member_id}}"
    assert artifact.checkpoint.extract == {"balance": "role=row[name=/^Current Balance:/] >> role=cell >> nth=1"}
    assert artifact.target.app_name == "127.0.0.1"
    assert "saved" in capsys.readouterr().out


def test_save_artifact_refuses_a_value_bound_run_and_writes_nothing(tmp_path, capsys):
    assert save_artifact(_completed_run(checkpoint_name="$4500.00"), _record_args(tmp_path), {"12345": "member_id"}) == 1
    assert not (tmp_path / "my_balance.json").exists()
    assert "Not recorded" in capsys.readouterr().err


def test_malformed_param_error_never_echoes_the_value():
    # A value-bearing malformed flag is most likely a credential; the message names the problem only.
    with pytest.raises(ValueError) as excinfo:
        parse_param_map(["=hunter2"])
    assert "hunter2" not in str(excinfo.value)
    with pytest.raises(ValueError, match="an empty value for 'username'"):
        parse_param_map(["username="])
