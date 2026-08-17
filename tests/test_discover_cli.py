"""Tests for the discover CLI's argument parsing and its offline-checkable guard (missing
ANTHROPIC_API_KEY). The actual discovery run this CLI triggers needs a real API key and a real
browser, so it isn't exercised here - see AgentLoop's own tests for the loop mechanics, tested
with a scripted client instead.
"""

from types import SimpleNamespace

import anthropic
import pytest

from capability_forge.discover import build_arg_parser, main


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
            "--headless", "--max-steps", "5", "--timeout-seconds", "30",
        ]
    )
    assert args.headless is True
    assert args.max_steps == 5
    assert args.timeout_seconds == 30.0


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
