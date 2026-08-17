"""Tests for the evidence writer: log.jsonl line shape and redaction, screenshot saving, and
transcript.json redaction (including the deliberate "value" field strip described in its
docstring)."""

import json
from types import SimpleNamespace

import pytest

from capability_forge.utils.evidence import EvidenceWriter, new_run_id


def make_writer(tmp_path, sensitive_fields=frozenset({"password"})):
    return EvidenceWriter(run_id="test_run", mode="discovery", root=tmp_path, sensitive_fields=sensitive_fields)


def read_log_lines(writer):
    return [json.loads(line) for line in (writer.run_dir / "log.jsonl").read_text().splitlines()]


def test_new_run_id_includes_mode_and_is_stable_shape():
    run_id = new_run_id("discovery")
    assert run_id.startswith("discovery_")
    assert run_id.split("_")[1].isdigit()


def test_writer_creates_run_and_screenshots_directories(tmp_path):
    writer = make_writer(tmp_path)
    assert writer.run_dir == tmp_path / "test_run"
    assert writer.run_dir.is_dir()
    assert writer.screenshots_dir.is_dir()


def test_log_step_appends_one_json_line_with_expected_fields(tmp_path):
    writer = make_writer(tmp_path)
    writer.log_step(
        step_id="step_1",
        action_attempted="click(role=button,name=Login)",
        locator_tier_used="role",
        expected_state="submit login",
        observed_state="click succeeded.",
        outcome_type="success",
        duration_ms=42,
        evidence_ref="screenshots/step_01.png",
    )
    lines = read_log_lines(writer)
    assert len(lines) == 1
    line = lines[0]
    assert line["run_id"] == "test_run"
    assert line["mode"] == "discovery"
    assert line["step_id"] == "step_1"
    assert line["action_attempted"] == "click(role=button,name=Login)"
    assert line["locator_tier_used"] == "role"
    assert line["outcome_type"] == "success"
    assert line["duration_ms"] == 42
    assert line["evidence_ref"] == "screenshots/step_01.png"
    assert "timestamp" in line


def test_log_step_redacts_sensitive_field_names(tmp_path):
    # A log line field itself named "password" (not realistic for this schema's fixed field set,
    # but proves the redact() pass is genuinely wired in, not skipped).
    writer = make_writer(tmp_path, sensitive_fields=frozenset({"observed_state"}))
    writer.log_step(
        step_id="step_1",
        action_attempted="type(role=textbox,name=)",
        locator_tier_used="role",
        expected_state="enter password",
        observed_state="typed secret123",
        outcome_type="success",
        duration_ms=10,
    )
    line = read_log_lines(writer)[0]
    assert line["observed_state"] == "[REDACTED:observed_state]"


def test_log_step_multiple_calls_append_not_overwrite(tmp_path):
    writer = make_writer(tmp_path)
    for i in range(3):
        writer.log_step(
            step_id=f"step_{i}",
            action_attempted="click(role=button,name=X)",
            locator_tier_used="role",
            expected_state="",
            observed_state="",
            outcome_type="success",
            duration_ms=1,
        )
    assert len(read_log_lines(writer)) == 3


def test_save_screenshot_writes_a_real_png_and_returns_relative_path(tmp_path, page):
    writer = make_writer(tmp_path)
    page.set_content("<h1>hello</h1>")
    ref = writer.save_screenshot(page)
    assert ref == "screenshots/step_01.png"
    saved_path = writer.run_dir / ref
    assert saved_path.is_file()
    assert saved_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature


def test_save_screenshot_numbers_increment_across_calls(tmp_path, page):
    writer = make_writer(tmp_path)
    page.set_content("<h1>hello</h1>")
    assert writer.save_screenshot(page) == "screenshots/step_01.png"
    assert writer.save_screenshot(page) == "screenshots/step_02.png"


def test_save_transcript_strips_value_field_from_tool_use_input(tmp_path):
    writer = make_writer(tmp_path)
    tool_use_block = SimpleNamespace(
        type="tool_use", id="t1", name="type",
        input={"role": "textbox", "name": "", "value": "super-secret-password", "risk": "safe_reversible"},
    )
    messages = [
        {"role": "user", "content": "Current page: ..."},
        {"role": "assistant", "content": [tool_use_block]},
    ]
    writer.save_transcript(messages)

    saved = json.loads((writer.run_dir / "transcript.json").read_text())
    assistant_content = saved[1]["content"]
    tool_use_entry = assistant_content[0]
    assert "value" not in tool_use_entry["input"]
    assert tool_use_entry["input"]["role"] == "textbox"


def test_save_transcript_redacts_by_field_name_after_normalizing(tmp_path):
    writer = make_writer(tmp_path, sensitive_fields=frozenset({"password"}))
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi", "password": "shouldnt-be-here"}]}]
    writer.save_transcript(messages)
    saved = json.loads((writer.run_dir / "transcript.json").read_text())
    assert saved[0]["content"][0]["password"] == "[REDACTED:password]"


def test_save_transcript_handles_text_blocks_and_plain_string_content(tmp_path):
    writer = make_writer(tmp_path)
    text_block = SimpleNamespace(type="text", text="I'll click login.")
    messages = [
        {"role": "user", "content": "plain string observation"},
        {"role": "assistant", "content": [text_block]},
    ]
    writer.save_transcript(messages)
    saved = json.loads((writer.run_dir / "transcript.json").read_text())
    assert saved[0]["content"] == "plain string observation"
    assert saved[1]["content"][0] == {"type": "text", "text": "I'll click login."}


# --- register_secret: scrubbing a typed value out of every subsequent write ----------------------
# Found live against ParaBank, not anticipated up front: a typed password resurfaced, unprompted,
# in a later raw accessibility-tree observation - Playwright exposes an <input>'s current value as
# part of its own description, even for type="password", which stripping the tool_use "value"
# field (the fix above) does nothing to prevent, since the leak isn't in the tool call at all.


def test_registered_secret_is_scrubbed_from_a_later_plain_string_observation(tmp_path):
    writer = make_writer(tmp_path, sensitive_fields=frozenset())
    writer.register_secret("hunter2")
    # Mirrors the real leak shape: the secret was typed several turns ago, then resurfaces
    # unprompted inside a later raw page observation with no field name to redact by.
    messages = [
        {"role": "user", "content": "Current page:\n...\n- textbox: hunter2\n- button \"Log In\""},
    ]
    writer.save_transcript(messages)
    saved_text = (writer.run_dir / "transcript.json").read_text()
    assert "hunter2" not in saved_text
    assert "[REDACTED_INPUT]" in saved_text


def test_registered_secret_is_scrubbed_from_log_step_free_text_fields(tmp_path):
    writer = make_writer(tmp_path, sensitive_fields=frozenset())
    writer.register_secret("hunter2")
    writer.log_step(
        step_id="step_1",
        action_attempted="type(role=textbox,name=)",
        locator_tier_used="role",
        expected_state="type hunter2 into the password field",
        observed_state="type succeeded.",
        outcome_type="success",
        duration_ms=1,
    )
    saved_text = (writer.run_dir / "log.jsonl").read_text()
    assert "hunter2" not in saved_text
    assert "[REDACTED_INPUT]" in saved_text


def test_unregistered_values_are_left_alone(tmp_path):
    writer = make_writer(tmp_path, sensitive_fields=frozenset())
    # Nothing registered - scrubbing must be a no-op, not an accidental blanket redaction.
    messages = [{"role": "user", "content": "Current page:\n- textbox: someone_else's_value"}]
    writer.save_transcript(messages)
    saved_text = (writer.run_dir / "transcript.json").read_text()
    assert "someone_else's_value" in saved_text


def test_register_secret_ignores_empty_values(tmp_path):
    writer = make_writer(tmp_path, sensitive_fields=frozenset())
    writer.register_secret("")
    assert writer._secret_values == []


def test_register_secret_scrubs_longest_match_first(tmp_path):
    # "16" is a substring of "16896" - the longer registered secret must win so "16896" doesn't
    # get partially replaced into "[REDACTED_INPUT]896", leaking the trailing digits.
    writer = make_writer(tmp_path, sensitive_fields=frozenset())
    writer.register_secret("16")
    writer.register_secret("16896")
    messages = [{"role": "user", "content": "account 16896 balance"}]
    writer.save_transcript(messages)
    saved = json.loads((writer.run_dir / "transcript.json").read_text())
    assert saved[0]["content"] == "account [REDACTED_INPUT] balance"


# The real leak was in a plain-string observation, but the whole point of register_secret() is
# that it's field-agnostic - it must catch the value wherever it lands, not just in the one field
# that happened to leak first. Parametrized over several different locations a secret could appear
# in a transcript, so the fix's generality is pinned down by a test rather than true by inspection.
_SECRET_LOCATIONS = {
    "plain_string_observation": [{"role": "user", "content": "Current page:\n- textbox: {secret}"}],
    "text_block": [{"role": "assistant", "content": [{"type": "text", "text": "I typed {secret} into the field."}]}],
    "tool_use_reasoning_field": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "click", "input": {"role": "button", "name": "Go", "reasoning": "Submitting with {secret} entered."}}]}
    ],
    "tool_result_content": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Field now reads {secret}."}]}],
    "nested_list_of_dicts": [{"role": "user", "content": [{"type": "text", "text": "outer", "extra": {"nested": ["deep", "{secret}", {"deeper": "{secret} again"}]}}]}],
}


@pytest.mark.parametrize("location", _SECRET_LOCATIONS, ids=list(_SECRET_LOCATIONS))
def test_registered_secret_never_survives_regardless_of_where_it_appears(tmp_path, location):
    writer = make_writer(tmp_path, sensitive_fields=frozenset())
    secret = "hunter2"
    writer.register_secret(secret)

    # Substitute the secret into whichever location this case targets - done via string
    # substitution on a JSON round-trip so the fixture data above stays plain and readable.
    template = json.dumps(_SECRET_LOCATIONS[location])
    messages = json.loads(template.replace("{secret}", secret))

    writer.save_transcript(messages)
    saved_text = (writer.run_dir / "transcript.json").read_text()
    assert secret not in saved_text, f"secret leaked via location={location!r}"
