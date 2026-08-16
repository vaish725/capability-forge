"""Tests for capability_forge.utils.redact: field-name masking (including nested structures) and
the SSN-shaped value pattern scan, plus the "input is never mutated" guarantee.
"""

from capability_forge.utils.redact import redact


def test_sensitive_top_level_field_is_masked():
    record = {"account_number": "123456789", "note": "hello"}
    result = redact(record)
    assert result["account_number"] == "[REDACTED:account_number]"
    assert result["note"] == "hello"


def test_matching_is_case_insensitive_on_key_name():
    record = {"Account_Number": "123456789"}
    result = redact(record)
    assert result["Account_Number"] == "[REDACTED:Account_Number]"


def test_non_sensitive_fields_pass_through_unchanged():
    record = {"member_id": "12345", "goal": "look up balance"}
    result = redact(record)
    assert result == record


def test_sensitive_field_nested_inside_dict_is_masked_wholesale():
    # A sensitive key whose value is itself a nested structure must be masked outright, not
    # recursed into (recursing would leave nested sub-fields like "hash" or "salt" exposed).
    record = {"password": {"hash": "abc123", "salt": "xyz"}}
    result = redact(record)
    assert result["password"] == "[REDACTED:password]"


def test_sensitive_field_inside_a_list_of_dicts_is_masked():
    record = {
        "steps": [
            {"step_id": "s1", "input_value": "safe text"},
            {"step_id": "s2", "input_value": "also safe", "password": "hunter2"},
        ]
    }
    result = redact(record)
    assert result["steps"][0]["input_value"] == "safe text"
    assert result["steps"][1]["password"] == "[REDACTED:password]"
    assert result["steps"][1]["input_value"] == "also safe"


def test_deeply_nested_sensitive_field_is_masked():
    record = {"a": {"b": {"c": {"ssn": "123-45-6789"}}}}
    result = redact(record)
    assert result["a"]["b"]["c"]["ssn"] == "[REDACTED:ssn]"


def test_custom_sensitive_fields_list_overrides_default():
    record = {"account_number": "123456789", "custom_secret": "shh"}
    result = redact(record, sensitive_fields={"custom_secret"})
    # account_number is not in the custom list, so it passes through; custom_secret is masked.
    assert result["account_number"] == "123456789"
    assert result["custom_secret"] == "[REDACTED:custom_secret]"


def test_ssn_shaped_substring_is_masked_even_in_an_unlisted_field():
    record = {"free_text_note": "customer SSN is 123-45-6789, please verify"}
    result = redact(record)
    assert result["free_text_note"] == "customer SSN is [REDACTED:SSN], please verify"


def test_non_ssn_shaped_digit_sequences_are_left_alone():
    record = {"note": "order number 123456789"}
    result = redact(record)
    assert result["note"] == "order number 123456789"


def test_redact_does_not_mutate_input():
    record = {"account_number": "123456789", "nested": {"password": "hunter2"}}
    original = {"account_number": "123456789", "nested": {"password": "hunter2"}}
    redact(record)
    assert record == original


def test_redact_handles_top_level_list():
    records = [{"password": "a"}, {"password": "b"}]
    result = redact(records)
    assert result == [{"password": "[REDACTED:password]"}, {"password": "[REDACTED:password]"}]


def test_redact_handles_non_string_non_dict_values():
    record = {"count": 5, "confirmed": True, "amount": 12.5, "note": None}
    result = redact(record)
    assert result == record
