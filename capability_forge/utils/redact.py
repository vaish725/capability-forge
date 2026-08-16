"""Redaction utilities.

Masks sensitive data before it is written to disk, in both structured log lines and artifacts.
Two independent layers, since a field can be sensitive by name (its key is "account_number") or
by shape (its value looks like an SSN even though nobody thought to name the field accordingly):

  - Field-name redaction: recursively walks the record; any dict key matching a configured
    sensitive field name (case-insensitive, exact match) has its value replaced outright with
    "[REDACTED:<field>]" - whatever the value's shape, so a sensitive field that happens to hold
    a nested object is masked wholesale rather than recursed into.
  - Pattern redaction: scans string values (in non-sensitive fields) for an SSN-shaped substring
    and masks just that substring, regardless of which field it was found in.

Known limitation, documented rather than silently assumed away: field-name matching is exact, so
a field named "acct_num" instead of "account_number" is not caught by name - only the pattern
layer offers any protection for unexpectedly-named or freeform fields, and only for the one shape
it knows about (SSNs). This is not a general PII scanner.
"""

import re
from typing import Any

# Sensitive field names to mask if the caller doesn't supply its own list. Normally callers pass
# the sensitive_fields list loaded from config/allowlist.yaml instead of relying on this default;
# it exists so redact() has a sane standalone behavior even without a loaded policy.
DEFAULT_SENSITIVE_FIELDS = frozenset({
    "account_number", "ssn", "full_name", "email", "password", "token",
})

# US SSN shape: 3 digits, dash, 2 digits, dash, 4 digits.
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

REDACTED_FIELD_TEMPLATE = "[REDACTED:{field}]"
REDACTED_SSN_VALUE = "[REDACTED:SSN]"


def redact(record: Any, sensitive_fields: frozenset[str] | set[str] | list[str] = DEFAULT_SENSITIVE_FIELDS) -> Any:
    """Return a redacted deep copy of record. Never mutates the input.

    record is typically a dict (a log line or an artifact's model_dump()), but any JSON-like
    structure (nested dicts/lists/primitives) is handled.
    """
    lowered_fields = {name.lower() for name in sensitive_fields}
    return _redact_value(record, lowered_fields)


def _redact_value(value: Any, lowered_fields: set[str]) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, val in value.items():
            if key.lower() in lowered_fields:
                # Sensitive key: mask the whole value regardless of its shape, don't recurse into it.
                result[key] = REDACTED_FIELD_TEMPLATE.format(field=key)
            else:
                result[key] = _redact_value(val, lowered_fields)
        return result
    if isinstance(value, list):
        return [_redact_value(item, lowered_fields) for item in value]
    if isinstance(value, str):
        return _SSN_PATTERN.sub(REDACTED_SSN_VALUE, value)
    return value
