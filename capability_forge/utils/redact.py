"""Redaction utilities.

Masks any field whose name matches a configured sensitive-field list (account_number, ssn,
full_name, email, password, token) before it is written to a log line or artifact, replacing the
value with "[REDACTED:<field>]". Applied at the point of construction, never as a post-hoc scrub.

TODO: implement redact(record: dict) -> dict.
"""
