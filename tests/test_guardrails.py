"""Tests for the guardrail layer: allowlist enforcement (domain + action type), the risk
classification heuristic (including its one-directional escalate-only behavior), and the config
loader's cross-validation against the schema's action type taxonomy.
"""

import pytest
from pydantic import ValidationError

from capability_forge.guardrails.policy import (
    AllowlistPolicy,
    Guardrail,
    PolicyViolation,
    _normalize_domain,
)


def make_policy(**overrides) -> AllowlistPolicy:
    base = {
        "allowed_domains": ["example.com"],
        "allowed_action_types": {"*": ["click", "type", "navigate", "wait", "select"]},
        "risky_keywords": ["confirm", "submit", "delete", "transfer"],
        "sensitive_fields": ["account_number", "ssn"],
    }
    base.update(overrides)
    return AllowlistPolicy.model_validate(base)


def make_guardrail(**overrides) -> Guardrail:
    return Guardrail(make_policy(**overrides))


# --- _normalize_domain -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("example.com", "example.com"),
        ("https://example.com/path", "example.com"),
        ("http://example.com:8080/path", "example.com"),
        ("EXAMPLE.COM", "example.com"),
        ("www.example.com", "www.example.com"),
    ],
)
def test_normalize_domain(value, expected):
    assert _normalize_domain(value) == expected


# --- AllowlistPolicy: config validated against the schema's action type taxonomy ---------------


def test_policy_rejects_unknown_action_type():
    with pytest.raises(ValidationError, match="unknown action type"):
        make_policy(allowed_action_types={"*": ["click", "extract"]})


def test_policy_accepts_all_currently_valid_action_types():
    policy = make_policy(allowed_action_types={"*": ["click", "type", "navigate", "wait", "select"]})
    assert "select" in policy.allowed_action_types["*"]


def test_policy_requires_at_least_one_allowed_domain():
    with pytest.raises(ValidationError):
        make_policy(allowed_domains=[])


def test_policy_rejects_unknown_top_level_field():
    with pytest.raises(ValidationError):
        AllowlistPolicy.model_validate(
            {
                "allowed_domains": ["example.com"],
                "allowed_action_types": {"*": ["click"]},
                "unexpected_field": True,
            }
        )


# --- domain allowlisting -----------------------------------------------------------------------


def test_exact_domain_is_allowed():
    guardrail = make_guardrail()
    assert guardrail.is_domain_allowed("example.com") is True


def test_subdomain_of_allowed_domain_is_allowed():
    guardrail = make_guardrail()
    assert guardrail.is_domain_allowed("www.example.com") is True
    assert guardrail.is_domain_allowed("https://portal.example.com/login") is True


def test_unrelated_domain_is_not_allowed():
    guardrail = make_guardrail()
    assert guardrail.is_domain_allowed("evil.com") is False


def test_domain_that_merely_contains_allowed_domain_as_substring_is_not_allowed():
    # "notexample.com" must not be allowed just because "example.com" is a substring of it.
    guardrail = make_guardrail()
    assert guardrail.is_domain_allowed("notexample.com") is False


# --- action type allowlisting -------------------------------------------------------------------


def test_wildcard_action_type_is_allowed_on_any_allowed_domain():
    guardrail = make_guardrail()
    assert guardrail.is_action_type_allowed("example.com", "click") is True


def test_action_type_not_in_wildcard_or_domain_list_is_not_allowed():
    guardrail = make_guardrail(allowed_action_types={"*": ["click"]})
    assert guardrail.is_action_type_allowed("example.com", "type") is False


def test_domain_specific_action_type_extends_wildcard():
    guardrail = make_guardrail(
        allowed_action_types={"*": ["click"], "example.com": ["type"]}
    )
    assert guardrail.is_action_type_allowed("example.com", "click") is True
    assert guardrail.is_action_type_allowed("example.com", "type") is True
    # A different domain doesn't inherit example.com's specific grant.
    assert guardrail.is_action_type_allowed("other.com", "type") is False


# --- check_action: the raise-based gate called before every act() ------------------------------


def test_check_action_passes_silently_when_allowed():
    guardrail = make_guardrail()
    guardrail.check_action("example.com", "click")  # should not raise


def test_check_action_raises_for_disallowed_domain():
    guardrail = make_guardrail()
    with pytest.raises(PolicyViolation) as exc_info:
        guardrail.check_action("evil.com", "click")
    assert exc_info.value.reason == "policy_violation"


def test_check_action_raises_for_disallowed_action_type():
    guardrail = make_guardrail(allowed_action_types={"*": ["click"]})
    with pytest.raises(PolicyViolation):
        guardrail.check_action("example.com", "type")


# --- risk classification: keyword heuristic can only escalate, never downgrade -----------------


def test_model_tagged_safe_with_no_risky_keyword_stays_safe():
    guardrail = make_guardrail()
    risk = guardrail.classify_risk("Search", model_tag="safe_reversible")
    assert risk == "safe_reversible"


def test_model_tagged_safe_but_risky_keyword_present_is_escalated():
    guardrail = make_guardrail()
    risk = guardrail.classify_risk("Confirm Transfer", model_tag="safe_reversible")
    assert risk == "risky_irreversible"


def test_keyword_match_is_case_insensitive():
    guardrail = make_guardrail()
    risk = guardrail.classify_risk("SUBMIT payment", model_tag="safe_reversible")
    assert risk == "risky_irreversible"


def test_model_tagged_risky_stays_risky_even_with_no_keyword_match():
    guardrail = make_guardrail()
    risk = guardrail.classify_risk("Continue", model_tag="risky_irreversible")
    assert risk == "risky_irreversible"


def test_invalid_model_tag_raises():
    guardrail = make_guardrail()
    with pytest.raises(ValueError):
        guardrail.classify_risk("Search", model_tag="not_a_real_tag")


@pytest.mark.parametrize("keyword_text", ["confirm", "submit", "delete", "transfer"])
def test_each_configured_keyword_escalates(keyword_text):
    guardrail = make_guardrail()
    risk = guardrail.classify_risk(f"{keyword_text} now", model_tag="safe_reversible")
    assert risk == "risky_irreversible"


# --- redact delegation -----------------------------------------------------------------------


def test_guardrail_redact_uses_policy_sensitive_fields():
    guardrail = make_guardrail(sensitive_fields=["custom_field"])
    result = guardrail.redact({"custom_field": "secret", "safe_field": "visible"})
    assert result["custom_field"] == "[REDACTED:custom_field]"
    assert result["safe_field"] == "visible"


# --- loading the real config/allowlist.yaml -----------------------------------------------------


def test_default_allowlist_yaml_loads_and_validates():
    # This is the actual repo config, not a fixture - if it's malformed or drifted out of sync
    # with the schema's action types, this test catches it.
    guardrail = Guardrail.from_yaml()
    assert isinstance(guardrail.policy, AllowlistPolicy)
    assert len(guardrail.policy.allowed_domains) >= 1
