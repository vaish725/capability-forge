"""Guardrail policy: allowlist enforcement, risk classification, and redaction.

Wraps every action proposed in discovery or replay with three checks, applied in this order:
  1. Allowlist - is this domain, and this action type on this domain, permitted at all? Checked
     before every act() call. A blocked action is the caller's cue to log a hard_failure with
     reason "policy_violation" (Section 8 of the design doc) rather than executing.
  2. Risk classification - is this action safe_reversible or risky_irreversible? The model tags
     its own proposed action at discovery time; a keyword heuristic can only escalate that tag
     (safe -> risky), never downgrade it, so a determined mislabel can at worst be caught late,
     never silently waved through as safer than it is.
  3. Redaction - delegated to utils.redact, using this policy's configured sensitive_fields.

Known limitation, worth stating plainly rather than leaving implicit: the risk heuristic is a
keyword substring match against the target element's visible text. It trusts the LLM's self-tagged
risk unless a keyword overrides it - an adversarial prompt or a mislabeled UI could still slip a
risky action through as "safe" if it also avoids every configured keyword. This is a known gap,
not a solved problem.
"""

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from capability_forge.schema.artifact import ACTION_TYPES
from capability_forge.utils.redact import DEFAULT_SENSITIVE_FIELDS
from capability_forge.utils.redact import redact as redact_record

# config/allowlist.yaml lives two directories up from this file (capability_forge/guardrails/).
DEFAULT_ALLOWLIST_PATH = Path(__file__).resolve().parents[2] / "config" / "allowlist.yaml"

RiskTier = Literal["safe_reversible", "risky_irreversible"]


class PolicyViolation(Exception):
    """Raised when a proposed action fails the allowlist check. Callers should catch this and log
    a hard_failure with reason="policy_violation", per the error taxonomy."""

    reason = "policy_violation"


class AllowlistPolicy(BaseModel):
    """The validated, in-memory form of config/allowlist.yaml."""

    model_config = ConfigDict(extra="forbid")

    allowed_domains: list[str] = Field(min_length=1)
    allowed_action_types: dict[str, list[str]]
    risky_keywords: list[str] = Field(default_factory=list)
    sensitive_fields: list[str] = Field(default_factory=lambda: list(DEFAULT_SENSITIVE_FIELDS))

    @field_validator("allowed_action_types")
    @classmethod
    def _action_types_must_be_known_to_the_schema(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        # Catches exactly the class of bug this project already hit once: a config action type
        # that no longer exists (or never did) on StepAction.action_type.
        for domain, action_types in value.items():
            unknown = set(action_types) - set(ACTION_TYPES)
            if unknown:
                raise ValueError(
                    f"allowed_action_types[{domain!r}] references unknown action type(s) "
                    f"{sorted(unknown)}; valid types are {sorted(ACTION_TYPES)}"
                )
        return value


def _normalize_domain(domain_or_url: str) -> str:
    """Extract a bare, lowercased hostname from either a full URL or an already-bare domain."""
    candidate = domain_or_url if "://" in domain_or_url else f"http://{domain_or_url}"
    hostname = urlparse(candidate).hostname
    return (hostname or domain_or_url).lower()


class Guardrail:
    """The runtime guardrail layer: one loaded policy, checked before every action."""

    def __init__(self, policy: AllowlistPolicy):
        self.policy = policy

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_ALLOWLIST_PATH) -> "Guardrail":
        """Load and validate config/allowlist.yaml (or a caller-supplied path) into a Guardrail."""
        data = yaml.safe_load(Path(path).read_text())
        return cls(AllowlistPolicy.model_validate(data))

    def is_domain_allowed(self, domain_or_url: str) -> bool:
        """A domain is allowed if it exactly matches an allowlisted domain, or is a subdomain of
        one (e.g. "www.example.com" is allowed if "example.com" is allowlisted)."""
        host = _normalize_domain(domain_or_url)
        for allowed in self.policy.allowed_domains:
            allowed_lower = allowed.lower()
            if host == allowed_lower or host.endswith("." + allowed_lower):
                return True
        return False

    def is_action_type_allowed(self, domain_or_url: str, action_type: str) -> bool:
        """An action type is allowed if it's listed under the "*" wildcard or under this specific
        domain's entry in allowed_action_types."""
        host = _normalize_domain(domain_or_url)
        wildcard_types = set(self.policy.allowed_action_types.get("*", []))
        domain_types = set(self.policy.allowed_action_types.get(host, []))
        return action_type in wildcard_types or action_type in domain_types

    def check_action(self, domain_or_url: str, action_type: str) -> None:
        """Raise PolicyViolation if this action is not permitted. Call before every act()."""
        if not self.is_domain_allowed(domain_or_url):
            raise PolicyViolation(f"domain not allowlisted: {_normalize_domain(domain_or_url)!r}")
        if not self.is_action_type_allowed(domain_or_url, action_type):
            raise PolicyViolation(
                f"action_type {action_type!r} not permitted for domain "
                f"{_normalize_domain(domain_or_url)!r}"
            )

    def classify_risk(self, element_text: str, model_tag: RiskTier) -> RiskTier:
        """Cross-check the model's self-tagged risk against the keyword heuristic. The heuristic
        can only escalate safe_reversible to risky_irreversible, never the reverse - once a step
        is risky, nothing here can talk it back down to safe."""
        if model_tag not in ("safe_reversible", "risky_irreversible"):
            raise ValueError(f"invalid model_tag: {model_tag!r}")
        if model_tag == "risky_irreversible":
            return "risky_irreversible"
        lowered_text = element_text.lower()
        for keyword in self.policy.risky_keywords:
            if keyword.lower() in lowered_text:
                return "risky_irreversible"
        return "safe_reversible"

    def redact(self, record: dict) -> dict:
        """Mask this policy's configured sensitive fields in a record before it hits disk."""
        return redact_record(record, self.policy.sensitive_fields)
