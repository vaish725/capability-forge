"""Guardrail policy.

Wraps every action proposed in discovery or replay with three checks, in order: allowlist
(domain/route/action-type permitted per config/allowlist.yaml), risk classification (safe_reversible
vs risky_irreversible, frozen into the artifact at recording time and never re-derived at replay),
and redaction (sensitive fields masked before anything is written to disk).

TODO: implement Guardrail.check(action, context) and the allowlist loader.
"""
