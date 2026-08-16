"""Escalation manager.

Owns the control-transfer state machine: AGENT_ACTIVE -> PAUSED_FOR_HUMAN -> HUMAN_ACTIVE ->
RESUMING -> AGENT_ACTIVE | DONE. Triggered by the dead-end guard, a hard_failure outcome, or a
risky/irreversible step awaiting confirmation. Exposes the live browser session (not a fresh one)
to the operator and appends a HandoffRecord to the run's evidence log.

TODO: implement EscalationManager and the state machine transitions above.
"""
