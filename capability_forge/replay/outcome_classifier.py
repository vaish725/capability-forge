"""Outcome classifier.

Maps what actually happened during a replay step to the closed outcome taxonomy: success,
business_outcome (a named expected non-error result), recoverable (a known transient condition
handled automatically), or hard_failure (stops the run and triggers escalation).

TODO: implement OutcomeClassifier.classify(step_result) -> OutcomeType.
"""
