"""Agent-facing capability API (stretch goal).

Thin FastAPI layer over the replay engine: GET /capabilities lists recorded artifacts with their
typed input/output schema, POST /capabilities/{artifact_id}/invoke runs a replay and returns the
resulting ReplayResult.

TODO: implement the FastAPI app and its two routes.
"""
