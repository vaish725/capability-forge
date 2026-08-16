"""Replay engine.

Executes a CapabilityArtifact deterministically: validates input params against the artifact's
typed schema, drives the Surface Driver through each step (falling back across locator tiers),
verifies the checkpoint, and returns a structured ReplayResult. No LLM is involved.

TODO: implement ReplayEngine.run(artifact, params) -> ReplayResult.
"""
