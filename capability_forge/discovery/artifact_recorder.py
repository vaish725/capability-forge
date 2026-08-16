"""Artifact recorder.

Turns a successful discovery run's step history into a validated CapabilityArtifact: normalizes
locators into ranked tiers, assigns per-step risk, and attaches the verified checkpoint. Only
called after a discovery run ends in success with its checkpoint confirmed.

TODO: implement ArtifactRecorder (step history -> CapabilityArtifact).
"""
