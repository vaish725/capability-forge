"""Goal-driven discovery loop.

Drives Claude through an observe -> decide -> act cycle against a live browser surface until the
goal is reached (or a stopping condition fires). Every proposed action is guardrail-checked before
it executes, and the resulting run is handed to the artifact recorder on success.

TODO: implement AgentLoop (observe/decide/act cycle, stopping conditions, dead-end guard).
"""
