"""Agent-facing capability API (Tier 1 stretch goal #1 of the design's Section 10).

Thin FastAPI layer over the replay engine: GET /capabilities lists every recorded artifact under
artifacts/ with its typed input/output schema (and reliability data, if a stability check has been
run - see replay/reliability.py), POST /capabilities/{artifact_id}/invoke runs a real replay
against a fresh browser and returns the resulting ReplayResult. This is the most direct possible
demonstration of the project's own thesis: a capability artifact is something an AI agent can call
programmatically, not just something a human replays from the CLI.

Deliberately out of scope for this thin layer, named rather than silently absent:
  - Escalation. An HTTP request/response cycle has no human sitting at a terminal to block on
    input() the way cli_operator_console does - CLI-style escalation is structurally the wrong
    mechanism for a programmatic caller. A production version of this API would need an
    async/webhook-based flow instead (return a "paused" status with a resume URL, notify an
    operator out of band). invoke_capability() runs with no escalation_manager: a risky step needs
    confirm_risky=true in the request body up front, and a hard_failure is simply returned as one,
    with no chance for a human to intervene and let it retry.
  - Authentication/authorization, rate limiting, concurrency control across simultaneous invoke
    calls, artifact hot-reloading. None of these are the point this stretch goal demonstrates.
  - A shared/pooled browser across requests. Each invoke call launches and closes its own
    Playwright browser - correct and simple at demo scale, not what a production version handling
    concurrent traffic would do.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from playwright.sync_api import sync_playwright
from pydantic import BaseModel

from capability_forge.guardrails.policy import Guardrail
from capability_forge.replay.engine import ParamValidationError, ReplayEngine, ReplayResult
from capability_forge.schema.artifact import CapabilityArtifact, InputParam, OutputField, ReliabilityInfo
from capability_forge.surfaces.playwright_driver import PlaywrightDriver

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def load_artifacts(directory: Path = ARTIFACTS_DIR) -> dict[str, CapabilityArtifact]:
    """Loads every *.json file under `directory` into a registry keyed by artifact_id. A file that
    fails schema validation is skipped (with a note printed to stderr) rather than taking the
    whole API down - one malformed artifact on disk shouldn't cost every other capability its
    listing."""
    registry: dict[str, CapabilityArtifact] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            artifact = CapabilityArtifact.load(path)
        except Exception as exc:  # noqa: BLE001 - any load/validation failure is equally "skip this one file"
            print(f"Skipping {path}: {exc}")
            continue
        registry[artifact.artifact_id] = artifact
    return registry


class CapabilitySummary(BaseModel):
    """What GET /capabilities exposes about each artifact: the typed contract an agent needs to
    decide whether and how to call it - not the internal step/locator detail it has no use for."""

    artifact_id: str
    goal_description: str
    inputs: list[InputParam]
    outputs: list[OutputField]
    risk_summary: str
    expected_outcome_type: str
    reliability: ReliabilityInfo | None = None


class InvokeRequest(BaseModel):
    params: dict[str, Any] = {}
    confirm_risky: bool = False


def create_app(artifacts: dict[str, CapabilityArtifact] | None = None) -> FastAPI:
    """Builds the FastAPI app. Accepts a pre-populated artifact registry - tests use this to
    exercise the routes against a controlled artifact without touching the real artifacts/
    directory. Defaults to load_artifacts() (the real directory) when not supplied, which is what
    running this module for real (`uvicorn capability_forge.api.main:app`) does via the
    module-level `app` object below."""
    app = FastAPI(title="capability-forge", description="Agent-facing interface to recorded, replayable capabilities.")
    registry = artifacts if artifacts is not None else load_artifacts()
    guardrail = Guardrail.from_yaml()

    @app.get("/capabilities", response_model=list[CapabilitySummary])
    def list_capabilities() -> list[CapabilitySummary]:
        return [
            CapabilitySummary(
                artifact_id=a.artifact_id,
                goal_description=a.goal_description,
                inputs=a.inputs,
                outputs=a.outputs,
                risk_summary=a.risk_summary,
                expected_outcome_type=a.expected_outcome_type,
                reliability=a.reliability,
            )
            for a in registry.values()
        ]

    @app.post("/capabilities/{artifact_id}/invoke")
    def invoke_capability(artifact_id: str, body: InvokeRequest) -> ReplayResult:
        artifact = registry.get(artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail=f"No capability named {artifact_id!r}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                driver = PlaywrightDriver(page)
                engine = ReplayEngine(driver, guardrail)
                try:
                    result = engine.run(artifact, params=body.params, confirm_risky=body.confirm_risky)
                except ParamValidationError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            finally:
                browser.close()
        return result

    return app


# What `uvicorn capability_forge.api.main:app` actually imports - built from the real artifacts/
# directory. create_app() itself stays a plain function precisely so tests aren't stuck with this
# one eagerly-loaded instance.
app = create_app()
