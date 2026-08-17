"""Multi-run stability check (Tier 1 stretch goal #2 of the design's Section 10).

Runs the same CapabilityArtifact + params N times, each against a fresh Playwright page, and
aggregates the results into a ReliabilityInfo attached to a *new* CapabilityArtifact -
CapabilityArtifact is treated as immutable by every other module in this project (discovery
records one, replay reads one, nothing mutates one in place), so this follows that convention
rather than breaking it for convenience.

Why a fresh page per run, not one page reused N times: ReplayEngine.run() already begins with an
implicit navigation to target.base_url, so reusing one page would still "work" for this project's
own fixture - but a stability check's whole point is to answer "does this artifact reliably
succeed as an independent invocation", and a real deployed capability API (Tier 1 stretch goal #1)
would never guarantee two calls share browser state. Testing under the same isolation a real
invocation gets is what makes the resulting pass_rate mean anything; testing under artificially
shared state (leftover cookies, session tokens) would make it a claim about this specific repeated
sequence, not about the capability itself.

pass_rate: the fraction of runs whose ReplayResult.status was anything other than "hard_failure" -
success, business_outcome, and recoverable_then_success (a run that needed a locator-tier fallback
but still finished) all count as a pass. This intentionally loses the "how cleanly did it pass"
detail that recoverable_then_success carries - a caller that wants that finer signal should read
each individual ReplayResult in StabilityCheckResult.runs directly (each one's own `.steps` already
carries per-step tier-fallback detail - see engine.py's own docstring on why that's not just folded
into the coarse status). ReliabilityInfo's own schema is one aggregate number per Section 6 of the
design; it was never meant to replace the per-run detail, only summarize it.

avg_duration_ms is each run's total wall-clock time (page creation through the returned
ReplayResult), averaged over sample_size - not a per-step breakdown, which again lives on each
individual ReplayResult already returned.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from capability_forge.guardrails.policy import Guardrail
from capability_forge.replay.engine import ReplayEngine, ReplayResult
from capability_forge.schema.artifact import CapabilityArtifact, ReliabilityInfo
from capability_forge.surfaces.playwright_driver import PlaywrightDriver

DEFAULT_SAMPLE_SIZE = 5  # matches the design's own stated config default


@dataclass
class StabilityCheckResult:
    """Everything one stability check produced: the artifact with ReliabilityInfo attached (the
    thing meant to be saved back to disk in place of the artifact passed in), plus every
    individual run's own ReplayResult, for a caller that wants more than one aggregate number."""

    artifact: CapabilityArtifact
    runs: list[ReplayResult] = field(default_factory=list)


def check_stability(
    artifact: CapabilityArtifact,
    page_factory: Callable[[], Any],
    guardrail: Guardrail,
    params: dict[str, Any] | None = None,
    confirm_risky: bool = False,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> StabilityCheckResult:
    """Replay `artifact` `sample_size` times, each against a fresh page from page_factory() (e.g.
    `lambda: browser.new_page()`) - this function owns that page's lifecycle, closing it after
    each run so a large sample_size doesn't accumulate open tabs. No escalation_manager or
    evidence_writer is passed to the underlying ReplayEngine runs: a stability check is meant to
    run unattended and repeatedly, and per-run evidence bundles for N throwaway attempts would
    bloat evidence/ for no benefit the aggregate ReliabilityInfo doesn't already capture - a
    caller wanting per-run evidence should call ReplayEngine directly instead of this function.

    Raises whatever the first run's ReplayEngine.run() raises (in practice, only
    ParamValidationError - everything else is caught internally and turned into a hard_failure
    ReplayResult) rather than catching and repeating it sample_size times: a param mismatch is a
    caller/artifact contract problem, not a reliability signal, and it will fail identically on
    every subsequent attempt."""
    runs: list[ReplayResult] = []
    durations_ms: list[float] = []

    for _ in range(sample_size):
        page = page_factory()
        try:
            driver = PlaywrightDriver(page)
            engine = ReplayEngine(driver, guardrail)
            run_start = time.monotonic()
            result = engine.run(artifact, params=params, confirm_risky=confirm_risky)
            durations_ms.append((time.monotonic() - run_start) * 1000)
            runs.append(result)
        finally:
            page.close()

    passed = sum(1 for r in runs if r.status != "hard_failure")
    reliability = ReliabilityInfo(
        pass_rate=passed / sample_size,
        avg_duration_ms=sum(durations_ms) / sample_size,
        sample_size=sample_size,
        last_checked=datetime.now(timezone.utc),
    )
    # model_copy(update=...), not mutating artifact.reliability directly - CapabilityArtifact is
    # treated as immutable everywhere else in this project (see module docstring).
    updated_artifact = artifact.model_copy(update={"reliability": reliability})
    return StabilityCheckResult(artifact=updated_artifact, runs=runs)
