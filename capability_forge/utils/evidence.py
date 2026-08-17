"""Evidence writer for discovery (and, later, replay) runs.

Writes the per-run evidence bundle to evidence/<run_id>/:
  - log.jsonl: one redacted JSON object per step (run_id, mode, step_id, timestamp,
    action_attempted, locator_tier_used, expected_state, observed_state, outcome_type,
    duration_ms, evidence_ref). Deliberately never includes a step's literal input_value (what was
    typed/selected) - action identity (role/name) is enough for debugging a locator, and omitting
    the value sidesteps ever needing to redact free-text input by content.
  - screenshots/step_<n>.png: one per step, referenced from that step's log line.
  - transcript.json: the raw LLM message history for the run, redacted before writing.

A caller that doesn't want evidence written (e.g. the scripted-client test suite, or any run that
shouldn't touch disk) simply doesn't construct one - nothing in AgentLoop requires it.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capability_forge.utils.redact import DEFAULT_SENSITIVE_FIELDS, redact

DEFAULT_EVIDENCE_ROOT = Path("evidence")


def new_run_id(mode: str) -> str:
    """A sortable, collision-resistant enough run_id for local, single-operator use - a
    timestamp is sufficient here, no need for a UUID's extra entropy."""
    return f"{mode}_{int(time.time())}"


@dataclass
class EvidenceWriter:
    run_id: str
    mode: str
    # default_factory (re-reading the module-level constant at construction time), not a plain
    # default - a dataclass default value is bound once at class-definition time, which would
    # make DEFAULT_EVIDENCE_ROOT un-monkeypatchable by tests (a real concern here, not
    # theoretical: without this, a test redirecting the evidence root to a tmp_path would silently
    # keep writing into the real repo's evidence/ directory instead).
    root: Path = field(default_factory=lambda: DEFAULT_EVIDENCE_ROOT)
    sensitive_fields: frozenset[str] | set[str] = field(default_factory=lambda: DEFAULT_SENSITIVE_FIELDS)

    def __post_init__(self) -> None:
        self.run_dir = self.root / self.run_id
        self.screenshots_dir = self.run_dir / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.run_dir / "log.jsonl"
        self._step_count = 0

    def log_step(
        self,
        step_id: str,
        action_attempted: str,
        locator_tier_used: str | None,
        expected_state: str,
        observed_state: str,
        outcome_type: str,
        duration_ms: int,
        evidence_ref: str | None = None,
    ) -> None:
        """Append one redacted JSON line to log.jsonl."""
        line = {
            "run_id": self.run_id,
            "mode": self.mode,
            "step_id": step_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action_attempted": action_attempted,
            "locator_tier_used": locator_tier_used,
            "expected_state": expected_state,
            "observed_state": observed_state,
            "outcome_type": outcome_type,
            "duration_ms": duration_ms,
            "evidence_ref": evidence_ref,
        }
        redacted_line = redact(line, self.sensitive_fields)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(redacted_line) + "\n")

    def save_screenshot(self, page: Any) -> str:
        """Save a full-page screenshot for the current step, returning its path relative to
        run_dir (the value stored as a log line's evidence_ref)."""
        self._step_count += 1
        filename = f"step_{self._step_count:02d}.png"
        page.screenshot(path=str(self.screenshots_dir / filename))
        return f"screenshots/{filename}"

    def save_transcript(self, messages: list[dict[str, Any]]) -> None:
        """Write the run's raw LLM message history, redacted, to transcript.json. Anthropic SDK
        response content blocks (assistant turns) aren't plain dicts, so they're normalized to
        dicts first; a tool_use block's "input" is stripped of any "value" key before redaction -
        a typed/selected literal could be anything, including something not caught by field-name
        redaction, so it's dropped outright rather than trusted to redact() alone."""
        normalized = [_normalize_message(m) for m in messages]
        redacted_transcript = redact(normalized, self.sensitive_fields)
        transcript_path = self.run_dir / "transcript.json"
        transcript_path.write_text(json.dumps(redacted_transcript, indent=2), encoding="utf-8")


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    if isinstance(content, list):
        return {"role": message["role"], "content": [_normalize_block(block) for block in content]}
    return message


def _normalize_block(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        normalized = dict(block)
    else:
        # An Anthropic SDK content block object (TextBlock, ToolUseBlock, ...) - pull out the
        # fields this project's transcript actually needs rather than depending on its own
        # (possibly non-JSON-serializable) internal representation.
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            normalized = {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)}
        elif block_type == "text":
            normalized = {"type": "text", "text": block.text}
        else:
            normalized = {"type": block_type or "unknown"}
    if normalized.get("type") == "tool_use" and isinstance(normalized.get("input"), dict):
        normalized = {**normalized, "input": {k: v for k, v in normalized["input"].items() if k != "value"}}
    return normalized
