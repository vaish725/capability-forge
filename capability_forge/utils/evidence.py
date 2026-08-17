"""Evidence writer for discovery and replay runs.

Writes the per-run evidence bundle to evidence/<run_id>/:
  - log.jsonl: one redacted JSON object per step (run_id, mode, step_id, timestamp,
    action_attempted, locator_tier_used, expected_state, observed_state, outcome_type,
    duration_ms, evidence_ref). Deliberately never includes a step's literal input_value (what was
    typed/selected) - action identity (role/name) is enough for debugging a locator, and omitting
    the value sidesteps ever needing to redact free-text input by content.
  - screenshots/step_<n>.png: one per step, referenced from that step's log line.
  - transcript.json: the raw LLM message history for the run, redacted before writing.
  - handoffs.jsonl: one redacted JSON object per completed human-in-the-loop escalation
    (escalation/manager.py's HandoffRecord). Kept as its own file rather than folded into
    log.jsonl: a handoff is a control-transfer record (who took control, what they decided, what
    they said they did), not a step outcome, and forcing it into log.jsonl's outcome_type enum
    (success | business_outcome | recoverable | hard_failure) would either stretch that enum to
    cover a fifth, structurally different kind of event or lose the handoff-specific fields
    (decision, notes, taken_at/released_at) that don't map onto a step's fields at all.

Two independent redaction layers, deliberately not just one:
  - Field-name redaction (redact(), from utils/redact.py): masks a value whose *key* matches a
    configured sensitive field name.
  - Secret-value scrubbing (register_secret()/the internal _scrub_secrets recursive walk, this
    module): masks a *literal value* wherever it appears, in any string, regardless of which field
    it's in. This exists because of a real leak found against the live ParaBank target: a typed
    password never appears under a field literally named "password" (the tool schema just calls it
    "value", already stripped from tool_use blocks - see save_transcript) - but the same literal
    text resurfaces later, unprompted, inside a raw page observation. Playwright's accessibility
    snapshot exposes an <input>'s current value as part of its own description, even for a
    type="password" field, which the screenshot's visual masking does nothing to protect against.
    Field-name redaction cannot catch this at all, since there is no field name to match; only
    knowing the literal value in advance (from watching what the run actually typed) and scrubbing
    for it everywhere closes this gap.

A caller that doesn't want evidence written (e.g. the scripted-client test suite, or any run that
shouldn't touch disk) simply doesn't construct one - nothing in AgentLoop requires it.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from capability_forge.utils.redact import DEFAULT_SENSITIVE_FIELDS, redact

if TYPE_CHECKING:
    # Only for the log_handoff() type hint - a real import would be circular, since
    # escalation/manager.py itself imports EvidenceWriter from this module.
    from capability_forge.escalation.manager import HandoffRecord

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
        # Literal values (typed/selected during the run) to scrub from every subsequent write -
        # see the module docstring's second redaction layer.
        self._secret_values: list[str] = []

    def register_secret(self, value: str) -> None:
        """Register a literal value to be scrubbed from every string written to log.jsonl or
        transcript.json from this point on, wherever it appears - not just in the field it was
        originally typed into. Call this as soon as a value is known (right after a type/select
        is dispatched), before it has a chance to resurface in a later page observation. Empty or
        already-registered values are skipped."""
        if value and value not in self._secret_values:
            self._secret_values.append(value)

    def _scrub_secrets(self, value: Any) -> Any:
        """Recursively replace every registered secret substring, in every string found anywhere
        in value (dict values, list items, or a bare string), with a fixed marker. Longest secret
        first, so a short one that happens to be a substring of a longer one doesn't get replaced
        first and corrupt the longer match. A no-op walk (returns value's own strings unchanged)
        if nothing has been registered yet."""
        if not self._secret_values:
            return value
        if isinstance(value, dict):
            return {k: self._scrub_secrets(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._scrub_secrets(v) for v in value]
        if isinstance(value, str):
            scrubbed = value
            for secret in sorted(self._secret_values, key=len, reverse=True):
                scrubbed = scrubbed.replace(secret, "[REDACTED_INPUT]")
            return scrubbed
        return value

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
        scrubbed_line = self._scrub_secrets(line)
        redacted_line = redact(scrubbed_line, self.sensitive_fields)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(redacted_line) + "\n")

    def log_handoff(self, record: "HandoffRecord") -> None:
        """Append one redacted JSON line to handoffs.jsonl for a completed human-in-the-loop
        escalation. See the module docstring for why this is a separate file from log.jsonl rather
        than another outcome_type value."""
        line = asdict(record)
        scrubbed_line = self._scrub_secrets(line)
        redacted_line = redact(scrubbed_line, self.sensitive_fields)
        handoff_path = self.run_dir / "handoffs.jsonl"
        with handoff_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(redacted_line) + "\n")

    def save_screenshot(self, page: Any) -> str:
        """Save a full-page screenshot for the current step, returning its path relative to
        run_dir (the value stored as a log line's evidence_ref)."""
        self._step_count += 1
        filename = f"step_{self._step_count:02d}.png"
        page.screenshot(path=str(self.screenshots_dir / filename))
        return f"screenshots/{filename}"

    def save_transcript(self, messages: list[dict[str, Any]]) -> None:
        """Write the run's raw LLM message history, redacted and secret-scrubbed, to
        transcript.json. Anthropic SDK response content blocks (assistant turns) aren't plain
        dicts, so they're normalized to dicts first; a tool_use block's "input" is stripped of any
        "value" key outright (see the module docstring's first layer), then the whole structure is
        scrubbed for every registered secret value (the second layer, catching a typed value that
        resurfaces in a later observation's raw page text rather than a tool_use input)."""
        normalized = [self._normalize_message(m) for m in messages]
        scrubbed = self._scrub_secrets(normalized)
        redacted_transcript = redact(scrubbed, self.sensitive_fields)
        transcript_path = self.run_dir / "transcript.json"
        transcript_path.write_text(json.dumps(redacted_transcript, indent=2), encoding="utf-8")

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if isinstance(content, str):
            return {"role": message["role"], "content": content}
        if isinstance(content, list):
            return {"role": message["role"], "content": [self._normalize_block(block) for block in content]}
        return message

    def _normalize_block(self, block: Any) -> dict[str, Any]:
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
