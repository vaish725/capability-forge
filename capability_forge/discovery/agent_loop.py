"""Goal-driven discovery loop.

Drives Claude through an observe -> decide -> act cycle against a live browser surface until the
goal is reached or a stopping condition fires. Produces a DiscoveryRun: an ordered list of
StepActions with literal (non-templated) input values, a checkpoint, and an outcome. Turning a
DiscoveryRun into a versioned, parameterized CapabilityArtifact (declaring InputParam/OutputField,
templating literal values into {{param}} placeholders, assigning schema_version and target) is
artifact_recorder.py's job, not this module's - this module's output is the raw material, not the
finished artifact.

Design decision: what observe() hands to Claude.
Every turn, Claude receives the current URL, title, and accessibility tree (Playwright's own
aria_snapshot() format - role, name, and nesting for every element), trimmed to a fixed character
budget. Deliberately NOT a screenshot: the design's own stated bias is accessibility/DOM-first
perception (screenshot+coordinates is explicitly the last-resort tier, not needed for the primary
target), and adding an image to every turn would roughly double token/latency cost for a
capability this loop doesn't currently need. Concretely, this means discovery-time action
proposals are role/css only - Claude is never asked to reason about pixel coordinates, and the
coordinate locator tier (while still fully supported by the schema and the driver) is not
something a discovery run can produce. If a future target genuinely can't be perceived through the
accessibility tree, that's the point at which to revisit this and add a screenshot channel - not
before, since it's a real ongoing cost for a capability nothing has needed yet.

Trimming is a hard character cutoff (DEFAULT_MAX_TREE_CHARS), not structural filtering (e.g.
dropping non-interactive nodes) - simpler to reason about and to unit test, at the cost of
sometimes truncating mid-structure on a very large page. Good enough for this project's scale;
worth revisiting with real usage data (see the design doc's own note that max_steps/timeout should
be tuned empirically once the loop is actually running).
"""

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic

from capability_forge.guardrails.policy import Guardrail, PolicyViolation
from capability_forge.schema.artifact import Checkpoint, LocatorTier, StepAction
from capability_forge.surfaces.playwright_driver import LocatorResolutionError, PlaywrightDriver

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_STEPS = 25
DEFAULT_TIMEOUT_SECONDS = 180
# How many times the exact same (url, accessibility tree) state must repeat before the loop
# concludes it's stuck and forces a stop, rather than letting Claude retry indefinitely.
DEFAULT_REPEATED_STATE_LIMIT = 3
DEFAULT_MAX_TREE_CHARS = 4000
DEFAULT_MAX_TOKENS = 1024

StopReason = Literal[
    "goal_complete",
    "business_outcome",
    "give_up",
    "max_steps_exceeded",
    "timeout_exceeded",
    "dead_end_detected",
]

# Anthropic tool-use schema for the loop. click/type/select target an element by accessibility
# role + accessible name (the terms Claude reads directly off the accessibility tree it's given,
# never a raw CSS selector or pixel coordinate - see the module docstring's design decision).
# click/type/select also carry the model's own risk self-tag, cross-checked against the
# guardrail's keyword heuristic before being frozen into the recorded step (Section 8 of the
# design doc: the model tags its own action, a keyword heuristic can only escalate it, never
# downgrade it). wait/navigate are structurally always safe (neither submits nor mutates
# anything), so they aren't asked to self-tag.
_RISK_PROPERTY = {
    "type": "string",
    "enum": ["safe_reversible", "risky_irreversible"],
    "description": (
        "Your own assessment of this action's risk. risky_irreversible if it writes, submits, "
        "confirms, deletes, or transfers anything; safe_reversible if it only reads, searches, "
        "or navigates."
    ),
}
_ELEMENT_TARGET_PROPERTIES = {
    "role": {"type": "string", "description": "The element's ARIA role, exactly as shown in the accessibility tree (e.g. 'button', 'textbox')."},
    "name": {"type": "string", "description": "The element's accessible name, exactly as shown in the accessibility tree."},
    "reasoning": {"type": "string", "description": "One sentence: why this action moves toward the goal."},
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "click",
        "description": "Click an element identified by its accessibility role and accessible name.",
        "input_schema": {
            "type": "object",
            "properties": {**_ELEMENT_TARGET_PROPERTIES, "risk": _RISK_PROPERTY},
            "required": ["role", "name", "risk", "reasoning"],
        },
    },
    {
        "name": "type",
        "description": "Type text into a form field identified by its accessibility role and accessible name.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_ELEMENT_TARGET_PROPERTIES,
                "value": {"type": "string", "description": "The literal text to type."},
                "risk": _RISK_PROPERTY,
            },
            "required": ["role", "name", "value", "risk", "reasoning"],
        },
    },
    {
        "name": "select",
        "description": "Choose an option in a dropdown identified by its accessibility role and accessible name.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_ELEMENT_TARGET_PROPERTIES,
                "value": {"type": "string", "description": "The option value to select."},
                "risk": _RISK_PROPERTY,
            },
            "required": ["role", "name", "value", "risk", "reasoning"],
        },
    },
    {
        "name": "navigate",
        "description": "Navigate directly to a URL. Only for an explicit destination change - use click for links.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["url", "reasoning"],
        },
    },
    {
        "name": "wait",
        "description": "Wait for an element identified by its accessibility role and accessible name to appear, e.g. after a slow-loading action.",
        "input_schema": {
            "type": "object",
            "properties": _ELEMENT_TARGET_PROPERTIES,
            "required": ["role", "name", "reasoning"],
        },
    },
    {
        "name": "extract",
        "description": "Read the text of an element identified by its role and accessible name, without performing any action. Use this to check a value before deciding what to do next, or to identify what the final output should be read from.",
        "input_schema": {
            "type": "object",
            "properties": _ELEMENT_TARGET_PROPERTIES,
            "required": ["role", "name", "reasoning"],
        },
    },
    {
        "name": "done",
        "description": "Declare the run finished. Must identify an element (the checkpoint) that proves this terminal state was actually reached - this is verified before the run is accepted, your word alone is not enough.",
        "input_schema": {
            "type": "object",
            "properties": {
                "outcome_type": {
                    "type": "string",
                    "enum": ["success", "business_outcome"],
                    "description": (
                        "'success' if the original goal was achieved. 'business_outcome' if a "
                        "named, expected non-error result was reached instead (e.g. the "
                        "requested record doesn't exist, or a requested amount exceeds what's "
                        "available) - this is a legitimate result, not a failure."
                    ),
                },
                "business_outcome_reason": {
                    "type": "string",
                    "description": "Required if outcome_type is business_outcome: a short snake_case name, e.g. 'member_not_found'.",
                },
                "checkpoint_role": {"type": "string", "description": "ARIA role of the element proving this terminal state was reached."},
                "checkpoint_name": {"type": "string", "description": "Accessible name of that element."},
                "summary": {"type": "string", "description": "One sentence describing what was accomplished or found."},
            },
            "required": ["outcome_type", "checkpoint_role", "checkpoint_name", "summary"],
        },
    },
    {
        "name": "give_up",
        "description": "Declare that the goal cannot be completed - use this if blocked by an unrecoverable error, or you've tried reasonable approaches without success. Do not repeat the same action indefinitely.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


def trim_accessibility_tree(tree: str, max_chars: int) -> str:
    """Hard character cutoff, not structural filtering - simpler to reason about and to unit
    test, at the cost of sometimes truncating mid-structure on a very large page. See the module
    docstring's design decision for why this is what observe() output looks like to Claude."""
    if len(tree) <= max_chars:
        return tree
    return tree[:max_chars] + "\n... [accessibility tree truncated for length]"


def build_system_prompt(goal_description: str) -> str:
    """The instructions Claude operates under for the whole run. Goal is embedded once here
    rather than repeated every turn, since it doesn't change during the run."""
    return f"""You are operating a legacy web application to accomplish a specific goal. You interact with the page only through the provided tools, identifying elements by their accessibility role and accessible name exactly as shown in the accessibility tree you're given after each action.

Goal: {goal_description}

Guidance:
- Read the accessibility tree carefully before each action. Role and name must match what is shown.
- If you encounter an unexpected notice, dialog, or interstitial blocking your path, look for a way to dismiss or continue past it (e.g. a "Continue", "OK", or "Dismiss" button) before giving up. This is often a normal, recoverable part of the flow, not a failure.
- If the goal cannot be completed because of the specific input you were given (for example, a record genuinely does not exist, or a requested amount exceeds what is available), that is a valid, expected result, not a failure. Call done with outcome_type="business_outcome" and a short business_outcome_reason. Do not retry a business outcome as if it were an error.
- Only call done with outcome_type="success" once you can point to a specific element (role and name) that proves the goal was actually achieved.
- If you are genuinely stuck - an unrecoverable error, or you've tried multiple reasonable approaches with no progress - call give_up with a clear reason rather than repeating the same action.
- Use extract to read a value off the page when you need to check something before deciding what to do next.
"""


@dataclass
class DiscoveryRun:
    """The raw output of a discovery run: what happened, not yet a reusable artifact."""

    goal_description: str
    target_url: str
    steps: list[StepAction]
    stop_reason: StopReason
    checkpoint: Checkpoint | None = None
    business_outcome_reason: str | None = None
    summary: str | None = None
    # Values read via the extract tool during the run, kept for a future artifact_recorder to
    # reference when deciding what belongs in checkpoint.extract - not interpreted by this module.
    extract_log: list[dict[str, str]] = field(default_factory=list)


@dataclass
class _ToolDispatchResult:
    """Internal result of executing one tool call. Either the loop continues (stop_reason is None,
    tool_result_text is fed back to Claude) or it ends here (stop_reason set, remaining fields
    populate the DiscoveryRun)."""

    tool_result_text: str = ""
    stop_reason: StopReason | None = None
    recorded_step: StepAction | None = None
    extract_entry: dict[str, str] | None = None
    checkpoint: Checkpoint | None = None
    business_outcome_reason: str | None = None
    summary: str | None = None


class AgentLoop:
    """Runs one discovery loop against a driver-controlled page."""

    def __init__(
        self,
        driver: PlaywrightDriver,
        guardrail: Guardrail,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        repeated_state_limit: int = DEFAULT_REPEATED_STATE_LIMIT,
        max_tree_chars: int = DEFAULT_MAX_TREE_CHARS,
    ):
        self.driver = driver
        self.guardrail = guardrail
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.repeated_state_limit = repeated_state_limit
        self.max_tree_chars = max_tree_chars

    def run(self, goal_description: str, target_url: str) -> DiscoveryRun:
        steps: list[StepAction] = []
        extract_log: list[dict[str, str]] = []
        state_counts: dict[str, int] = {}
        start_time = time.monotonic()
        system_prompt = build_system_prompt(goal_description)
        messages: list[dict[str, Any]] = []

        # Implicit initial navigation to the target - not itself a recorded StepAction (see the
        # schema's own design note: target.base_url is where a replay starts, before step 1).
        self.driver.page.goto(target_url)

        def finish(**kwargs: Any) -> DiscoveryRun:
            return DiscoveryRun(
                goal_description=goal_description,
                target_url=target_url,
                steps=steps,
                extract_log=extract_log,
                **kwargs,
            )

        for _ in range(self.max_steps):
            if time.monotonic() - start_time > self.timeout_seconds:
                return finish(stop_reason="timeout_exceeded")

            snapshot = self.driver.observe()
            state_key = f"{snapshot.url}::{snapshot.accessibility_tree}"
            state_counts[state_key] = state_counts.get(state_key, 0) + 1
            if state_counts[state_key] >= self.repeated_state_limit:
                return finish(stop_reason="dead_end_detected")

            tree = trim_accessibility_tree(snapshot.accessibility_tree, self.max_tree_chars)
            observation = f"Current page:\nURL: {snapshot.url}\nTitle: {snapshot.title}\n\nAccessibility tree:\n{tree}"
            messages.append({"role": "user", "content": observation})

            response = self.client.messages.create(
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
            if not tool_use_blocks:
                messages.append({"role": "user", "content": "Please respond by calling exactly one tool."})
                continue

            block = tool_use_blocks[0]  # only the first tool call per turn is acted on
            result = self._dispatch_tool_call(block.name, block.input, target_url, len(steps) + 1)

            if result.stop_reason is not None:
                return finish(
                    stop_reason=result.stop_reason,
                    checkpoint=result.checkpoint,
                    business_outcome_reason=result.business_outcome_reason,
                    summary=result.summary,
                )

            if result.recorded_step is not None:
                steps.append(result.recorded_step)
            if result.extract_entry is not None:
                extract_log.append(result.extract_entry)

            # Every tool_use block in the assistant's turn needs a matching tool_result in the
            # next message, or the API rejects the following request outright - confirmed against
            # the real API, not something the scripted-client tests exercise, since Claude
            # sometimes proposes more than one tool call in a single turn even though only the
            # first is ever acted on here. The unexecuted ones get a placeholder result so Claude
            # knows to reissue them next turn instead of assuming they ran.
            tool_results = [{"type": "tool_result", "tool_use_id": block.id, "content": result.tool_result_text}]
            for extra_block in tool_use_blocks[1:]:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": extra_block.id,
                        "content": "Not executed: only one tool call is processed per turn. Re-propose this if still needed, after seeing the result of the first.",
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return finish(stop_reason="max_steps_exceeded")

    def _dispatch_tool_call(self, name: str, tool_input: dict[str, Any], target_url: str, next_step_number: int) -> _ToolDispatchResult:
        if name == "done":
            return self._handle_done(tool_input)
        if name == "give_up":
            return _ToolDispatchResult(stop_reason="give_up", summary=tool_input.get("reason"))
        if name == "extract":
            return self._handle_extract(tool_input)
        if name == "navigate":
            return self._handle_navigate(tool_input, target_url, next_step_number)
        if name in ("click", "type", "select", "wait"):
            return self._handle_element_action(name, tool_input, target_url, next_step_number)
        return _ToolDispatchResult(tool_result_text=f"Unknown tool {name!r}.")

    def _handle_element_action(self, action_type: str, tool_input: dict[str, Any], target_url: str, step_number: int) -> _ToolDispatchResult:
        role = tool_input.get("role", "")
        name = tool_input.get("name", "")
        value = tool_input.get("value")
        model_risk_tag = tool_input.get("risk", "safe_reversible")

        try:
            self.guardrail.check_action(target_url, action_type)
        except PolicyViolation as exc:
            return _ToolDispatchResult(tool_result_text=f"Blocked by policy: {exc}")

        try:
            locators = self.driver.build_locator_tiers(role, name)
        except LocatorResolutionError:
            return _ToolDispatchResult(
                tool_result_text=(
                    f"Could not find an element with role={role!r} name={name!r}. "
                    "Check the accessibility tree and try again with an exact match."
                )
            )

        # Risk is frozen here, at proposal time, from the model's own tag cross-checked against
        # the keyword heuristic - never re-derived later. See Section 8 of the design doc.
        risk = self.guardrail.classify_risk(name, model_risk_tag)
        step = StepAction(
            step_id=f"step_{step_number}",
            action_type=action_type,
            locators=locators,
            input_value=value,
            risk=risk,
            description=tool_input.get("reasoning") or f"{action_type} on role={role} name={name}",
        )

        try:
            self.driver.act(step)
        except Exception as exc:  # noqa: BLE001 - genuinely any failure here should be reported back, not crash the run
            return _ToolDispatchResult(tool_result_text=f"Action failed: {exc}. Check the current page state and try a different approach.")

        return _ToolDispatchResult(recorded_step=step, tool_result_text=f"{action_type} succeeded.")

    def _handle_navigate(self, tool_input: dict[str, Any], target_url: str, step_number: int) -> _ToolDispatchResult:
        url = tool_input.get("url", "")
        try:
            self.guardrail.check_action(url or target_url, "navigate")
        except PolicyViolation as exc:
            return _ToolDispatchResult(tool_result_text=f"Blocked by policy: {exc}")

        step = StepAction(
            step_id=f"step_{step_number}",
            action_type="navigate",
            locators=[],
            input_value=url,
            risk="safe_reversible",
            description=tool_input.get("reasoning") or f"Navigate to {url}",
        )
        try:
            self.driver.act(step)
        except Exception as exc:  # noqa: BLE001
            return _ToolDispatchResult(tool_result_text=f"Navigation failed: {exc}.")
        return _ToolDispatchResult(recorded_step=step, tool_result_text=f"Navigated to {url}.")

    def _handle_extract(self, tool_input: dict[str, Any]) -> _ToolDispatchResult:
        role = tool_input.get("role", "")
        name = tool_input.get("name", "")
        locator = self.driver.resolve_role_name(role, name)
        if locator is None:
            return _ToolDispatchResult(tool_result_text=f"Could not find an element with role={role!r} name={name!r} to extract.")
        value = locator.inner_text()
        return _ToolDispatchResult(
            extract_entry={"role": role, "name": name, "value": value},
            tool_result_text=f"Extracted value: {value!r}",
        )

    def _handle_done(self, tool_input: dict[str, Any]) -> _ToolDispatchResult:
        checkpoint_role = tool_input.get("checkpoint_role", "")
        checkpoint_name = tool_input.get("checkpoint_name", "")
        outcome_type = tool_input.get("outcome_type", "success")
        summary = tool_input.get("summary")
        business_outcome_reason = tool_input.get("business_outcome_reason")

        # Never trust the model's own claim of success - verify the checkpoint element is
        # actually present before accepting the run as complete (design doc Section 4.2: emitted
        # only after a discovery run ends in success and its checkpoint is verified).
        locator = self.driver.resolve_role_name(checkpoint_role, checkpoint_name)
        if locator is None:
            return _ToolDispatchResult(
                tool_result_text=(
                    f"Could not verify: no element found with role={checkpoint_role!r} "
                    f"name={checkpoint_name!r}. The goal is not confirmed complete - check the "
                    "current page, or call give_up if truly stuck."
                )
            )

        role_value = f'role={checkpoint_role}[name="{checkpoint_name}"]'
        checkpoint = Checkpoint(
            description=summary or "Goal reached",
            locator=LocatorTier(strategy="role", value=role_value, confidence=0.9),
            extract=None,
        )
        stop_reason: StopReason = "goal_complete" if outcome_type == "success" else "business_outcome"
        return _ToolDispatchResult(
            stop_reason=stop_reason,
            checkpoint=checkpoint,
            business_outcome_reason=business_outcome_reason,
            summary=summary,
        )
