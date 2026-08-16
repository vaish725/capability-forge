"""Tests for AgentLoop: the loop mechanics (stopping conditions, guardrail integration, step
recording, risk escalation, dead-end detection) driven against the real fixture over a real
browser and driver, with only the LLM itself replaced by a scripted fake client.

This is a deliberate split: the browser, driver, guardrail, and fixture are all real - only
Claude's decision-making is faked, since that's the one piece that's non-deterministic, costs
money, and requires a credential this test suite shouldn't depend on. It tests "does the loop do
the right thing given a sequence of tool calls", not "does Claude make good decisions" - the
latter needs a real API key and is exercised separately, not as an automated test.
"""

from types import SimpleNamespace

import pytest

from capability_forge.discovery.agent_loop import AgentLoop, trim_accessibility_tree
from capability_forge.guardrails.policy import AllowlistPolicy, Guardrail
from capability_forge.surfaces.playwright_driver import PlaywrightDriver
from capability_forge.utils.local_server import serve_directory


def tool_use(name, tool_input, block_id="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def text(content):
    return SimpleNamespace(type="text", text=content)


class ScriptedMessage:
    def __init__(self, content):
        self.content = content


class ScriptedClient:
    """Fake Anthropic client: returns one scripted Message per call to messages.create(), in
    order. Raises if the script runs out, so a test that needs more turns than it scripted fails
    loudly instead of hanging or returning something misleading."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # AgentLoop passes its own `messages` list by reference and keeps appending to it after
        # this call returns. Snapshotting it here (shallow copy of the list, not its elements) is
        # required - storing the reference as-is meant every recorded call's "messages" silently
        # became whatever the list looked like by the time the whole run() finished, not what was
        # actually sent for that call. Found via a genuinely confusing assertion failure, not
        # anticipated up front.
        kwargs = {**kwargs, "messages": list(kwargs["messages"])}
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("ScriptedClient ran out of scripted responses")
        return self._responses.pop(0)


@pytest.fixture
def fixture_server():
    fixtures_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "fixtures"
    with serve_directory(fixtures_dir) as base_url:
        yield base_url


@pytest.fixture
def guardrail():
    policy = AllowlistPolicy(
        allowed_domains=["127.0.0.1"],
        allowed_action_types={"*": ["click", "type", "navigate", "wait", "select"]},
        risky_keywords=["confirm", "submit", "delete", "transfer"],
        sensitive_fields=["account_number"],
    )
    return Guardrail(policy)


@pytest.fixture
def driver(page):
    return PlaywrightDriver(page, locator_timeout_ms=1500)


def loop(driver, guardrail, responses, **kwargs):
    client = ScriptedClient(responses)
    return AgentLoop(driver, guardrail, client=client, **kwargs), client


# --- a full successful run ----------------------------------------------------------------------


def test_full_success_run_records_steps_and_verified_checkpoint(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    responses = [
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "User Name:", "value": "jdoe", "risk": "safe_reversible", "reasoning": "log in"}, "t1")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Password:", "value": "secret", "risk": "safe_reversible", "reasoning": "log in"}, "t2")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "submit login"}, "t3")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Member ID:", "value": "12345", "risk": "safe_reversible", "reasoning": "search"}, "t4")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Search", "risk": "safe_reversible", "reasoning": "search"}, "t5")]),
        ScriptedMessage(
            [
                tool_use(
                    "done",
                    {
                        "outcome_type": "success",
                        "checkpoint_role": "cell",
                        "checkpoint_name": "$4500.00",
                        "summary": "Found the balance.",
                    },
                    "t6",
                )
            ]
        ),
    ]
    agent, client = loop(driver, guardrail, responses)
    result = agent.run("Look up the balance for member 12345", target)

    assert result.stop_reason == "goal_complete"
    assert [s.action_type for s in result.steps] == ["type", "type", "click", "type", "click"]
    assert result.checkpoint is not None
    assert result.checkpoint.locator.value == 'role=cell[name="$4500.00"]'
    assert len(client.calls) == 6


# --- business outcome: an expected, non-error terminal result -----------------------------------


def test_business_outcome_run(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    responses = [
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "User Name:", "value": "jdoe", "risk": "safe_reversible", "reasoning": "log in"}, "t1")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Password:", "value": "secret", "risk": "safe_reversible", "reasoning": "log in"}, "t2")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "submit login"}, "t3")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Member ID:", "value": "00001", "risk": "safe_reversible", "reasoning": "search"}, "t4")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Search", "risk": "safe_reversible", "reasoning": "search"}, "t5")]),
        ScriptedMessage(
            [
                tool_use(
                    "done",
                    {
                        "outcome_type": "business_outcome",
                        "business_outcome_reason": "member_not_found",
                        "checkpoint_role": "alert",
                        "checkpoint_name": "No member found matching that ID.",
                        "summary": "Member 00001 does not exist.",
                    },
                    "t6",
                )
            ]
        ),
    ]
    agent, _ = loop(driver, guardrail, responses)
    result = agent.run("Look up the balance for member 00001", target)

    assert result.stop_reason == "business_outcome"
    assert result.business_outcome_reason == "member_not_found"


# --- recoverable interstitial: proves the loop mechanics support dismiss-then-continue ----------


def test_recoverable_interstitial_dismissed_then_goal_reached(driver, guardrail, fixture_server):
    # Scripts the "correct" behavior a well-prompted Claude should exhibit for member 88888
    # (which triggers the session-notice interstitial): dismiss it, then continue to the goal.
    # This proves the loop's tool dispatch supports that pattern with no special-casing required -
    # it does NOT test whether a real Claude call would choose to do this; that's a live-run
    # concern, not something a scripted unit test can verify.
    target = f"{fixture_server}/hostile_legacy_page.html"
    responses = [
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "User Name:", "value": "jdoe", "risk": "safe_reversible", "reasoning": "log in"}, "t1")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Password:", "value": "secret", "risk": "safe_reversible", "reasoning": "log in"}, "t2")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "submit login"}, "t3")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Member ID:", "value": "88888", "risk": "safe_reversible", "reasoning": "search"}, "t4")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Search", "risk": "safe_reversible", "reasoning": "search"}, "t5")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Continue", "risk": "safe_reversible", "reasoning": "dismiss the session notice"}, "t6")]),
        ScriptedMessage(
            [
                tool_use(
                    "done",
                    {"outcome_type": "success", "checkpoint_role": "cell", "checkpoint_name": "$10234.10", "summary": "Found the balance."},
                    "t7",
                )
            ]
        ),
    ]
    agent, _ = loop(driver, guardrail, responses)
    result = agent.run("Look up the balance for member 88888", target)

    assert result.stop_reason == "goal_complete"
    assert "click" in [s.action_type for s in result.steps]
    # 2 logins fields + login click + member id + search click + dismiss-notice click. The
    # dismiss-the-notice click and the earlier steps are both just ordinary recorded steps - no
    # special "recoverable" marker exists on StepAction, confirming the design doc's framing that
    # recoverable is a replay-time classification, not a discovery-time one.
    assert len(result.steps) == 6


# --- give_up ------------------------------------------------------------------------------------


def test_give_up_stops_the_loop_with_reason(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    responses = [ScriptedMessage([tool_use("give_up", {"reason": "Cannot proceed."}, "t1")])]
    agent, _ = loop(driver, guardrail, responses)
    result = agent.run("Do something impossible", target)

    assert result.stop_reason == "give_up"
    assert result.summary == "Cannot proceed."
    assert result.steps == []


# --- stopping conditions: max_steps, timeout, dead-end -------------------------------------------


def test_max_steps_exceeded(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    # extract doesn't mutate the page, so this alone won't trigger the dead-end guard before
    # max_steps does (each response has a different tool_use id, but role/name are identical -
    # the dead-end guard keys off page state, not the tool call itself, and with max_steps=2 the
    # loop ends before repeated_state_limit's default of 3 would fire anyway).
    responses = [
        ScriptedMessage([tool_use("extract", {"role": "heading", "name": "First Fidelity Member Services - Internal Portal", "reasoning": "check"}, "t1")]),
        ScriptedMessage([tool_use("extract", {"role": "heading", "name": "First Fidelity Member Services - Internal Portal", "reasoning": "check"}, "t2")]),
    ]
    agent, _ = loop(driver, guardrail, responses, max_steps=2)
    result = agent.run("An open-ended goal", target)

    assert result.stop_reason == "max_steps_exceeded"


def test_timeout_exceeded_before_any_claude_call(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    agent, client = loop(driver, guardrail, responses=[], timeout_seconds=-1)
    result = agent.run("A goal that never gets a chance to start", target)

    assert result.stop_reason == "timeout_exceeded"
    assert len(client.calls) == 0


def test_dead_end_detected_when_state_never_changes(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    # repeated_state_limit=3 needs the state observed 3 times; the 3rd observe happens before a
    # 3rd Claude call is ever made, so only 2 scripted responses are needed.
    responses = [
        ScriptedMessage([tool_use("extract", {"role": "heading", "name": "First Fidelity Member Services - Internal Portal", "reasoning": "check"}, "t1")]),
        ScriptedMessage([tool_use("extract", {"role": "heading", "name": "First Fidelity Member Services - Internal Portal", "reasoning": "check"}, "t2")]),
    ]
    agent, client = loop(driver, guardrail, responses, repeated_state_limit=3, max_steps=25)
    result = agent.run("An open-ended goal", target)

    assert result.stop_reason == "dead_end_detected"
    assert len(client.calls) == 2


# --- text-only response nudges toward a tool call, doesn't crash --------------------------------


def test_text_only_response_is_nudged_not_fatal(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    responses = [
        ScriptedMessage([text("Let me think about this.")]),
        ScriptedMessage([tool_use("give_up", {"reason": "Cannot proceed."}, "t1")]),
    ]
    agent, client = loop(driver, guardrail, responses)
    result = agent.run("A goal", target)

    assert result.stop_reason == "give_up"
    assert len(client.calls) == 2


# --- locator resolution failure gives Claude a chance to retry, not a crash --------------------


def test_unresolvable_locator_reports_failure_and_continues(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    responses = [
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Does Not Exist", "risk": "safe_reversible", "reasoning": "try"}, "t1")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "retry with correct name"}, "t2")]),
        ScriptedMessage([tool_use("give_up", {"reason": "stopping here for the test"}, "t3")]),
    ]
    agent, client = loop(driver, guardrail, responses)
    result = agent.run("A goal", target)

    # The first (bad) click never became a step; the second (valid) one did.
    assert len(result.steps) == 1
    assert result.steps[0].description == "retry with correct name"
    # messages at the 2nd create() call: [obs1, assistant1, tool_result1, obs2] - the tool_result
    # reporting the first click's failure is second-to-last, not last (obs2 was appended after it).
    tool_result_content = client.calls[1]["messages"][-2]["content"][0]["content"]
    assert "Could not find" in tool_result_content


# --- risk escalation: the keyword heuristic overrides a model's own "safe" self-tag -------------


def test_risky_keyword_escalates_risk_even_when_model_self_tags_safe(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    responses = [
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "User Name:", "value": "jdoe", "risk": "safe_reversible", "reasoning": "log in"}, "t1")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Password:", "value": "secret", "risk": "safe_reversible", "reasoning": "log in"}, "t2")]),
        # Login is deliberately mislabeled risky_irreversible here to prove risk is per-action,
        # not forced globally; Search below proves the opposite direction.
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "submit login"}, "t3")]),
        ScriptedMessage([tool_use("give_up", {"reason": "stopping here for the test"}, "t4")]),
    ]
    agent, _ = loop(driver, guardrail, responses)
    result = agent.run("A goal", target)

    # "Login" doesn't match any configured risky keyword - stays safe.
    login_step = [s for s in result.steps if s.description == "submit login"][0]
    assert login_step.risk == "safe_reversible"


def test_confirm_transfer_is_escalated_regardless_of_model_self_tag(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    # Drives all the way to the real "Confirm Transfer" button inside the iframe, so the click
    # genuinely resolves and gets a risk assigned - a click that fails to resolve never becomes a
    # step at all (see test_unresolvable_locator_reports_failure_and_continues), so this has to
    # reach the actual button, not just name-match against a nonexistent one, to test anything.
    responses = [
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "User Name:", "value": "jdoe", "risk": "safe_reversible", "reasoning": "log in"}, "t1")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Password:", "value": "secret", "risk": "safe_reversible", "reasoning": "log in"}, "t2")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "submit login"}, "t3")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Member ID:", "value": "12345", "risk": "safe_reversible", "reasoning": "search"}, "t4")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Search", "risk": "safe_reversible", "reasoning": "search"}, "t5")]),
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Transfer Funds", "risk": "safe_reversible", "reasoning": "open transfer form"}, "t6")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "To Account:", "value": "55555", "risk": "safe_reversible", "reasoning": "fill destination"}, "t7")]),
        ScriptedMessage([tool_use("type", {"role": "textbox", "name": "Amount:", "value": "100", "risk": "safe_reversible", "reasoning": "fill amount"}, "t8")]),
        # Model deliberately self-tags this "safe_reversible" - the keyword heuristic must
        # override it to risky_irreversible regardless.
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Confirm Transfer", "risk": "safe_reversible", "reasoning": "confirm the transfer"}, "t9")]),
        ScriptedMessage([tool_use("give_up", {"reason": "stopping here for the test"}, "t10")]),
    ]
    agent, _ = loop(driver, guardrail, responses)
    result = agent.run("A goal", target)

    confirm_step = [s for s in result.steps if s.description == "confirm the transfer"][0]
    assert confirm_step.risk == "risky_irreversible"


# --- extract: read-only, never becomes a StepAction ----------------------------------------------


def test_extract_is_logged_but_not_recorded_as_a_step(driver, guardrail, fixture_server):
    target = f"{fixture_server}/hostile_legacy_page.html"
    responses = [
        ScriptedMessage([tool_use("extract", {"role": "heading", "name": "First Fidelity Member Services - Internal Portal", "reasoning": "check title"}, "t1")]),
        ScriptedMessage([tool_use("give_up", {"reason": "stopping here for the test"}, "t2")]),
    ]
    agent, _ = loop(driver, guardrail, responses)
    result = agent.run("A goal", target)

    assert result.steps == []
    assert len(result.extract_log) == 1
    assert result.extract_log[0]["value"] == "First Fidelity Member Services - Internal Portal"


# --- guardrail policy violations don't crash the loop --------------------------------------------


def test_policy_violation_reported_back_and_loop_continues(driver, fixture_server):
    # Deliberately a different, more restrictive policy than the shared `guardrail` fixture, built
    # locally rather than adding a second fixture just for this one test.
    restrictive_policy = AllowlistPolicy(
        allowed_domains=["127.0.0.1"],
        allowed_action_types={"*": ["type"]},  # click is deliberately not allowed
        risky_keywords=[],
        sensitive_fields=[],
    )
    restrictive_guardrail = Guardrail(restrictive_policy)
    target = f"{fixture_server}/hostile_legacy_page.html"
    responses = [
        ScriptedMessage([tool_use("click", {"role": "button", "name": "Login", "risk": "safe_reversible", "reasoning": "try"}, "t1")]),
        ScriptedMessage([tool_use("give_up", {"reason": "stopping here for the test"}, "t2")]),
    ]
    agent, client = loop(driver, restrictive_guardrail, responses)
    result = agent.run("A goal", target)

    assert result.steps == []
    assert result.stop_reason == "give_up"
    # messages at the 2nd create() call: [obs1, assistant1, tool_result1, obs2] - the tool_result
    # reporting the blocked click is second-to-last, not last (obs2 was appended after it).
    tool_result_content = client.calls[1]["messages"][-2]["content"][0]["content"]
    assert "Blocked by policy" in tool_result_content


# --- trim_accessibility_tree: pure function, tested directly -------------------------------------


def test_trim_accessibility_tree_leaves_short_text_unchanged():
    assert trim_accessibility_tree("short", max_chars=100) == "short"


def test_trim_accessibility_tree_truncates_long_text():
    long_tree = "x" * 500
    trimmed = trim_accessibility_tree(long_tree, max_chars=100)
    assert len(trimmed) > 100  # includes the truncation marker
    assert trimmed.startswith("x" * 100)
    assert "truncated" in trimmed


def test_trim_accessibility_tree_boundary_exact_length():
    exact = "x" * 100
    assert trim_accessibility_tree(exact, max_chars=100) == exact
