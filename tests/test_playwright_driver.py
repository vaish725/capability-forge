"""Tests for PlaywrightDriver against the hostile legacy fixture: locator-tier resolution and
fallback (including the ambiguous-match-is-not-a-match rule), frame-awareness for the iframe
detail view, every action_type, checkpoint verification, and {{param}} templating end to end.
"""

from pathlib import Path

import pytest

from capability_forge.schema.artifact import Checkpoint, LocatorTier, StepAction
from capability_forge.surfaces.playwright_driver import (
    CheckpointNotReachedError,
    LocatorResolutionError,
    PlaywrightDriver,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "hostile_legacy_page.html"
FIXTURE_URL = f"file://{FIXTURE_PATH}"


def locator(strategy, value, confidence=0.9):
    return LocatorTier(strategy=strategy, value=value, confidence=confidence)


def step(step_id="step", action_type="click", locators=None, input_value=None, risk="safe_reversible", description="a step"):
    return StepAction(
        step_id=step_id,
        action_type=action_type,
        locators=locators if locators is not None else [locator("role", 'role=button[name="Login"]')],
        input_value=input_value,
        risk=risk,
        description=description,
    )


@pytest.fixture
def driver(page):
    page.goto(FIXTURE_URL)
    # Short locator timeout for the test suite: several tests deliberately use a broken or
    # nonexistent locator to exercise fallback/failure paths, and the fixture's own async content
    # (the iframe) settles in well under 100ms, so this is generous for correctness without making
    # the suite pay the 10s production default on every intentional miss.
    return PlaywrightDriver(page, locator_timeout_ms=1500)


def login_via_driver(driver, username="jdoe", password="secret"):
    """Drives the login form through the driver itself (dogfooding act()), rather than raw
    Playwright calls, so setup doubles as further exercise of the type/click actions."""
    driver.act(step(action_type="type", locators=[locator("css", "#txtUserName")], input_value=username))
    driver.act(step(action_type="type", locators=[locator("css", "#txtPassword")], input_value=password))
    driver.act(step(action_type="click", locators=[locator("role", 'role=button[name="Login"]')]))


def search_via_driver(driver, member_id):
    driver.act(step(action_type="type", locators=[locator("css", "#txtMemberId")], input_value=member_id))
    driver.act(step(action_type="click", locators=[locator("role", 'role=button[name="Search"]')]))


# --- observe -------------------------------------------------------------------------------


def test_observe_returns_url_title_and_accessibility_tree(driver):
    snapshot = driver.observe()
    assert snapshot.url == FIXTURE_URL
    assert snapshot.title == "First Fidelity Member Services"
    assert "Login" in snapshot.accessibility_tree
    assert "textbox" in snapshot.accessibility_tree


# --- click: role tier, css tier, fallback, ambiguity, coordinate ------------------------------


def test_click_via_role_tier(driver):
    login_via_driver(driver)
    assert driver.page.locator("#screenAccounts").is_visible()


def test_click_falls_back_to_css_tier_when_role_tier_is_wrong(driver):
    # First tier deliberately references a nonexistent role/name; the driver must fall through to
    # the css tier rather than failing outright.
    s = step(
        action_type="click",
        locators=[
            locator("role", 'role=button[name="Not A Real Button"]'),
            locator("css", "#btnLogin"),
        ],
    )
    driver.act(step(action_type="type", locators=[locator("css", "#txtUserName")], input_value="jdoe"))
    driver.act(step(action_type="type", locators=[locator("css", "#txtPassword")], input_value="secret"))
    driver.act(s)
    assert driver.page.locator("#screenAccounts").is_visible()


def test_ambiguous_css_tier_is_skipped_in_favor_of_a_later_unique_tier(driver):
    # ".btn" matches multiple buttons on the login screen alone - this must NOT resolve as a
    # match (which would risk clicking the wrong button); the driver should fall through to the
    # role tier, which does uniquely identify the Login button.
    login_locators = [
        locator("css", ".btn", confidence=0.3),
        locator("role", 'role=button[name="Login"]', confidence=0.9),
    ]
    assert driver.page.locator(".btn").count() > 1, "test assumption: .btn must be ambiguous on this screen"
    driver.act(step(action_type="type", locators=[locator("css", "#txtUserName")], input_value="jdoe"))
    driver.act(step(action_type="type", locators=[locator("css", "#txtPassword")], input_value="secret"))
    driver.act(step(action_type="click", locators=login_locators))
    assert driver.page.locator("#screenAccounts").is_visible()


def test_click_via_coordinate_tier_only(driver):
    driver.act(step(action_type="type", locators=[locator("css", "#txtUserName")], input_value="jdoe"))
    driver.act(step(action_type="type", locators=[locator("css", "#txtPassword")], input_value="secret"))
    box = driver.page.locator("#btnLogin").bounding_box()
    coord_value = f"{box['x'] + box['width'] / 2},{box['y'] + box['height'] / 2}"
    driver.act(step(action_type="click", locators=[locator("coordinate", coord_value)]))
    assert driver.page.locator("#screenAccounts").is_visible()


def test_locator_resolution_error_when_no_tier_resolves(driver):
    with pytest.raises(LocatorResolutionError):
        driver.act(step(action_type="click", locators=[locator("role", 'role=button[name="Does Not Exist"]')]))


# --- type: role, css, coordinate, and {{param}} templating ------------------------------------


def test_type_via_role_tier(driver):
    driver.act(
        step(
            action_type="type",
            locators=[locator("role", 'role=textbox[name="User Name:"]')],
            input_value="jdoe",
        )
    )
    assert driver.page.locator("#txtUserName").input_value() == "jdoe"


def test_type_via_coordinate_tier(driver):
    box = driver.page.locator("#txtUserName").bounding_box()
    coord_value = f"{box['x'] + box['width'] / 2},{box['y'] + box['height'] / 2}"
    driver.act(step(action_type="type", locators=[locator("coordinate", coord_value)], input_value="jdoe"))
    assert driver.page.locator("#txtUserName").input_value() == "jdoe"


def test_type_resolves_template_params(driver):
    s = step(
        action_type="type",
        locators=[locator("css", "#txtUserName")],
        input_value="{{username}}",
    )
    driver.act(s, params={"username": "jdoe"})
    assert driver.page.locator("#txtUserName").input_value() == "jdoe"


def test_type_missing_param_raises_value_error(driver):
    s = step(action_type="type", locators=[locator("css", "#txtUserName")], input_value="{{username}}")
    with pytest.raises(ValueError, match="unresolved param"):
        driver.act(s, params={})


# --- frame-awareness: acting inside the iframe without special handling -----------------------


def test_click_inside_iframe_via_role_tier(driver):
    login_via_driver(driver)
    search_via_driver(driver, "12345")
    driver.act(step(action_type="click", locators=[locator("role", 'role=button[name="Transfer Funds"]')]))
    frame = driver.page.frame_locator("#detailFrame")
    assert frame.locator("#txtToAccount").is_visible()


def test_select_via_role_tier_inside_iframe(driver):
    login_via_driver(driver)
    search_via_driver(driver, "12345")
    driver.act(step(action_type="click", locators=[locator("role", 'role=button[name="Transfer Funds"]')]))
    driver.act(
        step(
            action_type="select",
            locators=[locator("role", 'role=combobox[name="Transfer Type:"]')],
            input_value="savings",
        )
    )
    frame = driver.page.frame_locator("#detailFrame")
    assert frame.locator("#selTransferType").input_value() == "savings"


def test_select_via_coordinate_tier_is_unsupported(driver):
    login_via_driver(driver)
    search_via_driver(driver, "12345")
    driver.act(step(action_type="click", locators=[locator("role", 'role=button[name="Transfer Funds"]')]))
    with pytest.raises(ValueError, match="not supported via the coordinate locator tier"):
        driver.act(step(action_type="select", locators=[locator("coordinate", "10,10")], input_value="savings"))


# --- wait: the exact race the fixture test suite caught earlier, driven through the driver -----


def test_wait_blocks_until_iframe_content_is_actually_ready(driver):
    login_via_driver(driver)
    # Deliberately no pause here: search_via_driver's click returns as soon as the click itself is
    # processed, before the iframe's srcdoc has necessarily finished loading. The "wait" action
    # must block until .balance-value is really there, not just return immediately.
    search_via_driver(driver, "12345")
    driver.act(step(action_type="wait", locators=[locator("css", ".balance-value")]))
    frame = driver.page.frame_locator("#detailFrame")
    assert frame.locator(".balance-value").inner_text() == "$4500.00"


def test_wait_via_coordinate_tier_is_unsupported(driver):
    with pytest.raises(ValueError, match="not supported via the coordinate locator tier"):
        driver.act(step(action_type="wait", locators=[locator("coordinate", "10,10")]))


# --- navigate --------------------------------------------------------------------------------


def test_navigate_goes_to_the_resolved_destination(driver):
    s = step(action_type="navigate", locators=[], input_value="data:text/html,<h1>Navigated</h1>")
    driver.act(s)
    assert driver.page.locator("h1").inner_text() == "Navigated"


def test_navigate_resolves_template_params(driver):
    s = step(action_type="navigate", locators=[], input_value="data:text/html,<h1>{{label}}</h1>")
    driver.act(s, params={"label": "Templated"})
    assert driver.page.locator("h1").inner_text() == "Templated"


# --- checkpoint verification -------------------------------------------------------------------


def test_verify_checkpoint_extracts_declared_outputs(driver):
    login_via_driver(driver)
    search_via_driver(driver, "12345")
    checkpoint = Checkpoint(
        description="Balance page loaded",
        locator=locator("css", ".balance-value"),
        extract={"balance": "css=.balance-value", "member_name": "css=.member-name"},
    )
    outputs = driver.verify_checkpoint(checkpoint)
    assert outputs == {"balance": "$4500.00", "member_name": "Jane Doe"}


def test_verify_checkpoint_raises_when_locator_never_appears(driver):
    login_via_driver(driver)
    search_via_driver(driver, "00000")  # the fixture's deterministic hard-failure member id
    checkpoint = Checkpoint(
        description="Balance page loaded",
        locator=locator("css", ".balance-value"),
        extract=None,
    )
    with pytest.raises(CheckpointNotReachedError):
        driver.verify_checkpoint(checkpoint)
