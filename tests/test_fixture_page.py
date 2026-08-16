"""Tests for fixtures/hostile_legacy_page.html itself: every path the fixture is designed to
demonstrate (login outcomes, all four replay outcome types, the transfer flow inside its iframe,
and the CSS-ambiguity / coordinate-tier properties the fixture exists to prove).

These are deliberately independent of any driver code - they drive the page directly with raw
Playwright calls. The point is to lock the fixture's own behavior in as a regression-tested
contract, so an accidental edit made later while debugging playwright_driver.py against this same
page is caught immediately here rather than surfacing as a confusing driver failure much later.
"""

from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "hostile_legacy_page.html"
FIXTURE_URL = f"file://{FIXTURE_PATH}"


def login(page, username="jdoe", password="secret"):
    page.get_by_label("User Name:").fill(username)
    page.get_by_label("Password:").fill(password)
    page.get_by_role("button", name="Login").click()


@pytest.fixture
def fixture_page(page):
    """A page already navigated to the fixture, before login."""
    page.goto(FIXTURE_URL)
    return page


@pytest.fixture
def logged_in_page(fixture_page):
    """A page past login, sitting on the accounts/search screen."""
    login(fixture_page)
    return fixture_page


# --- login screen ------------------------------------------------------------------------------


def test_empty_login_shows_required_fields_message(fixture_page):
    fixture_page.get_by_role("button", name="Login").click()
    assert fixture_page.locator("#loginMessage").inner_text() == "User Name and Password are both required."


def test_locked_username_is_a_login_business_outcome(fixture_page):
    login(fixture_page, username="locked", password="x")
    assert "locked" in fixture_page.locator("#loginMessage").inner_text().lower()
    assert fixture_page.locator("#screenAccounts").is_visible() is False


def test_successful_login_reaches_accounts_screen_via_role_locator(fixture_page):
    login(fixture_page)
    assert fixture_page.get_by_label("Member ID:").is_visible()


# --- search: the four outcome types --------------------------------------------------------


def test_unknown_member_id_is_a_business_outcome(logged_in_page):
    logged_in_page.get_by_label("Member ID:").fill("00001")
    logged_in_page.get_by_role("button", name="Search").click()
    assert "No member found" in logged_in_page.locator("#searchResultArea").inner_text()


def test_hard_failure_member_id_shows_system_error_with_no_continuation(logged_in_page):
    logged_in_page.get_by_label("Member ID:").fill("00000")
    logged_in_page.get_by_role("button", name="Search").click()
    assert "SYS-500" in logged_in_page.locator("#searchResultArea").inner_text()
    assert logged_in_page.locator("#detailFrame").is_visible() is False


def test_recoverable_member_id_shows_interstitial_before_detail(logged_in_page):
    logged_in_page.get_by_label("Member ID:").fill("88888")
    logged_in_page.get_by_role("button", name="Search").click()
    assert logged_in_page.locator("#sessionNotice").is_visible()
    assert logged_in_page.locator("#detailFrame").is_visible() is False

    logged_in_page.get_by_role("button", name="Continue").click()
    frame = logged_in_page.frame_locator("#detailFrame")
    assert frame.locator(".member-name").inner_text() == "Maria Lopez"
    assert logged_in_page.locator("#sessionNotice").is_visible() is False


def test_valid_member_id_reaches_detail_via_the_checkpoint_locator(logged_in_page):
    logged_in_page.get_by_label("Member ID:").fill("12345")
    logged_in_page.get_by_role("button", name="Search").click()
    frame = logged_in_page.frame_locator("#detailFrame")
    # .balance-value is the fixture's literal implementation of the schema's own
    # checkpoint.extract worked example ({"balance": "css=.balance-value"}).
    assert frame.locator(".balance-value").inner_text() == "$4500.00"
    assert frame.locator(".member-name").inner_text() == "Jane Doe"
    assert frame.locator(".acct-no").inner_text() == "9988776655"


# --- transfer flow, inside the iframe ---------------------------------------------------------


def test_successful_transfer_shows_a_confirmation_reference(logged_in_page):
    logged_in_page.get_by_label("Member ID:").fill("12345")
    logged_in_page.get_by_role("button", name="Search").click()
    frame = logged_in_page.frame_locator("#detailFrame")
    frame.get_by_role("button", name="Transfer Funds").click()
    frame.get_by_label("To Account:").fill("55555")
    frame.get_by_label("Amount:").fill("100")
    frame.get_by_role("button", name="Confirm Transfer").click()
    result_text = frame.locator("#transferResult").inner_text()
    assert "Reference #: TXN-" in result_text
    # Transfer Type defaults to "Checking" if never touched - confirms the <select>'s default
    # option is honored, not just that a value happened to be present.
    assert "from checking account" in result_text


def test_transfer_type_select_can_be_changed(logged_in_page):
    logged_in_page.get_by_label("Member ID:").fill("12345")
    logged_in_page.get_by_role("button", name="Search").click()
    frame = logged_in_page.frame_locator("#detailFrame")
    frame.get_by_role("button", name="Transfer Funds").click()
    frame.get_by_label("To Account:").fill("55555")
    frame.get_by_label("Amount:").fill("100")
    frame.get_by_label("Transfer Type:").select_option("savings")
    frame.get_by_role("button", name="Confirm Transfer").click()
    assert "from savings account" in frame.locator("#transferResult").inner_text()


def test_over_balance_transfer_is_a_business_outcome(logged_in_page):
    logged_in_page.get_by_label("Member ID:").fill("67890")  # balance 250.75
    logged_in_page.get_by_role("button", name="Search").click()
    frame = logged_in_page.frame_locator("#detailFrame")
    frame.get_by_role("button", name="Transfer Funds").click()
    frame.get_by_label("To Account:").fill("55555")
    frame.get_by_label("Amount:").fill("999999")
    frame.get_by_role("button", name="Confirm Transfer").click()
    assert "Insufficient funds" in frame.locator("#transferResult").inner_text()


# --- css-tier: legacy WebForms-style ids, including inside the iframe -------------------------


def test_css_tier_ids_drive_the_login_and_search_form(fixture_page):
    fixture_page.locator("#txtUserName").fill("jdoe")
    fixture_page.locator("#txtPassword").fill("secret")
    fixture_page.locator("#btnLogin").click()
    assert fixture_page.locator("#screenAccounts").is_visible()

    fixture_page.locator("#txtMemberId").fill("12345")
    fixture_page.locator("#btnSearch").click()
    frame = fixture_page.frame_locator("#detailFrame")
    # is_visible() is a non-waiting snapshot query (unlike click()/fill()/inner_text(), which all
    # have built-in actionability waiting) - it can run before the iframe's srcdoc content has
    # finished loading. wait_for() makes this deterministic instead of racing the iframe load.
    frame.locator(".balance-value").wait_for(state="visible")
    assert frame.locator(".balance-value").is_visible()


def test_css_tier_ids_drive_the_transfer_flow_inside_the_iframe(logged_in_page):
    logged_in_page.get_by_label("Member ID:").fill("12345")
    logged_in_page.get_by_role("button", name="Search").click()
    frame = logged_in_page.frame_locator("#detailFrame")
    frame.locator("#btnTransferFunds").click()
    frame.locator("#txtToAccount").fill("1")
    frame.locator("#txtAmount").fill("50")
    frame.locator("#btnConfirmTransfer").click()
    assert "Reference #: TXN-" in frame.locator("#transferResult").inner_text()


def test_btn_class_alone_is_genuinely_ambiguous(logged_in_page):
    # This is the fixture's concrete demonstration of why CSS-only locators aren't the primary
    # tier: ".btn" is reused across every button on the page. If this count ever drops to 1, the
    # fixture no longer proves what it claims to and the driver's fallback design should be
    # re-justified against it.
    logged_in_page.get_by_label("Member ID:").fill("12345")
    logged_in_page.get_by_role("button", name="Search").click()
    frame = logged_in_page.frame_locator("#detailFrame")
    frame.get_by_role("button", name="Transfer Funds").click()
    top_level_count = logged_in_page.locator(".btn").count()
    frame_count = frame.locator(".btn").count()
    assert top_level_count + frame_count >= 3


# --- coordinate tier: unexercised by any locator-based test above, checked separately ----------


def test_coordinate_click_on_login_button_works(fixture_page):
    # The coordinate tier is the fallback most likely to be flaky in practice (viewport size,
    # element reflow), so it's worth proving it works against this fixture at all, independent of
    # any driver code. Finds the button's bounding box and clicks its center directly via
    # page.mouse, bypassing role/css locators entirely.
    fixture_page.locator("#txtUserName").fill("jdoe")
    fixture_page.locator("#txtPassword").fill("secret")
    box = fixture_page.locator("#btnLogin").bounding_box()
    assert box is not None, "bounding box must be available to test coordinate-based clicking"
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2
    fixture_page.mouse.click(center_x, center_y)
    assert fixture_page.locator("#screenAccounts").is_visible()


def test_coordinate_click_inside_iframe_works(logged_in_page):
    # Coordinates are captured relative to the full page viewport (as they would be from a
    # discovery-time screenshot, which composites iframe content into page-level pixels), so a
    # coordinate click on an element inside the iframe still uses page-level, not frame-level,
    # mouse coordinates. This confirms that assumption holds for a button actually inside the
    # iframe, not just one on the top-level page.
    logged_in_page.get_by_label("Member ID:").fill("12345")
    logged_in_page.get_by_role("button", name="Search").click()
    frame = logged_in_page.frame_locator("#detailFrame")
    box = frame.locator("#btnTransferFunds").bounding_box()
    assert box is not None
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2
    logged_in_page.mouse.click(center_x, center_y)
    assert frame.locator("#txtToAccount").is_visible()
