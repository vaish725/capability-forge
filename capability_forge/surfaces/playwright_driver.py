"""Playwright surface driver.

The Surface Driver implementation for real browser pages: observe() -> StateSnapshot, act(Action)
-> None, plus verify_checkpoint() for the "did this run reach its goal" check. Wraps an
already-created Playwright Page; launching the browser/context is the caller's responsibility (the
driver's job is surface mechanics, not session lifecycle - that belongs to whatever orchestrates
discovery or replay).

Locator resolution, in order per step:
  1. Try each of the step's locator tiers in turn (role, then css, by convention - the order is
     whatever the artifact recorded, this driver doesn't second-guess it).
  2. For role/css tiers, search the main frame first, then any child frames in document order
     (Playwright's page.frames includes the main frame). A tier only counts as resolved if it
     matches exactly one element - an ambiguous match (more than one element) is treated the same
     as no match. This matters concretely for this project's own test fixture: its ".btn" class is
     deliberately reused across every button on the page, so a naive "grab the first match" would
     silently click the wrong element instead of falling back to a tier that actually
     disambiguates. A locator that doesn't uniquely identify an element isn't a reliable locator.
  3. The coordinate tier is last resort and never fails to "resolve" (it's raw pixel data, not a
     DOM query) - its value is a page-viewport-relative "x,y" string, the same coordinate space a
     full-page screenshot would use, so it works the same whether the target visually happens to
     sit inside an iframe or not.
  4. If nothing resolves, raise LocatorResolutionError.

Resolution waits, per (tier, frame) combination, rather than checking instantaneously. This was a
deliberate fix, not the first design tried: an instant count()-based check meant clicking an
element that hadn't rendered yet (e.g. right after a click that triggers this project's own async
iframe content) failed immediately instead of giving it a moment to appear, defeating the whole
point of Playwright's own actionability waiting. With resolution itself wait-based, "wait" as an
action_type is just resolution with nothing performed on the result afterward - a way to insert an
explicit synchronization point in a recorded flow without it needing to double as a click or type.
"""

from dataclasses import dataclass

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator, Page

from capability_forge.schema.artifact import Checkpoint, LocatorTier, StepAction, render_template


class LocatorResolutionError(Exception):
    """Raised when none of a step's locator tiers resolve to a unique, actionable element."""


class CheckpointNotReachedError(Exception):
    """Raised when a checkpoint's locator (or one of its declared output extractions) can't be
    found. Callers should treat this as a hard_failure signal."""


@dataclass
class StateSnapshot:
    """A structural snapshot of the current page, meant to be handed to an LLM during discovery.
    accessibility_tree is Playwright's own YAML-like aria snapshot format (role, name, and nesting
    for every element), not a raw DOM dump - it's already close to what the design doc calls for
    as the primary perception mechanism."""

    url: str
    title: str
    accessibility_tree: str


@dataclass
class _ResolvedTarget:
    """Internal result of resolving a step's locator list: either a unique Locator (role/css
    tiers) or a page-viewport-relative pixel coordinate (coordinate tier), never both."""

    tier: LocatorTier
    locator: Locator | None
    coordinates: tuple[float, float] | None


class PlaywrightDriver:
    """Wraps one Playwright Page and implements the Surface Driver contract against it."""

    def __init__(self, page: Page, locator_timeout_ms: int = 10000):
        self.page = page
        # Per-(tier, frame) wait budget during resolution. Production-oriented default (real
        # legacy apps can be slow to render); tests exercising an intentionally-broken locator's
        # fallback pass a much smaller value so the suite doesn't pay this timeout for every
        # deliberate miss.
        self._locator_timeout_ms = locator_timeout_ms

    def observe(self) -> StateSnapshot:
        """Return the current page's URL, title, and accessibility tree. Token-budget trimming
        for LLM consumption is a discovery-loop concern layered on top of this, not this driver's
        job - this returns the full structural snapshot."""
        return StateSnapshot(
            url=self.page.url,
            title=self.page.title(),
            accessibility_tree=self.page.locator("body").aria_snapshot(),
        )

    def act(self, step: StepAction, params: dict[str, str] | None = None) -> None:
        """Execute one recorded step. params fills in any {{name}} placeholders in
        step.input_value; the schema already guarantees any such placeholder was declared as an
        artifact input, so a KeyError here means that check was skipped upstream, not that the
        param is legitimately optional."""
        try:
            resolved_value = render_template(step.input_value, params or {})
        except KeyError as exc:
            raise ValueError(f"step {step.step_id!r} references an unresolved param: {exc}") from exc

        if step.action_type == "navigate":
            # navigate has no element to locate - input_value is already the full destination
            # (the schema requires it non-empty for this action type).
            self.page.goto(resolved_value)
            return

        if step.action_type == "wait":
            # Resolution itself already waits (see module docstring); "wait" just resolves and
            # discards the result rather than acting on it. A coordinate tier "resolves" to a raw
            # pixel position with no visibility state to wait for, so it's explicitly rejected
            # here rather than silently treated as a no-op success.
            target = self._resolve_target(step.locators)
            if target.locator is None:
                raise ValueError("wait action_type is not supported via the coordinate locator tier")
            return

        target = self._resolve_target(step.locators)
        dispatch = {
            "click": self._click,
            "type": self._type,
            "select": self._select,
        }
        dispatch[step.action_type](target, resolved_value)

    def verify_checkpoint(self, checkpoint: Checkpoint) -> dict[str, str]:
        """Confirm the checkpoint's locator is present, then read every declared output. Returns
        the extracted {name: value} dict. Raises CheckpointNotReachedError if the checkpoint
        locator or any declared output's locator can't be found."""
        locator = self._find_unique(checkpoint.locator.value)
        if locator is None:
            raise CheckpointNotReachedError(f"checkpoint locator not found: {checkpoint.locator.value!r}")
        locator.wait_for(state="visible")

        outputs: dict[str, str] = {}
        for name, selector in (checkpoint.extract or {}).items():
            value_locator = self._find_unique(selector)
            if value_locator is None:
                raise CheckpointNotReachedError(f"declared output {name!r} not found via {selector!r}")
            outputs[name] = value_locator.inner_text()
        return outputs

    # --- locator resolution -----------------------------------------------------------------

    def _resolve_target(self, locators: list[LocatorTier]) -> _ResolvedTarget:
        tried = []
        for tier in locators:
            if tier.strategy == "coordinate":
                return _ResolvedTarget(tier=tier, locator=None, coordinates=self._parse_coordinates(tier.value))
            locator = self._find_unique(tier.value)
            if locator is not None:
                return _ResolvedTarget(tier=tier, locator=locator, coordinates=None)
            tried.append(f"{tier.strategy}={tier.value}")
        raise LocatorResolutionError(f"no locator tier resolved to a unique element; tried: {tried}")

    def _find_unique(self, selector: str) -> Locator | None:
        """Search the main frame, then child frames, for a selector that resolves to exactly one
        attached element - waiting up to self._locator_timeout_ms per frame rather than checking
        instantaneously, so a target that hasn't rendered yet still gets a fair chance to appear.
        Playwright's role= and css= prefixed selector strings (and bare CSS with no prefix) all
        work directly with .locator() - no separate parsing needed per strategy.

        Catches the base PlaywrightError, not just its TimeoutError subclass: Playwright's own
        strict-mode enforcement raises immediately (not a timeout) the moment a locator matches
        more than one element, which is exactly the ambiguous-match case this method needs to
        treat as "not resolved here, try the next frame" rather than letting propagate. This is a
        deliberately broad catch for a fallback chain, where any failure to resolve a given tier
        cleanly (ambiguous match, malformed selector, or nothing found) should mean "move on to
        the next tier", not crash the whole step - the trade-off is that a genuinely malformed
        locator string in an earlier tier fails silently into the next tier instead of surfacing
        as a configuration error; LocatorResolutionError's "tried: [...]" message is what a
        caller has to fall back on to notice that pattern.
        """
        for frame in self.page.frames:
            candidate = frame.locator(selector)
            try:
                candidate.wait_for(state="attached", timeout=self._locator_timeout_ms)
            except PlaywrightError:
                continue
            if candidate.count() == 1:
                return candidate
        return None

    def _parse_coordinates(self, value: str) -> tuple[float, float]:
        try:
            x_str, y_str = value.split(",")
            return float(x_str.strip()), float(y_str.strip())
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"coordinate locator value must be 'x,y', got {value!r}") from exc

    # --- action dispatch ---------------------------------------------------------------------

    def _click(self, target: _ResolvedTarget, _value: str | None) -> None:
        if target.locator is not None:
            target.locator.click()
        else:
            self.page.mouse.click(*target.coordinates)

    def _type(self, target: _ResolvedTarget, value: str | None) -> None:
        if value is None:
            raise ValueError("type action requires a resolved input_value")
        if target.locator is not None:
            target.locator.fill(value)
        else:
            # No coordinate-based "fill" exists; click to focus the field, then type via keyboard.
            self.page.mouse.click(*target.coordinates)
            self.page.keyboard.type(value)

    def _select(self, target: _ResolvedTarget, value: str | None) -> None:
        if value is None:
            raise ValueError("select action requires a resolved input_value")
        if target.locator is None:
            # A coordinate click can't drive a native <select>'s option list reliably; this
            # combination is unsupported rather than pretending to handle it.
            raise ValueError("select action_type is not supported via the coordinate locator tier")
        target.locator.select_option(value)
