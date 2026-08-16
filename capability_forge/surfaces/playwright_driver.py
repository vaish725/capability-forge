"""Playwright surface driver.

Implements the surface interface (observe() -> StateSnapshot, act(Action) -> None) for real
browser pages using Playwright's accessibility tree and DOM roles as the primary locator tier,
raw CSS as fallback, and screenshot coordinates as a last resort.

TODO: implement PlaywrightDriver.observe() and PlaywrightDriver.act().
"""
