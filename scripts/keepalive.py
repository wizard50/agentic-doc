#!/usr/bin/env python3
"""Keep a Streamlit Community Cloud app awake by visiting it with Playwright.

Detects the hibernation page ("Yes, get this app back up!") and clicks it when
present; otherwise confirms the app is already awake.
"""

from __future__ import annotations

import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout  # type: ignore[import-not-found]
from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]


APP_URL = os.environ.get("STREAMLIT_APP_URL", "https://agentic-doc.streamlit.app/")
WAKE_BUTTON_TEXT = "Yes, get this app back up!"
# Streamlit sleep wake can take a while; allow up to 2 minutes after click.
WAKE_TIMEOUT_MS = 120_000
PAGE_LOAD_TIMEOUT_MS = 60_000


def main() -> int:
    print(f"Visiting {APP_URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT_MS)
            # Give Streamlit a moment to render either the app or the sleep page.
            page.wait_for_timeout(3_000)

            wake_button = page.get_by_role("button", name=WAKE_BUTTON_TEXT)
            if wake_button.count() > 0 and wake_button.first.is_visible():
                print(f"WAKE: found '{WAKE_BUTTON_TEXT}' — clicking to restart app")
                wake_button.first.click()
                # Wait until the wake button is gone (app finished booting).
                try:
                    wake_button.first.wait_for(state="hidden", timeout=WAKE_TIMEOUT_MS)
                except PlaywrightTimeout:
                    # Button may navigate away; also accept main Streamlit chrome.
                    pass

                # Confirm the app shell is present (Streamlit root or no wake button).
                page.wait_for_timeout(5_000)
                if page.get_by_role("button", name=WAKE_BUTTON_TEXT).count() > 0:
                    still = page.get_by_role("button", name=WAKE_BUTTON_TEXT)
                    if still.count() > 0 and still.first.is_visible():
                        print("WAKE: FAILED — wake button still visible after timeout")
                        return 1
                print("WAKE: app is back up")
                return 0

            print("OK: app is already awake")
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
