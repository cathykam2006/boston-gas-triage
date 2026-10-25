"""Keep Streamlit Community Cloud apps awake.

Community Cloud puts an app to sleep after 12 hours without traffic, and a sleeping app only wakes
when someone clicks "Yes, get this app back up!". This opens each app in a headless browser (a real
visit, which counts as traffic), clicks that button if the app is asleep, and waits until the app
has loaded. Run on a schedule by .github/workflows/keep-awake.yml.

    APP_URLS="https://a.streamlit.app https://b.streamlit.app" python scripts/keep_awake.py
"""
import os
import re
import sys
import time

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

DEFAULT_URLS = [
    "https://boston-gas-triage-kljspjcnsvm6y8pc8uv3sp.streamlit.app/",
    "https://ne-gas-forecast-zmi45pfx6pdyncuxxazxcq.streamlit.app/",
    "https://dpu-filing-assistant-6tsrcudhmux8mpakqcnlux.streamlit.app/",
]
WAKE = re.compile(r"get this app back up", re.I)
LOAD_TIMEOUT = int(os.environ.get("LOAD_TIMEOUT", "300"))      # seconds to wait for a waking app


def app_loaded(page) -> bool:
    """The Streamlit app is rendered in the page or in its embedded frame."""
    return any(f.locator('[data-testid="stApp"]').count() > 0 for f in page.frames)


def visit(page, url: str) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=90_000)
    woke = False
    try:
        button = page.get_by_role("button", name=WAKE)
        button.wait_for(state="visible", timeout=15_000)
        button.click()
        woke = True
        print("  asleep: clicked 'Yes, get this app back up!'")
    except PWTimeout:
        pass
    deadline = time.time() + LOAD_TIMEOUT
    while time.time() < deadline:
        if app_loaded(page):
            return "woken and loaded" if woke else "awake"
        page.wait_for_timeout(5_000)
    return "did not load"


def main() -> int:
    urls = os.environ.get("APP_URLS", "").split() or DEFAULT_URLS
    failures = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for url in urls:
            page = browser.new_page()
            print(url)
            try:
                status = visit(page, url)
            except Exception as e:          # network hiccup: report and try the next app
                status = f"error: {e}"
            print(f"  {status}")
            if status not in ("awake", "woken and loaded"):
                failures += 1
            page.close()
        browser.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
