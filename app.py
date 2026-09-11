"""
SDM Lap Optimizer -- desktop launcher.

Runs lapsim.html in a real native window (Edge WebView2 on Windows, via
pywebview) instead of a browser tab: no address bar, no manual local server,
no artifact sandbox -- a proper installed-feeling app. localStorage and file
downloads both work exactly like they do in Edge/Chrome, because WebView2 IS
that engine.

Run from source:
    pip install -r requirements.txt
    python app.py

Build a standalone .exe (see build.py / README.md):
    python build.py
"""
import os
import sys

import webview

APP_TITLE = "SDM Lap Optimizer"
HTML_FILE = "lapsim.html"


def resource_path(rel_path):
    """Resolve a path next to this script, or inside a PyInstaller onefile bundle."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)


def main():
    html_path = resource_path(HTML_FILE)
    if not os.path.isfile(html_path):
        raise SystemExit("Could not find " + HTML_FILE + " next to app.py (looked in: " + html_path + ")")

    webview.create_window(
        APP_TITLE,
        html_path,
        width=1440,
        height=900,
        min_size=(960, 640),
        text_select=True,
        confirm_close=False,
    )
    # edgechromium = Edge WebView2, the modern Chromium engine bundled with
    # Windows 10 2004+/11. Falls back automatically on other platforms.
    gui = "edgechromium" if sys.platform.startswith("win") else None
    webview.start(gui=gui)


if __name__ == "__main__":
    main()
