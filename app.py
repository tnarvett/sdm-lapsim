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

APP_TITLE = "SDM Lap Optimizer"
HTML_FILE = "lapsim.html"


def _unblock_bundle():
    """Strip the NTFS "Mark of the Web" (Zone.Identifier) from every file next
    to this exe, if present.

    Windows tags every file extracted from a downloaded zip as "from the
    internet." pywebview's Windows backend hosts WebView2 through a .NET
    Framework AppDomain (via pythonnet), and .NET Framework refuses to load
    an assembly carrying that tag from inside a sandboxed AppDomain -- so a
    freshly-downloaded-and-unzipped build fails deep inside pythonnet with
    "Failed to resolve Python.Runtime.Loader.Initialize", even though the
    exact same build runs fine from a folder that was never downloaded. Doing
    this once at startup, before webview/pythonnet touch any .NET DLLs, makes
    the downloaded build self-heal instead of requiring the user to manually
    right-click each dll or run Unblock-File.

    Windows-only, and a no-op (fast) when nothing is tagged -- safe to call
    unconditionally, including when running from source.
    """
    if not sys.platform.startswith("win"):
        return
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    for dirpath, _dirnames, filenames in os.walk(base):
        for name in filenames:
            try:
                os.remove(os.path.join(dirpath, name) + ":Zone.Identifier")
            except OSError:
                pass  # not tagged, or in use -- nothing to strip


_unblock_bundle()

import webview  # noqa: E402  (must come after _unblock_bundle so pythonnet never sees a tagged dll)


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
