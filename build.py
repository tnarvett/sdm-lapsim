"""
Build a standalone Windows .exe: dist/SDM-Lap-Optimizer.exe

One file, nothing else needed on the machine that runs it (WebView2 is part
of Windows 10 2004+ / 11 already; on an older Windows it's a tiny Microsoft
download pywebview will prompt for automatically).

Usage:
    pip install -r requirements.txt
    python build.py

Output:
    dist/SDM-Lap-Optimizer.exe
"""
import os

import PyInstaller.__main__

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "lapsim.html")
ICON = os.path.join(HERE, "icon.ico")

args = [
    os.path.join(HERE, "app.py"),
    "--name=SDM-Lap-Optimizer",
    "--onefile",
    "--windowed",
    "--add-data=" + HTML + os.pathsep + ".",
    "--clean",
    "--noconfirm",
    "--distpath=" + os.path.join(HERE, "dist"),
    "--workpath=" + os.path.join(HERE, "build"),
    "--specpath=" + HERE,
]
if os.path.isfile(ICON):
    args.append("--icon=" + ICON)

if __name__ == "__main__":
    PyInstaller.__main__.run(args)
    print("\nBuilt: dist/SDM-Lap-Optimizer.exe")
