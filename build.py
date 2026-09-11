"""
Build a standalone Windows app: dist/SDM-Lap-Optimizer/SDM-Lap-Optimizer.exe,
zipped up as dist/SDM-Lap-Optimizer-win64.zip for release.

Deliberately --onedir, not --onefile: a onefile exe self-extracts to a temp
dir at launch, and that exact behavior is what makes antivirus / Windows
Defender / browser download-protection heuristics flag PyInstaller onefile
binaries as a trojan (a well-known false positive, not anything the app
actually does -- see PyInstaller's own docs on this). --onedir ships the
same bundle unpacked, which the same heuristics essentially never flag, at
the cost of shipping a zip instead of a single exe.

Nothing else needed on the machine that runs it (WebView2 is part of
Windows 10 2004+ / 11 already; on an older Windows it's a tiny Microsoft
download pywebview will prompt for automatically).

Usage:
    pip install -r requirements.txt
    python build.py

Output:
    dist/SDM-Lap-Optimizer/SDM-Lap-Optimizer.exe   (run this)
    dist/SDM-Lap-Optimizer-win64.zip                (upload this to Releases)
"""
import os
import shutil

import PyInstaller.__main__

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "lapsim.html")
ICON = os.path.join(HERE, "icon.ico")
DIST = os.path.join(HERE, "dist")

args = [
    os.path.join(HERE, "app.py"),
    "--name=SDM-Lap-Optimizer",
    "--onedir",
    "--windowed",
    "--add-data=" + HTML + os.pathsep + ".",
    "--clean",
    "--noconfirm",
    "--distpath=" + DIST,
    "--workpath=" + os.path.join(HERE, "build"),
    "--specpath=" + HERE,
]
if os.path.isfile(ICON):
    args.append("--icon=" + ICON)

if __name__ == "__main__":
    PyInstaller.__main__.run(args)

    zip_base = os.path.join(DIST, "SDM-Lap-Optimizer-win64")
    zip_path = shutil.make_archive(zip_base, "zip", DIST, "SDM-Lap-Optimizer")

    print("\nBuilt: dist/SDM-Lap-Optimizer/SDM-Lap-Optimizer.exe")
    print("Zipped for release: " + zip_path)
