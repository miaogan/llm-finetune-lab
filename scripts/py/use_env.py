"""Project environment setup — sets env vars and activates venv PATH.

Usage:
    python scripts/py/use_env.py

Run this first in every new terminal session.
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_SCRIPTS = os.path.join(PROJECT_ROOT, ".venv", "Scripts")

os.environ["PIP_CONFIG_FILE"] = os.path.join(PROJECT_ROOT, "pip.ini")
os.environ["DATA_DIR"] = os.path.join(PROJECT_ROOT, "data")
os.environ["VIRTUAL_ENV"] = os.path.join(PROJECT_ROOT, ".venv")

# Prepend venv\Scripts to PATH so swift / modelscope works
current_path = os.environ.get("PATH", "")
os.environ["PATH"] = VENV_SCRIPTS + os.pathsep + current_path

py_exe = os.path.join(VENV_SCRIPTS, "python.exe")
version = subprocess.check_output([py_exe, "--version"], text=True).strip()

print(f"[OK] venv: {os.environ['VIRTUAL_ENV']}")
print(f"[OK] pip index: Tsinghua mirror (project pip.ini)")
print(f"[OK] python: {version}")

swift_exe = os.path.join(VENV_SCRIPTS, "swift.exe")
if os.path.exists(swift_exe):
    print("[OK] ms-swift CLI: available")
else:
    print("[!] ms-swift CLI not installed in this venv.")
    print("    Run: python scripts/py/setup_env.py")
