"""Rebuild the virtual environment from scratch (only needed if .venv is broken/deleted).

Usage: python scripts/py/setup_env.py
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PIP_CONFIG_FILE"] = os.path.join(PROJECT_ROOT, "pip.ini")

# ms-swift 官方 setup.py 声明支持至 Python 3.12 并推荐 3.12；
# 3.13 虽有兼容提交但未正式声明，bitsandbytes 等量化依赖容易踩坑，故锁定 3.12。
SYSTEM_PYTHON = r"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe"
USE_LAUNCHER = False
if not os.path.exists(SYSTEM_PYTHON):
    import shutil
    # 依次尝试 py launcher 指定 3.12、PATH 上的 python
    for candidate in (shutil.which("py"), shutil.which("python")):
        if not candidate:
            continue
        try:
            out = subprocess.run([candidate, "-3.12", "--version"], capture_output=True, text=True)
            if out.returncode == 0 and "3.12" in out.stdout:
                SYSTEM_PYTHON = candidate
                USE_LAUNCHER = True
                break
        except Exception:
            continue
    if not USE_LAUNCHER:
        print("[!] Python 3.12 not found. ms-swift officially supports up to 3.12.")
        print("    Install Python 3.12 from python.org, then re-run this script.")
        sys.exit(1)

venv_path = os.path.join(PROJECT_ROOT, ".venv")
venv_py = os.path.join(venv_path, "Scripts", "python.exe")

# .venv 是可再生的构建产物，用 venv --clear 原地重建即可，无需手动删除整个目录。
# 若旧环境是别的 Python 版本（如迁移前的 3.13），--clear 会一并清掉残留的 site-packages。
venv_cmd_flags = []
if os.path.exists(venv_path):
    cfg = os.path.join(venv_path, "pyvenv.cfg")
    old_version = ""
    if os.path.exists(cfg):
        with open(cfg, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("version"):
                    old_version = line.split("=", 1)[1].strip()
    print(f"[!] Found existing venv (Python {old_version or 'unknown'}) at {venv_path}")
    print("    It will be rebuilt in place with Python 3.12 (--clear).")
    print("    Installed packages inside .venv will be discarded; nothing else is touched.")
    venv_cmd_flags.append("--clear")

print(f"[1/3] Creating virtual environment with {SYSTEM_PYTHON} ...")
create_cmd = [SYSTEM_PYTHON]
if USE_LAUNCHER:
    create_cmd.append("-3.12")
create_cmd += ["-m", "venv"] + venv_cmd_flags + [venv_path]
subprocess.run(create_cmd, check=True)

print("[2/3] Upgrading pip ...")
subprocess.run([venv_py, "-m", "pip", "install", "--upgrade", "pip"], check=True)

# ms-swift 会自动带上 torch / transformers / trl / peft；
# bitsandbytes 是 QLoRA 4bit 量化必需，需单独装；gradio 已从 ms-swift 依赖中移除，WebUI 需要它。
print("[3/3] Installing training stack (ms-swift + bitsandbytes + gradio, this may take a while) ...")
subprocess.run([venv_py, "-m", "pip", "install", "-U", "ms-swift", "bitsandbytes", "gradio"], check=True)

print("Done. Verify with: python scripts/py/use_env.py  then  swift sft --help")
