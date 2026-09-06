"""Download a dataset from ModelScope into data/.

Usage:
    python scripts/py/download_dataset.py                              # default COIG-CQIA
    python scripts/py/download_dataset.py m-a-p/COIG-CQIA mydir       # custom dataset + folder
    python scripts/py/download_dataset.py --proxy http://127.0.0.1:7890
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PIP_CONFIG_FILE"] = os.path.join(PROJECT_ROOT, "pip.ini")

DEFAULT_DATASET = "m-a-p/COIG-CQIA"
DEFAULT_FOLDER = "coig-cqia"


def detect_system_proxy():
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
        enable, _ = winreg.QueryValueEx(key, "ProxyEnable")
        if enable:
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
            winreg.CloseKey(key)
            return f"http://{server}"
        winreg.CloseKey(key)
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Download dataset from ModelScope")
    parser.add_argument("dataset_id", nargs="?", default=DEFAULT_DATASET, help="ModelScope dataset id")
    parser.add_argument("folder_name", nargs="?", default=DEFAULT_FOLDER, help="Local folder name under data/")
    parser.add_argument("--proxy", default=None, help="HTTP proxy URL")
    args = parser.parse_args()

    proxy = args.proxy or detect_system_proxy()
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"Using proxy: {proxy}")

    modelscope_cli = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "modelscope.exe")
    target_dir = os.path.join(PROJECT_ROOT, "data", args.folder_name)

    print(f"Downloading dataset {args.dataset_id} -> {target_dir}")
    subprocess.run([modelscope_cli, "download", "--dataset", args.dataset_id, "--local_dir", target_dir], check=True)
    print(f"Done: {target_dir}")


if __name__ == "__main__":
    main()
