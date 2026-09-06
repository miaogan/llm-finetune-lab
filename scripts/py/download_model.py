"""Download a model from ModelScope into models/.

Usage:
    python scripts/py/download_model.py                          # default Qwen3-4B-Instruct-2507
    python scripts/py/download_model.py Qwen/Qwen3-1.7B          # any ModelScope model id
    python scripts/py/download_model.py Qwen/Qwen3-0.6B --proxy http://127.0.0.1:7890

If your network has only IPv6 access (modelscope.cn has no IPv6),
start your proxy first or pass --proxy.
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PIP_CONFIG_FILE"] = os.path.join(PROJECT_ROOT, "pip.ini")

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


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
    parser = argparse.ArgumentParser(description="Download model from ModelScope")
    parser.add_argument("model_id", nargs="?", default=DEFAULT_MODEL, help="ModelScope model id")
    parser.add_argument("--proxy", default=None, help="HTTP proxy URL (e.g. http://127.0.0.1:7890)")
    args = parser.parse_args()

    proxy = args.proxy or detect_system_proxy()
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        print(f"Using proxy: {proxy}")

    modelscope_cli = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "modelscope.exe")
    target_dir = os.path.join(PROJECT_ROOT, "models", args.model_id.replace("/", os.sep))

    print(f"Downloading {args.model_id} -> {target_dir}")
    subprocess.run([modelscope_cli, "download", "--model", args.model_id, "--local_dir", target_dir], check=True)
    print(f"Done: {target_dir}")


if __name__ == "__main__":
    main()
