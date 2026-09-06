"""Launch ms-swift Web-UI (visual training & inference).

Usage:
    python scripts/py/webui.py              # 中文界面
    python scripts/py/webui.py --lang en    # 英文界面

ms-swift 的 Web-UI 是命令行的高层封装：界面上点「开始」实际会在后台另起一个
`swift sft ...` 进程，因此关掉界面不会中断训练。要停止训练需在 Runtime 标签页
选中任务并点 kill service。

注意：gradio 已从 ms-swift 的依赖中移除，需单独安装（setup_env.py 已包含）。
Web-UI 默认端口未在官方文档中声明，沿用 gradio 默认的 7860；实际地址以终端输出为准。
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PIP_CONFIG_FILE"] = os.path.join(PROJECT_ROOT, "pip.ini")
os.chdir(PROJECT_ROOT)

cli = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "swift.exe")

if not os.path.exists(cli):
    sys.exit(f"[!] {cli} not found.\n    Run: python scripts/py/setup_env.py")

parser = argparse.ArgumentParser(description="Launch ms-swift Web-UI")
parser.add_argument("--lang", default="zh", choices=["zh", "en"], help="界面语言")
parser.add_argument("--share", action="store_true", help="生成公网分享链接（慎用）")
args = parser.parse_args()

cmd = [cli, "web-ui", "--lang", args.lang]
if args.share:
    cmd += ["--share", "true"]

print("Starting ms-swift Web-UI (Ctrl+C to stop) ...")
print("浏览器地址以终端输出为准，通常为 http://localhost:7860")
subprocess.run(cmd, check=True)
