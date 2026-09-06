"""Start QLoRA fine-tuning.

Usage:
    python scripts/py/train.py                              # default config
    python scripts/py/train.py configs/my_config.yaml       # custom config
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PIP_CONFIG_FILE"] = os.path.join(PROJECT_ROOT, "pip.ini")
os.environ["DATA_DIR"] = os.path.join(PROJECT_ROOT, "data")
os.chdir(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Start QLoRA training")
    parser.add_argument("config", nargs="?", default="configs/qwen3_4b_qlora_sft.yaml", help="Config YAML path")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    cli = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "swift.exe")

    if not os.path.exists(cli):
        sys.exit(f"[!] {cli} not found.\n    Run: python scripts/py/setup_env.py")

    print(f"DATA_DIR = {os.environ['DATA_DIR']}")
    print(f"Training with config: {config_path}")
    # ms-swift 把 YAML 作为位置参数传入：swift sft <config.yaml>
    subprocess.run([cli, "sft", config_path], check=True)


if __name__ == "__main__":
    main()
