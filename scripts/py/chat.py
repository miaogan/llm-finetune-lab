"""Chat with the fine-tuned model (base + LoRA adapter) via `swift infer`.

Usage:
    python scripts/py/chat.py                                    # 自动取最新 adapter
    python scripts/py/chat.py outputs/qwen3_4b_qlora_sft/v0-xxx/checkpoint-12   # 指定 checkpoint
    python scripts/py/chat.py --merged                           # 已合并的完整权重（无需 adapter）

ms-swift 会从 adapter 目录里的 args.json 自动读回基座模型与 template，
所以这里不用再手写 --model / --template。
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swift_paths import (  # noqa: E402
    PROJECT_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_MERGED_DIR,
    resolve, swift_cli, pick_adapter,
)

os.environ["PIP_CONFIG_FILE"] = os.path.join(PROJECT_ROOT, "pip.ini")
os.chdir(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Chat with fine-tuned model")
    parser.add_argument("adapter", nargs="?", default=None,
                        help=f"LoRA adapter/checkpoint 路径（默认取 {DEFAULT_OUTPUT_DIR} 下最新的）")
    parser.add_argument("--merged", action="store_true",
                        help=f"改为加载已合并的完整权重目录（默认 {DEFAULT_MERGED_DIR}）")
    parser.add_argument("--merged-dir", default=None, help="自定义合并权重目录")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", default="0", help="采样温度，0 表示关闭随机性")
    args = parser.parse_args()

    cli = swift_cli()

    if args.merged or args.merged_dir:
        model_dir = resolve(args.merged_dir or DEFAULT_MERGED_DIR)
        if not os.path.isdir(model_dir):
            sys.exit(f"[!] 未找到合并后的权重目录: {model_dir}\n    先执行: python scripts/py/export.py")
        cmd = [cli, "infer", "--model", model_dir, "--stream", "true"]
        print(f"Starting chat with merged model: {model_dir}")
    else:
        adapter = pick_adapter(args.adapter, DEFAULT_OUTPUT_DIR)
        cmd = [cli, "infer", "--adapters", adapter, "--stream", "true"]
        print(f"Starting chat (base model + adapter: {adapter})")

    cmd += ["--max_new_tokens", str(args.max_new_tokens), "--temperature", str(args.temperature)]
    print("Type 'exit' to quit.")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
