"""Export (merge) the LoRA adapter into a full standalone model via `swift export`.

Usage:
    python scripts/py/export.py                            # 用默认配置文件
    python scripts/py/export.py configs/my_export.yaml     # 自定义配置文件
    python scripts/py/export.py --adapter outputs/xxx/v0-xxx/checkpoint-12   # 直接指定 adapter
    python scripts/py/export.py --adapter <path> --out models/my-merged      # 自定义输出目录

ms-swift 合并时用 --output_dir 指定结果目录。本脚本显式传该参数，不依赖框架默认命名，
完成后以实际落盘的权重文件为准来报告路径。
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swift_paths import (  # noqa: E402
    PROJECT_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_MERGED_DIR,
    resolve, swift_cli, ensure_swift_adapter, latest_adapter,
)

os.environ["PIP_CONFIG_FILE"] = os.path.join(PROJECT_ROOT, "pip.ini")
os.chdir(PROJECT_ROOT)


def read_yaml_scalar(config_path, key):
    """从 YAML 里读出某个标量字段。用正则而非 PyYAML，因为本脚本由系统 Python 运行，
    系统环境里不一定装了 yaml。读不出来就返回 None，交给调用方回退默认值。"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                m = re.match(r"^" + re.escape(key) + r"\s*:\s*(.+?)\s*$", line)
                if m:
                    return m.group(1).strip("'\"")
    except OSError:
        pass
    return None


def find_weights(directory):
    """在目录树里找合并产物，返回 (含权重的目录, 权重文件数)。找不到返回 (None, 0)。"""
    if not os.path.isdir(directory):
        return None, 0
    for root, _dirs, files in os.walk(directory):
        weights = [f for f in files if f.endswith(".safetensors") or f.endswith(".bin")]
        if weights:
            return root, len(weights)
    return None, 0


def main():
    parser = argparse.ArgumentParser(description="Export merged model")
    parser.add_argument("config", nargs="?", default=None, help="Export config YAML path")
    parser.add_argument("--adapter", default=None,
                        help=f"直接指定 adapter 路径（默认取 {DEFAULT_OUTPUT_DIR} 下最新的）")
    parser.add_argument("--out", default=None, help=f"合并结果输出目录（默认 {DEFAULT_MERGED_DIR}）")
    args = parser.parse_args()

    cli = swift_cli()

    # 给了 config 走配置文件；否则（含完全无参数）自动探测最新 adapter。
    # 因为 add_version 会让产物落在 vN-<时间戳>/ 子目录里，
    # 配置文件中写死的 adapters 路径容易过期。
    if args.config:
        config_path = resolve(args.config)
        if not os.path.exists(config_path):
            sys.exit(f"[!] 配置文件不存在: {config_path}")
        # 配置文件里写的 adapter 也要校验，避免对旧框架产物执行合并
        cfg_adapter = read_yaml_scalar(config_path, "adapters")
        if cfg_adapter:
            ensure_swift_adapter(resolve(cfg_adapter))
        print(f"Exporting with config: {config_path}")
        subprocess.run([cli, "export", config_path], check=True)
        out_dir = resolve(read_yaml_scalar(config_path, "output_dir") or DEFAULT_MERGED_DIR)
    else:
        adapter = (resolve(args.adapter) if args.adapter
                   else latest_adapter(resolve(DEFAULT_OUTPUT_DIR)))
        if adapter is None:
            sys.exit(f"[!] 在 {resolve(DEFAULT_OUTPUT_DIR)} 下没找到 ms-swift 产出的 adapter。\n"
                     "    先执行: python scripts/py/train.py")
        ensure_swift_adapter(adapter)
        out_dir = resolve(args.out or DEFAULT_MERGED_DIR)
        print(f"Merging adapter: {adapter}")
        print(f"Output dir: {out_dir}")
        # 显式传 --output_dir，不依赖框架的默认目录命名规则
        subprocess.run([cli, "export",
                        "--adapters", adapter,
                        "--merge_lora", "true",
                        "--output_dir", out_dir], check=True)

    # 以实际落盘的权重文件为准来报告结果，而不是假设目录名
    weight_dir, n_weights = find_weights(out_dir)
    if weight_dir:
        print(f"\nDone. 合并后的完整权重在: {weight_dir}  ({n_weights} 个权重文件)")
    else:
        print(f"\n[!] 命令执行成功，但在 {out_dir} 下没找到权重文件。")
        print("    请查看上方 swift 日志里实际写出的目录（框架可能使用了默认命名）。")
    print("可直接给 transformers / vLLM 加载；接入 Ollama 见 GUIDE.md（框架原生支持 to_ollama）。")


if __name__ == "__main__":
    main()
