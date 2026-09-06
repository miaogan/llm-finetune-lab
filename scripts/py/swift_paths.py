"""共享路径与 adapter 定位工具，供 chat.py / export.py 使用。

设计要点：
- ms-swift 靠 adapter 目录里的 args.json 自动读回基座模型、template、量化设置。
  旧框架（LLaMA-Factory）的产物没有 args.json，无法被 swift infer / swift export 加载。
- ms-swift 默认 add_version: true，产物落在 <output_dir>/vN-<时间戳>/checkpoint-N；
  若设为 false，则落在 <output_dir>/checkpoint-N。两种布局都要能找到。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 与 configs/qwen3_4b_qlora_sft.yaml 的 output_dir 保持一致
DEFAULT_OUTPUT_DIR = "outputs/qwen3_4b_qlora_sft"
# 与 configs/export_merged.yaml 的 output_dir 保持一致
DEFAULT_MERGED_DIR = "models/Qwen3-4B-finetuned"

MARKER = "args.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def swift_cli():
    cli = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "swift.exe")
    if not os.path.exists(cli):
        sys.exit(f"[!] {cli} not found.\n    Run: python scripts/py/setup_env.py")
    return cli


def _is_adapter(path):
    """真 adapter 目录须同时含 args.json 和 LoRA 权重文件。

    注意：ms-swift 的版本目录（vN-时间戳/）顶层也有 args.json（训练参数快照），
    但 LoRA 权重在它的 checkpoint-N/ 子目录里；只查 args.json 会把版本目录误判
    成 adapter，导致 swift export 报 "not an adapter"。
    """
    if not (os.path.isdir(path) and os.path.exists(os.path.join(path, MARKER))):
        return False
    return any(os.path.exists(os.path.join(path, w)) for w in ADAPTER_WEIGHTS)


def find_adapters(root):
    """在 root 下最多两层内查找含 args.json 的 adapter 目录（含 root 本身）。

    覆盖三种布局：
      root/                       (直接训练到 root，无 checkpoint 子目录)
      root/checkpoint-N           (add_version: false)
      root/vN-时间戳/checkpoint-N (add_version: true，ms-swift 默认)
    """
    found = []
    if not os.path.isdir(root):
        return found
    try:
        level1 = [e.path for e in os.scandir(root) if e.is_dir()]
    except OSError:
        return found
    if _is_adapter(root):
        found.append(root)
    for d in level1:
        if _is_adapter(d):
            found.append(d)
            continue
        try:
            for e in os.scandir(d):
                if e.is_dir() and _is_adapter(e.path):
                    found.append(e.path)
        except OSError:
            continue
    return found


def latest_adapter(root):
    """返回最近修改的 adapter 目录；找不到返回 None。"""
    found = find_adapters(root)
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def ensure_swift_adapter(path):
    """确认 path 是 ms-swift 产出的 adapter，否则给出可操作的错误提示。"""
    if not os.path.isdir(path):
        sys.exit(f"[!] 目录不存在: {path}\n    先执行: python scripts/py/train.py")
    if not _is_adapter(path):
        sys.exit(
            f"[!] {path} 里没有 {MARKER}，不是 ms-swift 训练产出的 adapter。\n"
            "    本项目已从 LLaMA-Factory 迁移到 ms-swift，旧 checkpoint 无法被 swift 直接加载。\n"
            "    请用 ms-swift 重新训练：python scripts/py/train.py\n"
            "    （若必须复用旧权重，需手动指定 --model 与 --template，见 GUIDE.md）"
        )


def pick_adapter(explicit=None, root=DEFAULT_OUTPUT_DIR):
    """确定要用的 adapter 路径。explicit 为空时自动取 root 下最新的。"""
    root_abs = resolve(root)
    if explicit:
        path = resolve(explicit)
        ensure_swift_adapter(path)
        return path
    path = latest_adapter(root_abs)
    if path is None:
        legacy = [e.name for e in os.scandir(root_abs) if e.is_dir()] if os.path.isdir(root_abs) else []
        hint = ""
        if legacy:
            hint = (f"\n    {root} 下现有目录（{', '.join(legacy[:5])}）均为旧框架产物，"
                    "缺少 args.json，不能用于 swift 推理/合并。")
        sys.exit(f"[!] 在 {root_abs} 下没找到 ms-swift 产出的 adapter。{hint}\n"
                 "    先执行: python scripts/py/train.py")
    return path
