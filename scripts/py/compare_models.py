"""对比微调前后模型的回答风格差异。

加载两个模型（基座 vs 微调合并版），对同一组问题各生成一次回答，
输出并排对比报告到 outputs/compare_base_vs_finetuned.md。

Usage:
    .venv\\Scripts\\python.exe scripts/py/compare_models.py

说明：
- 两个模型用完全相同的问题与生成参数（temperature=0.7, top_p=0.9, seed=42），保证公平对比
- 8GB 显存装不下 bf16 全量模型，device_map='auto' 会把放不下的层 offload 到内存，
  生成速度略慢属正常
- 微调模型用的是 checkpoint-783-merged（v2 知乎数据 3 轮训练后的完整合并权重）
"""
import gc
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_PATH = os.path.join(PROJECT_ROOT, "outputs", "compare_base_vs_finetuned.md")
SEED = 42

# ── 测试问题：前 3 个贴近知乎训练数据风格，最后 1 个作通用能力对照 ──
QUESTIONS = [
    "有哪些看似简单但特别高效的学习方法？",
    "为什么现在的年轻人越来越喜欢独处？",
    "第一次独自旅行，需要注意什么？",
    "用一句话解释什么是通货膨胀。",
]

GEN_KWARGS = dict(
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

MODELS = [
    ("基座 Qwen3-4B-Instruct-2507", os.path.join(PROJECT_ROOT, "models/Qwen3-4B-Instruct-2507")),
    ("微调后 v2（coig_cqia_zhihu × 3 轮）",
     os.path.join(PROJECT_ROOT, "outputs/qwen3_4b_qlora_sft/v2-20260906-151227/checkpoint-783-merged")),
]


def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "7GiB", "cpu": "24GiB"},
        attn_implementation="sdpa",
    )
    model.eval()
    return tok, model


def answer(tok, model, question):
    torch.manual_seed(SEED)  # 同一问题两个模型使用相同采样起点，尽量公平
    msgs = [{"role": "user", "content": question}]
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            **GEN_KWARGS,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip(), time.time() - t0


def main():
    results = {}  # (model_idx, q_idx) -> (answer, seconds)
    for mi, (name, path) in enumerate(MODELS):
        if not os.path.isdir(path):
            raise SystemExit(f"[!] 模型目录不存在: {path}")
        print(f"\n===== [{mi + 1}/{len(MODELS)}] 加载: {name} =====", flush=True)
        tok, model = load_model(path)
        for qi, q in enumerate(QUESTIONS):
            print(f"  Q{qi + 1}: {q[:24]}...", flush=True)
            ans, sec = answer(tok, model, q)
            results[(mi, qi)] = (ans, sec)
            print(f"      ({sec:.0f}s, {len(ans)} 字)", flush=True)
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    # ── 生成并排对比报告 ──
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = [
        "# 微调前后回答风格对比",
        "",
        f"- 生成参数：max_new_tokens={GEN_KWARGS['max_new_tokens']}, "
        f"temperature={GEN_KWARGS['temperature']}, top_p={GEN_KWARGS['top_p']}, seed={SEED}",
        f"- 对比模型：{' vs '.join(n for n, _ in MODELS)}",
        "",
    ]
    for qi, q in enumerate(QUESTIONS):
        lines += [f"## Q{qi + 1}. {q}", ""]
        for mi, (name, _) in enumerate(MODELS):
            ans, sec = results[(mi, qi)]
            lines += [f"### {name}（{sec:.0f}s）", "", ans, ""]
        lines.append("---")
        lines.append("")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nDone. 报告已写入: {REPORT_PATH}")


if __name__ == "__main__":
    main()
