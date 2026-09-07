"""从 COIG-CQIA 构造检索模型训练数据（embedding / rerank 通用）。

输出 FlagEmbedding 微调要求的 jsonl 格式，每行：
    {"query": "问题", "pos": ["对应答案段落"], "neg": ["随机负样本段落", ...]}

- query  = instruction（+ input 拼接）
- pos    = output（截断到 ~350 字，适配 bge 512 token 上限）
- neg    = 从全局段落池随机采样（embedding/rerank 的负样本，天然可得）
- 输出 train.jsonl / eval.jsonl（9:1 划分），embedding 与 rerank 共用

Usage:
    .venv\\Scripts\\python.exe scripts/py/build_retrieval_data.py
"""
import json
import os
import random

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PROJECT_ROOT, "data", "coig-cqia")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "retrieval")

# 混合子集与采样量：领域均衡，总约 9600 条
SUBSETS = {
    "zhihu/zhihu_score9.0-10_clean_v10.jsonl": 2561,   # 知乎高分问答（全取）
    "wikihow/wikihow.jsonl": 1485,                     # 生活技能（全取）
    "wiki/bkmy_medicine.jsonl": 2000,                  # 医学百科（采样）
    "wiki/bkmy_symptom.jsonl": 668,                    # 症状百科（全取）
    "wiki/zgbk.jsonl": 1706,                           # 中国大百科（全取）
    "exam/coig_exam_sampled_clean_v3.jsonl": 1000,     # 试题（采样）
    "zhihu/zhihu_expansion.jsonl": 171,                # 知乎扩展（全取）
}

N_NEG = 7                # 每个 query 的负样本数（配 train_group_size=8）
POS_MAX_CHARS = 350      # pos/neg 段落截断长度
QUERY_MAX_CHARS = 128    # query 截断长度


def load_and_clean():
    """读取各子集并清洗，返回 [(query, passage), ...]"""
    pairs = []
    for rel, take in SUBSETS.items():
        path = os.path.join(SRC, rel)
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = (d.get("instruction") or "").strip()
                extra = (d.get("input") or "").strip()
                if extra:
                    q = f"{q} {extra}".strip()
                a = (d.get("output") or "").strip()
                # 过滤：query 太短没区分度，答案太短当负样本/正样本都没意义
                if len(q) < 6 or len(a) < 30:
                    continue
                rows.append((q[:QUERY_MAX_CHARS], a[:POS_MAX_CHARS]))
        if len(rows) > take:
            rows = random.sample(rows, take)
        pairs.extend(rows)
        print(f"  {rel}: 取 {len(rows)} 条")
    return pairs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading COIG-CQIA subsets ...")
    pairs = load_and_clean()
    print(f"Total pairs: {len(pairs)}")

    # 负样本池：全部 passage
    pool = [p for _, p in pairs]

    records = []
    for q, pos in pairs:
        negs = []
        while len(negs) < N_NEG:
            cand = random.choice(pool)
            if cand != pos and cand not in negs:
                negs.append(cand)
        records.append({"query": q, "pos": [pos], "neg": negs})

    random.shuffle(records)
    n_eval = max(1, len(records) // 10)
    eval_set, train_set = records[:n_eval], records[n_eval:]

    for name, data in (("train.jsonl", train_set), ("eval.jsonl", eval_set)):
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} -> {path}")

    print("\nSample record:")
    s = train_set[0]
    print("  query:", s["query"][:60])
    print("  pos  :", s["pos"][0][:60])
    print("  neg0 :", s["neg"][0][:60])


if __name__ == "__main__":
    main()
