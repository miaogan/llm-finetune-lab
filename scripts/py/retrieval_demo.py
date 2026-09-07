"""迷你 RAG 检索演示：用微调后的 embedding 召回 + 微调后的 rerank 精排。

语料库：data/retrieval/eval.jsonl 的 952 个段落（知乎/医学/百科/生活技能）
流程：query → embedding 召回 top5 → reranker 精排 → 展示

Usage:
    .venv\\Scripts\\python.exe scripts/py/retrieval_demo.py                # 跑内置示例
    .venv\\Scripts\\python.exe scripts/py/retrieval_demo.py 你的问题      # 自定义查询
"""
import json
import os
import sys

from FlagEmbedding import FlagModel, FlagReranker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DATA = os.path.join(PROJECT_ROOT, "data", "retrieval", "eval.jsonl")
EMB_FT = os.path.join(PROJECT_ROOT, "outputs", "bge-large-zh-ft")
RR_FT = os.path.join(PROJECT_ROOT, "outputs", "bge-reranker-ft")

DEMO_QUERIES = [
    "长期熬夜对身体有什么危害",
    "新手学摄影应该先买什么镜头",
    "感冒了怎么办",
]


def main():
    queries = sys.argv[1:] or DEMO_QUERIES

    pool = []
    with open(EVAL_DATA, "r", encoding="utf-8") as f:
        for line in f:
            pool.append(json.loads(line)["pos"][0])
    print(f"语料库: {len(pool)} 个段落（COIG-CQIA 评估集）\n")

    emb = FlagModel(EMB_FT, use_fp16=True)
    reranker = FlagReranker(RR_FT, use_fp16=True)
    p_emb = emb.encode(pool, batch_size=64)

    for q in queries:
        print("=" * 70)
        print(f"Query: {q}")
        q_emb = emb.encode([q])
        scores = (q_emb @ p_emb.T)[0]
        top = scores.argsort()[-5:][::-1]                     # embedding 召回 top5
        pairs = [[q, pool[i]] for i in top]
        rr_scores = reranker.compute_score(pairs, batch_size=8)
        if not isinstance(rr_scores, list):
            rr_scores = [rr_scores]
        reranked = sorted(zip(top, rr_scores), key=lambda x: -x[1])   # rerank 精排

        print("-" * 70)
        for rank, (idx, rs) in enumerate(reranked[:3], 1):
            text = pool[idx].replace("\n", " ")
            print(f"  #{rank} [rerank={rs:.3f} | emb={scores[idx]:.3f}] {text[:80]}...")
        print()

    print("说明: emb=向量相似度(召回阶段), rerank=交叉编码器分(精排阶段)。")
    print("      rerank 分数高低即最终排序依据，可看到精排如何调整召回顺序。")


if __name__ == "__main__":
    main()
