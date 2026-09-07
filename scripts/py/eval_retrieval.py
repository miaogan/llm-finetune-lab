"""评估 embedding / rerank 模型微调前后的检索效果。

在 data/retrieval/eval.jsonl（952 条）上做两组实验：

  A. embedding 全局检索：952 个正样本构成候选池，每个 query 在全池中检索
     指标：Recall@1 / @5 / @10、MRR —— 对比基座 vs 微调
  B. rerank 精排：取 embedding 召回的 top-10 交给 reranker 重排
     指标：Recall@1、MRR —— 展示"召回 + 精排"完整链路

Usage:
    .venv\\Scripts\\python.exe scripts/py/eval_retrieval.py
"""
import json
import os
import time

import numpy as np
from FlagEmbedding import FlagModel, FlagReranker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DATA = os.path.join(PROJECT_ROOT, "data", "retrieval", "eval.jsonl")

MODELS = {
    "embedding_base": os.path.join(PROJECT_ROOT, "models", "BAAI", "bge-large-zh-v1.5"),
    "embedding_ft": os.path.join(PROJECT_ROOT, "outputs", "bge-large-zh-ft"),
    "rerank_base": os.path.join(PROJECT_ROOT, "models", "BAAI", "bge-reranker-base"),
    "rerank_ft": os.path.join(PROJECT_ROOT, "outputs", "bge-reranker-ft"),
}


def load_eval():
    queries, pos_pool = [], []
    with open(EVAL_DATA, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries.append(d["query"])
            pos_pool.append(d["pos"][0])
    return queries, pos_pool


def retrieval_metrics(queries, pool, model, topk=10):
    """全局检索：query 在整个池中找自己的正样本"""
    t0 = time.time()
    q_emb = model.encode(queries, batch_size=64)
    p_emb = model.encode(pool, batch_size=64)
    sim = q_emb @ p_emb.T                                  # (Q, P)
    gold = np.arange(len(pool))                            # 第 i 个 query 的正样本就是 pool[i]
    order = np.argsort(-sim, axis=1)[:, :topk]
    ranks = (order == gold[:, None]).argmax(axis=1) + 1    # 正样本排名（1-based）
    return {
        "Recall@1": float((ranks == 1).mean()),
        "Recall@5": float((ranks <= 5).mean()),
        "Recall@10": float((ranks <= 10).mean()),
        "MRR": float((1.0 / ranks).mean()),
        "sec": time.time() - t0,
    }


def rerank_metrics(queries, pool, emb_model, reranker, topk=10):
    """embedding 召回 top-10 → reranker 精排，看正样本能否被顶到第 1"""
    t0 = time.time()
    q_emb = emb_model.encode(queries, batch_size=64)
    p_emb = emb_model.encode(pool, batch_size=64)
    sim = q_emb @ p_emb.T
    gold = np.arange(len(pool))
    order = np.argsort(-sim, axis=1)[:, :topk]

    hits, mrrs = 0, []
    for i in range(len(queries)):
        cands = [pool[j] for j in order[i]]
        pairs = [[queries[i], c] for c in cands]
        scores = reranker.compute_score(pairs, batch_size=8)
        if not isinstance(scores, list):
            scores = [scores]
        rerank_order = np.argsort(-np.asarray(scores))
        reranked = [order[i][j] for j in rerank_order]
        rank = reranked.index(gold[i]) + 1 if gold[i] in reranked else topk + 1
        hits += int(rank == 1)
        mrrs.append(1.0 / rank)
    return {"Recall@1": hits / len(queries), "MRR": float(np.mean(mrrs)), "sec": time.time() - t0}


def main():
    queries, pool = load_eval()
    print(f"Eval set: {len(queries)} queries / {len(pool)} passages\n")

    # ── A. embedding：基座 vs 微调 ──
    emb_results = {}
    for tag in ("embedding_base", "embedding_ft"):
        if not os.path.isdir(MODELS[tag]):
            print(f"[skip] {tag} 不存在: {MODELS[tag]}")
            continue
        print(f"Encoding with {tag} ...")
        m = FlagModel(MODELS[tag], use_fp16=True)
        emb_results[tag] = retrieval_metrics(queries, pool, m)
        r = emb_results[tag]
        print(f"  Recall@1 {r['Recall@1']:.4f} | @5 {r['Recall@5']:.4f} | "
              f"@10 {r['Recall@10']:.4f} | MRR {r['MRR']:.4f} ({r['sec']:.0f}s)\n")
        del m
        import torch, gc
        torch.cuda.empty_cache(); gc.collect()

    # ── B. rerank：基座 vs 微调（各配自己的 embedding 召回）──
    pair_plan = [
        ("rerank_base", "embedding_base", "基座链路(base-emb召回+base-rerank)"),
        ("rerank_ft", "embedding_ft", "微调链路(ft-emb召回+ft-rerank)"),
    ]
    rerank_results = {}
    for rtag, etag, label in pair_plan:
        if not (os.path.isdir(MODELS[rtag]) and etag in emb_results):
            print(f"[skip] {rtag} 或其 embedding 不可用")
            continue
        print(f"Reranking: {label} ...")
        emb_m = FlagModel(MODELS[etag], use_fp16=True)
        rr = FlagReranker(MODELS[rtag], use_fp16=True)
        rerank_results[rtag] = rerank_metrics(queries, pool, emb_m, rr)
        r = rerank_results[rtag]
        print(f"  Recall@1 {r['Recall@1']:.4f} | MRR {r['MRR']:.4f} ({r['sec']:.0f}s)\n")
        del emb_m, rr
        import torch, gc
        torch.cuda.empty_cache(); gc.collect()

    # ── 汇总 ──
    print("=" * 60)
    print("Summary")
    for k, v in emb_results.items():
        print(f"  [embedding] {k:15s} R@1 {v['Recall@1']:.4f}  MRR {v['MRR']:.4f}")
    for k, v in rerank_results.items():
        print(f"  [rerank]   {k:15s} R@1 {v['Recall@1']:.4f}  MRR {v['MRR']:.4f}")


if __name__ == "__main__":
    main()
