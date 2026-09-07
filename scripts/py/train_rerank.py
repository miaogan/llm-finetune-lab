"""微调 rerank 模型（bge-reranker-base）— 交叉编码器排序学习。

数据：data/retrieval/train.jsonl（与 embedding 微调共用同一份三元组）
原理：cross-encoder 把 (query, 段落) 拼一起打相关性分，对 1正+7负 做
      softmax 交叉熵（listwise），学会把正样本排最前。RAG 二阶段召回后用它精排。

基座选型说明：8GB 显卡跑不动 568M 的 bge-reranker-v2-m3 全参微调
（fp32 参数+梯度+优化器静态就要 ~9GB，Windows WDDM 换页导致 90s+/步）。
bge-reranker-base（278M，中文专用）静态只占 ~2.5GB，速度正常。

Usage:
    .venv\\Scripts\\python.exe scripts/py/train_rerank.py

显存：278M 模型 + 8-bit 优化器 + 梯度检查点 ≈ 4GB
耗时：8.5k 样本 1 轮，RTX 4070 约 30~50 分钟
"""
import os

from FlagEmbedding.abc.finetune.reranker import (
    AbsRerankerDataArguments,
    AbsRerankerModelArguments,
    AbsRerankerTrainingArguments,
)
from FlagEmbedding.finetune.reranker.encoder_only.base import EncoderOnlyRerankerRunner

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "BAAI", "bge-reranker-base")
TRAIN_DATA = os.path.join(PROJECT_ROOT, "data", "retrieval", "train.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bge-reranker-ft")


def main():
    model_args = AbsRerankerModelArguments(
        model_name_or_path=MODEL_PATH,
        cache_dir=os.path.join(PROJECT_ROOT, "outputs", ".cache"),
    )
    data_args = AbsRerankerDataArguments(
        train_data=[TRAIN_DATA],  # 注意：必须是列表
        train_group_size=8,        # 1 正 + 7 负，listwise 交叉熵
        query_max_len=64,
        passage_max_len=512,
    )
    training_args = AbsRerankerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,                 # rerank 学得快，1 轮防过拟合
        # 显存策略同 embedding：小 batch + 大累积，避免 Windows WDDM 换页
        # （8GB 卡上 fp32 参数+梯度+优化器已占 ~5.2GB，激活只能小口喂）
        per_device_train_batch_size=1,      # 1 样本 × 8 候选 = 8 序列/微批
        gradient_accumulation_steps=32,     # 等效 batch 32（与原 4×8 相同）
        learning_rate=3e-5,
        # 568M 模型 fp32 静态占用 ~9GB 装不下：8-bit 优化器 + 梯度检查点双管齐下
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        sub_batch_size=2,                   # 每样本 8 候选拆成 4×2 子批前向，激活峰值降 4 倍
        warmup_steps=25,                     # ≈5% of 536 步（transformers 5.x 无 warmup_ratio）
        lr_scheduler_type="cosine",
        bf16=True,
        dataloader_num_workers=0,           # Windows 必须 0
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=42,
    )

    runner = EncoderOnlyRerankerRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    runner.run()
    print(f"\nDone. 微调后的 rerank 模型在: {OUTPUT_DIR}（最终权重在最后 checkpoint）")


if __name__ == "__main__":
    main()
