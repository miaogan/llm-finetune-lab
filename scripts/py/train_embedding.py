"""微调 embedding 模型（bge-large-zh-v1.5）— 双塔对比学习。

数据：data/retrieval/train.jsonl（{"query","pos","neg"}，由 build_retrieval_data.py 生成）
原理：把 query 和 1正+7负 段落编码成向量，InfoNCE 对比损失拉近正样本、推远负样本。

Usage:
    .venv\\Scripts\\python.exe scripts/py/train_embedding.py

显存：326M 模型 bf16 + batch 4 × (1q+7p) ≈ 5GB，8GB 显卡无压力
耗时：8.5k 样本 2 轮，RTX 4070 约 30~50 分钟
"""
import os

from FlagEmbedding.finetune.embedder.encoder_only.base import (
    EncoderOnlyEmbedderDataArguments,
    EncoderOnlyEmbedderModelArguments,
    EncoderOnlyEmbedderTrainingArguments,
    EncoderOnlyEmbedderRunner,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "BAAI", "bge-large-zh-v1.5")
TRAIN_DATA = os.path.join(PROJECT_ROOT, "data", "retrieval", "train.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bge-large-zh-ft")


def main():
    model_args = EncoderOnlyEmbedderModelArguments(
        model_name_or_path=MODEL_PATH,
        cache_dir=os.path.join(PROJECT_ROOT, "outputs", ".cache"),
        # query 指令前缀：训练与推理保持一致（此处 query 已是完整问题，留空）
    )
    data_args = EncoderOnlyEmbedderDataArguments(
        train_data=[TRAIN_DATA],  # 注意：必须是列表
        train_group_size=8,        # 1 正 + 7 负（与 build_retrieval_data.py 的 N_NEG 对应）
        query_max_len=64,
        passage_max_len=512,
    )
    training_args = EncoderOnlyEmbedderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        # 显存策略：8GB 卡被桌面占用约 0.6GB，batch 大会触发 Windows WDDM
        # 共享内存换页（GPU 100% 但每步 90s+）。用小 batch + 大累积，等效不变。
        per_device_train_batch_size=1,       # 1 样本 × 8 文本 = 8 序列/微批
        gradient_accumulation_steps=16,      # 等效 batch 16（与原 4×4 相同）
        learning_rate=2e-5,                  # bge 官方推荐微调学习率
        # 显存第二刀：Trainer 的 bf16 只是计算混合精度，参数+优化器状态仍是 fp32
        # （静态 ~5.2GB，8GB 卡必爆）。8-bit 优化器把 AdamW 状态压到 1/4。
        optim="adamw_bnb_8bit",
        warmup_steps=50,                     # ≈5% of 1072 步（transformers 5.x 无 warmup_ratio）
        lr_scheduler_type="cosine",
        temperature=0.02,                    # InfoNCE 温度（对比学习关键超参）
        bf16=True,
        dataloader_num_workers=0,            # Windows 必须 0
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
        seed=42,
    )

    runner = EncoderOnlyEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    runner.run()
    print(f"\nDone. 微调后的 embedding 模型在: {OUTPUT_DIR}（最终权重在最后 checkpoint）")


if __name__ == "__main__":
    main()
