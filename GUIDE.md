# 本地模型微调 CLI 速查（ms-swift）

> 项目路径：`D:\work_space\llm-finetune-lab`
> GPU: RTX 4070 Laptop 8GB | 模型: Qwen3-4B-Instruct-2507 | 框架: ms-swift 4.5
> 所有脚本用 Python，在项目根目录下运行。完整教程见 `README.md`，本手册供快速查阅。

---

## 0. 激活环境（每个新终端先跑一次）

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\use_env.py
```

设置 `PIP_CONFIG_FILE`、`DATA_DIR`、`PATH`。

> **首次使用 / 环境损坏**：先跑 `python scripts\py\setup_env.py` 用 Python 3.12 重建 `.venv` 并安装 `ms-swift` + `bitsandbytes` + `gradio`。

---

## 1. 下载模型 / 数据集（已就位可跳过）

```powershell
# 模型（默认 Qwen3-4B-Instruct-2507）
python D:\work_space\llm-finetune-lab\scripts\py\download_model.py

# 数据集
python D:\work_space\llm-finetune-lab\scripts\py\download_dataset.py

# 换其他模型（--proxy 可选，网络受限时用）
python D:\work_space\llm-finetune-lab\scripts\py\download_model.py Qwen/Qwen3-1.7B
python D:\work_space\llm-finetune-lab\scripts\py\download_model.py Qwen/Qwen3-1.7B --proxy http://127.0.0.1:7890
```

---

## 2. 训练

### 2.1 快速验证（demo 数据，约 1 分钟）

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\train.py
```

默认配置 `configs/qwen3_4b_qlora_sft.yaml`，用 30 条 demo_selfcognition 数据跑 3 轮。
等价裸命令：`.venv\Scripts\swift.exe sft configs\qwen3_4b_qlora_sft.yaml`

### 2.2 正式训练

```powershell
notepad D:\work_space\llm-finetune-lab\configs\qwen3_4b_qlora_sft.yaml
```

改 `dataset:` 一行即可（demo_selfcognition → 正式数据集别名），然后跑 `train.py`。

主配置关键字段速览（文件内有逐行注释）：

```yaml
model: models/Qwen3-4B-Instruct-2507   # 基座模型路径
custom_dataset_info: data/dataset_info.json  # 数据集注册表
dataset: coig_cqia_zhihu               # ★ 每次训练主要改这里；别名#N 可采样
quant_method: bnb                      # QLoRA 量化方法
quant_bits: 4                          # 4bit 量化（8GB 显存的关键）
tuner_type: lora                       # 微调方法
lora_rank: 8                           # LoRA 容量，数据多可升 16/32
target_modules: all-linear             # 对所有线性层挂 LoRA
max_length: 1024                       # 单样本最大 token；OOM 时降到 512
learning_rate: 1.0e-4                  # 学习率；loss 飙升就调小
num_train_epochs: 3.0                  # 训练轮数；过拟合就减少
per_device_train_batch_size: 1         # 保持 1（显存所限）
gradient_accumulation_steps: 8         # 等效 batch = 1×8 = 8
dataloader_num_workers: 0              # Windows 必须为 0
```

> `template` 无需手写，ms-swift 会按 config.json 的 architectures 自动匹配 Qwen3 模板。

可选数据集别名（注册于 `data/dataset_info.json`）：

| 别名 | 说明 |
|---|---|
| `demo_selfcognition` | 自我认知示例（~30 条） |
| `coig_cqia_full` | 全量中文指令（~4.5 万条） |
| `coig_cqia_zhihu` | 知乎高分问答（~2000 条） |
| `coig_cqia_wikihow` | 生活技能 |
| `coig_cqia_exam` | 考试试题 |
| `coig_cqia_medicine` / `_poem` / `_chengyu` / `_ruozhiba` / `_segmentfault` / `_zhihu_expansion` / `_xhs` | 各领域中文指令 |

采样训练（只取前 N 条）：`dataset: coig_cqia_full#1000`

自定义配置文件：`python scripts\py\train.py configs\my_config.yaml`

### 2.3 训练产物在哪、怎么看

```
outputs/qwen3_4b_qlora_sft/v2-20260906-151227/   # 每次训练一个目录
├── checkpoint-200/  checkpoint-400/  ...        # 每 save_steps 步存一个（自动保留最近 3 个）
│   └── args.json + adapter_model.safetensors    # ★ chat/export 靠这两个文件工作
├── logging.jsonl        # 每步 loss 日志（文本，可直接打开看）
├── images/              # 训练曲线 png（train_loss.png 等）
└── runs/                # tensorboard 日志
```

- **看曲线**：`.venv\Scripts\tensorboard.exe --logdir outputs\qwen3_4b_qlora_sft`，浏览器开 `http://localhost:6006`
- **判断在学好**：loss 缓慢下降、token_acc 缓慢上升；loss 飙升 = 学习率太大
- **判断训练完成**：终端打印 `train_runtime` / `train_loss` 总结行，且出现最终 checkpoint

### 2.4 断点续训 / OOM 调参

中断后续训（配置中加一行，路径换成实际 checkpoint）：

```yaml
resume_from_checkpoint: outputs/qwen3_4b_qlora_sft/v2-20260906-151227/checkpoint-400
```

OOM 时：

```yaml
max_length: 512                        # 从 1024 降
gradient_accumulation_steps: 16        # 补偿 batch size
```

确认 `quant_bits: 4`、`per_device_train_batch_size: 1`、`gradient_checkpointing: true`。
配置已设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解显存碎片。

---

## 3. 对话测试

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\chat.py
```

自动加载 `outputs/` 下最新的 ms-swift adapter（读 `args.json` 还原基座模型与模板）。

```powershell
# 指定 checkpoint
python D:\work_space\llm-finetune-lab\scripts\py\chat.py outputs\qwen3_4b_qlora_sft\v2-xxx\checkpoint-400

# 加载已合并的完整权重（无需 adapter）
python D:\work_space\llm-finetune-lab\scripts\py\chat.py --merged
```

输入 `exit` 退出。

---

## 4. 导出合并模型

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\export.py
```

自动探测最新 checkpoint，合并后默认输出到 `models/Qwen3-4B-finetuned/`（约 8GB，以脚本打印的实际落盘目录为准）。

```powershell
# 指定 adapter 与输出目录
python D:\work_space\llm-finetune-lab\scripts\py\export.py --adapter outputs\qwen3_4b_qlora_sft\v2-xxx\checkpoint-783 --out models\my-merged
```

接入 Ollama：在 `configs/export_merged.yaml` 取消 `to_ollama: true` 注释后重新导出，生成带 Modelfile 的目录，再执行：

```powershell
ollama create my-model -f models\Qwen3-4B-finetuned-ollama\Modelfile
ollama run my-model
```

---

## 5. 使用自有数据

ms-swift 原生支持 alpaca 三字段格式（`instruction`/`input`/`output` 自动映射为 query/response）。在 `data/` 下创建 JSON/JSONL：

```json
{"instruction": "翻译成英文", "input": "今天天气真好", "output": "The weather is nice today."}
```

方式 A —— 注册别名（在 `data/dataset_info.json`，**列表**格式）：

```json
{
  "dataset_name": "my_dataset",
  "dataset_path": "my_data.jsonl"
}
```

> ⚠️ `dataset_path` 相对路径以 `dataset_info.json` 所在目录（`data/`）为基准，**不要写 `data/` 前缀**（否则拼成 `data/data/...` 报 FileNotFoundError）。

然后训练配置改 `dataset: my_dataset`。

方式 B —— 不注册，直接在配置里写路径：

```yaml
dataset: data/my_data.jsonl
```

跑 `python scripts\py\train.py`。

---

## 6. 可视化训练 / 辅助命令

```powershell
# Web-UI（界面训练/推理，后台另起进程，关界面不中断训练；停训在 Runtime 页 kill service）
python D:\work_space\llm-finetune-lab\scripts\py\webui.py
python D:\work_space\llm-finetune-lab\scripts\py\webui.py --lang en

# 重建环境（.venv 损坏或首次使用）
python D:\work_space\llm-finetune-lab\scripts\py\setup_env.py
```

---

## 速查表

| 操作 | 命令 |
|---|---|
| 激活环境 | `python scripts\py\use_env.py` |
| 重建环境 | `python scripts\py\setup_env.py` |
| 下载模型 | `python scripts\py\download_model.py` |
| 下载数据 | `python scripts\py\download_dataset.py` |
| 训练 | `python scripts\py\train.py` |
| 对话 | `python scripts\py\chat.py` |
| 导出合并 | `python scripts\py\export.py` |
| Web-UI | `python scripts\py\webui.py` |
| 看训练曲线 | `.venv\Scripts\tensorboard.exe --logdir outputs\qwen3_4b_qlora_sft` |

## 常见报错速查

| 报错关键词 | 原因与解决 |
|---|---|
| `FileNotFoundError: ... data\data\...` | 数据集注册表里 `dataset_path` 多写了 `data/` 前缀（正确写法见第 5 节） |
| `CUDA out of memory` | 按 2.4 节降 `max_length`；确认 `quant_bits: 4` |
| `没有 args.json` | 指向的不是 ms-swift 产物（旧框架产物已清理）；用 `train.py` 重新训练 |
| `swift: command not found` | 没激活环境；先跑 `use_env.py`，或直接用 `.venv\Scripts\swift.exe` |
| flash-attention 警告 | Windows 正常现象，已配 `attn_impl: sdpa`，忽略 |
| pip 安装超时 | 先 `$env:PIP_CONFIG_FILE='D:\work_space\llm-finetune-lab\pip.ini'` 再装 |
