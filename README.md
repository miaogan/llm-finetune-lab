# LLM 微调实验室（llm-finetune-lab）

本地大模型微调与管理一体化项目，基于阿里魔搭 [ms-swift](https://github.com/modelscope/ms-swift) 框架，专为 8GB 显存（RTX 4070 Laptop）调优。另含检索模型（embedding / rerank）微调模块，基于智源 [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) 框架。

## 这个项目能做什么？

用 **QLoRA**（一种低显存微调技术）在你的笔记本 GPU 上训练属于自己的大模型：

- **微调**：让 Qwen3-4B 学会你想要的行为（如特定问答风格、领域知识）
- **测试**：训练完立刻对话验证效果
- **导出**：把训练成果合并成可独立部署的完整模型，还能一键接入 Ollama 本地运行
- **检索微调**：微调 BGE 系列的 embedding（召回）与 rerank（精排）模型，搭出迷你 RAG 检索链路

> **给初学者的三个概念**：
> - **基座模型**：出厂的通用大模型（本项目用 `Qwen3-4B-Instruct-2507`，约 8GB）
> - **LoRA 适配器**：训练产出的"补丁"，只存模型中很少的可训练参数（几十 MB），推理时叠加到基座上生效
> - **QLoRA**：先把基座压缩到 4bit 再训练 LoRA，让 8GB 显存也能微调 4B 模型

## 已配置环境

| 组件 | 说明 |
|---|---|
| Python 虚拟环境 | `.venv`（Python 3.12，ms-swift 官方推荐版本） |
| 微调框架 | ms-swift 4.5 + bitsandbytes（4bit QLoRA） |
| 基座模型 | Qwen3-4B-Instruct-2507（`models/` 目录，已就位） |
| 数据集 | COIG-CQIA 高质量中文指令集（`data/coig-cqia/`，已就位） |
| 显存策略 | QLoRA 4bit 量化 + LoRA，训练约占用 5GB 显存 |

> **Python 版本**：ms-swift 官方支持 `>=3.10`、推荐 `3.12`。本项目锁定 3.12。
>
> **网络提示**：当前 IPv4 出口正常，魔搭（modelscope.cn）可直接访问。若网络受限（如仅 IPv6 可用，魔搭无 IPv6 地址），请先启动本机代理软件（`系统代理` 模式）再运行下载脚本。pip 依赖走清华源，无需代理即可安装。

## 目录结构

```
llm-finetune-lab/
├── .venv/                          # Python 虚拟环境（可随时用 setup_env.py 重建）
├── configs/                        # 训练/导出配置（YAML，所有超参都在这里改）
│   ├── qwen3_4b_qlora_sft.yaml         # QLoRA 微调主配置（★ 最常编辑）
│   └── export_merged.yaml              # 合并导出配置
├── data/                           # 数据集目录
│   ├── dataset_info.json               # ms-swift 数据集注册表（列表格式）
│   ├── demo_selfcognition.json         # 自我认知示例数据（~30 条，快速试跑）
│   ├── coig-cqia/                      # COIG-CQIA 中文指令数据集（各领域 jsonl）
│   └── retrieval/                      # 检索微调数据（query/pos/neg 三元组）
│       ├── train_demo.jsonl                # 小样本（100 条，格式参考/试跑）
│       └── eval_demo.jsonl                 # 小样本（100 条）
├── models/                         # 模型存放目录
│   ├── Qwen3-4B-Instruct-2507/         # 基座模型
│   └── BAAI/                           # BGE 检索模型（bge-large-zh-v1.5 / bge-reranker-base）
├── outputs/                        # 训练输出（每次训练一个带时间戳的子目录）
├── scripts/py/                     # 常用脚本（每个都可用 --help 查看用法）
│   ├── use_env.py                      # 激活环境（每个新终端先跑一次）
│   ├── setup_env.py                    # 重建虚拟环境（环境损坏时用）
│   ├── download_model.py               # 下载模型
│   ├── download_dataset.py             # 下载数据集
│   ├── swift_paths.py                  # 内部工具：adapter 自动定位（勿直接运行）
│   ├── train.py                        # 训练入口（LLM 指令微调）
│   ├── chat.py                         # 对话测试
│   ├── export.py                       # 导出合并模型
│   ├── webui.py                        # 启动可视化训练界面
│   ├── build_retrieval_data.py         # 从 COIG-CQIA 构造检索三元组数据
│   ├── train_embedding.py              # 微调 embedding 模型（bge-large-zh-v1.5）
│   ├── train_rerank.py                 # 微调 rerank 模型（bge-reranker-base）
│   ├── eval_retrieval.py               # 评估微调前后检索效果（Recall/MRR）
│   └── retrieval_demo.py               # 迷你 RAG 演示（召回 + 精排）
├── pip.ini                         # 项目专用 pip 源（清华源）
├── README.md                        # 本文件：完整使用指南
├── GUIDE.md                         # CLI 速查手册（配合本文件使用）
└── EXPORT.md                        # 模型导出与部署详解（原理+实操+踩坑记录）
```

## 新手完整流程（五步走）

> 全程在项目根目录 `D:\work_space\llm-finetune-lab` 下操作。

### 第 0 步：激活环境（每个新终端都要做）

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\use_env.py
```

**预期输出**：三行 `[OK]`（venv 路径、pip 源、python 版本）+ ms-swift CLI 可用。

> 这一步只是设置环境变量，不会改变你的系统。`.venv` 已就位时无需再做别的。

### 第 1 步：demo 试跑（验证环境，约 1 分钟）

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\train.py
```

默认配置用 30 条自我认知数据跑 3 轮，只为验证环境没问题。

**预期过程**：
1. 终端打印完整参数列表（`SftArguments(...)`）
2. 加载基座模型（`Loading weights: 100%`）
3. 生成数据集（`Generating train split: 29 examples`）
4. 出现训练进度条 `Train: 100%|...| 12/12`，每 10 步打印一次 loss
5. 结束时保存 checkpoint 并打印 `train_loss`、`eval_loss`

**怎么判断成功**：loss 从约 4.0 降到 3.5 左右、显存占用 < 3GB、`outputs/qwen3_4b_qlora_sft/` 下出现新的 `vN-时间戳/checkpoint-12/` 目录。跑通即可进入下一步。

### 第 2 步：正式训练（换数据集）

编辑主配置：

```powershell
notepad D:\work_space\llm-finetune-lab\configs\qwen3_4b_qlora_sft.yaml
```

只改一行——`dataset:` 换成正式数据集别名：

```yaml
dataset: coig_cqia_zhihu        # 或 coig_cqia_wikihow / coig_cqia_exam 等，见下表
```

然后再次运行：

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\train.py
```

**训练要多久**：以知乎数据集（2091 条、3 轮、783 步）为例，约 80 分钟；全量数据集会更久。可先采样小跑：`dataset: coig_cqia_zhihu#500` 只取 500 条。

**怎么判断模型在学好**：看终端每隔 10 步打印的 `loss`——总体应缓慢下降（如 2.8 → 2.5），`token_acc` 缓慢上升。偶尔波动正常。

**看训练曲线**（可选）：

```powershell
.venv\Scripts\tensorboard.exe --logdir outputs\qwen3_4b_qlora_sft
```

浏览器打开提示的地址（通常 `http://localhost:6006`），SCALARS 页有 loss / lr / token_acc 曲线；每个版本目录的 `images/` 下也自动存了 png 曲线图。

**中断了怎么办**：在配置中加一行 `resume_from_checkpoint: outputs/qwen3_4b_qlora_sft/v2-xxxx/checkpoint-N`（用实际存在的 checkpoint 路径），重新运行 `train.py` 即可从断点续训。

### 第 3 步：对话测试微调效果

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\chat.py
```

自动加载 `outputs/` 下**最新**的 LoRA 适配器（读取 checkpoint 里的 `args.json` 自动还原基座模型与模板），进入交互对话。输入 `exit` 退出。

**指定某个 checkpoint 测试**：

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\chat.py outputs\qwen3_4b_qlora_sft\v1-20260906-150321\checkpoint-12
```

### 第 4 步：导出合并模型 / 接入 Ollama

LoRA 适配器不能独立部署，需合并回基座得到完整权重：

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\export.py
```

自动探测最新 checkpoint，合并后输出到 `models/Qwen3-4B-finetuned/`（约 8GB，脚本会打印实际落盘路径）。

**接入 Ollama 本地运行**：编辑 `configs/export_merged.yaml`，取消 `to_ollama: true` 和对应 `output_dir` 两行注释，重新运行 `export.py`，然后：

```powershell
ollama create my-model -f models\Qwen3-4B-finetuned-ollama\Modelfile
ollama run my-model
```

### 可选：可视化训练（新手友好界面）

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\webui.py
```

浏览器打开终端提示的地址（通常 `http://localhost:7860`）。界面里每个超参都带 `--xxx` 前缀，和命令行参数一致。注意：WebUI 点「开始」后在后台另起独立训练进程，**关掉界面不会中断训练**；要停训需在 Runtime 标签页选中任务点 kill service。

## 训练产物结构解读

```
outputs/qwen3_4b_qlora_sft/
└── v2-20260906-151227/            # 每次训练一个目录：v序号-时间戳
    ├── checkpoint-200/            # 每 200 步存一个（save_steps 控制）
    │   ├── adapter_model.safetensors  # ★ LoRA 权重（几十 MB）
    │   ├── adapter_config.json        # LoRA 结构参数
    │   ├── args.json                  # ★ 完整训练参数（chat/export 靠它自动还原）
    │   └── optimizer.pt 等            # 断点续训用的状态（占大头，可删）
    ├── checkpoint-400/
    ├── args.json                    # 本次训练参数（顶层）
    ├── logging.jsonl                # 每步日志（loss 等，可直接文本查看）
    ├── val_dataset.jsonl            # 自动划分出的验证集
    ├── images/                      # 训练曲线 png
    └── runs/                        # tensorboard 日志
```

> **清理旧 checkpoint**：`save_total_limit: 3` 已自动只保留最近 3 个 checkpoint。想节省磁盘，可删掉 checkpoint 目录里的 `optimizer.pt`（每个几百 MB，只影响断点续训，不影响推理）。

## 数据集说明

### 内置数据集（别名见 `data/dataset_info.json`）

| 别名 | 内容 | 规模 | 用途 |
|---|---|---|---|
| `demo_selfcognition` | 自我认知问答 | ~30 条 | 快速验证训练流程 |
| `coig_cqia_full` | COIG-CQIA 全量中文指令 | ~4.5 万条 | 正式指令微调 |
| `coig_cqia_zhihu` | 知乎高分问答 | ~2000 条 | 正式指令微调 |
| `coig_cqia_wikihow` | 生活技能 | - | 正式指令微调 |
| `coig_cqia_exam` | 考试试题 | - | 正式指令微调 |
| `coig_cqia_medicine` / `_poem` / `_chengyu` / `_ruozhiba` / `_segmentfault` / `_zhihu_expansion` / `_xhs` | 各领域中文指令 | - | 按需组合 |

采样控制数据量：`dataset: 别名#1000` 表示只取 1000 条。

### 数据格式（alpaca）与自有数据接入

ms-swift 原生支持 alpaca 三字段格式，自动把 `instruction`+`input` 合并为 query、`output` 作为 response。新增自有数据两步：

1. 准备 JSON/JSONL 文件放入 `data/`，每条格式：

```json
{"instruction": "你的指令", "input": "补充输入（可为空）", "output": "期望回答"}
```

2. 在 `data/dataset_info.json`（**列表**格式）中追加一项：

```json
{
  "dataset_name": "my_dataset",
  "dataset_path": "my_data.jsonl"
}
```

> **路径基准（重要）**：`dataset_path` 里的相对路径以 `dataset_info.json` 所在目录（即 `data/`）为基准解析，**不要再写 `data/` 前缀**（写了会拼成 `data/data/...` 导致 FileNotFoundError）。
>
> 也可以完全不注册，直接在训练配置里写 `dataset: data/my_data.jsonl`（此处相对路径按运行命令时的工作目录解析），ms-swift 会自动识别格式。

## 显存与调参建议（8GB）

| 现象 | 调整 |
|---|---|
| 显存溢出（OOM） | `max_length` 从 1024 降到 512；`per_device_train_batch_size` 保持 1；确认 `quant_bits: 4` |
| 训练太慢 | 减少数据量（`dataset: 别名#1000`）；或先用 `demo_selfcognition` 跑通 |
| 想训更大模型 | QLoRA 上限约 7B（8GB 很紧张）；建议仍用 4B |
| 过拟合（loss 降但对话变差） | 减少 `num_train_epochs`（如 3.0 → 1.0）；增加数据量 |

关键超参的通俗解释（详见主配置文件内注释）：

| 参数 | 作用 | 新手建议 |
|---|---|---|
| `learning_rate` | 每步学习的"步幅" | 保持 1e-4；太大学崩（loss 飙升），太小学不动 |
| `num_train_epochs` | 全量数据过几遍 | 指令微调 1~3 轮；过多会过拟合 |
| `lora_rank` | LoRA 补丁的"容量" | 8 够用；数据多可到 16/32（显存略增） |
| `max_length` | 单条样本最大 token 数 | 1024；样本短可降到 512 省显存 |

> 配置里已开启 `gradient_checkpointing: true` 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片化 OOM。Windows 下 `dataloader_num_workers` 必须为 0（官方要求）。

## 检索模型微调（embedding + rerank）

除了对话模型，本项目还能微调 RAG 检索链路里的两类模型（基于 FlagEmbedding 框架，已装进 `.venv`）：

> **两个概念（与 LLM 微调的 LoRA 不同，这里都是全参微调）**：
> - **embedding 模型**（双塔）：把 query 和段落各自编码成向量，靠相似度召回。本项微调 `bge-large-zh-v1.5`（326M）
> - **rerank 模型**（交叉编码器）：把 (query, 段落) 拼一起打相关性分，用于召回后的精排。本项目微调 `bge-reranker-base`（278M）

### 第 1 步：构造三元组数据

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\build_retrieval_data.py
```

从已下载的 COIG-CQIA 各领域子集采样约 9600 条，生成 FlagEmbedding 要求的格式（每行一个 json）：

```json
{"query": "问题", "pos": ["正确段落"], "neg": ["随机负样本段落", "...共7个"]}
```

输出到 `data/retrieval/train.jsonl`（8.6k 条）与 `eval.jsonl`（952 条）。仓库里附有 `train_demo.jsonl` / `eval_demo.jsonl`（各 100 条小样本），可直接看格式。

### 第 2 步：微调 embedding 模型（约 30~50 分钟）

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\train_embedding.py
```

- 原理：对比学习（InfoNCE），把 query 与 1 正 + 7 负段落编码成向量，拉近正样本、推远负样本
- 显存策略：`adamw_bnb_8bit` 8-bit 优化器（把 AdamW 状态压到 1/4）+ 小 batch 大累积，约 5GB
- 产物：`outputs/bge-large-zh-ft/`（最终权重在最后一个 checkpoint 子目录）

### 第 3 步：微调 rerank 模型（约 30~50 分钟）

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\train_rerank.py
```

- 原理：listwise 交叉熵，对 1 正 + 7 负打分排序，学会把正样本顶到最前
- 基座选型：8GB 显存跑不动 568M 的 `bge-reranker-v2-m3`（fp32 静态就要 ~9GB），故用 278M 的 `bge-reranker-base`
- 显存策略：8-bit 优化器 + 梯度检查点 + `sub_batch_size`（每样本 8 候选拆子批前向），约 4GB
- 产物：`outputs/bge-reranker-ft/`

### 第 4 步：评估微调效果

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\eval_retrieval.py
```

在 eval 集 952 个段落构成的候选池上做全局检索，对比基座 vs 微调的 `Recall@1/@5/@10` 和 `MRR`，并评估"召回 + 精排"完整链路。本机实测结果：

| 模型 | Recall@1 | MRR | 说明 |
|---|---|---|---|
| bge-large-zh-v1.5 基座 | 0.9596 | 0.9764 | embedding 召回 |
| bge-large-zh 微调后 | 0.9601 | 0.9767 | 与基座持平（基座已很强） |
| 基座链路（base 召回 + base 精排） | 0.9548 | 0.9713 | rerank 精排 |
| 微调链路（ft 召回 + ft 精排） | **0.9664** | **0.9793** | 优于基座链路 |

> 解读：embedding 召回层面基座本已接近天花板，微调收益有限属正常；但"召回 + 精排"链路上微调组合有明确提升（R@1 +1.2pt）。若换到自己的垂直领域数据（医疗、法律等），微调收益会显著得多。

### 第 5 步：迷你 RAG 检索演示

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\retrieval_demo.py              # 内置示例问题
python D:\work_space\llm-finetune-lab\scripts\py\retrieval_demo.py 感冒了怎么办  # 自定义查询
```

流程：query → 微调 embedding 召回 top5 → 微调 reranker 精排 → 打印每个候选的 emb 分与 rerank 分，可直观看到精排如何调整召回顺序。

## 常见问题

**Q: 首次使用要装什么？**
运行 `python scripts/py/setup_env.py`，它会用 Python 3.12 重建 `.venv` 并安装 `ms-swift` + `bitsandbytes` + `gradio`。

**Q: pip 安装卡住/超时？**
本机全局 pip.ini 配置了不可达的阿里云备用源。本项目脚本已自动设置 `PIP_CONFIG_FILE` 指向项目内 `pip.ini`（纯清华源）。手动装包前请先执行：
```powershell
$env:PIP_CONFIG_FILE='D:\work_space\llm-finetune-lab\pip.ini'
```

**Q: 训练时报 flash-attention 相关警告？**
Windows 下不装 flash-attn 是正常的。配置已设 `attn_impl: sdpa`，不影响训练。

**Q: WebUI 打不开？**
确认端口未被占用（以终端输出的地址为准，通常 7860）；gradio 未装会导致启动失败，重跑 `setup_env.py`。

**Q: 想换模型？**
```powershell
python scripts\py\download_model.py Qwen/Qwen3-1.7B
```
下载后修改训练配置中的 `model:` 即可。

**Q: 训练时电脑很卡？**
正常，QLoRA 训练会吃满 GPU。若需临时用电脑，可 Ctrl+C 停训、之后用 `resume_from_checkpoint` 续训（见第 2 步）。

## 历史说明（LLaMA-Factory → ms-swift）

本项目早期使用 LLaMA-Factory，现已完全迁移到 ms-swift，旧框架产物已清理。若你在别处见到旧参数名，对照如下：`llamafactory-cli train` → `swift sft`；`model_name_or_path` → `model`；`cutoff_len` → `max_length`；`quantization_bit` → `quant_bits`；`val_size` → `split_dataset_ratio`；`llamafactory-cli export` → `swift export --merge_lora true`。
