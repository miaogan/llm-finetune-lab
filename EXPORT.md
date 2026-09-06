# 模型导出与部署详解（EXPORT.md）

> 本文记录从训练产物（LoRA adapter）到可用模型的完整链路：**原理 + 实操 + 排坑**。
> 快速命令请看 `GUIDE.md` 第 4 节；本文是详解版，含 2026-09-06 v2 模型的完整导出实录。
> 适用环境：ms-swift 4.5.2 / Qwen3-4B / RTX 4070 8GB / LM Studio & Ollama

---

## 1. 三种使用微调模型的方式

| 方式 | 命令 | 输入 | 适合场景 |
|---|---|---|---|
| 本地对话测试 | `python scripts/py/chat.py` | checkpoint 目录 | 快速验证效果，最省事 |
| LM Studio（GGUF） | 见第 4 节 | GGUF 文件 | 图形界面日常使用，推荐 |
| Ollama | `configs/export_merged.yaml` 开 `to_ollama` | Modelfile 目录 | 命令行/服务化使用 |

三者的输入不同：chat.py 直接吃 checkpoint（LoRA adapter）；LM Studio 只认 **GGUF** 格式；Ollama 吃 Modelfile（内部自动转 GGUF）。

## 2. 原理：从 LoRA 到 GGUF 发生了什么

### 2.1 LoRA 合并（merge_lora）

LoRA 训练时冻结基座权重 `W`（4B 参数），只训练旁路的低秩矩阵 `B`(d×r) 和 `A`(r×d)，r=8 远小于 d=2560。推理时等效于：

```
W' = W + B × A        ← 训练学到的"补丁"
```

合并（`swift export --merge_lora true`）就是把这次加法**永久固化**进权重：

1. 读 checkpoint 里的 `args.json`，自动还原基座路径、LoRA 配置（rank/alpha/target_modules）
2. 加载基座（bf16）+ adapter，执行 peft 的 `merge_and_unload()`
3. 把 `W + BA` 的结果存成标准 HF safetensors —— 之后不再需要基座和 adapter，就是一个普通完整模型

产物是标准 HF 格式，可给 transformers / vLLM 直接加载；LM Studio 需要下一步转换。

### 2.2 GGUF 转换（llama.cpp）

GGUF 是 llama.cpp 系（LM Studio / Ollama 底层）的模型容器格式。转换脚本 `convert_hf_to_gguf.py` 做三件事：

1. 逐层把 safetensors 权重改写成 ggml 二进制布局
2. **量化**（可选）：把 bf16 权重按 64 元素一组压成定点数。Q8_0 = 每组 8bit + 一个 fp16 缩放因子，4B 模型 7.7GB → 4.3GB，精度损失可忽略
3. 把 **tokenizer + Qwen3 聊天模板内嵌**进 GGUF —— 所以 LM Studio 加载后无需任何配置即可正常对话

### 2.3 量化档位怎么选（4B 模型 / 8GB 显卡）

| 档位 | 体积 | 质量 | 说明 |
|---|---|---|---|
| bf16/f16 | ~7.7GB | 无损 | 8GB 显卡装不下（还有 KV cache），不推荐 |
| **Q8_0** | ~4.3GB | 几乎无损 | **推荐**，8GB 可全 GPU 加载，convert 脚本一步输出 |
| Q4_K_M | ~2.5GB | 轻微损失 | 需额外下载 llama.cpp release 用 `llama-quantize` 二次量化；显存余量大 |

## 3. 实操实录（2026-09-06，v2 知乎模型）

> 训练背景：v2 = `coig_cqia_zhihu`（约 2091 条），3 轮 783 步，train_loss 2.71，
> 最佳 eval_loss 2.735 在 step 400；最终模型 = checkpoint-783。

### 第 1 步：合并 LoRA

```powershell
python D:\work_space\llm-finetune-lab\scripts\py\export.py
```

脚本自动探测最新 checkpoint（`v2-20260906-151227/checkpoint-783`）并执行合并，耗时约 70 秒。

**产出两个目录，用途不同（重要）：**

| 产物 | 格式 | 大小 | 用途 |
|---|---|---|---|
| `outputs/.../v2-.../checkpoint-783-merged/` | **bf16 标准 HF** | 7.7GB | ★ GGUF 转换 / vLLM / 高精度推理的**唯一正确输入** |
| `models/Qwen3-4B-finetuned/` | bnb 4bit 量化 | 2.5GB | 仅 transformers 低显存推理；**不能**转 GGUF |

> 为什么会有第二个：checkpoint 的 `args.json` 记录了训练时的 4bit 量化设置，
> swift export 读回后又把合并模型量化了一遍存到默认 output_dir。
> 想要它也是 bf16，可用 `--adapters` 指定 checkpoint 后再显式传 `--output_dir` 指到别处。

### 第 2 步：转 GGUF（Q8_0）

```powershell
cd D:\work_space\llm-finetune-lab\llama.cpp   # llama.cpp 仓库已克隆在项目内
..\.venv\Scripts\python.exe convert_hf_to_gguf.py `
  "..\outputs\qwen3_4b_qlora_sft\v2-20260906-151227\checkpoint-783-merged" `
  --outfile "..\models\gguf\Qwen3-4B-finetuned-Q8_0.gguf" `
  --outtype q8_0
cd ..
```

前置依赖（已装好，重装环境后需重做）：`.venv\Scripts\pip install gguf` + `git clone --depth 1 https://github.com/ggml-org/llama.cpp.git`

实测：398 个张量，约 53 秒（80MB/s 写盘），产物 4.27GB。

### 第 3 步：部署到 LM Studio

LM Studio 模型目录是 `发布者/模型名/文件.gguf` 三层结构：

```powershell
# 方式 A：默认 C 盘目录
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.lmstudio\models\miaogan\qwen3-4b-finetuned" | Out-Null
Copy-Item "models\gguf\Qwen3-4B-finetuned-Q8_0.gguf" "$env:USERPROFILE\.lmstudio\models\miaogan\qwen3-4b-finetuned\"

# 方式 B：自定义目录（本机实际使用 E 盘）
robocopy "models\gguf" "E:\LM_MODELS\lmstudio\miaogan\qwen3-4b-finetuned" "Qwen3-4B-finetuned-Q8_0.gguf" /MT:8
```

E 盘目录需在 LM Studio 设置（Ctrl+,）→ My Models 中添加为模型目录。

### 第 4 步：加载对话

打开 LM Studio → 顶部搜索框输入 `qwen3-4b-finetuned` → 点击加载 → 对话。
聊天模板已内嵌，无需手写 system prompt。

## 4. 备选方案：导出"最佳"checkpoint-400

v2 训练中 eval_loss 最低点在 step 400（2.735 vs 783 步的 2.739），后 383 步存在轻微过拟合。若最终版回答偏机械/重复，可换 400 步版本对比：

```powershell
# 合并（显式指定 checkpoint-400，输出到独立目录避免覆盖）
.venv\Scripts\swift.exe export `
  --adapters outputs\qwen3_4b_qlora_sft\v2-20260906-151227\checkpoint-400 `
  --merge_lora true `
  --output_dir outputs\qwen3_4b_qlora_sft\v2-20260906-151227\checkpoint-400-merged

# 转 GGUF（把上节的输入路径换成 checkpoint-400-merged，文件名区分开）
cd llama.cpp
..\.venv\Scripts\python.exe convert_hf_to_gguf.py `
  "..\outputs\qwen3_4b_qlora_sft\v2-20260906-151227\checkpoint-400-merged" `
  --outfile "..\models\gguf\Qwen3-4B-finetuned-ck400-Q8_0.gguf" `
  --outtype q8_0
cd ..
```

两个版本在 LM Studio 里共存，各聊几轮同样的问题对比即可。

> 注意：merge 产物目录名 = checkpoint 目录名 + `-merged`，与本文件第 3 节的默认行为一致。

## 5. 踩坑记录（2026-09-06 实际遇到）

| 坑 | 现象 | 解决 |
|---|---|---|
| 版本目录被误判为 adapter | export 报 `... v2-... is not an adapter` | `swift_paths.py` 的 `_is_adapter()` 已修复：须同时含 `args.json` **和** `adapter_model.safetensors`（版本目录顶层也有 args.json，但权重在 checkpoint-N/ 里） |
| bnb 4bit 产物误用于 GGUF | convert 脚本报错/输出异常 | GGUF 转换**只用** `checkpoint-N-merged/`（bf16）；`models/Qwen3-4B-finetuned/` 是 bnb 4bit，仅 transformers 可读 |
| 直接写 LM Studio 目录被拦 | 工具沙箱限制 C 盘用户目录写入 | 先输出到项目内 `models/gguf/`，再手动/robocopy 拷贝 |
| dataset_path 带 data/ 前缀 | 训练报 `FileNotFoundError: ... data\data\...` | `dataset_info.json` 中相对路径以 `data/` 为基准，**不写** `data/` 前缀（详见 GUIDE.md 第 5 节） |
| 旧版转换脚本是单文件 | 下载的 convert_hf_to_gguf.py 只有 13KB 且无法运行 | 新版依赖仓库内 `conversion/` 模块，**必须克隆整个 llama.cpp 仓库**运行（已在项目内） |

## 6. 快速速查（下次训练后照抄）

```powershell
# ① 合并（自动取最新 checkpoint；产物在 checkpoint-N-merged/）
python scripts\py\export.py

# ② 转 GGUF（把 vN-时间戳/checkpoint-N 换成新的）
cd llama.cpp
..\.venv\Scripts\python.exe convert_hf_to_gguf.py "..\outputs\qwen3_4b_qlora_sft\<vN-时间戳>\checkpoint-N-merged" --outfile "..\models\gguf\Qwen3-4B-finetuned-Q8_0.gguf" --outtype q8_0
cd ..

# ③ 拷贝到 LM Studio（二选一）
Copy-Item "models\gguf\Qwen3-4B-finetuned-Q8_0.gguf" "$env:USERPROFILE\.lmstudio\models\miaogan\qwen3-4b-finetuned\"
robocopy "models\gguf" "E:\LM_MODELS\lmstudio\miaogan\qwen3-4b-finetuned" "Qwen3-4B-finetuned-Q8_0.gguf" /MT:8

# ④ LM Studio 搜索 qwen3-4b-finetuned → 加载 → 对话
```

> 磁盘管理提示：每轮导出会新增 merged 7.7GB + GGUF 4.3GB。确认部署成功后，
> 可删 `checkpoint-N-merged/` 省空间（GGUF 已含全部信息，重转只需几分钟）；
> checkpoint 里的 `optimizer.pt`（每个数百 MB）同样可删，不影响推理与合并。
