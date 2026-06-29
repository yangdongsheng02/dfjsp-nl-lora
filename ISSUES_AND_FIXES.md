# 问题与解决方案记录

记录本实验在 **任务设计、训练、推理、环境与数据格式** 上遇到的问题及处理情况。实验数据见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)，项目概述见 [README.md](README.md)。

---

## 1. 实验结论相关（未解决）

### 1.1 题干时刻 t 与标签全局时间轴不一致

**现象：** Base 与 LoRA 生成结果大量出现「18时/21时/24时开始」，与标签中多数工序 `start < t` 不符；strict 工序召回约 0%。

**原因：** 题干以插队时刻 `t` 及机台状态为主线；标签为从 0 起的全局最优完整排程。自由生成时，模型倾向于复用题干中的 `t` 与可选加工时间，而非推理全局排程。

**已做修改：** 在 `INSTRUCTION` 与标签首行注明「全局时间轴从 0 起算，非当前时刻 t」。

**结果：** 未改善生成质量；单样本过拟合 loss 接近 0 时 strict recall 仍为 0%。

---

### 1.2 训练 loss 收敛但自由生成无效

**现象：** 全量 500 step 及单样本过拟合实验中，训练 loss 均可压至接近 0。

**原因：** SFT 在 teacher forcing 下可续写标签 token；从空 assistant 段自回归生成时分布不一致（exposure bias）。单条输出约 30 道工序，序列长，问题更明显。

**已做修改：** 仅对 assistant 排程段计算 loss（`label_start_index`、标签掩码）。

**结果：** 训练指标正常，生成指标仍无效；评估必须以自由生成 + strict 工序召回为准。

---

### 1.3 LoRA 后完整方案率低于 Base（98% → 36%）

**现象：** LoRA 解析成功率 100%，完整方案率 36%；Base 为 98%。

**指标定义：** 「完整」指至少解析出 1 条工序，且输出包含 `预计总完工时间：XX`（`scheduling_format.parse_schedule_completion`）。

**原因：** LoRA 生成过程中更易回写题干内容（如「急单 J7」「工序1：可选」），触发 `_nl_generation_done` 提前停止，未输出 makespan 行。50 条 LoRA 样本中 32 条不完整，均无 makespan 行，其中 27 条含题干复述特征。Base 与 LoRA 平均预测工序数均约 10 条（标签约 30 条）；差异主要在是否写完 makespan，而非条数。

**结果：** LoRA 未提升调度正确性，完整方案率反而下降。

---

### 1.4 解析成功率不能代表排程正确

**现象：** Base / LoRA 解析成功率均为 100%。

**原因：** 解析器只要求输出符合 `- Jx-Ox 机器Mx，…时开始，加工…单位` 格式，不要求与标签一致。

**处理：** 以 strict op recall、makespan 匹配为主指标；oracle 模式 50/50 用于校验评估代码（`eval_schedule.py`、`validate_pipeline.py`）。

---

### 1.5 过拟合门控未通过仍执行全量训练

**现象：** `overfit_sanity.py` 在单样本 400 step 后 strict op recall 0%、jom recall 29%，低于 50% 门槛。

**结果：** 全量 500 step 训练后指标与门控预判一致，未出现门控误判。

---

## 2. 训练

### 2.1 HF QLoRA 训练过慢

**现象：** `train_hf_qlora.py` 约 125 s/step，500 step 在 4GB 显卡上不可行。

**处理：** 全量训练改用 `train_unsloth.py`（约 14 s/step）。HF 脚本保留未删。

---

### 2.2 Windows 下 Triton compile 导致 backward 失败

**现象：** 训练 backward 阶段报错（与 triton_key / torch.compile 相关）。

**处理：** 设置环境变量 `UNSLOTH_COMPILE_DISABLE=1`、`TORCH_COMPILE_DISABLE=1`（`train_unsloth.py` 内亦有默认设置）。

---

### 2.3 4GB 显存下多进程占满 GPU

**现象：** 显存占用高、GPU 利用率长期 0%，进程无进展。

**原因：** 4GB 环境下不宜同时运行两个占用 GPU 的 Python 进程；亦可能是异常退出后的残留进程。

**处理：** 训练或评估前结束其它 python 进程；`nvidia-smi` 确认显存释放。单进程训练峰值约 1.74 GB。

---

### 2.4 LoRA 加载 base 模型命名空间

**现象：** LoRA 评估时若 base 与训练时不一致，加载或生成异常。

**处理：** 训练与 LoRA 推理使用 `unsloth/Qwen2.5-1.5B-Instruct`；Base 零样本评估使用 `Qwen/Qwen2.5-1.5B-Instruct`（见 `local_inference.py`、`eval_schedule.py`）。

---

### 2.5 合并 4bit 权重时的舍入

**现象：** `save_pretrained_merged` 提示 merge 到 4bit 可能存在舍入误差。

**处理：** 评估以 adapter + 对应 base 为准；`merged_hf` 用于导出时再核对生成结果。

---

## 3. 推理与评估

### 3.1 Windows 下 Unsloth generate 不可用

**现象：** Unsloth  patched `generate` 输出乱码或重复。

**处理：** 评估统一经 `local_inference.py` 调用 HuggingFace `model.generate`（`scheduling_format.generate_fast` / `generate_completion`）。

---

### 3.2 Base 与 LoRA 解码方式不同

**约定：** Base 使用 `decode=fast`；LoRA 使用 `decode=padded`（与训练时 padding 方式一致）。见 `eval_schedule.py`。

**说明：** LoRA 完整方案率下降的主要原因见 1.3，并非单纯 decode 差异所致。

---

### 3.3 生成提前停止（复述题干）

**机制：** `_nl_generation_done` 在检测到 `预计总完工时间`、或出现「作业 Jx」「工序1：可选」等题干特征时停止生成。

**影响：** LoRA 更易触发停止，导致缺少 makespan 行（见 1.3）。

---

### 3.4 对比评估耗时

**现象：** 50 条 × base + LoRA，逐步生成，Step 3 可运行数小时。

**处理：** 调试使用 `--max-samples` 限制条数；正式结果 `--max-samples 0`；日志写入 `eval_nl_compare.log`。

---

## 4. 环境与依赖

### 4.1 控制台 UTF-8 / 日志乱码

**现象：** PowerShell 管道重定向时出现 `UnicodeDecodeError` 或中文乱码。

**处理：** 设置 `PYTHONUTF8=1`；`run_nl_pipeline.ps1` 将 stdout/stderr 重定向到文件，而非 `Tee-Object` 管道。

---

### 4.2 HuggingFace 模型下载

**处理：** 可选 `HF_ENDPOINT=https://hf-mirror.com`；`HF_HUB_DOWNLOAD_TIMEOUT=300`（`train_unsloth.py` 默认）。

---

### 4.3 依赖安装顺序

**处理：** 按 `install.ps1`：PyTorch cu118 → requirements → xformers/accelerate（`--no-deps`）→ unsloth（`--no-deps`），避免覆盖 torch。安装后运行 `check_env.py`。

---

## 5. 流水线

### 5.1 训练前校验

`validate_pipeline.py` 检查 NL 编解码、oracle 指标、jsonl token 长度；`train_unsloth.py` 启动时自动执行。

### 5.2 建议执行顺序

```text
validate_pipeline（自动）→ overfit_sanity（可选）→ Step1 评估 → Step2 训练 → Step3 对比
```

---

## 6. 数据与 Prompt 调整记录

| 阶段 | 问题 | 修改 |
|------|------|------|
| 早期 | Alpaca 模板与 Qwen2.5-Instruct 不匹配 | 改为 `apply_chat_template`，用 `CHAT_ASSISTANT_MARKER` 定位标签 |
| 早期 | 模型照抄 system 内具体示例工序 | `OUTPUT_EXAMPLE` 改为占位符，并注明勿照抄 |
| 中期 | 开始时刻理解混乱 | 标签首行增加「全局时间轴从 0 起算」说明 |
| — | 仍抄题干 `t` | 见 1.1，未解决 |

单条样本规模：标签 23–37 道工序（中位 30），7–8 个作业，含 prompt 约 1200–1450 tokens。

---

## 7. 状态汇总

| 问题 | 状态 |
|------|------|
| 题干 t 与标签全局 0 轴冲突 | 未解决 |
| loss 收敛但生成无效 | 未解决 |
| LoRA 完整方案率低于 Base | 未解决 |
| strict 工序召回约 0% | 未解决 |
| 解析率不能代表正确性 | 已调整评估指标 |
| Unsloth Windows generate | 已改用 HF generate |
| HF QLoRA 过慢 | 已改用 Unsloth 训练 |
| Triton compile 报错 | 已禁用 compile |
| 4GB 多进程占 GPU | 需单进程运行 |
| LoRA base 命名空间 | 已统一 |
| UTF-8 / 日志 | 已按上述方式处理 |
| Chat 模板与占位符 | 已修改 |

---

## 附录：相关日志

| 文件 | 内容 |
|------|------|
| `overfit_sanity.log` | 单样本过拟合门控 |
| `train_nl.log` | 500 step 训练 |
| `eval_nl_base.log` | Base 评估 |
| `eval_nl_compare.log` | Base vs LoRA |
| `eval_nl_comparison.csv` | 汇总指标 |
| `eval_nl_samples.csv` | 逐样本输出片段 |
