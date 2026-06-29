# 实验记录报告：DFJSP 自然语言端到端 LoRA 微调

> **项目介绍**（背景、工程价值、实现说明）见 **[README.md](README.md)**。  
> 本文件为 **实验过程与结果的完整记录**。

**报告日期：** 2026-06-26  
**硬件：** NVIDIA GeForce RTX 3050 Laptop GPU（4 GB VRAM）  
**基础模型：** Qwen2.5-1.5B-Instruct（4-bit QLoRA，Unsloth 训练）  
**仓库：** https://github.com/yangdongsheng02/dfjsp-nl-lora

---

## 1. 研究目标

验证在 **消费级 4 GB 显存** 条件下，能否通过 **LoRA 微调** 使小参数指令模型完成以下端到端任务：

| 维度 | 设定 |
|------|------|
| **输入** | 自然语言描述的动态柔性作业车间调度问题（含初始作业、急单插队、当前时刻 t、机器状态等） |
| **输出** | 自然语言形式的 **完整重调度方案**（逐条列出全部工序的 J-O-M-开始时刻-加工时长，并给出 makespan） |
| **标签来源** | OR-Tools CP-SAT 近似最优解 |
| **成功标准** | LoRA 在测试集上 **严格工序召回率 / makespan** 等指标 **显著优于** 未微调 base 模型（而非仅提高解析成功率） |

本研究 **不** 采用「自然语言进、结构化 JSON 出」或「模型只出机器分配、求解器算时间」等混合方案；目标是 **纯自然语言 I/O 的全排程生成**。

---

## 2. 问题与数据

### 2.1 问题规模与实例复杂度

本研究实例为 **动态柔性作业车间（Dynamic FJSP）重调度**：初始 6 作业执行中插入 1–2 急单，需对 **全部作业的全部工序** 给出机器指派与开工时刻，最小化 makespan。

| 维度 | 规格 |
|------|------|
| 机器 | 4 台（M1–M4），互斥占用 |
| 初始作业 | 6 个，每作业 3–5 道工序；每道工序 1–4 台候选机，加工时间异机不同 |
| 急单 | 1–2 个，在初始计划约 50% 进度到达（`insertion_time = max(1, initial_makespan // 2)`） |
| 总作业 | 7–8 个（含急单） |
| **监督标签：完整排程工序数** | **23–37 条/样本**（训练集中位 **30**，测试集中位 **30**） |
| 每工序标注 | (job, operation, machine, start, duration) 五字段 + 全局 makespan |
| 求解器 | OR-Tools CP-SAT，单实例 3s 时限 |

**与题干的区别（重要）**：题干描述插队时刻 `t` 下的 **现场状态**（已完成约十余道工序、若干道进行中、十余道未开工）；标签却是 **从时刻 0 起的全局最优完整排程**（含已过去工序的正确开始时刻，其中大量 `start < t`）。模型输出需 **逐条列出约 30 道工序**，而非仅输出「当前尚未完成的十几道」。

**数据集统计**：

| 指标 | train（200） | test（50） |
|------|--------------|------------|
| 工序数 / 样本 | 23–37（中位 30） | 26–34（中位 30） |
| makespan | 34–82（中位 51） | 33–80（中位 54） |
| insertion_time t | 12–34（中位 20） | 11–30（中位 20） |
| token 长度（含 prompt） | 1180–1456 | 1226–1394 |

### 2.2 数据集

| 集合 | 样本数 | 文件 |
|------|--------|------|
| 训练集 | 200 | `train_data.jsonl` |
| 测试集 | 50 | `test_data.jsonl` |

每条样本包含：`instance_id`、`description`（自然语言题干）、`optimal_action`（含 `insertion_time`、`makespan`、`schedule` 列表）。

### 2.3 输出格式（标签）

示例（`scheduling_format.py` → `format_schedule_nl`）：

```
重调度方案（全局时间轴从0起算，非当前时刻t）：
- J2-O1 机器M1，0时开始，加工1单位
- J4-O1 机器M2，0时开始，加工1单位
…
预计总完工时间：56
```

系统提示（`INSTRUCTION`）明确要求：开始时刻为全局最优排程时刻，**不是** 题目中的 t；并给出占位符格式示意（禁止照抄）。

---

## 3. 技术方案

### 3.1 流水线（三步）

```
Step 1  base 评估  →  eval_nl_base.log
Step 2  LoRA 训练  →  train_nl.log, lora_output/
Step 3  base vs LoRA 对比  →  eval_nl_compare.log, eval_nl_comparison.csv
```

编排脚本：`run_nl_pipeline.ps1`（Step 1/2/3）。

### 3.2 训练

| 项 | 配置 |
|----|------|
| 框架 | Unsloth + TRL `SFTConfig` + `UnslothTrainer` |
| 模型 | `unsloth/Qwen2.5-1.5B-Instruct`，`load_in_4bit=True` |
| LoRA | r=8, alpha=16，target 全部 attention + MLP 投影 |
| 序列长度 | 1536 |
| batch | 1 × 梯度累积 8 |
| 步数 | 500（约 25 epoch / 160 训练条） |
| 学习率 | 1.5e-4，linear，warmup 15 |
| 损失掩码 | 仅对 assistant 排程标签段计算 loss（`label_start_index`） |
| 产物 | `lora_output/`（adapter）+ `lora_output/merged_hf`（合并 4bit，供 HF 推理） |

训练前运行 `validate_pipeline.py`：校验 NL 编解码、oracle 50/50、训练/测试 token 长度与标签完整性。

### 3.3 评估

| 项 | 配置 |
|----|------|
| 推理 | HuggingFace Transformers 4-bit（`local_inference.py`），规避 Windows 上 Unsloth `generate` 问题 |
| base | `Qwen/Qwen2.5-1.5B-Instruct`，`decode=fast` |
| LoRA | `unsloth/Qwen2.5-1.5B-Instruct` + adapter，`decode=padded` |
| 生成预算 | 最多约 784 tokens（完整排程），支持 early-stop |
| 指标 | 解析率、工序非空、完整方案率、严格 5 字段工序召回/精确、makespan 匹配与偏差；辅助 jom（仅 J-O-M）匹配 |

Oracle 模式（标签作预测）在测试集 **50/50 解析且方案完全一致**，证明 **评分与解析管线正确**。

### 3.4 过拟合门控（未通过全量训练前门槛）

`overfit_sanity.py`：在 **单条** 训练样本上训练 400 步（LoRA r=16），合并后 HF 子进程评估。

| 指标 | 结果 | 门槛 |
|------|------|------|
| strict op_recall | **0.0%** | ≥ 50% |
| jom_recall | **29.0%** | ≥ 50% |

结论：**loss 可压至接近 0，但自由生成仍无法复现标签内容**。门控判定 FAIL。后续跳过门控仍执行了 500 步全量训练，结果与门控一致。

---

## 4. 实验结果

### 4.1 测试集汇总（50 条，`eval_nl_comparison.csv`）

| 指标 | Base | LoRA | Oracle（管线校验） |
|------|------|------|-------------------|
| 自然语言解析成功率 | **100%** | **100%** | 100% |
| 工序非空率 | 98% | **100%** | — |
| 完整方案率 | **98%** | 36% | — |
| 方案完全一致率 | 0% | 0% | 50/50 |
| makespan 完全匹配率 | 0% | 0% | 50/50 |
| makespan 平均绝对偏差 | 20.7 | 23.2 | 0 |
| 严格工序召回（均值） | 0.56% | **0.20%** | 100% |
| 严格工序精确（均值） | 1.02% | 0.90% | 100% |

**结论：LoRA 未达成研究目标；在核心调度指标上未优于 base，完整方案率反而大幅下降。**

### 4.2 训练过程（`train_nl.log`）

| 项 | 数值 |
|----|------|
| 总时长 | 约 1 小时 56 分（6934 s） |
| 速度 | 约 13.9 s/step |
| step 25 loss | 1.08 |
| step 500 loss | ≈ -0.0001 |
| train_loss（汇总） | 0.055 |
| 峰值 VRAM | 约 1.74 GB allocated / 4 GB |

训练在 **teacher forcing** 下充分收敛；与 **自回归自由生成** 表现严重脱节。

### 4.3 典型失败模式（定性）

**Base 与 LoRA 共性：**

1. **抄写当前时刻 t**：将题干中的 `t=18/21/24` 当作多数工序的开始时刻，与标签中大量 `start < t` 矛盾。
2. **从题干抄机器与加工时间**：可选机器集合中的数字被填入输出，而非最优指派。
3. **makespan 荒谬**：常出现「预计总完工时间：3/4」等远小于真实 makespan（50+）的数值。
4. **方案不完整**：LoRA 更易在生成中途 **回抄题干**，完整方案率降至 36%。

**Base 偶发模式：** 按 J1,J2,… 顺序编造递增时间轴，形式完整但语义仍错。

---

## 5. 原因分析

### 5.1 任务–标签–题干语义冲突（主因）

题干叙事以 **「当前时刻 t 的现场状态」** 为中心；标签要求 **「从 0 起的全局最优完整排程」**。模型最省力策略是 **利用上下文中显式出现的 t 与可选加工时间**，而非推理全局优化。

### 5.2 输出复杂度过高

单条标签约 **30 道工序 × 5 字段**，序列约 **1200–1450 tokens**。在 200 条训练样本、1.5B 参数规模下，模型更易学习表面格式，难以内化组合优化。

### 5.3 Teacher forcing 与自由生成鸿沟

过拟合单样本时 loss → 0，但 **strict op_recall 仍为 0%**（jom 最高约 29%），属 **exposure bias** 在长结构化输出下的典型表现。

### 5.4 数据量与模型容量

| 因素 | 本实验 | 经验需求 |
|------|--------|----------|
| 训练样本 | 200 | 完整排程常需 10³–10⁴+ |
| LoRA 可训练参数 | ~9.2M | 够拟格式，难学优化 |
| 硬件 | 4 GB | 限制模型规模与 batch |

### 5.5 备选改法与原始目标关系

| 改法 | 与原始目标关系 |
|------|----------------|
| 相对时间 + 仅输出 t 之后工序 | 改变输出语义 |
| 只预测 J-O-M | 降级子任务 |
| JSON / 结构化输出 | 改变输出模态 |
| LLM + OR-Tools 混合 | 非端到端 NL 出 |

本报告记录的是 **坚持原始 NL 全方案设定下的负结果**。

---

## 6. 工程侧已验证内容

1. **数据生成**：`data_generator.py` 稳定产出 CP-SAT 标签 NL 样本。  
2. **格式与解析**：`scheduling_format.py` 指标与 Qwen 模板。  
3. **4 GB 训练**：Unsloth 500 step × seq 1536 可完成。  
4. **评估一致性**：Windows HF 推理；base / LoRA / oracle 流程跑通。  
5. **门控价值**：单样本即可预判全量训练无效。

---

## 7. 总体结论

在 **RTX 3050 4 GB + Qwen2.5-1.5B + 200 条 NL 全排程标签** 下：

1. **未能证明** LoRA 可学会自然语言端到端最优排程。  
2. 严格工序召回 ≈ 0%，LoRA **未优于** base。  
3. 主因：**题干 t 与标签全局 0 轴冲突** + 输出过长 + 样本过少 + 模型容量不足。  
4. **负结果可复现**；流水线可供后续 ablation 复用。

---

## 8. 附录

### 8.1 关键文件

| 文件 | 说明 |
|------|------|
| `scheduling_format.py` | Prompt、标签、解析、指标 |
| `data_generator.py` | 数据与 CP-SAT 标签 |
| `train_unsloth.py` | LoRA 训练 |
| `eval_schedule.py` | 评估与对比 |
| `local_inference.py` | HF 4-bit 推理 |
| `overfit_sanity.py` | 单样本过拟合门控 |
| `validate_pipeline.py` | 训练前校验 |
| `run_nl_pipeline.ps1` | 三步流水线 |

### 8.2 关键日志与结果

| 文件 | 内容 |
|------|------|
| `eval_nl_base.log` | Step 1 base 评估 |
| `train_nl.log` | Step 2 全量训练 |
| `eval_nl_compare.log` | Step 3 对比评估 |
| `eval_nl_comparison.csv` | 汇总指标 |
| `eval_nl_samples.csv` | 逐样本明细 |
| `overfit_sanity.log` | 门控失败记录 |

### 8.3 复现命令

```powershell
cd scheduling_research
$env:PYTHONUTF8="1"
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:UNSLOTH_COMPILE_DISABLE="1"
$env:TORCH_COMPILE_DISABLE="1"

.\run_nl_pipeline.ps1 -Step 1
.\.venv\Scripts\python.exe train_unsloth.py
.\run_nl_pipeline.ps1 -Step 3
```

---

*实验记录结束 · 项目介绍见 [README.md](README.md)*
