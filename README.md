# DFJSP-NL-LoRA

> **自然语言进、自然语言出的动态柔性作业车间调度（DFJSP）端到端实验**  
> 在 RTX 3050 4GB 上，用 Qwen2.5-1.5B + LoRA 验证小模型能否学会「读题干 → 写完整最优排程」。  
> **结论：本设定下未成功（negative result）**；仓库保留完整流水线、数据与可复现脚本。

[实验记录报告](EXPERIMENT_REPORT.md) · [结果 CSV](eval_nl_comparison.csv)

---

## 目录

- [背景与工程价值](#背景与工程价值)
- [实验目的](#实验目的)
- [问题场景与数据规模](#问题场景与数据规模)
- [主要结果](#主要结果)
- [实现说明](#实现说明)
- [快速开始](#快速开始)
- [常见警示](#常见警示)
- [展望](#展望)

---

## 背景与工程价值

### 排程在产线里解决什么问题

离散制造里，**作业车间 / 柔性车间调度（JSP / FJSP）** 要在机器互斥、工序先后顺序约束下，决定每道工序上哪台机、何时开工，以压低 **makespan（总完工时间）**、拖期和换型成本。产线一旦跑起来，还会遇到 **急单插队、设备异常、插单改优先级**——需要 **动态重调度**，而不是只做一次静态排程。

传统做法依赖 APS/MES、启发式或 MIP/CP-SAT（如 OR-Tools）。效果好，但 **计划员学习成本高**：要懂甘特图、约束编码，改一次现场情况往往要回到系统里点选半天。

### 大模型带来的工程侧机会

2023 年以来，业界在探索 **大语言模型 + 排程/运筹**，对生产的实际价值主要在：

| 方向 | 工程价值 |
|------|----------|
| **自然语言交互** | 计划员用口语描述「两台急单、M2 还要 1 小时」→ 系统出调整方案，降低 APS 使用门槛 |
| **意图理解** | 把非结构化通知（邮件、群消息）转成可计算的扰动事件 |
| **边缘小模型** | 车间侧 4–8 GB 工控机/笔记本即可微调推理，**数据不出厂**、少依赖云端大 API |
| **与求解器协同** | LLM 解析现场 + 传统优化算最优（主流落地路径）；本仓库则试探 **更激进的纯 NL 端到端** |

### 技术路线演进（简述）

| 路线 | 特点 | 量产成熟度 |
|------|------|------------|
| CP-SAT / MIP | 解质量好，规模受限 | 高 |
| 元启发式 | 快、可定制 | 高 |
| RL / GNN 策略 | 研究多，落地少 | 低 |
| **LLM 端到端生成排程** | 交互最自然，**可行性与最优性难保证** | 探索期 |

**本仓库定位**：不是 APS 产品，而是一次 **可复现的工程试验**——在 **4 GB + 1.5B** 条件下，纯 NL 全排程是否值得继续投；负结果同样能指导 **该走「LLM+求解器」还是继续砸端到端**。

---

## 实验目的

### 要验证什么

> 消费级 4 GB 显存上，用约 200 条 **完整自然语言排程标签** LoRA 微调 Qwen2.5-1.5B 后，**自由生成** 的排程是否在 **严格工序匹配 / makespan** 上 **明显优于** 未微调 base？

### 刻意坚持的设定

- **输入输出均为中文自然语言**（完整约 30 道工序列表 + makespan），不用 JSON 中间格式、不在推理时调 OR-Tools「补算」。
- 成功标准看 **调度硬指标**，不看「解析率 100%」或「loss 很低」。

### 未达成（详见实验报告）

LoRA **未优于** base；训练 loss 可→0，但生成仍大量 **抄写题干时刻 t**。完整数字见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

---

## 问题场景与数据规模

动态 FJSP + 急单插队（与 `data_generator.py` 一致）：

| 维度 | 规格 |
|------|------|
| 机器 | **4** 台（M1–M4） |
| 初始作业 | **6** 个，每作业 **3–5** 道工序，柔性（每道 1–4 台候选机，工时不同） |
| 急单 | **1–2** 个，约在初始计划 **50%** 进度到达 |
| 总作业 | **7–8** |
| **标签工序数** | **23–37 条/样本**，中位 **30**（**全局完整排程**，非仅未完成段） |
| 标签来源 | OR-Tools CP-SAT，3s/实例 |
| 数据 | 训练 200 / 测试 50；单条约 **1200–1450 tokens** |

题干含 **当前时刻 t、机台状态、完成/进行中统计**；标签为 **从 0 起的全局最优排程**（许多 `start < t`）——这是本实验最重要的设计张力之一。

---

## 主要结果

测试集 50 条（[eval_nl_comparison.csv](eval_nl_comparison.csv)）：

| 指标 | Base | LoRA |
|------|------|------|
| 解析成功率 | 100% | 100% |
| 完整方案率 | 98% | 36% |
| 严格工序召回（均值） | 0.56% | 0.20% |
| makespan 完全匹配 | 0% | 0% |

训练 500 step ≈ 1h56m，VRAM 峰值 ~1.74 GB。典型失败：题干 `t=18` → 输出「18时开始」；makespan 常写成 3、4。

---

## 实现说明

### 流水线总览

```
data_generator.py  →  train/test.jsonl
       ↓
validate_pipeline.py（训练前校验）
       ↓
train_unsloth.py   →  lora_output/ + merged_hf
       ↓
eval_schedule.py   →  eval_nl_comparison.csv
```

编排入口：`run_nl_pipeline.ps1 -Step 1|2|3`；门控可选 `overfit_sanity.py` / `run_gate_then_train.ps1`。

### 核心模块

| 模块 | 作用 |
|------|------|
| `data_generator.py` | 随机 FJSP 实例 → 初始排程 → 50% 处插急单 → CP-SAT 重调度 → `description` + `optimal_action` JSONL |
| `scheduling_format.py` | Qwen chat 模板、`format_schedule_nl` 标签、`parse_schedule_completion` 解析、strict/jom/makespan 指标、SFT 标签掩码 |
| `train_unsloth.py` | Unsloth 4-bit QLoRA：seq 1536，LoRA r=8，500 step，仅 assistant 段 loss；保存 adapter + `merged_hf` |
| `train_hf_qlora.py` | HF 原生 QLoRA 备选（4GB 上过慢，未用于全量） |
| `local_inference.py` | Windows 下 HF `model.generate` 4-bit 推理；base 用 `Qwen/`，LoRA 用 `unsloth/` 命名空间 |
| `eval_schedule.py` | `--mode base|lora|compare|oracle`；compare 输出 CSV |
| `overfit_sanity.py` | 单样本过拟合 + 子进程 HF 评估，预判全量训练是否有效 |
| `validate_pipeline.py` | 编解码 round-trip、oracle 50/50、token 长度检查 |

### 训练与推理要点（工程向）

| 项 | 取值 / 说明 |
|----|-------------|
| 模型 | `unsloth/Qwen2.5-1.5B-Instruct` 4-bit |
| LoRA | r=8, α=16，attention + MLP 全投影 |
| batch | 1 × grad_accum 8 |
| Windows | `UNSLOTH_COMPILE_DISABLE=1`；评估勿用 Unsloth `generate` |
| 4GB | 勿同时跑两个占 GPU 的 Python；峰值 ~1.74 GB 正常 |
| 权重 | `lora_output/` 本地训练生成，仓库未上传（~1.5GB） |

### 评估指标（生产验收应采用的思路）

- **解析成功**：NL 能否被解析成工序列表（必要非充分）
- **strict op recall**：(J,O,M,start,duration) 五元组与标签一致比例
- **makespan exact / gap**：完工时间是否可信
- **oracle 50/50**：证明打分管线无误

---

## 快速开始

```powershell
git clone https://github.com/yangdongsheng02/dfjsp-nl-lora.git
cd dfjsp-nl-lora
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\install.ps1

$env:HF_ENDPOINT = "https://hf-mirror.com"   # 可选
$env:PYTHONUTF8 = "1"
$env:UNSLOTH_COMPILE_DISABLE = "1"

.\run_nl_pipeline.ps1 -Step 1    # base 评估
.\run_nl_pipeline.ps1 -Step 2    # LoRA 训练（约 2h）
.\run_nl_pipeline.ps1 -Step 3    # 对比
```

---

## 常见警示

1. **题干 t vs 标签全局 0 轴**：模型易抄 t；设计 NL 任务时输入输出坐标系要一致。  
2. **loss→0 ≠ 会排产**：必须自由生成 + strict 指标；teacher forcing 会骗人。  
3. **解析率 100% 幻觉**：格式对、内容可全错。  
4. **先跑 overfit 门控**：小数据长输出，单条不过就别训满 500 step。

更多失败模式与复现命令 → [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

---

## 展望

**短期量产**：更现实的是 **LLM 理解现场 + CP-SAT/MIP 出解** + LLM 解释对比；纯 NL 全排程难当 APS 内核。

**若继续端到端**：需更大模型/数据、约束解码、或分阶段（先 J-O-M 再时间）；边缘 4GB 更适合 **短输出、Tool-call**，而非 30 行自由最优排程。

**本仓库价值**：负结果 + 可跑通流水线，帮团队 **少踩「loss 很低」的坑**，明确小模型边缘部署的能力边界。

---

## 许可与引用

MIT · [yangdongsheng02](https://github.com/yangdongsheng02)

```bibtex
@misc{dfjsp_nl_lora_2026,
  title        = {DFJSP-NL-LoRA: Natural Language End-to-End Scheduling with LoRA on 4GB GPU},
  author       = {yangdongsheng02},
  year         = {2026},
  howpublished = {\url{https://github.com/yangdongsheng02/dfjsp-nl-lora}}
}
```

---

## English

**DFJSP-NL-LoRA**: Can a **1.5B LLM + LoRA** on **4GB VRAM** learn **NL-in / full-NL-schedule-out** for dynamic FJSP? **No** under this setup (op recall ≈ 0%). Full experiment log: [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).
