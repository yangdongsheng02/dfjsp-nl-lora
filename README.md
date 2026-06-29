# DFJSP-NL-LoRA

> 探索 **大语言模型** 在 **动态柔性作业车间排程** 中的端到端能力：自然语言进、自然语言出，完整约 30 道工序的全局排程。  
> **结论：本设定下未成功（negative result）**；开源完整流水线供复现与讨论。

**作者：** [yangdongsheng02](https://github.com/yangdongsheng02)

---

## 文档（两份，分工明确）

| 文档 | 适合谁读 | 内容 |
|------|----------|------|
| **[PROJECT_INTRO.md](PROJECT_INTRO.md)** | 想了解 **行业背景、LLM×排程发展、本项目探索动机、问题讨论与未来展望** | 项目介绍（主线：大模型在排程领域的探索） |
| **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)** | 想复现或审稿 **实验细节** | 实验记录：数据规模、训练/评估配置、结果表、失败分析、复现命令 |

结果数据：[eval_nl_comparison.csv](eval_nl_comparison.csv) · [eval_nl_samples.csv](eval_nl_samples.csv)

---

## 本项目做了什么（摘要）

- **场景**：4 机、6 初始作业 + 1–2 急单，动态重调度；标签为 OR-Tools CP-SAT 最优解转中文 NL（**23–37 道工序/样本，中位 30**）。
- **模型**：Qwen2.5-1.5B-Instruct + 4-bit LoRA（Unsloth），RTX 3050 **4 GB**。
- **结果**：训练 loss 收敛，但 **严格工序召回 ≈ 0%**；LoRA 未优于 base。详见实验报告。

---

## 快速开始

```powershell
git clone https://github.com/yangdongsheng02/dfjsp-nl-lora.git
cd dfjsp-nl-lora
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\install.ps1

# 可选：国内镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONUTF8 = "1"

.\run_nl_pipeline.ps1 -Step 1   # base 评估
.\run_nl_pipeline.ps1 -Step 2   # LoRA 训练（约 2h）
.\run_nl_pipeline.ps1 -Step 3   # 对比评估
```

---

## 仓库结构

```
├── PROJECT_INTRO.md          # 项目介绍（行业 + LLM 排程 + 讨论 + 展望）
├── EXPERIMENT_REPORT.md      # 实验记录报告
├── data_generator.py         # OR-Tools 数据生成
├── scheduling_format.py      # NL 格式 / 解析 / 指标
├── train_unsloth.py          # LoRA 训练
├── eval_schedule.py          # 评估
├── run_nl_pipeline.ps1       # 三步流水线
├── train_data.jsonl          # 200 训练样本
└── test_data.jsonl           # 50 测试样本
```

模型权重 `lora_output/` 需本地训练生成（未上传仓库）。

---

## 许可

[MIT](LICENSE)

```bibtex
@misc{dfjsp_nl_lora_2026,
  title        = {DFJSP-NL-LoRA: Exploring LLM End-to-End Scheduling on 4GB GPU},
  author       = {yangdongsheng02},
  year         = {2026},
  howpublished = {\url{https://github.com/yangdongsheng02/dfjsp-nl-lora}}
}
```

---

## English

Open-source **negative result** on whether a **1.5B LLM + LoRA** on **4GB VRAM** can learn **NL-in / full-NL-schedule-out** for dynamic FJSP. See [PROJECT_INTRO.md](PROJECT_INTRO.md) for context and [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for experiment logs.
