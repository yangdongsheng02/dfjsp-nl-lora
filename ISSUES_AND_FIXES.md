# 问题与解决方案记录

> **文档性质**：开发与实验过程中遇到的 **问题、现象、原因、解决办法** 汇总，便于汇报与后续避坑。  
> 实验数字与结论见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)；项目介绍见 [README.md](README.md)。

**最后更新：** 2026-06-26

---

## 目录

- [一、任务与模型层面（未完全解决）](#一任务与模型层面未完全解决)
- [二、训练与微调](#二训练与微调)
- [三、推理与评估](#三推理与评估)
- [四、Windows 与环境](#四windows-与环境)
- [五、流水线与脚本](#五流水线与脚本)
- [六、数据、Prompt 与标签](#六数据prompt-与标签)
- [七、开源与 Git](#七开源与-git)
- [八、问题状态总表](#八问题状态总表)

---

## 一、任务与模型层面（未完全解决）

### 1.1 题干「当前时刻 t」与标签「全局从 0 起算」冲突

| 项 | 内容 |
|----|------|
| **现象** | Base/LoRA 生成都大量写「18时/21时/24时开始」，与标签中 `start < t` 的工序矛盾；strict 工序召回 ≈ 0%。 |
| **原因** | 题干强调插队现场 `t`；标签要求完整全局最优排程。模型自由生成时 **抄题干里最显眼的数字 t** 最省力；system prompt 写「不要用 t」难以抵消。 |
| **尝试** | 在 `INSTRUCTION` 与标签头中强调「全局时间轴从 0 起算，非当前时刻 t」。 |
| **结果** | **未解决**；过拟合单条 loss→0 仍 strict recall 0%。 |
| **后续建议** | 统一坐标系（相对 t 或题干不突出 t）；或改为 LLM+求解器，不坚持纯 NL 全排程。 |

### 1.2 训练 loss→0，但自由生成不会排程

| 项 | 内容 |
|----|------|
| **现象** | 500 step 后 step loss ≈ 0；过拟合门控单样本同样 loss 极低。 |
| **原因** | **Teacher forcing** 下能续写标签 token；**自回归从空 assistant 生成** 时分布不同（exposure bias）。长结构化输出（~30 行）尤甚。 |
| **尝试** | 仅对 assistant 段算 loss（`label_start_index` + mask）。 |
| **结果** | **未解决**；必须用自由生成 + strict 指标验收。 |
| **后续建议** | 过拟合门控先测生成；考虑 DPO 惩罚抄 t；缩短输出或分阶段训练。 |

### 1.3 LoRA 后「完整方案率」反而低于 Base（98% → 36%）

| 项 | 内容 |
|----|------|
| **现象** | LoRA 解析率 100%，但完整方案率仅 36%；Base 为 98%。 |
| **定义** | `完整` = 至少 1 条可解析工序 **且** 输出含 `预计总完工时间：XX` 行（见 `scheduling_format.parse_schedule_completion`）。 |
| **原因** | LoRA 更易 **中途抄回题干**（`急单 J7`、`工序1：可选`），触发 `_nl_generation_done` early-stop，**来不及写 makespan 行**。32/50 条 LoRA 不完整样本 **均无 makespan 行**，其中 27 条含抄题特征。Base 常能凑完 makespan 行（数值常错如「3」），故形式上更「完整」。 |
| **尝试** | 全量 LoRA 训练 500 step。 |
| **结果** | **调度未变好，形式完整性变差**；平均预测工序数 Base/LoRA 均 ~10 条（标签 ~30），非条数少而是 **收尾失败**。 |
| **后续建议** | 不以完整方案率单独论优劣；以 strict recall / makespan 为准；微调若加重抄题干，可能损害生成行为。 |

### 1.4 解析率 100% 的误导

| 项 | 内容 |
|----|------|
| **现象** | Base/LoRA 解析成功率均 100%。 |
| **原因** | 只要输出「像排程」的中文列表即可被 `NL_OP_RE` 解析；内容与标签可完全无关。 |
| **解决办法** | 以 **strict op recall、makespan exact/gap** 为主指标；oracle 50/50 验证打分管线。 |
| **状态** | ✅ 流程已落实（`eval_schedule.py` + `validate_pipeline.py`）。 |

### 1.5 过拟合门控 FAIL 仍做全量训练

| 项 | 内容 |
|----|------|
| **现象** | `overfit_sanity.py`：单样本 400 step 后 strict 0%、jom 29%，未达 50% 门槛。 |
| **原因** | 同上，loss 与生成脱节。 |
| **决定** | 用户确认跳过门控，仍跑 500 step 全量训练；**结果与门控预判一致**。 |
| **教训** | 小数据 + 长输出，**务必先过拟合 1 条看生成**；可节省约 2 小时无效训练。 |

---

## 二、训练与微调

### 2.1 HF 原生 QLoRA 在 4GB 上过慢

| 项 | 内容 |
|----|------|
| **现象** | `train_hf_qlora.py` 约 **125 s/step**，全量 500 step 不现实。 |
| **原因** | 未用 Unsloth 优化路径；4GB batch=1 仍慢。 |
| **解决办法** | 全量训练改用 **`train_unsloth.py`**（约 14 s/step）。HF 脚本保留作备选。 |
| **状态** | ✅ 已切换 |

### 2.2 Windows + Triton：`torch.compile` / cut_cross_entropy 崩 backward

| 项 | 内容 |
|----|------|
| **现象** | 训练 backward 报错（triton_key 等）。 |
| **解决办法** | 训练前设置环境变量（`train_unsloth.py` 内也有 default）：<br>`UNSLOTH_COMPILE_DISABLE=1`<br>`TORCH_COMPILE_DISABLE=1` |
| **状态** | ✅ 已固化 |

### 2.3 4GB 显存：双 Python 进程挂死

| 项 | 内容 |
|----|------|
| **现象** | GPU 利用率 0%、显存仍占满，训练/评估无进展。 |
| **原因** | 4GB 上同时跑两个占 GPU 的 Python（如训练 + 评估、或僵尸进程）。 |
| **解决办法** | 开训/评估前结束其它 `python`；任务管理器或 `nvidia-smi` 确认显存释放。训练单进程峰值 ~1.74 GB 属正常。 |
| **状态** | ✅ 操作规范 |

### 2.4 LoRA 评估 base 命名空间不一致

| 项 | 内容 |
|----|------|
| **现象** | LoRA 加载或生成异常（与 Unsloth 训练权重不匹配）。 |
| **原因** | 训练用 `unsloth/Qwen2.5-1.5B-Instruct`；若评估 base 用 `Qwen/Qwen2.5-1.5B-Instruct` 挂 adapter 可能不一致。 |
| **解决办法** | LoRA 评估：**base = `unsloth/...` + adapter**；Base 零样本：**`Qwen/...`**（见 `local_inference.py` / `eval_schedule.py`）。 |
| **状态** | ✅ 已对齐 |

### 2.5 `save_pretrained_merged` 4bit 舍入

| 项 | 内容 |
|----|------|
| **现象** | PEFT 警告 merge 到 4bit 可能有舍入误差，生成与 adapter 略有差异。 |
| **解决办法** | 评估优先 **adapter + 同 base**；`merged_hf` 便于部署时再验一轮。 |
| **状态** | ⚠️ 已知限制 |

### 2.6 训练占用显存「看起来很高」但 GPU 利用率为 0

| 项 | 内容 |
|----|------|
| **现象** | 显存 ~3.8GB 占用，GPU 利用率长时间 0%。 |
| **原因** | 多为 **卡死/僵尸进程**，非正常训练。 |
| **解决办法** | 杀进程重跑；正常训练时利用率应周期性升高（~100%）。 |
| **状态** | ✅ 已识别 |

---

## 三、推理与评估

### 3.1 Unsloth `generate` 在 Windows 上不可用

| 项 | 内容 |
|----|------|
| **现象** | Unsloth  patched generate 输出乱码/重复。 |
| **原因** | Windows + CUDA 11.8 与 Unsloth generate 路径不兼容。 |
| **解决办法** | 评估统一走 **`local_inference.py` → HF `model.generate`** / `scheduling_format.generate_fast`；不在 Windows 上用 Unsloth generate。 |
| **状态** | ✅ 已切换 |

### 3.2 Base 与 LoRA 使用不同 decode 模式

| 项 | 内容 |
|----|------|
| **现象** | 对比时解码策略不一致。 |
| **约定** | Base：`decode=fast`（`model.generate` + stopping criteria）<br>LoRA：`decode=padded`（逐 token，对齐训练 padding） |
| **说明** | 有意为之；LoRA 完整率下降主因是 **抄题干 early-stop**，非单纯 decode 差异。 |
| **状态** | ✅ 文档化于 `eval_schedule.py` |

### 3.3 生成 early-stop：抄题干即停

| 项 | 内容 |
|----|------|
| **现象** | 生成在「急单 J7」「工序1：可选」等处截断。 |
| **原因** | `_nl_generation_done()` 检测到 **复述问题描述** 则停止，避免无限胡写。 |
| **副作用** | LoRA 更易触发表述 → **完整方案率下降**（见 1.3）。 |
| **状态** | ✅ 设计如此；是否放宽需权衡指标 |

### 3.4 评估极慢（50×2 模式）

| 项 | 内容 |
|----|------|
| **现象** | Step 3 对比评估可跑 **数小时**（50 样本 × base + LoRA，逐条生成 ~30 行预算）。 |
| **解决办法** | 调试时用 `--max-samples 5`；正式结果用 `--max-samples 0`。日志重定向到 `eval_nl_compare.log`。 |
| **状态** | ✅ 可接受 |

### 3.5 Oracle 校验

| 项 | 内容 |
|----|------|
| **目的** | 排除「解析/打分代码写错」导致假阴性。 |
| **结果** | Oracle 50/50 完全一致。 |
| **状态** | ✅ `validate_pipeline.py` + `eval_schedule.py --mode oracle` |

---

## 四、Windows 与环境

### 4.1 PowerShell `Tee-Object` UTF-8 解码错误

| 项 | 内容 |
|----|------|
| **现象** | `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb2`；日志乱码。 |
| **原因** | 管道编码与 Unsloth 输出二进制/GBK 混杂。 |
| **解决办法** | `$env:PYTHONUTF8="1"`；`train_unsloth.py` 设 `PYTHONIOENCODING=utf-8`；**重定向到文件** 优于 Tee（`run_nl_pipeline.ps1` 用 `Start-Process -RedirectStandardOutput`）。 |
| **状态** | ✅ 可绕过；训练不受影响 |

### 4.2 国内 HuggingFace 下载慢/超时

| 项 | 内容 |
|----|------|
| **解决办法** | `$env:HF_ENDPOINT="https://hf-mirror.com"`；`HF_HUB_DOWNLOAD_TIMEOUT=300`（`train_unsloth.py` 默认）。 |
| **状态** | ✅ 已用 |

### 4.3 `install.ps1` 依赖顺序

| 项 | 内容 |
|----|------|
| **要点** | PyTorch cu118 → requirements → xformers/accelerate **--no-deps** → unsloth **--no-deps**，避免 torch 被覆盖。 |
| **验证** | `python check_env.py` |
| **状态** | ✅ 见 `install.ps1` |

---

## 五、流水线与脚本

### 5.1 `run_nl_pipeline.ps1` 日志不刷新

| 项 | 内容 |
|----|------|
| **现象** | 管道读端到进程结束才显示日志。 |
| **解决办法** | `Start-Process` + `-RedirectStandardOutput` / `-RedirectStandardError` 到 `.log` 文件。 |
| **状态** | ✅ 已改 |

### 5.2 训练前校验门

| 项 | 内容 |
|----|------|
| **作用** | `validate_pipeline.py`：NL 编解码、oracle、jsonl token 长度、标签完整。 |
| **状态** | ✅ `train_unsloth.py` 开头自动调用 |

### 5.3 推荐执行顺序

```text
validate（自动）→ overfit_sanity（建议）→ Step1 base eval → Step2 train → Step3 compare
```

---

## 六、数据、Prompt 与标签

### 6.1 Alpaca 模板 → Qwen chat 模板

| 项 | 内容 |
|----|------|
| **现象** | 早期格式与 Qwen2.5-Instruct 不对齐，生成/训练不一致。 |
| **解决办法** | `scheduling_format.py` 使用 `tokenizer.apply_chat_template`；`CHAT_ASSISTANT_MARKER` 定位标签起点。 |
| **状态** | ✅ 已改 |

### 6.2 系统示例照抄（占位符 J1-O1）

| 项 | 内容 |
|----|------|
| **现象** | 模型照抄 system 里具体示例工序。 |
| **解决办法** | `OUTPUT_EXAMPLE` 改为 **占位符** `{工件}-{工序}`，并注明勿照抄。 |
| **状态** | ✅ 已改；仍难阻止抄题干 t。 |

### 6.3 标签头标明全局时间轴

| 项 | 内容 |
|----|------|
| **改动** | 标签首行：`重调度方案（全局时间轴从0起算，非当前时刻t）：` |
| **结果** | loss 可降，**生成仍抄 t**（见 1.1）。 |
| **状态** | ⚠️ 部分缓解，未根治 |

### 6.4 单条样本规模（避免误解为「小玩具题」）

| 项 | 内容 |
|----|------|
| **事实** | 每样本标签 **23–37 道工序**（中位 30）；7–8 个作业；序列 **1200–1450 tokens**。 |
| **文档** | 已写入 README / EXPERIMENT_REPORT。 |

---

## 七、开源与 Git

### 7.1 `gh` 未登录

| 项 | 内容 |
|----|------|
| **解决办法** | `gh auth login`（设备码 https://github.com/login/device） |
| **状态** | ✅ 已登录 yangdongsheng02 |

### 7.2 `git push` 报错 `remote-https is not a git command`

| 项 | 内容 |
|----|------|
| **原因** | 本机 Git 安装不完整，缺少 remote-https helper。 |
| **解决办法** | `winget install Git.Git` 升级；`gh auth setup-git`；推送前确保 PATH 中优先 `Git\cmd\git.exe` 而非 `git-core\git.exe`。 |
| **状态** | ✅ 已推送至 https://github.com/yangdongsheng02/dfjsp-nl-lora |

### 7.3 大文件不入库

| 项 | 内容 |
|----|------|
| **规则** | `.gitignore` 排除 `lora_output/*.safetensors`、`*.bin`、`.venv`、`*.log` 等。 |
| **说明** | 克隆后需本地 `train_unsloth.py` 生成权重。 |

---

## 八、问题状态总表

| # | 问题 | 状态 | 文档/代码位置 |
|---|------|------|----------------|
| 1 | 题干 t vs 标签全局 0 轴 | ❌ 未解决 | `scheduling_format.INSTRUCTION` |
| 2 | loss→0 但生成不会排程 | ❌ 未解决 | `overfit_sanity.py` |
| 3 | LoRA 完整方案率 < Base | ❌ 未解决（行为退化） | `eval_nl_samples.csv` |
| 4 | strict 召回 ≈ 0% | ❌ 未解决 | `eval_nl_comparison.csv` |
| 5 | 解析率误导 | ✅ 指标纠偏 | `eval_schedule.py` |
| 6 | Unsloth Windows generate | ✅ 绕过 | `local_inference.py` |
| 7 | HF QLoRA 过慢 | ✅ 换 Unsloth | `train_unsloth.py` |
| 8 | Triton compile 崩 | ✅ 禁 compile | 环境变量 |
| 9 | 4GB 双进程挂死 | ✅ 操作规范 | — |
| 10 | LoRA base 命名空间 | ✅ 对齐 | `local_inference.py` |
| 11 | UTF-8 / 日志乱码 | ✅ 绕过 | `run_nl_pipeline.ps1` |
| 12 | Chat 模板 / 占位符 | ✅ 已改 | `scheduling_format.py` |
| 13 | 过拟合门控 | ⚠️ 可用但曾跳过 | `overfit_sanity.py` |
| 14 | Git push / gh 登录 | ✅ 已解决 | `GITHUB_PUBLISH.md` |

**图例：** ✅ 已解决/已规避　❌ 实验层面未达成　⚠️ 已知限制或建议未严格执行

---

## 附录：相关日志与结果文件

| 文件 | 用途 |
|------|------|
| `overfit_sanity.log` | 单样本过拟合门控 FAIL 记录 |
| `train_nl.log` | 500 step 全量训练 |
| `eval_nl_base.log` | Base 评估 |
| `eval_nl_compare.log` | Base vs LoRA 对比 |
| `eval_nl_comparison.csv` | 汇总指标 |
| `eval_nl_samples.csv` | 逐条 raw 预览（含截断/抄题样例） |

---

*若新问题在复现中出现，建议在本文件末尾按同一表格格式追加条目。*
