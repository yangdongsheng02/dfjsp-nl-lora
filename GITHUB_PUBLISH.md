# 发布到 GitHub

本地仓库已 `git init` 并完成首次提交。按下列步骤 **登录一次** 后即可推送。

## 1. 登录 GitHub

在 PowerShell 中执行（按提示选择 GitHub.com → HTTPS → 浏览器授权）：

```powershell
gh auth login
```

## 2. 一键创建公开仓库并推送

在 `scheduling_research` 目录下：

```powershell
.\publish_to_github.ps1
```

默认仓库名：`dfjsp-nl-lora`。自定义名称：

```powershell
.\publish_to_github.ps1 -RepoName my-dfjsp-nl-lora
```

## 3. 手动方式（可选）

```powershell
cd D:\work\scheduling_research
gh repo create dfjsp-nl-lora --public --source=. --remote=origin `
  --description "DFJSP natural-language LoRA on 4GB GPU (negative result, reproducible pipeline)" `
  --push
```

## 4. 推送后请做

1. 打开 GitHub 仓库 **Settings → General**，确认仓库为 Public。  
2. 将 `README.md` 里 BibTeX 的 `YOUR_USERNAME` 改成你的 GitHub 用户名，再提交一次。  
3. （可选）添加 Topics：`job-shop-scheduling`, `llm`, `lora`, `unsloth`, `negative-result`, `operations-research`

## 说明

- **模型权重**（`.safetensors` / `.bin`）已在 `.gitignore` 中排除；克隆后需本地运行 `train_unsloth.py` 训练。  
- 若远程已有同名仓库，改用：`git remote add origin https://github.com/USER/REPO.git` 后 `git push -u origin master`（或 `main`）。
