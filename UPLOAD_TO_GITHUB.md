# GitHub 上传操作指南

> 操作目录：`D:\Study\DynRiskFramework\`

---

## 第 0 步：在 GitHub 新建空仓库

1. 打开 https://github.com/new
2. 仓库名建议：`DynRiskFramework`（或其他你想要的名字）
3. **不要**勾选"Add a README file"
4. 点击"Create repository"，复制仓库地址备用

---

## 第 1 步：初始化并提交

```powershell
cd D:\Study\DynRiskFramework

git init
git checkout -b main
git add -A
git commit -m "feat: two-stage integrated framework for financial prediction and tail risk

Stage 1 - DS-LDE: chaotic feature analysis + dual-stream Lévy-SDE path prediction
Stage 2 - ETGPD-Transformer: sparse attention + GPD tail correction for VaR/ES
- data/: raw daily K-line CSV for 11 assets
- results/: full experimental results (baseline, ablation, rolling backtest)
- figures/: thesis figures (Lévy vs Brownian, phase space, attention heatmaps)"
```

---

## 第 2 步：推送到 GitHub

```powershell
# 把 URL 换成你在第 0 步复制的地址
git remote add origin https://github.com/你的用户名/DynRiskFramework.git
git push -u origin main
```

---

## 第 3 步：验证

打开 GitHub 仓库页，确认：

- `stage1/models/ds_lde.py` — DS-LDE 主模型 ✓
- `stage2/models/etgpd_transformer.py` — ETGPD-Transformer ✓
- `data/` — 原始 CSV 数据 ✓
- `results/stage1/` `results/stage2/` — 实验结果 ✓
- `figures/` — 论文配图 ✓
- `README.md` — 完整框架说明 ✓

---

## 发给导师的内容

```
仓库地址：https://github.com/你的用户名/DynRiskFramework

克隆命令：
  git clone https://github.com/你的用户名/DynRiskFramework.git

运行说明：
  cd DynRiskFramework
  pip install -r requirements.txt
  python stage1/models/ds_lde.py          # 第一阶段：DS-LDE 预测
  python stage2/main_pipeline.py          # 第二阶段：ETGPD 风险度量

实验结果见 results/ 目录，论文配图见 figures/ 目录。
```
