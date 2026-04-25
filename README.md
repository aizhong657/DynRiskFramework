# DynRiskFramework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**DynRiskFramework** 是一个金融时序预测与尾部风险度量的两阶段集成框架，融合了 **神经随机微分方程 (Neural SDE)** 与 **极值理论 (EVT)**。该框架专门针对金融市场的非线性、混沌特征以及极端事件（厚尾风险）设计。

---

## 🌟 核心特性

### 第一阶段：DS-LDE (Dual-Stream Lévy-SDE)
- **双流融合架构**：随机流（Lévy 驱动的 SDE）捕捉价格随机演化，特征流（动力学辅助特征）提取确定性混沌规律。
- **厚尾建模**：采用 **Lévy 稳定分布 ($\alpha=1.2$)** 代替传统布朗运动，更精准地刻画金融危机的极端价格跳跃。
- **注意力融合**：利用 Multi-head Attention 动态加权双流特征，增强模型在不同市场状态下的适应性。

### 第二阶段：ETGPD-Transformer
- **稀疏注意力机制**：采用 Sparse Attention 过滤噪声，聚焦关键历史交易时点。
- **尾部校正 (EVT)**：将 Transformer 的分位数回归与 **广义帕累托分布 (GPD)** 结合，通过后验校正显著提升 VaR 和 ES 的度量精度。
- **可解释性分析**：集成 SHAP 归因分析，揭示金融危机期间各动力学特征对尾部风险的贡献度。

---

## 📂 目录结构

```text
DynRiskFramework/
├── stage1/                      # 第一阶段：DS-LDE 路径预测
│   ├── chaotic_analysis/        # 混沌特征分析（Lyapunov, Cao, GP 等）
│   ├── models/                  # DS-LDE 主模型及消融变体
│   ├── baselines/               # 对比基线（ARIMA, LSTM, SDE 等）
│   └── experiments/             # 批量实验与稳健性验证脚本
├── stage2/                      # 第二阶段：ETGPD-Transformer 风险度量
│   ├── models/                  # ETGPD-Transformer 主模型
│   ├── features/                # 12 维动力学特征提取
│   ├── training/                # 滚动回测与四项统计检验 (Kupiec, DQ 等)
│   └── shap_analysis/           # SHAP 危机归因分析
├── data/                        # 包含 11 种资产（A股、BTC）的原始数据
├── results/                     # 预生成的实验结果汇总 (CSV/XLSX)
└── figures/                     # 论文配图（路径预测、注意力热图、SHAP 等）
```

---

## 🚀 快速开始

### 1. 环境准备
```bash
git clone https://github.com/aizhong657/DynRiskFramework.git
cd DynRiskFramework
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置文件
复制 `.env.example` 并重命名为 `.env`，通常默认配置即可运行。

### 3. 运行第一阶段 (DS-LDE)
```bash
# 训练并运行主模型
python stage1/models/ds_lde.py
# 运行全部对比实验
python stage1/experiments/runner.py
```

### 4. 运行第二阶段 (ETGPD-Transformer)
```bash
# 运行端到端流水线（特征提取 -> 训练 -> 回测 -> 检验）
python stage2/main_pipeline.py
```

---

## 📊 数据说明
项目内置了 11 种资产的日频 K 线数据，涵盖：
- **股指**：上证 50 指数、沪深 300 指数
- **个股**：贵州茅台、招商银行、中国平安、宁德时代等
- **加密货币**：BTC (Bitcoin)

---

## 📜 许可证
本项目采用 [MIT License](LICENSE) 许可。

## 📧 联系方式
如有任何疑问，请联系：[aizhong657@example.com](mailto:aizhong657@example.com)
