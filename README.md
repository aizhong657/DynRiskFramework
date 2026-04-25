# 动力学条件路径预测——条件尾部风险度量两阶段集成框架

> 融合非线性动力学 · 神经随机微分方程 · Transformer · 极值理论  
> 解决金融时序多步预测与极端尾部风险度量难题

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 研究背景与核心问题

传统金融风险模型存在三个根本缺陷：

1. **分布假设失效**：依赖正态分布，无法捕捉金融序列尖峰厚尾、波动聚集、非连续跳跃特征；
2. **路径与风险割裂**：预测模型与风险度量相互独立，动力学状态信息未有效转化为尾部风险估计；
3. **极端分位数不稳定**：深度学习在极端分位数估计缺乏与极值理论的深度整合。

本研究提出**两阶段集成框架**，打通路径预测与尾部风险度量全链路：

```
市场数据
   │
   ▼
[第一阶段] DS-LDE 双流融合神经随机微分方程
   │  混沌特征分析 → 相空间重构 → Lévy-SDE 路径预测
   │  输出：多步条件均值 + 波动率 + 动力学特征向量
   │
   ▼
[第二阶段] ETGPD-Transformer 条件尾部风险度量
   │  稀疏注意力编码 → Pinball 分位数回归 → GPD 尾部校正
   │
   ▼
条件 VaR / ES（99% 置信水平）+ 四项监管回测检验
```

---

## 仓库结构

```
DynRiskFramework/
├── config.py                    # 统一路径与超参数配置 ← 首次运行必改
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量示例
│
├── data/                        # 原始股票日频 K 线 CSV
│   ├── sz50_index_data.csv      # 上证 50 指数（主数据集）
│   ├── hs300_index_data.csv     # 沪深 300 指数
│   ├── corr_dim_scaled.csv      # 预计算关联维数特征
│   └── ...（共 19 个资产）
│
├── stage1/                      # 第一阶段：DS-LDE 路径预测
│   ├── chaotic_analysis/        # 混沌特征分析（第 3 章）
│   │   ├── log_return_distribution.py   # 3.1 对数收益率分布
│   │   ├── ami_embedding_delay.py       # 3.2 AMI 选取时延 τ
│   │   ├── cao_embedding_dim.py         # 3.3 Cao 方法嵌入维数 m
│   │   ├── grassberger_procaccia_d2.py  # 3.4 G-P 关联维数 D₂
│   │   ├── lyapunov_bootstrap.py        # 3.5 Lyapunov 指数 Bootstrap CI
│   │   └── rosenstein_lambda_max.py     # 3.6 Rosenstein λ_max
│   │
│   ├── models/                  # 预测模型（演进脉络完整保留）
│   │   ├── ds_lde.py            # ★ DS-LDE 主模型（双流融合 Lévy-SDE）
│   │   ├── ldenet_dual_stream_v2.py     # DS-LDE 前身（双流 v2）
│   │   ├── ldenet_dual_stream.py        # 双流 v1
│   │   ├── ldenet_v1.py                 # LDENet 初版
│   │   ├── sdenet_v1.py                 # SDENet（高斯驱动，对比用）
│   │   ├── ablation_no_psr.py           # 消融：移除相空间重构
│   │   ├── ablation_no_sde.py           # 消融：移除 SDE（换 MLP）
│   │   └── ablation_no_attention.py     # 消融：移除 D₂ 门控注意力
│   │
│   ├── baselines/               # 基线对比模型
│   │   ├── arima.py             # ARIMA 多步预测
│   │   ├── lstm_psr.py          # LSTM + 相空间重构
│   │   ├── lstm_simple.py       # 普通双层 LSTM
│   │   ├── naive_forecast.py    # 朴素基线
│   │   └── five_models_compare.py  # 五模型横向对比实验
│   │
│   └── experiments/             # 实验脚本
│       ├── sensitivity_analysis.py  # 参数敏感性分析
│       ├── residual_analysis.py     # 多数据集残差分析
│       ├── multi_asset_validation.py # 多资产鲁棒性验证
│       ├── robustness_seeds.py      # 多随机种子稳健性
│       └── runner.py            # 一键批量运行所有实验
│
├── stage2/                      # 第二阶段：ETGPD-Transformer 风险度量
│   ├── main_pipeline.py         # ★ 端到端流水线入口
│   ├── models/
│   │   └── etgpd_transformer.py # 稀疏注意力 + 时变 GPD 尾部校正主模型
│   ├── training/
│   │   └── trainer.py           # 训练器 + Kupiec/Christoffersen/DQ/ESR 四项检验
│   ├── features/
│   │   └── dynamical_features.py # 12 维动力学特征（Hurst、PermEn、ξ 等）
│   └── shap_analysis/
│       └── crisis_shap.py       # SHAP 危机截面特征归因分析
│
├── results/                     # 实验结果
│   ├── stage1/                  # DS-LDE 预测实验
│   │   ├── results_baseline_sz50.csv    # 基准模型对比（t+1~t+4）
│   │   ├── results_ablation_sz50.csv    # 消融实验结果
│   │   ├── results_multiasset.csv       # 多资产验证
│   │   ├── results_robustness_sz50.csv  # 多种子鲁棒性
│   │   └── ssz50_*_results.xlsx         # 逐日预测明细（含混淆矩阵/VaR）
│   └── stage2/                  # ETGPD 风险度量实验
│       ├── backtest_report.csv          # 四项监管回测统计汇总
│       ├── rolling_predictions.csv      # 滚动回测逐日 VaR/ES 预测值
│       └── shap_compare.csv            # 危机 vs 正常期特征重要性对比
│
└── figures/                     # 论文配图
    ├── stage1/                  # Lévy vs Brownian、相空间重构示意等
    └── stage2/                  # 注意力热图、SHAP 归因图等
```

---

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/你的用户名/DynRiskFramework.git
cd DynRiskFramework
python -m venv .venv
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. 配置路径

```bash
copy .env.example .env          # Windows
```

编辑 `.env`（默认 `./data/` 无需修改，直接运行即可）：

```
DATA_DIR=./data
OUTPUT_DIR=./outputs
```

### 3. 运行第一阶段：混沌特征分析

```bash
python stage1/chaotic_analysis/log_return_distribution.py
python stage1/chaotic_analysis/cao_embedding_dim.py
python stage1/chaotic_analysis/grassberger_procaccia_d2.py
```

### 4. 运行第一阶段：DS-LDE 主模型

```bash
python stage1/models/ds_lde.py
```

### 5. 运行第一阶段：完整对比实验

```bash
python stage1/baselines/five_models_compare.py    # 五模型横向对比
python stage1/experiments/runner.py               # 一键跑全部实验
```

### 6. 运行第二阶段：ETGPD-Transformer 端到端流水线

```bash
python stage2/main_pipeline.py
```

---

## 核心模型

### 第一阶段：DS-LDE（`stage1/models/ds_lde.py`）

双流融合神经随机微分方程：

| 模块 | 说明 |
|---|---|
| 相空间重构（PSR）| Takens 嵌入，还原隐动力学流形 |
| Lévy-SDE 分支 | α=1.2 稳定分布驱动，捕捉跳跃厚尾 |
| 动力学特征分支 | 关联维数 D₂ 门控调节扩散强度 |
| 双流自注意力 | 动态整合两路，输出多步条件均值+波动率 |
| 联合损失 | MSE + NLL + 方向惩罚 + 交叉熵 |

### 第二阶段：ETGPD-Transformer（`stage2/models/etgpd_transformer.py`）

极值理论增强的条件风险度量模型：

| 模块 | 说明 |
|---|---|
| 输入层 | DS-LDE 输出 + 12 维动力学特征 + 价格/宏观特征（共 33 维） |
| 稀疏注意力 | Top-k 过滤，O(T·k) 复杂度，提取长程依赖 |
| 训练阶段 | Pinball Loss 分位数回归，获取基准 VaR |
| 推断阶段 | GPD 尾部校正，输出 99% 条件 VaR + ES |

---

## 主要实验结论

**第一阶段（DS-LDE）**

- 上证 50 指数 t+1~t+4 多步预测，RMSE / MAE 均优于 LSTM、高斯 SDE、普通 LDE；
- 消融实验验证：PSR、Lévy 过程、D₂ 门控三个模块各自独立贡献显著；
- 多资产（沪深 300、中石油、招商银行）鲁棒性验证通过。

**第二阶段（ETGPD-Transformer）**

- 99% 置信水平违例率 **0.95%**（理论目标 1%），优于 LSTM-Pinball（1.77%）、GARCH-EVT（1.52%）；
- 全部通过 **Kupiec、Christoffersen、DQ、ESR** 四项监管统计检验；
- 厚尾 + 同质化极端市场情景（四象限Ⅱ区）下，风险度量效果显著优于传统模型；
- 动力学指标（Hurst 指数、排列熵）可提前识别市场体制切换，是危机预警先行信号。

---

## 数据说明

`data/` 目录包含全部原始日频 K 线 CSV，来源 [baostock](http://baostock.com/)，字段：`date, open, high, low, close, volume`。

| 文件 | 资产 |
|---|---|
| `sz50_index_data.csv` | 上证 50 指数（主实验数据集）|
| `hs300_index_data.csv` | 沪深 300 指数 |
| `cnpc_data.csv` | 中石油（601857）|
| `cmb_data.csv` | 招商银行（600036）|
| `maotai_data.csv` | 贵州茅台（600519）|
| `yili_data.csv` | 伊利股份（600887）|
| `pingan_data.csv` | 中国平安（601318）|
| `gree_data.csv` | 格力电器（000651）|
| `ningde_data.csv` | 宁德时代 |
| `dongfang_data.csv` | 东方财富 |
| `BTC_data.csv` | 比特币（跨市场鲁棒性）|

---

## 引用

```bibtex
@phdthesis{author2026dynrisk,
  title  = {动力学条件路径预测与条件尾部风险度量两阶段集成框架},
  author = {<作者>},
  school = {<学校>},
  year   = {2026},
}
```

## License

MIT License，见 [LICENSE](./LICENSE)。
