# DynRiskFramework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

金融时序预测与尾部风险度量框架，融合神经随机微分方程与极值理论。包含两个核心模型：

- **DS-LDE**：双流融合 Lévy-SDE，用于多步价格路径预测
- **ETGPD-Transformer**：稀疏注意力 + 广义帕累托分布，用于条件 VaR / ES 度量

---

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA（可选，CPU 也可运行）

---

## 安装

```bash
git clone https://github.com/你的用户名/DynRiskFramework.git
cd DynRiskFramework
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

---

## 配置路径

```bash
copy .env.example .env    # Windows
```

`.env` 默认内容如下，`data/` 目录已包含全部数据，**通常无需修改**：

```
DATA_DIR=./data
OUTPUT_DIR=./outputs
```

运行以下命令可验证数据路径是否正确：

```bash
python config.py
```

---

## 目录结构

```
DynRiskFramework/
├── config.py                        # 路径与超参数统一配置
├── requirements.txt
├── .env.example
│
├── data/                            # 原始日频 K 线 CSV（11 个资产）
│
├── stage1/
│   ├── chaotic_analysis/            # 混沌特征分析工具
│   │   ├── log_return_distribution.py
│   │   ├── ami_embedding_delay.py
│   │   ├── cao_embedding_dim.py
│   │   ├── grassberger_procaccia_d2.py
│   │   ├── lyapunov_bootstrap.py
│   │   └── rosenstein_lambda_max.py
│   ├── models/                      # DS-LDE 主模型及消融变体
│   │   ├── ds_lde.py                # ★ 主模型
│   │   ├── ablation_no_psr.py
│   │   ├── ablation_no_sde.py
│   │   ├── ablation_no_attention.py
│   │   ├── sdenet_v1.py
│   │   ├── ldenet_v1.py
│   │   ├── ldenet_dual_stream.py
│   │   └── ldenet_dual_stream_v2.py
│   ├── baselines/                   # 对比基线模型
│   │   ├── arima.py
│   │   ├── lstm_simple.py
│   │   ├── lstm_psr.py
│   │   ├── naive_forecast.py
│   │   └── five_models_compare.py
│   └── experiments/                 # 实验脚本
│       ├── sensitivity_analysis.py
│       ├── residual_analysis.py
│       ├── multi_asset_validation.py
│       ├── robustness_seeds.py
│       └── runner.py                # 一键批量运行
│
├── stage2/
│   ├── main_pipeline.py             # ★ 端到端入口
│   ├── models/
│   │   └── etgpd_transformer.py     # 稀疏注意力 + 时变 GPD
│   ├── training/
│   │   └── trainer.py               # 训练 + 四项回测检验
│   ├── features/
│   │   └── dynamical_features.py    # 12 维动力学特征
│   └── shap_analysis/
│       └── crisis_shap.py           # SHAP 危机归因分析
│
├── results/
│   ├── stage1/                      # DS-LDE 实验结果 CSV / XLSX
│   └── stage2/                      # 回测报告与预测明细 CSV
│
└── figures/
    ├── stage1/                      # 路径预测相关配图
    └── stage2/                      # 注意力热图、SHAP 图等
```

---

## 运行 DS-LDE 路径预测

**混沌特征分析**（嵌入维数、关联维数、Lyapunov 指数）：

```bash
python stage1/chaotic_analysis/cao_embedding_dim.py
python stage1/chaotic_analysis/grassberger_procaccia_d2.py
python stage1/chaotic_analysis/lyapunov_bootstrap.py
```

**训练主模型**：

```bash
python stage1/models/ds_lde.py
```

**运行对比实验**（DS-LDE vs LSTM / ARIMA / SDE）：

```bash
python stage1/baselines/five_models_compare.py
```

**一键运行全部实验**（基线、消融、多资产、稳健性）：

```bash
python stage1/experiments/runner.py
```

输出保存到 `outputs/` 目录；汇总结果已在 `results/stage1/` 中提供。

---

## 运行 ETGPD-Transformer 风险度量

**端到端流水线**（特征提取 → 训练 → 滚动回测 → 四项检验）：

```bash
python stage2/main_pipeline.py
```

**SHAP 危机截面归因分析**：

```bash
python stage2/shap_analysis/crisis_shap.py
```

输出保存到 `outputs/`；回测报告已在 `results/stage2/` 中提供。

---

## 数据说明

`data/` 目录包含全部原始日频 K 线，来源 [baostock](http://baostock.com/)，字段：`date, open, high, low, close, volume`。

| 文件 | 资产 |
|---|---|
| `sz50_index_data.csv` | 上证 50 指数 |
| `hs300_index_data.csv` | 沪深 300 指数 |
| `cnpc_data.csv` | 中石油（601857）|
| `cmb_data.csv` | 招商银行（600036）|
| `maotai_data.csv` | 贵州茅台（600519）|
| `yili_data.csv` | 伊利股份（600887）|
| `pingan_data.csv` | 中国平安（601318）|
| `gree_data.csv` | 格力电器（000651）|
| `ningde_data.csv` | 宁德时代 |
| `dongfang_data.csv` | 东方财富 |
| `BTC_data.csv` | 比特币 |
| `corr_dim_scaled.csv` | 预计算关联维数特征 |

---

## 结果文件说明

| 文件 | 内容 |
|---|---|
| `results/stage1/results_baseline_sz50.csv` | DS-LDE 与基线模型 t+1~t+4 误差对比 |
| `results/stage1/results_ablation_sz50.csv` | 消融实验（移除 PSR / SDE / Attention）|
| `results/stage1/results_multiasset.csv` | 多资产鲁棒性验证结果 |
| `results/stage1/ssz50_LDE_results.xlsx` | 逐日预测明细，含混淆矩阵与区间覆盖率 |
| `results/stage2/backtest_report.csv` | Kupiec / Christoffersen / DQ / ESR 四项检验汇总 |
| `results/stage2/rolling_predictions.csv` | 滚动回测逐日 VaR / ES 预测值 |
| `results/stage2/shap_compare.csv` | 危机期 vs 正常期特征重要性对比 |

---

## License

MIT License，见 [LICENSE](./LICENSE)。
