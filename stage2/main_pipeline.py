"""
ETGPD-Transformer — 完整端到端流水线
将动力系统特征、模型训练、回测检验、SHAP 归因串联为一个入口

运行：
    python main_pipeline.py

依赖：
    pip install torch numpy pandas scipy shap matplotlib nolds antropy
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch

from models.etgpd_transformer  import ETGPDTransformer
from training.trainer          import (ETGPDTrainer, ReturnDataset,
                                       BacktestSuite, rolling_backtest)
from shap_analysis.crisis_shap import (CrisisShapAnalyzer,
                                       ALL_FEATURE_NAMES, FEATURE_NAMES_DYN)
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════
# 0. 模拟数据生成（替换为真实数据即可）
# ══════════════════════════════════════════════════════════════════

def generate_mock_data(n: int = 2000, d: int = 33,
                       seed: int = 42) -> tuple:
    """
    生成模拟的收益率序列和特征矩阵
    加入：GARCH 波动率聚集 + 危机跳跃 + 动力系统特征的模拟变化

    返回
    ----
    features : [N, 33]  已标准化
    returns  : [N]      对数收益率
    dates    : DatetimeIndex
    """
    np.random.seed(seed)

    # GARCH(1,1) 波动率
    vol = np.ones(n)
    for t in range(1, n):
        eps = np.random.randn()
        vol[t] = np.sqrt(max(1e-6,
            0.00001 + 0.08 * (eps * vol[t-1])**2 + 0.90 * vol[t-1]**2))

    returns = np.random.randn(n) * vol

    # 注入危机跳跃（模拟 2008、2020 类事件）
    crisis_idx = [int(n * 0.35), int(n * 0.72)]
    for ci in crisis_idx:
        returns[ci:ci+40] *= 3.5
        returns[ci:ci+40] -= np.abs(returns[ci:ci+40]) * 0.5

    # 生成 33 维特征（前21维随机，后12维动力特征模拟）
    features = np.zeros((n, d))

    # A+B+C 类（21维）
    features[:, :21] = np.column_stack([
        returns,                                    # r_t
        np.roll(returns, 1),                        # r_t-1
        np.roll(returns, 2),                        # r_t-2
        np.roll(returns, 3),                        # r_t-3
        np.roll(returns, 4),                        # r_t-4
        np.roll(returns, 5),                        # r_t-5
        pd.Series(returns).rolling(5).std().fillna(0).values,  # RV_5d
        pd.Series(returns).rolling(22).std().fillna(0).values, # RV_22d
        np.random.randn(n) * 0.002,                 # delta_rf
        np.abs(np.random.randn(n)) * 0.5,           # CS_IG
        np.abs(np.random.randn(n)) * 1.2,           # CS_HY
        np.random.randn(n) * 0.01,                  # FX_ret
        np.random.randn(n) * 0.02,                  # Comdty_ret
        np.clip(pd.Series(returns).rolling(60).corr(
            pd.Series(np.random.randn(n))).fillna(0).values, -1, 1),
        np.clip(pd.Series(returns**2).rolling(22).mean().fillna(0).values * 0.1, 0, 5),
        np.abs(np.random.randn(n)) * 15 + 15,       # IV_t (VIX-like)
        np.random.randn(n) * 0.5,                   # term_slope
        np.random.randn(n) * 5 + 100,               # SKEW_idx
        np.clip(np.random.beta(2, 20, n), 0, 1),    # n_u_ratio
        np.abs(np.random.randn(n)) * 15 + 15,       # IV_lag1
        np.abs(np.random.randn(n)) * 3,             # IV_roll_std
    ])

    # D 类：动力系统特征（12维，危机期异常化）
    dyn = np.random.randn(n, 12) * 0.3
    dyn[:, 0]  = 2.5 + np.random.randn(n) * 0.3     # D2_corr_dim
    dyn[:, 3]  = 0.5 + np.random.randn(n) * 0.05    # H_rs
    dyn[:, 4]  = 0.5 + np.random.randn(n) * 0.05    # H_dfa
    dyn[:, 6]  = 1.5 + np.random.randn(n) * 0.2     # SampEn
    dyn[:, 7]  = 0.85+ np.random.randn(n) * 0.05    # PermEn
    dyn[:, 10] = 0.05+ np.abs(np.random.randn(n)) * 0.01  # RR

    # 危机期特征异变
    for ci in crisis_idx:
        s, e = ci, ci + 40
        dyn[s:e, 0] += 1.5     # 关联维数骤升
        dyn[s:e, 3] -= 0.2     # Hurst 下降（去趋势）
        dyn[s:e, 4] -= 0.18    # DFA Hurst 下降
        dyn[s:e, 5] += 0.3     # 多重分形谱宽增大
        dyn[s:e, 6] -= 0.8     # 样本熵下降（市场同质化）
        dyn[s:e, 7] -= 0.06    # 排列熵下降
        dyn[s:e, 9] += 0.15    # Lyapunov 上升（混沌增大）
        dyn[s:e, 10]-= 0.025   # 递归率下降（体制切换）

    features[:, 21:] = dyn

    # 滚动标准化
    for j in range(d):
        s = pd.Series(features[:, j])
        mu = s.rolling(250, min_periods=30).mean().fillna(0)
        sd = s.rolling(250, min_periods=30).std().fillna(1).replace(0, 1)
        features[:, j] = ((s - mu) / sd).clip(-4, 4).values

    dates = pd.date_range("2016-01-04", periods=n, freq="B")
    return features, returns, dates


# ══════════════════════════════════════════════════════════════════
# 1. 模型配置
# ══════════════════════════════════════════════════════════════════

MODEL_CONFIG = dict(
    input_dim     = 33,     # 21原有 + 12动力系统
    tail_feat_dim = 6,      # C类尾部信号
    d_model       = 64,
    n_heads       = 4,
    n_layers      = 4,
    d_ff          = 256,
    top_k         = 10,
    seq_len       = 60,
    dropout       = 0.1,
    confidence    = 0.99,
)

TRAIN_CONFIG = dict(
    train_size  = 800,
    val_size    = 100,
    test_step   = 50,
    seq_len     = 60,
    epochs      = 60,
    batch_size  = 32,
    conf        = 0.99,
)


# ══════════════════════════════════════════════════════════════════
# 2. 主流水线
# ══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║        ETGPD-Transformer  端到端流水线               ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 0: 数据 ─────────────────────────────────────────────
    print("► Step 0: 生成/加载数据")
    features, returns, dates = generate_mock_data(n=1500)
    print(f"  特征矩阵: {features.shape}  收益率: {returns.shape}")
    print(f"  日期范围: {dates[0].date()} → {dates[-1].date()}\n")

    # ── Step 1: 滚动回测 ─────────────────────────────────────────
    print("► Step 1: 滚动回测（扩展窗口）")
    model = ETGPDTransformer(**MODEL_CONFIG)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    all_ret, all_var, all_es = rolling_backtest(
        model, features, returns, verbose=True, **TRAIN_CONFIG
    )
    print(f"\n  回测完成：预测样本数 = {len(all_ret)}")

    # ── Step 2: 统计检验 ─────────────────────────────────────────
    print("\n► Step 2: 统计一致性回测检验")
    suite    = BacktestSuite()
    report   = suite.full_report(all_ret, all_var, all_es, conf=0.99,
                                  model_name="ETGPD-Transformer")

    # 与基准模型对比（模拟数据）
    violations = -all_ret > all_var
    print(f"  实际违例率: {violations.mean():.3%}  "
          f"（理论值 1.00%，样本数 {len(violations)}）")

    # ── Step 3: SHAP 危机归因 ────────────────────────────────────
    print("\n► Step 3: SHAP 危机期动力特征归因")

    # 重新训练一个固定模型用于 SHAP 分析
    model_shap = ETGPDTransformer(**MODEL_CONFIG)
    trainer    = ETGPDTrainer(model_shap, lr=1e-3)

    tr_size = TRAIN_CONFIG["train_size"]
    ds_tr = ReturnDataset(features[:tr_size], returns[:tr_size],
                          seq_len=MODEL_CONFIG["seq_len"])
    ds_vl = ReturnDataset(features[tr_size:tr_size+100],
                          returns[tr_size:tr_size+100],
                          seq_len=MODEL_CONFIG["seq_len"])
    dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, drop_last=True)
    dl_vl = DataLoader(ds_vl, batch_size=32, shuffle=False)

    print("  训练 SHAP 分析专用模型...")
    trainer.fit(dl_tr, dl_vl, epochs=40, patience=8, verbose=True)

    # SHAP 分析
    analyzer = CrisisShapAnalyzer(
        model_shap,
        seq_len      = MODEL_CONFIG["seq_len"],
        feature_names = ALL_FEATURE_NAMES,
    )

    # 快速近似（调试用）；换用 compute_shap_values() 获得精确值
    test_feats = features[tr_size: tr_size + len(all_ret)]
    test_dates = dates[tr_size: tr_size + len(all_ret)]
    analyzer.wrapper.set_history(features[:MODEL_CONFIG["seq_len"]])

    print("  计算近似 SHAP 值（快速模式）...")
    n_shap = min(300, len(test_feats))
    shap_vals = analyzer.compute_shap_fast(
        test_feats[:n_shap], test_dates[:n_shap]
    )

    # 识别危机窗口
    crisis_mask = analyzer.identify_crisis_windows(
        violations[:n_shap], test_dates[:n_shap]
    )
    n_crisis = crisis_mask.sum()
    
    # ── Step 4: 可视化输出 ───────────────────────────────────────
    print("► Step 4: 生成可视化报告")

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 对比分析
    compare_df = analyzer.compare_crisis_normal(crisis_mask)
    analyzer.plot_dynamical_importance(
        compare_df,
        save_path=os.path.join(output_dir, "shap_crisis_analysis.png"),
    )

    # 注意力热图 (取测试集第一个序列作为展示)
    sample_x = features[tr_size: tr_size + MODEL_CONFIG["seq_len"]]
    analyzer.attention_heatmap(
        model_shap, sample_x,
        save_path=os.path.join(output_dir, "attention_heatmap.png"),
    )

    # ── Step 5: 保存结果 ─────────────────────────────────────────
    print("\n► Step 5: 保存结果")

    compare_df.to_csv(os.path.join(output_dir, "shap_compare.csv"),
                      index=False, encoding="utf-8-sig")
    report.to_csv(os.path.join(output_dir, "backtest_report.csv"),
                  index=False, encoding="utf-8-sig")

    result_df = pd.DataFrame({
        "date":    test_dates[:len(all_ret)],
        "return":  all_ret,
        "VaR_99":  all_var,
        "ES_99":   all_es,
        "violation": (-all_ret > all_var).astype(int),
    })
    result_df.to_csv(os.path.join(output_dir, "rolling_predictions.csv"),
                     index=False, encoding="utf-8-sig")

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  全部完成！输出文件：                                 ║")
    print(f"║  • {os.path.join(output_dir, 'shap_crisis_analysis.png')}  — SHAP 三图             ║")
    print(f"║  • {os.path.join(output_dir, 'attention_heatmap.png')}     — 注意力热图             ║")
    print(f"║  • {os.path.join(output_dir, 'shap_compare.csv')}          — 特征归因明细           ║")
    print(f"║  • {os.path.join(output_dir, 'backtest_report.csv')}       — 回测检验报告           ║")
    print(f"║  • {os.path.join(output_dir, 'rolling_predictions.csv')}   — 滚动预测结果           ║")
    print("╚══════════════════════════════════════════════════════╝")

    return compare_df, report, result_df


if __name__ == "__main__":
    compare_df, report, result_df = main()
