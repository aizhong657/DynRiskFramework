
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

import pandas as pd
import torch

from models.etgpd_transformer  import ETGPDTransformer
from training.trainer          import (ETGPDTrainer, ReturnDataset,
                                       BacktestSuite, rolling_backtest)
from shap_analysis.crisis_shap import (CrisisShapAnalyzer,
                                       ALL_FEATURE_NAMES, FEATURE_NAMES_DYN)
from torch.utils.data import DataLoader


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

def main(asset: str = "sz50",
         bridge_path: str = "sde_bridge_features.npz",
         macro_csv: str | None = None):
    print("╔══════════════════════════════════════════════════════╗")
    print("║        ETGPD-Transformer  端到端流水线               ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 0: 数据 ─────────────────────────────────────────────
    print("► Step 0: 加载真实数据")
    # 真实数据：真实 OHLCV + 12 维真实动力学特征 + DS-LDE 桥接先验
    from features.real_data_loader import load_real_data
    features, returns, dates = load_real_data(
        asset=asset, bridge_path=bridge_path, macro_csv=macro_csv,
    )
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

    # 违例率统计
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
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--asset", default="sz50")
    _p.add_argument("--bridge_path", default="sde_bridge_features.npz")
    _p.add_argument("--macro_csv", default=None)
    _a = _p.parse_args()
    compare_df, report, result_df = main(
        asset=_a.asset, bridge_path=_a.bridge_path,
        macro_csv=_a.macro_csv,
    )

