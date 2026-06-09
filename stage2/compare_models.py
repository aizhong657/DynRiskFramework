
"""
第四章 滚动回测对比驱动
======================================================================
复现知识库第 9 节表格：HS / EVT / LSTM / TCN / 正态参数法 / 本文模型
在同一测试区间上的 4 项监管检验（Kupiec / Christoffersen / DQ / ESR）对比。

运行
----
    # 真实数据（需先跑 Stage 1 生成 sde_bridge_features.npz）
    python stage2/compare_models.py --asset sz50 \
        --bridge_path sde_bridge_features.npz

    # 仅跑统计型基准 + 本文模型的快速版（深度基准 epoch 调小）
    python stage2/compare_models.py --fast
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines import (rolling_backtest_baseline,
                       ALL_BASELINES, BASELINE_LABELS)
from training.trainer import BacktestSuite, rolling_backtest
from models.etgpd_transformer import ETGPDTransformer


def _summarize(suite: BacktestSuite, name: str,
               ret, var, es, conf=0.99) -> dict:
    losses     = -np.asarray(ret)
    violations = losses > np.asarray(var)
    kup = suite.kupiec_pof(violations, conf)
    cc  = suite.christoffersen(violations)
    dq  = suite.dq_test(violations, np.asarray(var))
    esr = suite.esr_test(losses, np.asarray(es), np.asarray(var), conf)
    n_pass = sum([bool(kup.get("pass")), bool(cc.get("pass")),
                  bool(dq.get("pass")), bool(esr.get("pass"))])
    return {
        "模型":     name,
        "违例率":   f"{kup.get('viol_rate', float('nan')):.2%}"
                      if kup.get('viol_rate') is not None else "—",
        "Kupiec":   kup.get("pvalue"),
        "CC独立性": cc.get("pvalue"),
        "DQ":       dq.get("pvalue"),
        "ESR":      esr.get("esr"),
        "通过数":   f"{n_pass}/4",
    }


def run_comparison(features, returns, conf=0.99,
                   deep_epochs=40, proposed_epochs=60,
                   verbose=True) -> pd.DataFrame:
    suite = BacktestSuite()
    rows = []

    # ── 5 个基准 ────────────────────────────────────────────────
    for m in ALL_BASELINES:
        if verbose:
            print(f"\n{'='*60}\n  基准: {BASELINE_LABELS[m]}\n{'='*60}")
        ep = deep_epochs if m in ("lstm", "tcn") else 0
        ret_b, var_b, es_b = rolling_backtest_baseline(
            m, returns, conf=conf, epochs=ep, verbose=verbose
        )
        rows.append(_summarize(suite, BASELINE_LABELS[m],
                               ret_b, var_b, es_b, conf))

    # ── 本文模型 ────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*60}\n  本文模型: ETGPD-Transformer\n{'='*60}")
    model = ETGPDTransformer(
        input_dim=33, d_model=64, n_heads=4, n_layers=4,
        d_ff=256, top_k=10, seq_len=60, dropout=0.1, confidence=conf,
    )
    ret_p, var_p, es_p = rolling_backtest(
        model, features, returns, conf=conf,
        epochs=proposed_epochs, verbose=verbose,
    )
    rows.append(_summarize(suite, "本文模型(ETGPD-Transformer)",
                           ret_p, var_p, es_p, conf))

    df = pd.DataFrame(rows)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="sz50")
    p.add_argument("--bridge_path", default="sde_bridge_features.npz")
    p.add_argument("--macro_csv", default=None)
    p.add_argument("--conf", type=float, default=0.99)
    p.add_argument("--fast", action="store_true",
                   help="快速模式：深度基准与本文模型 epoch 调小")
    args = p.parse_args()

    from features.real_data_loader import load_real_data
    features, returns, _ = load_real_data(
        asset=args.asset, bridge_path=args.bridge_path,
        macro_csv=args.macro_csv,
    )

    de = 10 if args.fast else 40
    pe = 15 if args.fast else 60
    df = run_comparison(features, returns, conf=args.conf,
                        deep_epochs=de, proposed_epochs=pe)

    print("\n" + "=" * 72)
    print("  滚动回测对比（第 9 节）")
    print("=" * 72)
    print(df.to_string(index=False))

    out_dir = os.path.join(_THIS, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_comparison.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] {out_path}")
    return df


if __name__ == "__main__":
    main()

