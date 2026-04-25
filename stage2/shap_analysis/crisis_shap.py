"""
ETGPD-Transformer — SHAP 危机期动力特征归因 v3（修复版）

修复内容（SHAP 值全部相同问题）：
  根因：compute_shap_fast 中所有 N 个扰动样本共享同一段 history（seq_len-1步），
  而不同扰动样本对应不同的时间步，它们的历史应该不同。
  当 history 固定时，模型对所有扰动输出几乎相同的 VaR → SHAP 差异极小 → 全部相同。

  修正：
  1. wrapper 增加 set_history_batch() 方法，按样本索引切取对应历史
  2. compute_shap_fast 传入完整特征矩阵和起始索引，每个样本用自己的历史
  3. 扰动时只改变目标列，其余列保持原值（而不是共享一段固定 history）
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


# ── 特征名称 ──────────────────────────────────────────────────────

FEATURE_NAMES_ORIG = [
    "r_t","r_t-1","r_t-2","r_t-3","r_t-4","r_t-5","RV_5d","RV_22d",
    "delta_rf","CS_IG","CS_HY","FX_ret","Comdty_ret","corr_roll","corr_stress",
    "IV_t","term_slope","SKEW_idx","n_u_ratio","IV_lag1","IV_roll_std",
]
FEATURE_NAMES_DYN = [
    "D2_corr_dim","D_higuchi","D_boxcount",
    "H_rs","H_dfa","DeltaH_multifrac",
    "SampEn","PermEn","TE_cross",
    "lambda1","RR","m_star_FNN",
]
ALL_FEATURE_NAMES = FEATURE_NAMES_ORIG + FEATURE_NAMES_DYN

DYN_FEAT_DESC = {
    "D2_corr_dim":      "关联维数 D₂（吸引子复杂度）",
    "D_higuchi":        "Higuchi 分形维数",
    "D_boxcount":       "盒计数维数",
    "H_rs":             "R/S Hurst 指数（长记忆）",
    "H_dfa":            "DFA Hurst（去趋势长记忆）",
    "DeltaH_multifrac": "多重分形谱宽 ΔH",
    "SampEn":           "样本熵（序列规律性）",
    "PermEn":           "排列熵（有序模式复杂度）",
    "TE_cross":         "转移熵（跨资产信息流）",
    "lambda1":          "最大 Lyapunov 指数（混沌度）",
    "RR":               "递归率（体制稳定性）",
    "m_star_FNN":       "FNN 嵌入维数（相空间复杂度）",
}

CRISIS_PERIODS = {
    "GFC_2008":      ("2008-09-01", "2009-03-31"),
    "EuroDebt_2011": ("2011-07-01", "2012-01-31"),
    "COVID_2020":    ("2020-02-01", "2020-06-30"),
    "Rate_2022":     ("2022-01-01", "2022-10-31"),
}


# ══════════════════════════════════════════════════════════════════
# 1. SHAP Wrapper（修复版）
# ══════════════════════════════════════════════════════════════════

class ETGPDShapWrapper:
    """
    修复版：支持按样本索引取独立历史

    正确用法：
        wrapper.set_full_features(all_features)
        wrapper.set_start_indices(start_idx_array)  # 每个样本对应的起始位置
        shap_vals = analyzer.compute_shap_fast(...)
    """

    def __init__(self, model, seq_len: int = 60, device: str = "cpu"):
        self.model        = model.eval().to(device)
        self.seq_len      = seq_len
        self.device       = device
        self._full_feats: np.ndarray = None  # [N_total, d] 完整特征矩阵
        self._start_idx:  np.ndarray = None  # [N_explain] 每个样本的历史起始
        # 兼容旧接口
        self._x_hist: np.ndarray = None

    def set_history(self, x_hist: np.ndarray):
        """兼容旧接口（单段历史）"""
        self._x_hist = x_hist

    def set_full_features(self, full_feats: np.ndarray):
        """设置完整特征矩阵（用于按样本切历史）"""
        self._full_feats = full_feats

    def set_start_indices(self, start_idx: np.ndarray):
        """每个解释样本在 full_feats 中的历史起始位置"""
        self._start_idx = np.asarray(start_idx)

    def _get_history_for(self, sample_i: int) -> np.ndarray:
        """取第 sample_i 个样本的历史（seq_len-1 步）"""
        if self._full_feats is not None and self._start_idx is not None:
            si = int(self._start_idx[sample_i])
            return self._full_feats[si: si + self.seq_len - 1]
        elif self._x_hist is not None:
            return self._x_hist[-(self.seq_len - 1):]
        else:
            raise RuntimeError("请先调用 set_full_features() 或 set_history()")

    def __call__(self, x_query: np.ndarray) -> np.ndarray:
        """
        x_query : [N, d] — N 个扰动样本（最后一步特征向量）
        返回    : [N]    — VaR 预测
        """
        N, d    = x_query.shape
        results = []

        for i in range(N):
            hist   = self._get_history_for(i)              # [T-1, d]
            x_full = np.vstack([hist, x_query[i:i+1]])     # [T, d]

            x_t    = torch.tensor(x_full, dtype=torch.float32
                                  ).unsqueeze(0).to(self.device)
            tail   = torch.tensor(x_query[i, -6:],
                                  dtype=torch.float32
                                  ).unsqueeze(0).to(self.device)

            r_hist  = x_full[:, 0]
            mu_p    = float(r_hist.mean())
            sig_p   = float(r_hist.std()) + 1e-6
            # 动力特征索引：H_dfa = col 25（21+4），SampEn = col 27（21+6）
            h_dfa   = float(x_query[i, 25]) if d > 25 else 0.5
            samp_en = float(x_query[i, 27]) if d > 27 else 1.0
            cond    = torch.tensor([[mu_p, sig_p, h_dfa, samp_en]],
                                   dtype=torch.float32).to(self.device)

            with torch.no_grad():
                out = self.model(x_t, tail, cond)
            results.append(float(out["var"].cpu()))

        return np.array(results)


# ══════════════════════════════════════════════════════════════════
# 2. SHAP 计算器（修复版）
# ══════════════════════════════════════════════════════════════════

class CrisisShapAnalyzer:

    def __init__(self, model, seq_len: int = 60,
                 feature_names: List[str] = None,
                 device: str = "cpu"):
        self.wrapper      = ETGPDShapWrapper(model, seq_len, device)
        self.seq_len      = seq_len
        self.feat_names   = feature_names or ALL_FEATURE_NAMES
        self.shap_values: np.ndarray = None
        self.dates:  pd.DatetimeIndex = None

    def compute_shap_fast(
        self,
        features:  np.ndarray,
        dates:     pd.DatetimeIndex,
        n_samples: int = 200,
        seed:      int = 42,
    ) -> np.ndarray:
        """
        修复版快速 SHAP

        关键修正：
        - 每个样本 i 对应 features[i: i+seq_len] 窗口中的最后一步
        - 历史 = features[i: i+seq_len-1]（独立，不共享）
        - 扰动只改当前步（第 j 列），历史不变
        - 基线预测和扰动预测使用相同的 history，差异来自当前特征扰动

        返回 [n_samples, d] SHAP 近似值
        """
        np.random.seed(seed)
        N, d = features.shape
        # 有效起始索引：需要 seq_len 步历史
        max_start = N - self.seq_len
        if max_start <= 0:
            raise ValueError(f"特征矩阵长度 {N} 不足 seq_len={self.seq_len}")

        n    = min(n_samples, max_start)
        idxs = np.sort(np.random.choice(max_start, n, replace=False))
        self.dates = dates[idxs + self.seq_len - 1] if dates is not None else None

        # 当前步特征（每个样本的最后一步）
        x_current = features[idxs + self.seq_len - 1]  # [n, d]

        # 设置完整特征矩阵和起始索引（历史对齐）
        self.wrapper.set_full_features(features)
        self.wrapper.set_start_indices(idxs)

        # 基线预测
        base_pred = self.wrapper(x_current)
        med_base  = float(np.nanmedian(base_pred)) if np.any(np.isfinite(base_pred)) else 0.0
        base_pred = np.where(np.isfinite(base_pred), base_pred, med_base)

        importances = np.zeros((n, d))

        for j in range(d):
            x_perturb       = x_current.copy()
            # 只扰动第 j 列（随机置换），历史不变
            x_perturb[:, j] = np.random.permutation(x_perturb[:, j])

            # 每列扰动使用相同的 history 设置
            self.wrapper.set_full_features(features)
            self.wrapper.set_start_indices(idxs)

            perturb_pred = self.wrapper(x_perturb)
            med_p = float(np.nanmedian(perturb_pred)) \
                    if np.any(np.isfinite(perturb_pred)) else med_base
            perturb_pred = np.where(np.isfinite(perturb_pred),
                                    perturb_pred, med_p)

            importances[:, j] = base_pred - perturb_pred

        self.shap_values = importances
        return self.shap_values

    def compute_shap_values(
        self,
        features:     np.ndarray,
        dates:        pd.DatetimeIndex,
        n_background: int = 50,
        n_explain:    int = 100,
        verbose:      bool = True,
    ) -> np.ndarray:
        """精确 KernelSHAP（需要 shap 库）"""
        if not SHAP_OK:
            raise ImportError("pip install shap")

        N, d     = features.shape
        max_s    = N - self.seq_len
        bg_idxs  = np.random.choice(max_s, min(n_background, max_s), replace=False)
        ex_idxs  = np.sort(np.random.choice(max_s, min(n_explain, max_s), replace=False))
        self.dates = dates[ex_idxs + self.seq_len - 1]

        bg_data  = features[bg_idxs + self.seq_len - 1]
        ex_data  = features[ex_idxs + self.seq_len - 1]

        self.wrapper.set_full_features(features)
        self.wrapper.set_start_indices(bg_idxs)   # 背景用bg历史
        explainer = shap.KernelExplainer(self.wrapper, bg_data)

        self.wrapper.set_start_indices(ex_idxs)   # 解释用ex历史
        sv = explainer.shap_values(ex_data, nsamples=100, silent=not verbose)
        self.shap_values = np.array(sv)
        return self.shap_values

    # ────────────────────────────────────────────────────────────

    def identify_crisis_windows(
        self, violations, dates,
        min_viol=3, window_days=20, known_crises=True,
    ) -> pd.Series:
        s         = pd.Series(violations.astype(int), index=dates)
        roll_viol = s.rolling(window_days, min_periods=1).sum()
        flag      = roll_viol >= min_viol
        if known_crises:
            for name, (start, end) in CRISIS_PERIODS.items():
                try:
                    flag[(dates >= start) & (dates <= end)] = True
                except Exception:
                    pass
        return flag

    def compare_crisis_normal(self, crisis_mask: pd.Series) -> pd.DataFrame:
        assert self.shap_values is not None
        n         = min(len(self.shap_values), len(crisis_mask))
        sv        = self.shap_values[:n]
        cm        = crisis_mask.values[:n]
        crisis_sv = sv[cm]
        normal_sv = sv[~cm]

        rows = []
        for j, name in enumerate(self.feat_names[:sv.shape[1]]):
            c_col   = crisis_sv[:, j]
            n_col   = normal_sv[:, j]
            c_valid = c_col[np.isfinite(c_col)]
            n_valid = n_col[np.isfinite(n_col)]
            c_imp   = float(np.mean(np.abs(c_valid))) if len(c_valid) > 0 else 0.0
            n_imp   = float(np.mean(np.abs(n_valid))) if len(n_valid) > 0 else 1e-8
            ratio   = c_imp / (n_imp + 1e-8)
            c_dir   = ("↑风险" if (len(c_valid) > 0 and np.mean(c_valid) > 0)
                       else "↓风险")
            rows.append({
                "feature":          name,
                "crisis_imp":       round(c_imp, 6),
                "normal_imp":       round(n_imp, 6),
                "ratio":            round(ratio, 3),
                "is_dynamical":     name in FEATURE_NAMES_DYN,
                "crisis_direction": c_dir,
                "description":      DYN_FEAT_DESC.get(name, name),
            })

        return pd.DataFrame(rows).sort_values(
            "crisis_imp", ascending=False).reset_index(drop=True)

    def plot_dynamical_importance(self, compare_df, save_path=None, top_n=12):
        if not MATPLOTLIB_OK:
            return
        dyn_df  = compare_df[compare_df["is_dynamical"]].copy()
        all_top = compare_df.head(top_n).copy()
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.patch.set_facecolor("#0f1117")
        for ax in axes:
            ax.set_facecolor("#0f1117")
            ax.tick_params(colors="#c8c8d0", labelsize=9)
            for sp in ["bottom", "left"]:
                ax.spines[sp].set_color("#2a2a3a")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # 图1
        ax = axes[0]
        colors = ["#7f77dd" if r else "#2a8fff" for r in all_top["is_dynamical"]]
        ax.barh(range(len(all_top)), all_top["crisis_imp"].values,
                color=colors, height=0.7)
        ax.set_yticks(range(len(all_top)))
        ax.set_yticklabels(all_top["feature"].values, fontsize=9, color="#c8c8d0")
        ax.invert_yaxis()
        ax.set_xlabel("平均|SHAP|（危机期）", color="#9090a0", fontsize=9)
        ax.set_title("危机期特征重要性 Top N\n(紫色=动力特征)",
                     color="#e0e0f0", fontsize=11, pad=12)
        ax.legend(handles=[
            mpatches.Patch(color="#7f77dd", label="动力系统特征"),
            mpatches.Patch(color="#2a8fff", label="原有特征"),
        ], loc="lower right", facecolor="#1a1a2e",
           edgecolor="#2a2a3a", labelcolor="#c8c8d0", fontsize=8)

        # 图2
        ax = axes[1]
        ds = dyn_df.sort_values("crisis_imp", ascending=True)
        y  = np.arange(len(ds))
        w  = 0.38
        ax.barh(y - w/2, ds["crisis_imp"].values, w, color="#d85a30",
                alpha=0.9, label="危机期")
        ax.barh(y + w/2, ds["normal_imp"].values, w, color="#1d9e75",
                alpha=0.9, label="正常期")
        ax.set_yticks(y)
        ax.set_yticklabels(ds["feature"].values, fontsize=9, color="#c8c8d0")
        ax.set_xlabel("平均|SHAP|", color="#9090a0", fontsize=9)
        ax.set_title("动力特征\n危机 vs 正常期", color="#e0e0f0",
                     fontsize=11, pad=12)
        ax.legend(facecolor="#1a1a2e", edgecolor="#2a2a3a",
                  labelcolor="#c8c8d0", fontsize=8)

        # 图3
        ax = axes[2]
        dd = dyn_df.sort_values("ratio", ascending=False)
        bar_c = ["#e24b4a" if d == "↑风险" else "#1d9e75"
                 for d in dd["crisis_direction"]]
        ax.barh(range(len(dd)), dd["ratio"].values, color=bar_c, height=0.7)
        ax.axvline(1.0, color="#888790", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_yticks(range(len(dd)))
        ax.set_yticklabels(dd["feature"].values, fontsize=9, color="#c8c8d0")
        ax.invert_yaxis()
        ax.set_xlabel("危机/正常 比值", color="#9090a0", fontsize=9)
        ax.set_title("危机期相对重要性\n(>1=危机更重要)",
                     color="#e0e0f0", fontsize=11, pad=12)
        ax.legend(handles=[
            mpatches.Patch(color="#e24b4a", label="↑增大风险"),
            mpatches.Patch(color="#1d9e75", label="↓降低风险"),
        ], loc="lower right", facecolor="#1a1a2e",
           edgecolor="#2a2a3a", labelcolor="#c8c8d0", fontsize=8)

        plt.tight_layout(pad=2.5)
        # Windows 兼容的临时保存路径
        if not save_path:
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_outputs")
            os.makedirs(temp_dir, exist_ok=True)
            save = os.path.join(temp_dir, "shap_crisis.png")
        else:
            save = save_path
        plt.savefig(save, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SHAP] 图表已保存至 {save}")

    def attention_heatmap(self, model, x_sample, date_labels=None, save_path=None):
        if not MATPLOTLIB_OK:
            return
        x_t  = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out  = model(x_t)
        attn = out["attn"].squeeze(0).mean(0).cpu().numpy()
        T    = attn.shape[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        im = ax.imshow(attn, cmap="magma", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.tick_params(colors="#c8c8d0", labelsize=8)
        ax.set_title("稀疏注意力热图（最后编码层，多头平均）",
                     color="#e0e0f0", fontsize=12, pad=10)
        ax.set_xlabel("Key 时间步", color="#9090a0")
        ax.set_ylabel("Query 时间步", color="#9090a0")
        plt.tight_layout()
        # Windows 兼容的临时保存路径
        if not save_path:
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_outputs")
            os.makedirs(temp_dir, exist_ok=True)
            save = os.path.join(temp_dir, "attn_heatmap.png")
        else:
            save = save_path
        plt.savefig(save, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SHAP] 注意力热图已保存至 {save}")

    def print_crisis_report(self, compare_df, top_n=6):
        dyn = compare_df[compare_df["is_dynamical"]].head(top_n)
        print("\n" + "═"*62)
        print("  危机期动力系统特征 SHAP 归因报告")
        print("═"*62)
        print(f"  {'特征':<22} {'危机':>9} {'正常':>9} {'比值':>6}  方向")
        print("─"*62)
        for _, r in dyn.iterrows():
            print(f"  {r['feature']:<22} {r['crisis_imp']:>9.5f} "
                  f"{r['normal_imp']:>9.5f} {r['ratio']:>6.2f}  "
                  f"{r['crisis_direction']}")
        print("═"*62)
        print("\n  关键发现：")
        for _, r in dyn[dyn["ratio"] > 1.5].iterrows():
            desc = DYN_FEAT_DESC.get(r["feature"], r["feature"])
            print(f"  • {desc}")
            print(f"    危机期重要性是正常期的 {r['ratio']:.1f}×，{r['crisis_direction']}")
        print()
