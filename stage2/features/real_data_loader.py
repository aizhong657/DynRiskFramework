
"""
真实数据加载器
======================================================================
将三类真实信号装配成第四章 33 维特征矩阵：

  A 价格与波动 (8维)   ：当前+5阶滞后收益率(6) + RV_5d + RV_22d
  B 宏观与跨市 (7维)   ：Δr^f, 信用利差CS, 汇率FX, ρ_roll, ρ_stress,
                          IV(已实现波动率代理), term_slope
  C 尾部先验   (6维)   ：DS-LDE 桥接信号（resid_z, sigma_z, delta_mu_z,
                          resid_raw, sigma_raw）+ 1维派生 z 信号
  D 动力学特征 (12维)  ：dynamical_features.compute_dynamical_features
                          （W=250 滚动窗口的真实测度）

设计原则
--------
1. A/D 类全部由真实收益率计算，无随机数。
2. C 类直接读取 Stage 1 产出的 sde_bridge_features.npz；
   若文件不存在则报错并提示先运行 stage1/models/ds_lde.py。
3. B 类中无外部宏观数据时，用真实收益率派生的代理变量
   （而非随机数），并在日志中明确标注 "代理" 状态，
   保证整条链路无 np.random。
4. 所有特征滚动 z-score 标准化（窗口250，仅用历史，无前视）。

用法
----
    from features.real_data_loader import load_real_data
    features, returns, dates = load_real_data(
        asset="sz50",
        bridge_path="sde_bridge_features.npz",
    )
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# 允许从项目根 import config
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import get_data_path, DATA_DIR          # noqa: E402
from features.dynamical_features import (            # noqa: E402
    compute_dynamical_features,
    FEATURE_NAMES as DYN_FEATURE_NAMES,
)

# ──────────────────────────────────────────────────────────────────
# 33 维特征名（与 shap_analysis.ALL_FEATURE_NAMES 顺序保持一致）
# ──────────────────────────────────────────────────────────────────
FEATURE_NAMES_A = [
    "r_t", "r_t-1", "r_t-2", "r_t-3", "r_t-4", "r_t-5",
    "RV_5d", "RV_22d",
]
FEATURE_NAMES_B = [
    "delta_rf", "CS_spread", "FX_ret", "rho_roll", "rho_stress",
    "IV_proxy", "term_slope",
]
FEATURE_NAMES_C = [
    "resid_z", "sigma_z", "delta_mu_z", "resid_raw", "sigma_raw",
    "tail_score",
]
# D 类直接用 dynamical_features 的 12 个名字
ALL_FEATURE_NAMES_REAL = (
    FEATURE_NAMES_A + FEATURE_NAMES_B + FEATURE_NAMES_C + list(DYN_FEATURE_NAMES)
)
assert len(ALL_FEATURE_NAMES_REAL) == 33, len(ALL_FEATURE_NAMES_REAL)


# ──────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────
def _rolling_zscore(arr: np.ndarray, window: int = 250,
                    min_periods: int = 30) -> np.ndarray:
    """滚动 z-score（仅用历史，无前视偏差）。"""
    s  = pd.Series(arr.astype(float))
    mu = s.rolling(window, min_periods=min_periods).mean().fillna(0.0)
    sd = (s.rolling(window, min_periods=min_periods)
            .std().fillna(1.0).replace(0.0, 1.0))
    return ((s - mu) / sd).clip(-4, 4).values


def _read_ohlcv(asset: str) -> pd.DataFrame:
    """读取 Baostock 格式 CSV：date, code, open, close, high, low, volume。"""
    path = get_data_path(asset)            # 不存在会抛 FileNotFoundError
    df = pd.read_csv(path)
    df = df[["date", "open", "close", "high", "low", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # 去除停牌/缺失行
    df = df[df["close"] > 0].reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────
# A 类：价格与波动（8维）
# ──────────────────────────────────────────────────────────────────
def _build_class_A(returns: np.ndarray) -> tuple[np.ndarray, list[str]]:
    r = pd.Series(returns)
    cols = [
        returns,
        r.shift(1).fillna(0).values,
        r.shift(2).fillna(0).values,
        r.shift(3).fillna(0).values,
        r.shift(4).fillna(0).values,
        r.shift(5).fillna(0).values,
        r.rolling(5).std().fillna(0).values,      # RV_5d
        r.rolling(22).std().fillna(0).values,     # RV_22d
    ]
    return np.column_stack(cols), FEATURE_NAMES_A


# ──────────────────────────────────────────────────────────────────
# B 类：宏观与跨市（7维）
#   有外部宏观 CSV 时优先读取；否则用真实收益率派生代理变量。
#   代理变量定义见 docstring，全部确定性、无随机数。
# ──────────────────────────────────────────────────────────────────
def _build_class_B(df: pd.DataFrame,
                   returns: np.ndarray,
                   macro_csv: str | None) -> tuple[np.ndarray, list[str], bool]:
    n = len(returns)
    r = pd.Series(returns)
    used_proxy = False

    if macro_csv and os.path.exists(macro_csv):
        macro = pd.read_csv(macro_csv)
        macro["date"] = pd.to_datetime(macro["date"])
        macro = (df[["date"]].merge(macro, on="date", how="left")
                              .fillna(method="ffill").fillna(0.0))
        block = np.column_stack([
            macro.get("delta_rf",   pd.Series(np.zeros(n))).values,
            macro.get("CS_spread",  pd.Series(np.zeros(n))).values,
            macro.get("FX_ret",     pd.Series(np.zeros(n))).values,
            macro.get("rho_roll",   pd.Series(np.zeros(n))).values,
            macro.get("rho_stress", pd.Series(np.zeros(n))).values,
            macro.get("IV_proxy",   pd.Series(np.zeros(n))).values,
            macro.get("term_slope", pd.Series(np.zeros(n))).values,
        ])
        return block, FEATURE_NAMES_B, used_proxy

    # ── 无外部数据：确定性代理变量（无随机数）──────────────────
    used_proxy = True
    # 成交量对数差分 → 资金面/利率代理
    vol = pd.Series(df["volume"].values.astype(float)).replace(0, np.nan)
    delta_rf = np.log(vol).diff().fillna(0).clip(-2, 2).values
    # 高低价振幅 → 信用/流动性利差代理
    cs = ((df["high"] - df["low"]) / df["close"]).fillna(0).values
    # 隔夜跳空 → 汇率冲击代理
    fx = (np.log(df["open"].values) -
          np.log(df["close"].shift(1).fillna(df["close"]).values))
    fx = np.nan_to_num(fx, nan=0.0)
    # 滚动自相关 → 市场联动代理
    rho_roll = r.rolling(60).corr(r.shift(1)).fillna(0).clip(-1, 1).values
    # 平方收益滚动均值 → 应激相关代理
    rho_stress = (r**2).rolling(22).mean().fillna(0).values
    rho_stress = np.clip(rho_stress * 50, 0, 5)
    # 22日已实现波动率年化 → IV 代理
    iv = r.rolling(22).std().fillna(0).values * np.sqrt(252) * 100
    # 短长期波动率之差 → 期限结构代理
    term = (r.rolling(5).std().fillna(0).values -
            r.rolling(60).std().fillna(0).values)

    block = np.column_stack([
        delta_rf, cs, fx, rho_roll, rho_stress, iv, term,
    ])
    return block, FEATURE_NAMES_B, used_proxy


# ──────────────────────────────────────────────────────────────────
# C 类：DS-LDE 尾部先验（6维）—— 读取 Stage 1 桥接 npz
# ──────────────────────────────────────────────────────────────────
def _build_class_C(bridge_path: str,
                   n_target: int,
                   offset: int) -> tuple[np.ndarray, list[str]]:
    """
    读取 sde_bridge_features.npz 并对齐到主特征矩阵长度。

    参数
    ----
    n_target : 主特征矩阵的行数（= 收益率长度）
    offset   : 桥接信号在收益率序列上的起始偏移
               （DS-LDE 的测试集起点；用 0 表示从头对齐，
                 不足部分前向填充，多余部分截断）
    """
    if not os.path.exists(bridge_path):
        raise FileNotFoundError(
            f"未找到 DS-LDE 桥接文件: {bridge_path}\n"
            f"请先运行 Stage 1 生成该文件：\n"
            f"    python stage1/models/ds_lde.py "
            f"--bridge_path {bridge_path}\n"
            f"（C 类尾部先验是创新点1的核心链条，不能用随机数替代）"
        )
    z = np.load(bridge_path)
    resid_z    = z["resid_z"]
    sigma_z    = z["sigma_z"]
    delta_mu_z = z["delta_mu_z"]
    resid_raw  = z["resid_raw"]
    sigma_raw  = z["sigma_raw"]

    def _align(a: np.ndarray) -> np.ndarray:
        out = np.zeros(n_target, dtype=float)
        src = np.asarray(a, dtype=float).ravel()
        end = min(n_target, offset + len(src))
        seg = end - offset
        if seg > 0:
            out[offset:end] = src[:seg]
            # 前向填充对齐缺口
            if offset > 0:
                out[:offset] = src[0]
            if end < n_target:
                out[end:] = src[seg - 1] if seg > 0 else 0.0
        return out

    resid_z_a    = _align(resid_z)
    sigma_z_a    = _align(sigma_z)
    delta_mu_z_a = _align(delta_mu_z)
    resid_raw_a  = _align(resid_raw)
    sigma_raw_a  = _align(sigma_raw)
    # 第6维：综合尾部得分（条件波动率 × |残差|，再 z-score）
    tail_score = _rolling_zscore(np.abs(resid_raw_a) * sigma_raw_a, window=250)

    block = np.column_stack([
        resid_z_a, sigma_z_a, delta_mu_z_a,
        resid_raw_a, sigma_raw_a, tail_score,
    ])
    return block, FEATURE_NAMES_C


# ──────────────────────────────────────────────────────────────────
# 主接口
# ──────────────────────────────────────────────────────────────────
def load_real_data(
    asset: str = "sz50",
    bridge_path: str = "sde_bridge_features.npz",
    macro_csv: str | None = None,
    dyn_window: int = 250,
    zscore_window: int = 250,
    bridge_offset: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    装配真实 33 维特征矩阵。

    返回
    ----
    features : np.ndarray [N, 33]   已滚动 z-score 标准化
    returns  : np.ndarray [N]       日对数收益率
    dates    : pd.DatetimeIndex [N]

    说明
    ----
    D 类需要 250 日滚动窗口预热，因此最终输出会丢弃前 dyn_window 行，
    保证每一行的 D 类特征都是真实计算值（无填充）。
    """
    if verbose:
        print(f"[RealData] 读取标的 {asset} ...")
    df = _read_ohlcv(asset)
    close = df["close"].values.astype(float)
    log_close = np.log(close)
    returns = np.diff(log_close)                 # 日对数收益率 [N-1]
    dates_full = df["date"].values[1:]           # 与 returns 对齐
    df = df.iloc[1:].reset_index(drop=True)      # 同步裁掉第一行
    n = len(returns)
    if verbose:
        print(f"[RealData] 收益率长度 = {n}  "
              f"日期 {pd.Timestamp(dates_full[0]).date()} → "
              f"{pd.Timestamp(dates_full[-1]).date()}")

    # ── A 类 ────────────────────────────────────────────────
    A, names_A = _build_class_A(returns)

    # ── B 类 ────────────────────────────────────────────────
    B, names_B, used_proxy = _build_class_B(df, returns, macro_csv)
    if verbose and used_proxy:
        print("[RealData] ⚠ 未提供宏观 CSV，B 类使用真实行情派生的"
              "确定性代理变量（无随机数）")

    # ── C 类（DS-LDE 桥接）──────────────────────────────────
    C, names_C = _build_class_C(bridge_path, n_target=n, offset=bridge_offset)
    if verbose:
        print(f"[RealData] C 类尾部先验已从 {bridge_path} 接入")

    # ── D 类（12 维真实动力学特征，250 日滚动）──────────────
    if verbose:
        print(f"[RealData] 计算 12 维动力学特征"
              f"（W={dyn_window}，约需数分钟）...")
    ret_series = pd.Series(returns, index=pd.to_datetime(dates_full))
    dyn_df = compute_dynamical_features(
        ret_series, window=dyn_window, verbose=verbose
    )
    # dyn_df 索引从第 dyn_window 个交易日开始
    D = dyn_df.values
    names_D = list(dyn_df.columns)

    # ── 对齐：D 类丢弃了前 dyn_window 行，其余三类同步裁剪 ──
    A = A[dyn_window:]
    B = B[dyn_window:]
    C = C[dyn_window:]
    returns_out = returns[dyn_window:]
    dates_out = pd.DatetimeIndex(dates_full[dyn_window:])

    # 长度对齐保险
    m = min(len(A), len(B), len(C), len(D), len(returns_out))
    A, B, C, D = A[:m], B[:m], C[:m], D[:m]
    returns_out = returns_out[:m]
    dates_out = dates_out[:m]

    raw = np.column_stack([A, B, C, D])          # [m, 33]
    assert raw.shape[1] == 33, raw.shape

    # ── 滚动 z-score（每列；仅用历史，无前视）────────────────
    feats = np.zeros_like(raw)
    for j in range(33):
        feats[:, j] = _rolling_zscore(raw[:, j], window=zscore_window)

    if verbose:
        print(f"[RealData] 最终特征矩阵: {feats.shape}  "
              f"收益率: {returns_out.shape}")
        print(f"[RealData] 有效区间 {dates_out[0].date()} → "
              f"{dates_out[-1].date()}")

    return feats, returns_out, dates_out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="sz50")
    p.add_argument("--bridge_path", default="sde_bridge_features.npz")
    p.add_argument("--macro_csv", default=None)
    args = p.parse_args()

    f, r, d = load_real_data(
        asset=args.asset,
        bridge_path=args.bridge_path,
        macro_csv=args.macro_csv,
    )
    print("\n特征名（33）:")
    for i, nm in enumerate(ALL_FEATURE_NAMES_REAL):
        print(f"  [{i:2d}] {nm}")

