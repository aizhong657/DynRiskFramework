"""
Rosenstein 小数据量算法  —  上证50指数 close 序列的最大 Lyapunov 指数 λ_max 与可预测期间估计
==================================================================================
参考：Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993).
      A practical method for calculating largest Lyapunov exponents from
      small data sets. Physica D, 65(1–2), 117–134.
"""

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


# ─────────────────────────────────────────────────────────────────────
# 1. 数据加载：上证50指数 close 列
# ─────────────────────────────────────────────────────────────────────

def load_close_series(csv_path: str, close_col: str = "close") -> np.ndarray:
    df = pd.read_csv(csv_path)
    col_map = {str(c).strip().lower(): c for c in df.columns}
    if close_col.strip().lower() not in col_map:
        raise KeyError(f"未找到列 '{close_col}'，可用列：{list(df.columns)}")

    close = df[col_map[close_col.strip().lower()]].dropna().to_numpy(dtype=np.float64)
    if close.size < 10:
        raise ValueError(f"close 数据点过少（N={close.size}），无法估计。")
    return close


# ─────────────────────────────────────────────────────────────────────
# 2. 相空间重构（Takens 时延嵌入）
# ─────────────────────────────────────────────────────────────────────

def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    将标量序列嵌入 m 维相空间。
    返回形状 (L, m) 的矩阵，L = len(x) - (m-1)*tau。
    """
    L = len(x) - (m - 1) * tau
    return np.stack([x[i*tau:i*tau+L] for i in range(m)], axis=1)


# ─────────────────────────────────────────────────────────────────────
# 3. Rosenstein 算法核心
# ─────────────────────────────────────────────────────────────────────

def rosenstein(x: np.ndarray, m: int = 3, tau: int = 3,
               p_bar: int = 15, max_steps: int = 60,
               dt: float = 0.01):
    """
    用 Rosenstein 算法估计最大 Lyapunov 指数 λ_max。

    参数
    ----
    x        : 原始标量时间序列
    m        : 嵌入维数
    tau      : 时延（采样步数）
    p_bar    : 最近邻时间排除窗口（防止时间相关假近邻）
    max_steps: 追踪最大步数
    dt       : 采样时间间隔（秒）

    返回
    ----
    times    : 时间轴 (秒)
    y        : 平均对数发散 ⟨ln d_j(i)⟩
    lambda_max      : 拟合得到的 λ_max（nats/s）
    predictability  : 可预测期间 = 1/λ_max（秒）
    fit_range: (start, end) 拟合区间索引
    """
    # ── 2.1 嵌入 ──────────────────────────────────────────────────────
    Y = embed(x, m, tau)           # (L, m)
    L = len(Y)

    # ── 2.2 KD-Tree 加速最近邻搜索 ───────────────────────────────────
    tree = KDTree(Y)
    # 查询每点最近的 p_bar+2 个邻居（含自身），再过滤时间窗口
    k_query = min(p_bar + 5, L)
    dists, idxs = tree.query(Y, k=k_query)  # (L, k_query)

    nn = np.full(L, -1, dtype=int)
    nn_dist = np.full(L, np.inf)

    for j in range(L):
        for rank in range(1, k_query):          # rank=0 是自身
            k = idxs[j, rank]
            if abs(j - k) > p_bar:              # 满足时间排除条件
                nn[j] = k
                nn_dist[j] = dists[j, rank]
                break

    # ── 2.3 追踪近邻对发散 ───────────────────────────────────────────
    steps = min(max_steps, L // 5)
    log_div = np.zeros(steps)
    count   = np.zeros(steps, dtype=int)

    for j in range(L):
        k = nn[j]
        if k < 0:
            continue
        for i in range(steps):
            if j + i >= L or k + i >= L:
                break
            d = np.linalg.norm(Y[j+i] - Y[k+i])
            if d > 0:
                log_div[i] += np.log(d)
                count[i] += 1

    # 平均（避免除零）
    valid = count > 0
    y = np.where(valid, log_div / np.maximum(count, 1), np.nan)

    times = np.arange(steps) * dt

    # ── 2.4 线性拟合（取中段，跳过初始过渡和末端饱和）──────────────
    fit_start = max(3, steps // 10)
    fit_end   = min(steps - 1, steps // 2)
    mask = valid[fit_start:fit_end+1]
    t_fit = times[fit_start:fit_end+1][mask]
    y_fit = y[fit_start:fit_end+1][mask]

    if len(t_fit) < 4:
        raise RuntimeError("有效数据点不足，请增加序列长度或调整参数。")

    coeffs = np.polyfit(t_fit, y_fit, 1)
    lambda_max    = coeffs[0]                        # 斜率 = λ_max
    predictability = 1.0 / lambda_max if lambda_max > 0 else np.inf

    # R² 评估拟合质量
    y_pred = np.polyval(coeffs, t_fit)
    ss_res = np.sum((y_fit - y_pred)**2)
    ss_tot = np.sum((y_fit - y_fit.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "times":          times,
        "y":              y,
        "lambda_max":     lambda_max,
        "predictability": predictability,
        "fit_range":      (fit_start, fit_end),
        "coeffs":         coeffs,
        "r2":             r2,
        "L":              L,
        "q":              int(np.sum(nn >= 0)),
        "steps":          steps,
    }


# ─────────────────────────────────────────────────────────────────────
# 4. 主程序
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = DATA_DIR / "sz50_index_data.csv"
    DT   = 1.0           # 采样间隔（按 1 个交易日计）
    M    = 3             # 嵌入维数
    TAU  = 3             # 时延（步数）
    PBAR = 20            # 近邻时间排除窗口

    print("═" * 60)
    print("  Rosenstein 最大 Lyapunov 指数估计（上证50指数 close）")
    print("═" * 60)

    # 加载上证50指数 close 时间序列
    print(f"\n[1] 读取数据：{DATA_PATH}")
    x = load_close_series(DATA_PATH, close_col="close")
    print(f"    close 序列长度：{len(x)}")

    # 运行 Rosenstein 算法
    print(f"\n[2] 运行 Rosenstein 算法")
    print(f"    嵌入维数 m={M}, 时延 τ={TAU}, 排除窗口 p̄={PBAR}")
    res = rosenstein(x, m=M, tau=TAU, p_bar=PBAR,
                     max_steps=80, dt=DT)

    # 输出结果
    print("\n[3] 估计结果")
    print("─" * 40)
    print(f"  λ̂_max             = {res['lambda_max']:>10.5f}  nats/day")
    print(f"  Predictability T*  = {res['predictability']:>10.4f}  day")
    print(f"                   ≈ {res['predictability']/DT:>8.1f}  采样步")
    print(f"  R²（线性拟合） = {res['r2']:>10.4f}")
    print(f"  有效近邻对数 q = {res['q']}")
    print("─" * 40)
    lam = res["lambda_max"]
    if lam > 0:
        print(f"  判断：λ̂_max > 0  →  混沌运动")
    else:
        print("  判断：λ̂_max ≤ 0  →  非混沌（周期或准周期）")

    print("\n完成。")
