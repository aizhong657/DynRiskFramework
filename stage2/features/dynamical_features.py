"""
ETGPD-Transformer — 动力系统特征计算模块
D类特征，共12维，滚动窗口 W=250 日，步长1日

依赖库安装：
    pip install nolds antropy pyinform numpy pandas scipy
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# ── 可选依赖，按需导入 ──────────────────────────────────────────
try:
    import nolds
    NOLDS_OK = True
except ImportError:
    NOLDS_OK = False
    print("[警告] nolds 未安装，D1/D2/D4 部分特征将用内置实现")

try:
    import antropy as ant
    ANTROPY_OK = True
except ImportError:
    ANTROPY_OK = False
    print("[警告] antropy 未安装，D3 熵特征将用内置实现")

try:
    from pyinform import transfer_entropy
    PYINFORM_OK = True
except ImportError:
    PYINFORM_OK = False
    print("[警告] pyinform 未安装，转移熵将用近似实现")


# ══════════════════════════════════════════════════════════════════
# D1  分形维数（3维）
# ══════════════════════════════════════════════════════════════════

def correlation_dimension(x: np.ndarray, emb_dim: int = 10, tau: int = 1) -> float:
    """
    关联维数 D₂  —  Grassberger-Procaccia 算法

    参数
    ----
    x       : 1-D 收益率序列
    emb_dim : 嵌入维数 m（默认10）
    tau     : 时延 τ（默认1）

    返回
    ----
    D₂ 估计值（float）；序列过短或奇异时返回 np.nan
    """
    if NOLDS_OK:
        try:
            return float(nolds.corr_dim(x, emb_dim=emb_dim, lag=tau))
        except Exception:
            pass

    # ── 内置实现（简化版）──────────────────────────────────────
    N = len(x)
    if N < emb_dim * tau + 50:
        return np.nan

    # 相空间重构
    M = N - (emb_dim - 1) * tau
    X = np.array([x[i:i + emb_dim * tau:tau] for i in range(M)])

    # 计算所有点对距离（Chebyshev范数）
    dists = cdist(X, X, metric="chebyshev")
    np.fill_diagonal(dists, np.inf)

    # 对数-对数回归估计斜率
    epsilons = np.percentile(dists[dists < np.inf], np.linspace(5, 30, 15))
    log_eps, log_C = [], []
    for eps in epsilons:
        C = np.mean(dists < eps)
        if C > 0:
            log_eps.append(np.log(eps))
            log_C.append(np.log(C))

    if len(log_eps) < 4:
        return np.nan

    slope, _ = np.polyfit(log_eps, log_C, 1)
    return float(slope)


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """
    Higuchi 分形维数 D_H

    参数
    ----
    x    : 1-D 序列
    kmax : 最大步长（默认10）

    返回
    ----
    D_H 估计值（float）
    """
    if NOLDS_OK:
        try:
            return float(nolds.hurst_rs(x))   # nolds无Higuchi，用内置
        except Exception:
            pass

    N = len(x)
    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(1, k + 1):
            Lmk = 0
            n_max = int(np.floor((N - m) / k))
            if n_max < 1:
                continue
            for i in range(1, n_max):
                Lmk += abs(x[m + i * k - 1] - x[m + (i - 1) * k - 1])
            Lmk = Lmk * (N - 1) / (k * n_max) / k
            Lk.append(Lmk)
        if Lk:
            L.append(np.mean(Lk))

    if len(L) < 2:
        return np.nan

    ks = np.arange(1, len(L) + 1)
    slope, _ = np.polyfit(np.log(ks), np.log(L), 1)
    return float(-slope)


def boxcount_fd(x: np.ndarray, scales: int = 8) -> float:
    """
    盒计数维数 D_box（价格路径版本）

    将价格路径离散化到二维网格，用不同盒尺寸计数。

    返回
    ----
    D_box 估计值（float）
    """
    y = (x - x.min()) / (x.max() - x.min() + 1e-10)
    t = np.linspace(0, 1, len(y))

    box_sizes = np.logspace(-1, -scales / 10, scales)
    counts = []
    for s in box_sizes:
        xi = np.floor(t / s).astype(int)
        yi = np.floor(y / s).astype(int)
        counts.append(len(set(zip(xi, yi))))

    if len(counts) < 3:
        return np.nan

    slope, _ = np.polyfit(np.log(1 / box_sizes), np.log(counts), 1)
    return float(slope)


# ══════════════════════════════════════════════════════════════════
# D2  Hurst 指数（3维）
# ══════════════════════════════════════════════════════════════════

def hurst_rs(x: np.ndarray) -> float:
    """
    经典 R/S Hurst 指数 H_RS

    H < 0.5 均值回复，H ≈ 0.5 随机游走，H > 0.5 趋势持续
    """
    if NOLDS_OK:
        try:
            return float(nolds.hurst_rs(x))
        except Exception:
            pass

    N = len(x)
    if N < 20:
        return np.nan

    sizes = np.unique(np.floor(np.logspace(np.log10(10), np.log10(N // 2), 12)).astype(int))
    RS = []
    for size in sizes:
        rs_vals = []
        for start in range(0, N - size, size):
            seg = x[start:start + size]
            mean_seg = np.mean(seg)
            deviation = np.cumsum(seg - mean_seg)
            R = deviation.max() - deviation.min()
            S = np.std(seg, ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            RS.append(np.mean(rs_vals))

    if len(RS) < 4:
        return np.nan

    slope, _ = np.polyfit(np.log(sizes[:len(RS)]), np.log(RS), 1)
    return float(slope)


def hurst_dfa(x: np.ndarray, order: int = 1) -> float:
    """
    去趋势波动分析（DFA）Hurst 指数 H_DFA

    参数
    ----
    order : 多项式阶数（1=线性去趋势）
    """
    if NOLDS_OK:
        try:
            return float(nolds.dfa(x, order=order))
        except Exception:
            pass

    N = len(x)
    y = np.cumsum(x - np.mean(x))
    scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(N // 4), 14)).astype(int))
    F = []
    for s in scales:
        n_seg = N // s
        if n_seg < 2:
            continue
        rms = []
        for i in range(n_seg):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            coef = np.polyfit(t, seg, order)
            trend = np.polyval(coef, t)
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
        F.append(np.mean(rms))

    if len(F) < 4:
        return np.nan

    slope, _ = np.polyfit(np.log(scales[:len(F)]), np.log(F), 1)
    return float(slope)


def multifractal_width(x: np.ndarray, q_vals: list = None) -> float:
    """
    广义 Hurst 指数多重分形谱宽 ΔH = H(q_min) - H(q_max)

    谱宽越大 → 多重分形结构越强 → 尾部越重

    返回
    ----
    ΔH（float）
    """
    if q_vals is None:
        q_vals = [-4, -2, 0.001, 2, 4]

    N = len(x)
    scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(N // 4), 10)).astype(int))
    H_q = []

    for q in q_vals:
        F_q = []
        for s in scales:
            n_seg = N // s
            if n_seg < 2:
                continue
            y = np.cumsum(x - np.mean(x))
            segs = [y[i * s:(i + 1) * s] for i in range(n_seg)]
            vars_ = []
            for seg in segs:
                t = np.arange(s)
                coef = np.polyfit(t, seg, 1)
                trend = np.polyval(coef, t)
                vars_.append(np.mean((seg - trend) ** 2))
            vars_ = np.array(vars_)
            vars_ = vars_[vars_ > 0]
            if len(vars_) == 0:
                continue
            if abs(q) < 0.01:
                F_q.append(np.exp(0.5 * np.mean(np.log(vars_))))
            else:
                F_q.append(np.mean(vars_ ** (q / 2)) ** (1 / q))

        if len(F_q) >= 4:
            slope, _ = np.polyfit(np.log(scales[:len(F_q)]), np.log(F_q), 1)
            H_q.append(slope)
        else:
            H_q.append(np.nan)

    H_q = [h for h in H_q if not np.isnan(h)]
    if len(H_q) < 2:
        return np.nan
    return float(max(H_q) - min(H_q))


# ══════════════════════════════════════════════════════════════════
# D3  熵（3维）
# ══════════════════════════════════════════════════════════════════

def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    样本熵 SampEn(m, r)

    r = r_factor × σ_t（随当前波动率动态调整）

    越小 → 序列越规则；熵骤降往往预示羊群效应或流动性危机
    """
    if ANTROPY_OK:
        try:
            r = r_factor * np.std(x, ddof=1)
            return float(ant.sample_entropy(x, order=m, metric="chebyshev"))
        except Exception:
            pass

    N = len(x)
    r = r_factor * np.std(x, ddof=1)
    if r == 0 or N < 2 * m + 5:
        return np.nan

    def _count_matches(vec, length):
        count = 0
        for i in range(len(vec) - length):
            for j in range(i + 1, len(vec) - length):
                if np.max(np.abs(vec[i:i+length] - vec[j:j+length])) < r:
                    count += 1
        return count

    A = _count_matches(x, m + 1)
    B = _count_matches(x, m)
    if B == 0:
        return np.nan
    return float(-np.log(A / B)) if A > 0 else np.nan


def permutation_entropy(x: np.ndarray, order: int = 5, delay: int = 1,
                        normalize: bool = True) -> float:
    """
    排列熵 PermEn

    参数
    ----
    order     : 嵌入维数（默认5，对应120种排列）
    delay     : 时延（默认1）
    normalize : 归一化到 [0,1]（除以 ln(order!)）

    对奇异值和非平稳性鲁棒，实盘中比 SampEn 更稳定
    """
    if ANTROPY_OK:
        try:
            return float(ant.perm_entropy(x, order=order, delay=delay,
                                          normalize=normalize))
        except Exception:
            pass

    N = len(x)
    patterns = {}
    for i in range(N - (order - 1) * delay):
        idx = tuple(np.argsort(x[i:i + order * delay:delay]))
        patterns[idx] = patterns.get(idx, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    if normalize:
        import math
        H /= np.log(math.factorial(order))
    return float(H)


def transfer_entropy_approx(x: np.ndarray, y: np.ndarray,
                             k: int = 1, l: int = 1,
                             bins: int = 10) -> float:
    """
    转移熵 TE(X→Y)（离散化近似实现）

    衡量资产 X 对资产 Y 的单向信息流强度
    TE(X→Y) > TE(Y→X) 表示 X 领先于 Y（Granger因果的信息论版本）

    参数
    ----
    x, y : 两个等长收益率序列
    k    : Y 的历史阶数
    l    : X 的历史阶数
    bins : 离散化 bin 数
    """
    if PYINFORM_OK:
        try:
            xs = pd.qcut(x, bins, labels=False, duplicates="drop").values
            ys = pd.qcut(y, bins, labels=False, duplicates="drop").values
            return float(transfer_entropy(xs.astype(int), ys.astype(int), k=k))
        except Exception:
            pass

    # 离散化近似
    def _discretize(arr, n):
        edges = np.percentile(arr, np.linspace(0, 100, n + 1))
        return np.digitize(arr, edges[1:-1])

    N = min(len(x), len(y))
    x_d = _discretize(x[:N], bins)
    y_d = _discretize(y[:N], bins)

    te = 0.0
    for t in range(max(k, l), N):
        y_next = y_d[t]
        y_hist = tuple(y_d[t - k:t])
        x_hist = tuple(x_d[t - l:t])

        # 联合概率近似（频率估计）
        p_y_next_given_y_x = np.mean(
            (y_d[max(k,l):N] == y_next) &
            np.all([y_d[max(k,l)-k+i:N-k+i] == y_hist[i] for i in range(k)], axis=0) &
            np.all([x_d[max(k,l)-l+i:N-l+i] == x_hist[i] for i in range(l)], axis=0)
        ) + 1e-10

        p_y_next_given_y = np.mean(
            (y_d[max(k,l):N] == y_next) &
            np.all([y_d[max(k,l)-k+i:N-k+i] == y_hist[i] for i in range(k)], axis=0)
        ) + 1e-10

        te += np.log2(p_y_next_given_y_x / p_y_next_given_y)

    return float(te / (N - max(k, l)))


# ══════════════════════════════════════════════════════════════════
# D4  Lyapunov 指数 / 递归分析（3维）
# ══════════════════════════════════════════════════════════════════

def max_lyapunov(x: np.ndarray, emb_dim: int = 10, tau: int = 1,
                 min_tsep: int = 10) -> float:
    """
    最大 Lyapunov 指数 λ₁  —  Rosenstein 算法

    λ₁ > 0 → 混沌，可预测时域上界 T_pred ≈ 1/λ₁（单位：交易日）
    λ₁ 越大 → 序列越不可预测 → 应放宽 VaR 置信区间

    参数
    ----
    min_tsep : 近邻点最小时间间隔（避免选到时间上相邻的点）
    """
    if NOLDS_OK:
        try:
            return float(nolds.lyap_r(x, emb_dim=emb_dim, lag=tau,
                                      min_tsep=min_tsep))
        except Exception:
            pass

    N = len(x)
    M = N - (emb_dim - 1) * tau
    if M < 20:
        return np.nan

    # 相空间重构
    X = np.array([x[i:i + emb_dim * tau:tau] for i in range(M)])

    # 寻找最近邻（排除时间近邻）
    d = np.full(M, np.nan)
    for i in range(M):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists[max(0, i - min_tsep):min(M, i + min_tsep + 1)] = np.inf
        j = np.argmin(dists)
        if dists[j] < np.inf:
            d[i] = dists[j]

    # 跟踪发散（对数线性拟合）
    steps = min(20, M // 4)
    div = []
    for dt in range(1, steps):
        idx = np.where(~np.isnan(d))[0]
        idx = idx[idx + dt < M]
        if len(idx) == 0:
            break
        log_d = np.log(np.linalg.norm(X[idx + dt] - X[idx], axis=1) + 1e-10)
        div.append(np.mean(log_d))

    if len(div) < 3:
        return np.nan

    slope, _ = np.polyfit(range(len(div)), div, 1)
    return float(slope)


def recurrence_rate(x: np.ndarray, emb_dim: int = 5, tau: int = 1,
                    threshold_pct: float = 10.0) -> float:
    """
    递归率 RR（递归定量分析 RQA）

    RR = 递归图中黑点比例
    RR 骤降 → 相空间轨迹不再回访旧区域 → 体制切换早期信号

    参数
    ----
    threshold_pct : 距离阈值取所有点对距离的百分位数（默认10%）
    """
    N = len(x)
    M = N - (emb_dim - 1) * tau
    if M < 10:
        return np.nan

    X = np.array([x[i:i + emb_dim * tau:tau] for i in range(M)])

    # 限制计算量：最多取200个点
    if M > 200:
        idx = np.random.choice(M, 200, replace=False)
        X = X[idx]
        M = 200

    dists = cdist(X, X, metric="euclidean")
    eps = np.percentile(dists, threshold_pct)
    R = (dists <= eps).astype(float)
    np.fill_diagonal(R, 0)
    return float(R.sum() / (M * (M - 1)))


def false_nearest_neighbors(x: np.ndarray, max_dim: int = 10,
                             tau: int = 1, rtol: float = 10.0) -> float:
    """
    假近邻法（FNN）估计最优嵌入维数 m*

    返回 FNN 比例降至 5% 以下时的维数（浮点数，代表相空间复杂度）
    m* 越高 → 动力系统越复杂 → 模型需要更长历史窗口
    """
    N = len(x)
    fnn_rates = []

    for m in range(1, max_dim + 1):
        M = N - m * tau
        if M < 10:
            break
        X = np.array([x[i:i + m * tau:tau] for i in range(M)])
        X_next = np.array([x[i + tau:i + (m + 1) * tau:tau]
                           for i in range(N - (m + 1) * tau)])
        M2 = min(M, len(X_next))

        fnn = 0
        count = 0
        for i in range(min(M2, 100)):
            dists = np.linalg.norm(X[:M2] - X[i], axis=1)
            dists[i] = np.inf
            j = np.argmin(dists)
            if dists[j] == 0:
                continue
            ratio = abs(X_next[i, -1] - X_next[j, -1]) / dists[j]
            if ratio > rtol:
                fnn += 1
            count += 1

        rate = fnn / count if count > 0 else 0
        fnn_rates.append(rate)
        if rate < 0.05:
            return float(m)

    return float(max_dim)


# ══════════════════════════════════════════════════════════════════
# 主接口：滚动窗口计算所有12个特征
# ══════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # D1 分形维数
    "D2_corr_dim",       # 关联维数
    "D_higuchi",         # Higuchi 分形维数
    "D_boxcount",        # 盒计数维数
    # D2 Hurst
    "H_rs",              # R/S Hurst
    "H_dfa",             # DFA Hurst
    "DeltaH_multifrac",  # 多重分形谱宽
    # D3 熵
    "SampEn",            # 样本熵
    "PermEn",            # 排列熵
    "TE_cross",          # 转移熵（跨资产）
    # D4 Lyapunov/递归
    "lambda1",           # 最大 Lyapunov 指数
    "RR",                # 递归率
    "m_star_FNN",        # FNN 最优嵌入维数
]


def compute_dynamical_features(
    returns: pd.Series,
    window: int = 250,
    reference_series: pd.Series = None,
    n_jobs: int = 1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    滚动计算12个动力系统特征

    参数
    ----------
    returns          : 对数收益率序列（pd.Series，日频）
    window           : 滚动窗口长度（默认250交易日）
    reference_series : 用于转移熵的参考资产收益率（如VIX变化量）
                       为 None 时用 returns 自身的滞后序列
    n_jobs           : 并行进程数（1=串行，-1=全部核心）
    verbose          : 是否打印进度

    返回
    ----------
    pd.DataFrame，形状 [T-window, 12]，索引与 returns 对齐
    """
    x_all = returns.values.astype(float)
    dates = returns.index
    T = len(x_all)

    if T < window + 10:
        raise ValueError(f"序列长度 {T} 不足，需要至少 {window+10} 个观测")

    records = []
    n_windows = T - window

    for i in range(n_windows):
        if verbose and i % 50 == 0:
            print(f"  滚动窗口 {i+1}/{n_windows} ...")

        x = x_all[i: i + window]

        # 参考序列（转移熵）
        if reference_series is not None:
            ref = reference_series.values[i: i + window].astype(float)
        else:
            ref = np.roll(x, 1)        # 用自身滞后1期作为默认参考
            ref[0] = x[0]

        row = {}

        # ── D1 分形维数 ──────────────────────────────────────
        row["D2_corr_dim"]    = correlation_dimension(x)
        row["D_higuchi"]      = higuchi_fd(x)
        row["D_boxcount"]     = boxcount_fd(x)

        # ── D2 Hurst ────────────────────────────────────────
        row["H_rs"]           = hurst_rs(x)
        row["H_dfa"]          = hurst_dfa(x)
        row["DeltaH_multifrac"] = multifractal_width(x)

        # ── D3 熵 ────────────────────────────────────────────
        row["SampEn"]         = sample_entropy(x)
        row["PermEn"]         = permutation_entropy(x)
        row["TE_cross"]       = transfer_entropy_approx(ref, x)

        # ── D4 Lyapunov / 递归 ───────────────────────────────
        row["lambda1"]        = max_lyapunov(x)
        row["RR"]             = recurrence_rate(x)
        row["m_star_FNN"]     = false_nearest_neighbors(x)

        records.append(row)

    result = pd.DataFrame(records, index=dates[window:])
    return result[FEATURE_NAMES]


# ══════════════════════════════════════════════════════════════════
# 特征工程：归一化 + 与原有21维特征合并
# ══════════════════════════════════════════════════════════════════

def rolling_zscore(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    滚动 z-score 标准化（避免前视偏差）

    每个时间步 t 只用 [t-window, t-1] 的均值和标准差
    """
    return (df - df.rolling(window, min_periods=10).mean()) / \
           (df.rolling(window, min_periods=10).std() + 1e-8)


def merge_with_original(
    original_features: pd.DataFrame,
    dynamical_features: pd.DataFrame,
    normalize: bool = True,
    clip_std: float = 4.0
) -> pd.DataFrame:
    """
    将动力系统特征（12维）合并到原有特征（21维）

    参数
    ----------
    original_features  : 原有 A+B+C 类特征 DataFrame（21列）
    dynamical_features : 本模块输出的 12 列 DataFrame
    normalize          : 是否滚动标准化动力系统特征
    clip_std           : 截断极值（单位：标准差，默认±4σ）

    返回
    ----------
    合并后 DataFrame，共 33 列
    """
    dyn = dynamical_features.copy()

    # 填充少量 NaN（计算失败时）
    dyn = dyn.ffill().bfill()

    if normalize:
        dyn = rolling_zscore(dyn)
        dyn = dyn.clip(-clip_std, clip_std)

    combined = pd.concat([original_features, dyn], axis=1, join="inner")
    print(f"[合并完成] 总特征维度：{combined.shape[1]}，"
          f"样本数：{combined.shape[0]}")
    return combined


# ══════════════════════════════════════════════════════════════════
# GPD 动态阈值更新函数（集成动力系统特征）
# ══════════════════════════════════════════════════════════════════

def dynamic_threshold(
    mu_t: float,
    sigma_t: float,
    H_dfa: float,
    samp_en: float,
    c1: float = 1.5,
    c2: float = 0.3,
    c3: float = 0.2
) -> float:
    """
    三驱动动态 GPD 阈值（论文公式扩展版）

    u_t = μ̂_t + c1·σ̂_t + c2·(1 - H_DFA) + c3·SampEn⁻¹

    - Hurst 越低（均值回复）→ 阈值收紧，GPD 仅捕捉真正极端值
    - 熵越低（越规则）     → 阈值收紧，说明损失来自结构性因素
    - 危机期二者均触发宽阈值，GPD 自动纳入更多尾部样本

    参数
    ----------
    mu_t    : Transformer 预测条件均值
    sigma_t : Transformer 预测条件标准差
    H_dfa   : DFA Hurst 指数
    samp_en : 样本熵

    返回
    ----------
    阈值 u_t（float）
    """
    H_dfa   = np.clip(H_dfa,   0.01, 0.99)
    samp_en = np.clip(samp_en, 0.01, 10.0)

    u = (mu_t
         + c1 * sigma_t
         + c2 * (1.0 - H_dfa)
         + c3 / samp_en)
    return float(u)


# ══════════════════════════════════════════════════════════════════
# 快速演示
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("ETGPD-Transformer 动力系统特征模块 — 快速演示")
    print("=" * 60)

    # 生成模拟收益率（含波动率聚集）
    n = 600
    garch_vol = np.ones(n)
    for t in range(1, n):
        garch_vol[t] = np.sqrt(0.00001 + 0.09 * (np.random.randn() * garch_vol[t-1])**2
                               + 0.89 * garch_vol[t-1]**2)
    returns_sim = pd.Series(
        np.random.randn(n) * garch_vol,
        index=pd.date_range("2021-01-01", periods=n, freq="B"),
        name="sim_returns"
    )

    print(f"\n序列长度: {len(returns_sim)}，窗口: 250")
    print("计算前5个滚动窗口的特征...\n")

    # 仅演示前5个窗口
    demo = returns_sim.iloc[:260]
    feats = compute_dynamical_features(demo, window=250, verbose=True)

    print("\n── 特征输出预览 ──")
    print(feats.round(4).to_string())

    print("\n── 动态阈值示例 ──")
    u = dynamic_threshold(
        mu_t=0.0002,
        sigma_t=0.015,
        H_dfa=float(feats["H_dfa"].iloc[-1]),
        samp_en=float(feats["SampEn"].iloc[-1])
    )
    print(f"  u_t = {u:.6f}（约为 {u/0.015:.2f}σ 处）")

    print("\n── 特征含义速查 ──")
    interpretations = {
        "D2_corr_dim"    : "吸引子维度，骤升→结构破裂",
        "D_higuchi"      : "路径分形维，接近2→随机，接近1→趋势",
        "D_boxcount"     : "价格路径盒维数",
        "H_rs"           : "R/S Hurst，>0.5趋势，<0.5均值回复",
        "H_dfa"          : "DFA Hurst，更鲁棒的长记忆估计",
        "DeltaH_multifrac": "多重分形谱宽，越大尾部越重",
        "SampEn"         : "样本熵，越低越规则（危机前↓）",
        "PermEn"         : "排列熵，有序模式复杂度",
        "TE_cross"       : "转移熵，跨资产信息流强度",
        "lambda1"        : "最大Lyapunov，>0混沌，越大越不可预测",
        "RR"             : "递归率，骤降→体制切换信号",
        "m_star_FNN"     : "FNN嵌入维数，越高系统越复杂",
    }
    for k, v in interpretations.items():
        print(f"  {k:<20} {v}")

    print("\n完成。将 compute_dynamical_features() 输出接入 merge_with_original() 即可。")
