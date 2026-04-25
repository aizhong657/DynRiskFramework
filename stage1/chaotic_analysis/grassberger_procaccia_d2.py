#关联维数
from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.spatial import KDTree
from scipy.stats import linregress

# ════════════════════════════════════════════════════════════
# 1. Load data & compute log returns
# ════════════════════════════════════════════════════════════
df    = pd.read_csv(DATA_DIR / "sz50_index_data.csv")
dates = pd.to_datetime(df['date'].values)
close = df['close'].dropna().values.astype(float)

r     = np.diff(np.log(close))         # log returns, length N-1
dates_r = dates[1:]                    # aligned dates

N    = len(r)
print(f"Log-return series length: {N}")

# ════════════════════════════════════════════════════════════
# 2. Delay embedding
# ════════════════════════════════════════════════════════════
def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    return x[np.arange(n)[:, None] + np.arange(m)[None, :] * tau]

# ════════════════════════════════════════════════════════════
# 3. Correlation integral C(r) via KDTree + GP slope
# ════════════════════════════════════════════════════════════
def correlation_dimension(x, m, tau, n_radii=30, pct_lo=5, pct_hi=40):
    """
    Grassberger-Procaccia estimator.
    Fit log C(r) ~ D2 * log(r) in the scaling region [pct_lo, pct_hi]
    percentiles of pairwise distances.
    Returns estimated D2 (correlation dimension).
    """
    Y = embed(x, m, tau)
    n = len(Y)
    if n < 50:
        return np.nan

    # subsample if too large (speed)
    MAX_PTS = 600
    if n > MAX_PTS:
        idx = np.random.choice(n, MAX_PTS, replace=False)
        Y   = Y[idx]
        n   = MAX_PTS

    tree  = KDTree(Y)
    # pairwise distances (upper triangle)
    dists = tree.sparse_distance_matrix(tree, max_distance=np.inf,
                                         output_type='coo_matrix').data
    if len(dists) == 0:
        return np.nan

    r_lo = np.percentile(dists, pct_lo)
    r_hi = np.percentile(dists, pct_hi)
    if r_lo <= 0 or r_hi <= r_lo:
        return np.nan

    radii = np.geomspace(r_lo, r_hi, n_radii)
    C     = np.array([np.mean(dists <= rad) for rad in radii])
    C     = C[C > 0]
    radii = radii[:len(C)]

    if len(C) < 4:
        return np.nan

    slope, *_ = linregress(np.log(radii), np.log(C))
    return slope

# ════════════════════════════════════════════════════════════
# 4. Rolling window
# ════════════════════════════════════════════════════════════
tau    = 5
m      = 22
window = 500
step   = 5          # compute every `step` days for speed

np.random.seed(42)

roll_dates = []
roll_D2    = []

indices = range(window, N + 1, step)
total   = len(list(indices))
print(f"Computing {total} windows...")

for k, end in enumerate(range(window, N + 1, step)):
    seg = r[end - window: end]
    D2  = correlation_dimension(seg, m=m, tau=tau)
    roll_dates.append(dates_r[end - 1])
    roll_D2.append(D2)
    if (k + 1) % 50 == 0:
        print(f"  {k+1}/{total}  date={dates_r[end-1].date()}  D2={D2:.3f}" if not np.isnan(D2) else f"  {k+1}/{total}  D2=NaN")

roll_dates = np.array(roll_dates)
roll_D2    = np.array(roll_D2, dtype=float)

print(f"\nD2 range: [{np.nanmin(roll_D2):.3f}, {np.nanmax(roll_D2):.3f}]")
print(f"D2 mean : {np.nanmean(roll_D2):.3f}")

# ════════════════════════════════════════════════════════════
# 5. Publication-quality plot (中文)
# ════════════════════════════════════════════════════════════
GOLDEN = (1 + 5**0.5) / 2
fig, ax = plt.subplots(figsize=(9.0, 9.0 / GOLDEN), dpi=300)

# smooth (7-point moving average for visual clarity)
def moving_avg(x, w=7):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')

D2_smooth = moving_avg(roll_D2, w=7)

# rolling std band (30-point window, centered)
roll_std = pd.Series(roll_D2).rolling(window=30, center=True, min_periods=5).std().values
upper = D2_smooth + roll_std
lower = D2_smooth - roll_std
ax.fill_between(roll_dates, lower, upper,
                color='#4a90d9', alpha=0.18, zorder=1,
                label=r'移动平均 $\pm$ 滚动标准差 (30点)')

# raw blue line (clearly visible)
ax.plot(roll_dates, roll_D2,
        color='#4a90d9', linewidth=1.2, alpha=0.80, zorder=2,
        label='$D_2$ (原始)')

# smoothed dark line (main trend)
ax.plot(roll_dates, D2_smooth,
        color='#1a1a2e', linewidth=1.9, zorder=3,
        label='$D_2$ (7点移动平均)')

# mean line
mean_D2 = np.nanmean(roll_D2)
ax.axhline(mean_D2, color='#c0392b', linewidth=0.9,
           linestyle='--', dashes=(5, 4), alpha=0.80, zorder=4,
           label=f'平均 $D_2 = {mean_D2:.3f}$')

# 标注特定时间段（透明浅蓝色）
import matplotlib.dates as mdates

# 定义时间段 [开始日期, 结束日期]
time_periods = [
    ('2007-01-01', '2007-12-31'),
    ('2011-01-01', '2012-12-31'),
    ('2015-01-01', '2016-01-01'),
    ('2020-01-01', '2021-01-01'),
    ('2023-01-01', '2024-12-31')
]

# 为每个时间段添加透明深灰色背景
for i, (start_date, end_date) in enumerate(time_periods):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    # 只在第一个时间段添加图例标签
    label = '事件' if i == 0 else None
    ax.axvspan(start, end, alpha=0.15, color='#666666', zorder=0, label=label)

# axes
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

ax.set_xlabel('日期', fontsize=10.5, labelpad=6)
ax.set_ylabel('关联维数 $D_2$',
              fontsize=10.5, labelpad=6)
ax.set_title('上证50指数滚动关联维数 ($D_2$)\n'
             r'窗口 $= 500$,  $\tau = 2$,  $m = 29$  (Grassberger–Procaccia算法)',  
             fontsize=10.5, pad=10, color='#1a1a2e')

ax.xaxis.set_major_formatter(
    matplotlib.dates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
ax.xaxis.set_minor_locator(matplotlib.dates.YearLocator(1))
ax.tick_params(axis='x', which='major', labelsize=10.5,
               direction='in', length=4, width=0.7, pad=4, rotation=30)
ax.tick_params(axis='y', which='major', labelsize=10.5,
               direction='in', length=4, width=0.7, pad=4)
ax.tick_params(axis='both', which='minor', direction='in',
               length=2.5, width=0.5)

for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['bottom', 'left']:
    ax.spines[sp].set_linewidth(0.8)

ax.grid(axis='y', linestyle=':', linewidth=0.45,
        color='#bbbbbb', alpha=0.9, zorder=0)
ax.legend(fontsize=10.5, frameon=False, loc='lower right')
ax.set_xlim(roll_dates[0], roll_dates[-1])

ax.set_facecolor('white')
fig.patch.set_facecolor('white')
import matplotlib.dates

fig.autofmt_xdate(rotation=30, ha='right')
plt.tight_layout()

# 创建输出目录
import os
output_dir = str(OUTPUT_DIR)
os.makedirs(output_dir, exist_ok=True)

out_img  = str(OUTPUT_DIR / "rolling_D2_cmb.png")
out_code = str(OUTPUT_DIR / "rolling_D2_cmb.py")
plt.savefig(out_img, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\nSaved → {out_img}")

import shutil, __main__
shutil.copy(__main__.__file__, out_code)
print(f"Code  → {out_code}")
