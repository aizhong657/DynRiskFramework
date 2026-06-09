#计算AMI
from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 1. Load raw close ────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "sz50_index_data.csv")
close = df['close'].dropna().values.astype(float)
INDEX_NAME = '上证50指数'
  

# ── 2. 计算对数收益率 ───────────────────────────────────────────────────────
log_return = np.diff(np.log(close)) 

# ── 3. AMI via histogram estimator ──────────────────────────────────────────
def ami(x, lag, bins=150):
    x1 = x[:-lag]
    x2 = x[lag:]
    p_xy, xe, ye = np.histogram2d(x1, x2, bins=bins, density=False)
    p_xy = p_xy / p_xy.sum()                        # joint PMF
    p_x  = p_xy.sum(axis=1, keepdims=True)          # marginal X
    p_y  = p_xy.sum(axis=0, keepdims=True)          # marginal Y
    mask = p_xy > 0
    mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / (p_x * p_y)[mask]))
    return mi

max_lag = 60
lags = np.arange(1, max_lag + 1)
ami_values = np.array([ami(log_return, lag) for lag in lags])

# ── 3. First local minimum ───────────────────────────────────────────────────
def first_local_min(arr):
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
            return i
    return int(np.argmin(arr))

idx_min  = first_local_min(ami_values)
lag_min  = lags[idx_min]
ami_min  = ami_values[idx_min]
print(f"First local minimum → lag = {lag_min},  AMI = {ami_min:.4f}")

# ── 4. Figure ────────────────────────────────────────────────────────────────
GOLDEN = (1 + 5**0.5) / 2
fig, ax = plt.subplots(figsize=(7.0, 7.0 / GOLDEN), dpi=300)

# curve + fill
ax.plot(lags, ami_values,
        color='#1a1a2e', linewidth=1.5,
        solid_capstyle='round', solid_joinstyle='round')
ax.fill_between(lags, ami_values, alpha=0.06, color='#1a1a2e')

# local minimum marker
ax.scatter([lag_min], [ami_min],
           zorder=6, s=55, color='#c0392b',
           edgecolors='#c0392b', linewidths=0.8)
ax.axvline(lag_min, color='#c0392b', linewidth=0.9,
           linestyle='--', dashes=(4, 3), alpha=0.70)

# annotation — offset direction depends on position in plot
y_range = ami_values.max() - ami_values.min()
ann_dy  = y_range * 0.08
ann_dx  = 1.5 if lag_min < max_lag * 0.75 else -8

ax.annotate(
    f'$\\tau^* = {lag_min}$\n$I = {ami_min:.3f}$ nats',
    xy=(lag_min, ami_min),
    xytext=(lag_min + ann_dx, ami_min + ann_dy),
    fontsize=8.5, color='#c0392b', va='bottom', ha='left',
    arrowprops=dict(arrowstyle='->', color='#c0392b',
                    lw=0.8, connectionstyle='arc3,rad=0.15'),
)

# axes
ax.set_xlabel('时间延迟 $\\tau$ (交易日)',
              fontsize=10.5, labelpad=6)
ax.set_ylabel('平均互信息 $I(\\tau)$ (nats)',
              fontsize=10.5, labelpad=6)
ax.set_title('平均互信息 vs. 时间延迟\n'
             f'{INDEX_NAME} — 日对数收益率',
             fontsize=10.5, pad=10, color='#1a1a2e')

ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.tick_params(axis='both', which='major', labelsize=8.5,
               direction='in', length=4, width=0.7, pad=4)
ax.tick_params(axis='both', which='minor', direction='in',
               length=2.5, width=0.5)

for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['bottom', 'left']:
    ax.spines[sp].set_linewidth(0.8)

ax.grid(axis='y', linestyle=':', linewidth=0.45, color='#bbbbbb', alpha=0.9)
ax.set_xlim(0.5, max_lag + 0.5)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
out_img  = str(OUTPUT_DIR / "sz50_ami_log_return.png")
out_code = str(OUTPUT_DIR / "sz50_ami_log_return.py")
plt.savefig(out_img, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"图片已保存 → {out_img}")

# copy script to outputs
import shutil, __main__
shutil.copy(__main__.__file__, out_code)
print(f"代码已保存 → {out_code}")