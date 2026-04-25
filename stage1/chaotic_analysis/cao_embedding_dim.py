#计算嵌入维数
from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.spatial import KDTree
import os

# ── 1. Load & compute log returns ───────────────────────────────────────────
data_path = DATA_DIR / "sz50_index_data.csv"
_path_lower = data_path.lower()
if 'sz50' in _path_lower or 'ssz50' in _path_lower:
    series_name = '上证50'
elif 'cmb' in _path_lower:
    series_name = '招商银行'
else:
    series_name = os.path.splitext(os.path.basename(data_path))[0]

df  = pd.read_csv(data_path)
close = df['close'].dropna().values.astype(float)
r   = np.diff(np.log(close))

# ── 2. Cao's method ──────────────────────────────────────────────────────────
def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    return x[np.arange(n)[:, None] + np.arange(m)[None, :] * tau]

def cao_method(x, tau, max_m=30):
    N = len(x)
    a = np.zeros(max_m)
    b = np.zeros(max_m)

    for m in range(1, max_m + 1):
        Ym  = embed(x, m,     tau)
        Ym1 = embed(x, m + 1, tau)
        n   = min(len(Ym), len(Ym1))
        Ym  = Ym[:n];  Ym1 = Ym1[:n]

        tree = KDTree(Ym)
        _, inds = tree.query(Ym, k=2, workers=-1, p=np.inf)
        nn = inds[:, 1]

        d_m  = np.max(np.abs(Ym  - Ym[nn]),  axis=1)
        d_m1 = np.max(np.abs(Ym1 - Ym1[nn]), axis=1)

        mask      = d_m > 0
        ratio     = np.ones(n)
        ratio[mask] = d_m1[mask] / d_m[mask]
        a[m - 1]  = ratio.mean()

        # E2 numerator: extra-coordinate distance
        ei  = np.arange(n) + m * tau
        enn = nn            + m * tau
        valid = (ei < N) & (enn < N)
        diff  = np.zeros(n)
        diff[valid] = np.abs(x[ei[valid]] - x[enn[valid]])
        b[m - 1] = diff.mean()

    E1 = a[1:] / a[:-1]
    E2 = b[1:] / b[:-1]
    ms = np.arange(1, max_m)
    return ms, E1, E2

tau   = 2
max_m = 30
ms, E1, E2 = cao_method(r, tau=tau, max_m=max_m)

print("m  :", ms)
print("E1 :", np.round(E1, 4))
print("E2 :", np.round(E2, 4))

# ── 3. 正确的 m* 判断：E1曲线进入平台区的转折点 ──────────────────────────────
plateau_eps = 0.0045
close_to_one_idx = np.where(np.abs(E1 - 1.0) <= plateau_eps)[0]
if len(close_to_one_idx) > 0:
    opt_idx = int(close_to_one_idx[0])
    opt_m = int(ms[opt_idx])
else:
    threshold = 0.001
    dE1 = np.diff(E1)
    start_level = 0.99
    start_idx_candidates = np.where(E1 >= start_level)[0]
    search_start = int(start_idx_candidates[0]) if len(start_idx_candidates) > 0 else 0
    plateau_candidates = np.where(dE1[search_start:] < threshold)[0]

    if len(plateau_candidates) > 0:
        opt_idx = search_start + int(plateau_candidates[0]) + 1
        opt_m = int(ms[opt_idx])
    else:
        opt_idx = int(np.argmax(dE1)) + 2
        opt_m = int(ms[min(opt_idx, len(ms) - 1)])

e1_opt = E1[opt_idx]
e2_opt = E2[opt_idx]

print(f"最优嵌入维数  m* = {opt_m}")
print(f"m* 处 E1 = {e1_opt:.4f},  E2 = {e2_opt:.4f}")

# ── 4. 绘图 ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':          ['SimHei', 'Microsoft YaHei', 'DejaVu Serif'],
    'font.size':            10,
    'axes.linewidth':       0.7,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.labelsize':       11,
    'xtick.labelsize':      9,
    'ytick.labelsize':      9,
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'xtick.major.size':     4,
    'ytick.major.size':     4,
    'xtick.minor.size':     2,
    'ytick.minor.size':     2,
    'xtick.minor.visible':  True,
    'ytick.minor.visible':  True,
    'figure.dpi':           600,
    'savefig.dpi':          600,
    'figure.figsize':       (7.5, 4.8),
    'figure.facecolor':     'white',
    'savefig.facecolor':    'white',
    'savefig.bbox':         'tight',
    'axes.unicode_minus':   False,
})

fig, axes = plt.subplots(
    2, 1,
    figsize=(7.5, 6.0),
    gridspec_kw={'height_ratios': [3, 1.4], 'hspace': 0.08},
    sharex=True
)

ax1, ax2 = axes

ax1.axvspan(opt_m, ms[-1] + 0.5, color='#F0F0F0', zorder=0, label='平台区')
ax1.axhline(1.0, color='#AAAAAA', linewidth=0.7,
            linestyle=(0, (5, 4)), zorder=2)

ax1.plot(ms, E1,
         color='#1A1A1A', linewidth=1.4,
         marker='o', markersize=4.5,
         markerfacecolor='white', markeredgecolor='#1A1A1A',
         markeredgewidth=0.9,
         label='$E_1(m)$', zorder=4, clip_on=False)

ax1.axvline(opt_m, color='#C0392B', linewidth=0.9,
            linestyle=(0, (4, 3)), zorder=3)

ax1.scatter([opt_m], [e1_opt],
            s=60, color='#C0392B', zorder=6,
            edgecolors='white', linewidths=0.8)

label_x = opt_m - 0.8
ax1.annotate(
    f'$m^* = {opt_m}$\n$E_1 = {e1_opt:.3f}$',
    xy=(opt_m, e1_opt),
    xytext=(label_x - 2.5, e1_opt - 0.12),
    fontsize=8.5, color='#C0392B', va='top', ha='center',
    arrowprops=dict(arrowstyle='->', color='#C0392B',
                    lw=0.9, connectionstyle='arc3,rad=0.0'),
    zorder=7
)

ax1.text(ms[-1] + 0.3, 1.003, '$E_1=1$',
         fontsize=8, color='#AAAAAA', va='bottom', ha='left')

ax1.set_ylabel('$E_1(m)$', labelpad=6)
ax1.set_ylim(-0.02, 1.10)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax1.grid(axis='y', linestyle=':', linewidth=0.35, color='#BBBBBB', alpha=0.8)

handles = [
    plt.Line2D([0], [0], color='#1A1A1A', linewidth=1.4,
               marker='o', markersize=4, markerfacecolor='white',
               markeredgecolor='#1A1A1A', label='$E_1(m)$'),
    plt.Line2D([0], [0], color='#AAAAAA', linewidth=0.7,
               linestyle=(0, (5, 4)), label='$E_1=1$'),
    plt.Rectangle((0, 0), 1, 1, fc='#F0F0F0', ec='none', label='平台区'),
]
ax1.legend(handles=handles, fontsize=8.5, frameon=False,
           loc='lower right', handlelength=2.2)

ax1.set_title(f'Cao 方法 — 最优嵌入维数（{series_name}）',
              fontsize=11, pad=10, fontweight='normal')

ax2.axhline(1.0, color='#AAAAAA', linewidth=0.7,
            linestyle=(0, (5, 4)), zorder=2)
ax2.axvspan(opt_m, ms[-1] + 0.5, color='#F0F0F0', zorder=0)

ax2.plot(ms, E2,
         color='#2471A3', linewidth=1.2,
         marker='s', markersize=3.5,
         markerfacecolor='white', markeredgecolor='#2471A3',
         markeredgewidth=0.8,
         label='$E_2(m)$', zorder=4, clip_on=False)

ax2.axvline(opt_m, color='#C0392B', linewidth=0.9,
            linestyle=(0, (4, 3)), zorder=3)

ax2.scatter([opt_m], [e2_opt],
            s=45, color='#2471A3', zorder=6,
            edgecolors='white', linewidths=0.8)

ax2.annotate(
    f'$E_2={e2_opt:.3f}$',
    xy=(opt_m, e2_opt),
    xytext=(opt_m + 2.0, e2_opt + 0.015),
    fontsize=8, color='#2471A3', va='bottom', ha='left',
    arrowprops=dict(arrowstyle='->', color='#2471A3',
                    lw=0.8, connectionstyle='arc3,rad=-0.2'),
    zorder=7
)

ax2.set_xlabel('嵌入维数 $m$', labelpad=6)
ax2.set_ylabel('$E_2(m)$', labelpad=6)

e2_lo = min(0.90, E2.min() - 0.03)
e2_hi = max(1.10, E2.max() + 0.03)
ax2.set_ylim(e2_lo, e2_hi)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.grid(axis='y', linestyle=':', linewidth=0.35, color='#BBBBBB', alpha=0.8)

ax2.legend(fontsize=8.5, frameon=False, loc='upper left', handlelength=2.2)

ax1.set_xlim(ms[0] - 0.5, ms[-1] + 0.5)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))

fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.08, hspace=0.08)

output_dir = str(OUTPUT_DIR)
os.makedirs(output_dir, exist_ok=True)

_tag = 'sz50' if series_name == '上证50' else ('cmb' if series_name == '招商银行' else series_name)
out_img  = os.path.join(output_dir, f"cao_E1E2_{_tag}.png")
out_code = os.path.join(output_dir, f"cao_E1E2_{_tag}.py")

plt.savefig(out_img, dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"图片已保存 → {out_img} (600 DPI)")

import shutil, __main__
shutil.copy(__main__.__file__, out_code)
print(f"代码已保存 → {out_code}")
