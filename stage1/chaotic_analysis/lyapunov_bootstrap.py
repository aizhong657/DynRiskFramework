"""
Bootstrap 95% CI for the Largest Lyapunov Exponent (λ_max)
of SSE 50 close price series.

Algorithm:
  - Rosenstein et al. (1993) nearest-neighbour divergence method
  - τ=3, m=7 (from Cao analysis)
  - 100 bootstrap resamples (block bootstrap, block=20 to preserve local structure)
"""

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress

# ════════════════════════════════════════════════════════════
# 1. Load close price
# ════════════════════════════════════════════════════════════
# 1. Load close price
df    = pd.read_csv(DATA_DIR / "cmb_data.csv")
close = df['close'].dropna().values.astype(float)
N     = len(close)
print(f"Close series length: {N}")

# ════════════════════════════════════════════════════════════
# 2. Delay embedding
# ════════════════════════════════════════════════════════════
def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    return x[np.arange(n)[:, None] + np.arange(m)[None, :] * tau]

# ════════════════════════════════════════════════════════════
# 3. Rosenstein λ_max estimator
# ════════════════════════════════════════════════════════════
def lyapunov_rosenstein(x, m=7, tau=3, max_iter=50, min_dist=1e-10):
    """
    Rosenstein (1993) algorithm.
    Returns λ_max (slope of mean log-divergence vs iteration).
    """
    Y = embed(x, m, tau)
    n = len(Y)
    if n < 2 * max_iter:
        return np.nan

    # for each point find nearest neighbour (temporal exclusion W = mean period)
    W = max(1, int(1.0 / max(np.std(np.diff(x)), 1e-12)))   # rough Theiler window

    dists = np.full(n, np.inf)
    nn_idx = np.zeros(n, dtype=int)
    for i in range(n):
        d = np.max(np.abs(Y - Y[i]), axis=1)
        d[max(0, i - W): min(n, i + W + 1)] = np.inf
        j = np.argmin(d)
        dists[i] = d[j]
        nn_idx[i] = j

    # track divergence over max_iter steps
    valid = dists > min_dist
    log_div = np.zeros(max_iter)
    counts  = np.zeros(max_iter, dtype=int)

    for i in np.where(valid)[0]:
        j = nn_idx[i]
        for k in range(max_iter):
            if i + k >= n or j + k >= n:
                break
            d = np.max(np.abs(Y[i + k] - Y[j + k]))
            if d > min_dist:
                log_div[k] += np.log(d)
                counts[k]  += 1

    good = counts > 0
    if good.sum() < 4:
        return np.nan

    ld = np.where(good, log_div / np.maximum(counts, 1), np.nan)
    iters = np.arange(max_iter)[good]
    slope, *_ = linregress(iters, ld[good])
    return slope

# ════════════════════════════════════════════════════════════
# 4. Compute λ_max on original series
# ════════════════════════════════════════════════════════════
TAU   = 2
M     = 29
ITERS = 50

print("Computing λ_max on original series...")
lam_orig = lyapunov_rosenstein(close, m=M, tau=TAU, max_iter=ITERS)
print(f"  λ_max (original) = {lam_orig:.5f}")

# ════════════════════════════════════════════════════════════
# 5. Block Bootstrap (B=100, block size=20)
# ════════════════════════════════════════════════════════════
B          = 100
BLOCK      = 20
np.random.seed(2025)

lam_boot = np.full(B, np.nan)
n_blocks = N // BLOCK

print(f"\nRunning {B} bootstrap resamples (block size={BLOCK})...")
for b in range(B):
    # draw n_blocks random starting positions
    starts = np.random.randint(0, N - BLOCK + 1, size=n_blocks)
    idx    = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:N]
    x_boot = close[idx]
    lam_boot[b] = lyapunov_rosenstein(x_boot, m=M, tau=TAU, max_iter=ITERS)
    if (b + 1) % 10 == 0:
        valid_so_far = lam_boot[:b+1][~np.isnan(lam_boot[:b+1])]
        print(f"  [{b+1:3d}/100]  mean={np.mean(valid_so_far):.5f}  "
              f"std={np.std(valid_so_far):.5f}")

lam_boot_valid = lam_boot[~np.isnan(lam_boot)]
ci_lo = np.percentile(lam_boot_valid, 2.5)
ci_hi = np.percentile(lam_boot_valid, 97.5)
print(f"\n── Results ──────────────────────────────────")
print(f"  λ_max (original)     = {lam_orig:.5f}")
print(f"  Bootstrap mean       = {np.mean(lam_boot_valid):.5f}")
print(f"  Bootstrap std        = {np.std(lam_boot_valid):.5f}")
print(f"  95% CI               = [{ci_lo:.5f},  {ci_hi:.5f}]")
print(f"  Valid resamples      = {len(lam_boot_valid)}/{B}")

# ════════════════════════════════════════════════════════════
# 6. Publication-quality figure (中文)
# ════════════════════════════════════════════════════════════
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

GOLDEN = (1 + 5**0.5) / 2
fig, axes = plt.subplots(1, 2, figsize=(11.0, 11.0 / GOLDEN / 1.1), dpi=300)

# ── Left: histogram of bootstrap λ_max ──────────────────────
ax = axes[0]
bins = np.linspace(lam_boot_valid.min() - 0.0005,
                   lam_boot_valid.max() + 0.0005, 28)
ax.hist(lam_boot_valid, bins=bins,
        color='#4a90d9', edgecolor='white', linewidth=0.4,
        alpha=0.85, zorder=3)

# CI shading
mask_ci = (bins[:-1] >= ci_lo) & (bins[1:] <= ci_hi)
n_vals, _ = np.histogram(lam_boot_valid, bins=bins)
for i, (lo_b, hi_b, cnt) in enumerate(zip(bins[:-1], bins[1:], n_vals)):
    if lo_b >= ci_lo and hi_b <= ci_hi:
        ax.bar(lo_b, cnt, width=hi_b - lo_b, align='edge',
               color='#1a6fb0', alpha=0.55, zorder=3)

# vertical lines
ax.axvline(lam_orig, color='#c0392b', linewidth=1.4,
           linestyle='-', zorder=5,
           label=r'原始 $\lambda_{\max} = ' + f'{lam_orig:.4f}$')
ax.axvline(ci_lo, color='#e67e22', linewidth=1.1,
           linestyle='--', dashes=(4, 3), zorder=5,
           label=r'95% 置信区间  [' + f'{ci_lo:.4f}, {ci_hi:.4f}]$')
ax.axvline(ci_hi, color='#e67e22', linewidth=1.1,
           linestyle='--', dashes=(4, 3), zorder=5)
ax.axvline(np.mean(lam_boot_valid), color='#1a1a2e', linewidth=1.0,
           linestyle=':', dashes=(2, 2), zorder=5,
           label=r'Bootstrap 均值 $= ' + f'{np.mean(lam_boot_valid):.4f}$')

ax.set_xlabel(r'最大李雅普诺夫指数 $\lambda_{\max}$', fontsize=11, labelpad=5)
ax.set_ylabel('频数', fontsize=11, labelpad=5)
ax.set_title(r'Bootstrap $\lambda_{\max}$ 分布' + '\n($B = 100$, 块大小 $= 20$)',
             fontsize=10, pad=8, color='#1a1a2e')
ax.legend(fontsize=8, frameon=False, loc='upper left')
ax.tick_params(axis='both', which='major', labelsize=8.5,
               direction='in', length=4, width=0.7)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
for sp in ['bottom', 'left']:
    ax.spines[sp].set_linewidth(0.8)
ax.grid(axis='y', linestyle=':', linewidth=0.45, color='#bbbbbb', alpha=0.8)
ax.set_facecolor('white')

# ── Right: sorted bootstrap values with CI band ──────────────
ax2 = axes[1]
sorted_boot = np.sort(lam_boot_valid)
x_idx = np.arange(1, len(sorted_boot) + 1)

ax2.scatter(x_idx, sorted_boot,
            s=18, color='#4a90d9', alpha=0.75, zorder=3,
            label=r'Bootstrap $\lambda_{\max}$')
ax2.axhline(lam_orig, color='#c0392b', linewidth=1.4,
            linestyle='-', zorder=5,
            label=r'原始 $\lambda_{\max} = ' + f'{lam_orig:.4f}$')
ax2.axhline(ci_lo, color='#e67e22', linewidth=1.1,
            linestyle='--', dashes=(4, 3), zorder=4)
ax2.axhline(ci_hi, color='#e67e22', linewidth=1.1,
            linestyle='--', dashes=(4, 3), zorder=4,
            label=r'95% 置信区间  [' + f'{ci_lo:.4f}, {ci_hi:.4f}]$')
ax2.fill_between(x_idx, ci_lo, ci_hi,
                 color='#e67e22', alpha=0.08, zorder=1)
ax2.axhline(0, color='#888888', linewidth=0.8,
            linestyle=':', dashes=(2, 3), alpha=0.7,
            label=r'$\lambda_{\max} = 0$')

ax2.set_xlabel('Bootstrap 排序索引', fontsize=11, labelpad=5)
ax2.set_ylabel(r'最大李雅普诺夫指数 $\lambda_{\max}$', fontsize=11, labelpad=5)
ax2.set_title(r'Bootstrap 估计值排序 (含95%置信区间)' + '\n'
              r'cmb_data.csv,  $\tau=2$,  $m=29$',
              fontsize=10, pad=8, color='#1a1a2e')
ax2.legend(fontsize=8, frameon=False, loc='upper left')
ax2.tick_params(axis='both', which='major', labelsize=8.5,
                direction='in', length=4, width=0.7)
for sp in ['top', 'right']:
    ax2.spines[sp].set_visible(False)
for sp in ['bottom', 'left']:
    ax2.spines[sp].set_linewidth(0.8)
ax2.grid(axis='y', linestyle=':', linewidth=0.45, color='#bbbbbb', alpha=0.8)
ax2.set_facecolor('white')
ax2.set_xlim(0, len(sorted_boot) + 1)

fig.patch.set_facecolor('white')
plt.tight_layout(w_pad=3.0)

# 创建输出目录
import os
output_dir = str(OUTPUT_DIR)
os.makedirs(output_dir, exist_ok=True)

# 保存图片和代码
out_img  = str(OUTPUT_DIR / "lyapunov_bootstrap_cmb.png")
out_code = str(OUTPUT_DIR / "lyapunov_bootstrap_cmb.py")
plt.savefig(out_img, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\n图片已保存 → {out_img}")

import shutil, __main__
shutil.copy(__main__.__file__, out_code)
print(f"代码已保存 → {out_code}")