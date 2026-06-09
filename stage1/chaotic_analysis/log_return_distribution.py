# 第三章的数据处理
from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.stats import norm, kurtosis, skew

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ── 配置（按需修改）─────────────────────────────────────────────────────────
CSV_PATH   = DATA_DIR / "yili_data.csv"          # 数据文件路径
INDEX_CODE = 'SH.600887'                     # 指数代码（用于标题）
INDEX_NAME = '伊利股份指数'                 # 指数名称（用于标题）
OUTPUT_PATH = 'yili_log_return_distribution.png'

# ── 读取数据 ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# ── 计算日对数收益率 ─────────────────────────────────────────────────────────
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
r = df['log_return'].dropna().values

# ── 描述性统计 ───────────────────────────────────────────────────────────────
mu, sigma = r.mean(), r.std(ddof=1)
sk        = skew(r)
ku        = kurtosis(r)          # 超额峰度（excess kurtosis）
jb_stat, jb_p = stats.jarque_bera(r)
n         = len(r)

print(f"样本量        N  = {n:,}")
print(f"均值          μ  = {mu:.6f}")
print(f"标准差        σ  = {sigma:.6f}")
print(f"偏度    Skewness  = {sk:.4f}")
print(f"超额峰度  Ex.Kurt = {ku:.4f}")
print(f"JB 统计量        = {jb_stat:.4f}")
print(f"JB p 值          = {jb_p:.4e}")

# ── 绘图 ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 直方图
ax.hist(r, bins=80, density=True,
        color='#4878CF', edgecolor='white', linewidth=0.3, alpha=0.72, zorder=2)

# 正态分布曲线
x = np.linspace(r.min() - 0.002, r.max() + 0.002, 600)
ax.plot(x, norm.pdf(x, mu, sigma),
        color='#C44E52', linewidth=1.8, linestyle='-',
        label=r'正态分布 $\mathcal{N}(\hat{\mu},\,\hat{\sigma}^2)$', zorder=3)

# 核密度估计（KDE）
kde = stats.gaussian_kde(r, bw_method='silverman')
ax.plot(x, kde(x),
        color='#222222', linewidth=1.2, linestyle='--',
        label='核密度估计', zorder=4)

# 统计信息标注框
stats_text = (
    f"$N$ = {n:,}\n"
    f"$\\hat{{\\mu}}$ = {mu:.5f}\n"
    f"$\\hat{{\\sigma}}$ = {sigma:.5f}\n"
    f"偏度 = {sk:.4f}\n"
    f"超额峰度 = {ku:.4f}\n"
    f"JB统计量 = {jb_stat:.2f}  ($p$ < 0.001)"
)
ax.text(0.973, 0.97, stats_text,
        transform=ax.transAxes, fontsize=7.8,
        verticalalignment='top', horizontalalignment='right', linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                  edgecolor='#AAAAAA', linewidth=0.8, alpha=0.92))

# 坐标轴标签与标题
ax.set_xlabel('对数收益率', fontsize=12, labelpad=6)
ax.set_ylabel('密度', fontsize=12, labelpad=6)
ax.set_title(f'{INDEX_NAME}日对数收益率分布 ({INDEX_CODE})',
             fontsize=12, fontweight='bold', pad=10)

# 图例
ax.legend(fontsize=9, framealpha=0.9, edgecolor='#BBBBBB',
          loc='upper left', handlelength=1.8)

# 坐标轴样式
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('#333333')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:.3f}'))
ax.tick_params(axis='both', labelsize=9, direction='out', length=3, width=0.7)
ax.grid(axis='y', linestyle=':', linewidth=0.5, color='#CCCCCC', alpha=0.7, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout(pad=1.2)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"图片已保存至：{OUTPUT_PATH}")