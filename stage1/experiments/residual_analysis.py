# ============================================================
#  LDE 模型在 4 个数据集上的验证集残差分析
#  数据集：上证50 / 沪深300 / 中证500 / 创业板指
#  图表：残差时序折线图 / 分布直方图 / QQ图 / 箱线图
#  输出：residuals_<dataset>.npz  +  residual_stats.csv
# ============================================================

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import levy_stable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn.functional as F
from torch import nn, optim

# ============================================================
#  0. 数据集配置（修改为实际路径）
# ============================================================
DATASETS = {
    '上证50':  DATA_DIR / "sz50_index_data.csv",
    '沪深300': DATA_DIR / "hs300_index_data.csv",
    '中石油':  DATA_DIR / "cnpc_data.csv",
    '招行':    DATA_DIR / "cmb_data.csv",
}

# ============================================================
#  1. 固定超参数（与 chapter52.py 完全一致）
# ============================================================
D, TAU      = 22, 1
N_STEPS     = 4
LAYER_DEPTH = 25
SIGMA       = 0.5
ALPHA, BETA = 1.2, 0
LR, LR2     = 1e-4, 0.01
ITER        = 50
EPOCHS      = 100
SEED        = 4
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {device}')

# ============================================================
#  2. 通用工具
# ============================================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_and_split(csv_path):
    """加载 CSV，提取收盘价，按 70/10/20 划分，返回 scaler 和各段 DataFrame。"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    close = df['close'].values.reshape(-1, 1)
    n = len(close)
    n_train = int(np.ceil(TRAIN_RATIO * n))
    n_val   = int(np.ceil(VAL_RATIO   * n))

    scaler = StandardScaler()
    train_sc = scaler.fit_transform(close[:n_train]).flatten()
    val_sc   = scaler.transform(close[n_train:n_train+n_val]).flatten()

    dates = df['date'].values
    def mkdf(d, v): return pd.DataFrame({'Date': d, 'closescale': v})
    return (scaler,
            mkdf(dates[:n_train],                          train_sc),
            mkdf(dates[n_train:n_train+n_val],             val_sc),
            n_train, n_val)

def build_xy(df, tau, d, T, drop_tail=0):
    vals = np.array(df)[:, 1].astype(float)
    n = len(vals)
    width = n - (d - 1) * tau - T
    if width < 1:
        raise ValueError(f"tau/d/T 过大，无法构造样本：n={n}, tau={tau}, d={d}, T={T}")
    Xn = np.stack([vals[i*tau: i*tau+width] for i in range(d)], axis=1)
    Yn = vals[T + (d - 1) * tau: T + (d - 1) * tau + width]
    arr = np.column_stack([Xn, Yn[:width]])
    if drop_tail > 0:
        arr = arr[:-drop_tail]
    return arr[:, :d].astype(np.float64), arr[:, d].astype(np.float64)

def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float64)).float().to(device)

# ============================================================
#  3. LDE 模型定义（与 chapter52 完全一致）
# ============================================================
class DriftNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True))
    def forward(self, t, x): return self.net(x)

class DiffusionNet(nn.Module):
    def __init__(self, dim, hidden=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(True), nn.Linear(hidden, 1))
    def forward(self, t, x): return self.net(x)

class LDENet(nn.Module):
    def __init__(self, dim, layer_depth=25, sigma=0.5):
        super().__init__()
        self.layer_depth = layer_depth
        self.sigma = sigma
        self.delta_t = 1.0 / layer_depth
        self.downsample = nn.Linear(dim, dim)
        self.drift      = DriftNet(dim)
        self.diffusion  = DiffusionNet(dim)

    def forward(self, x, training_diffusion=False):
        out = self.downsample(x)
        if not training_diffusion:
            t = 0.0
            diff_term = self.sigma * torch.sigmoid(self.diffusion(t, out))
            for i in range(self.layer_depth):
                t = float(i) / self.layer_depth
                noise = torch.from_numpy(
                    levy_stable.rvs(ALPHA, BETA, size=out.shape[-1], scale=0.1)
                ).to(device=x.device, dtype=x.dtype).clamp(-10, 10)
                out = out + self.drift(t, out) * self.delta_t \
                          + diff_term * (self.delta_t ** (1/ALPHA)) * noise
            return self.drift(t, out), out
        else:
            return self.diffusion(0.0, out.detach())

def nll_loss(y, mean, sigma):
    sigma = sigma.clamp(min=1e-3)
    return torch.mean(torch.log(sigma) + (y-mean)**2/(2*sigma**2))

TAIL_CUTS = list(range(N_STEPS-1, -1, -1))

def fuse(raw, attn_):
    outs = [raw[i][1] for i in range(N_STEPS)]
    aligned = [outs[i][:outs[-1].shape[0]-TAIL_CUTS[i] if TAIL_CUTS[i]>0
                        else outs[-1].shape[0]] for i in range(N_STEPS)]
    min_len = min(a.shape[0] for a in aligned)
    cat  = torch.cat([a[:min_len] for a in aligned], dim=1)
    final = attn_(cat)
    means  = [final[:, i]                          for i in range(N_STEPS)]
    sigmas = [F.softplus(final[:, N_STEPS+i])+1e-3 for i in range(N_STEPS)]
    return means, sigmas, min_len

# ============================================================
#  4. 单数据集训练 + 残差提取
# ============================================================
def run_dataset(name, csv_path):
    print(f'\n{"="*55}')
    print(f'  数据集：{name}')
    print(f'{"="*55}')
    setup_seed(SEED)

    scaler, train_df, val_df, n_train, n_val = load_and_split(csv_path)

    # 构造 PSR 数据
    x_trains, y_trains = [], []
    x_vals,   y_vals   = [], []
    for T in range(1, N_STEPS+1):
        drop = T-1
        xtr, ytr = build_xy(train_df, TAU, D, T, drop)
        xva, yva = build_xy(val_df,   TAU, D, T, drop)
        x_trains.append(xtr); y_trains.append(ytr)
        x_vals.append(xva);   y_vals.append(yva)

    X_trains = [to_tensor(x) for x in x_trains]
    Y_trains = [to_tensor(y) for y in y_trains]
    X_vals   = [to_tensor(x) for x in x_vals]
    Y_vals   = [to_tensor(y) for y in y_vals]

    # 初始化模型
    nets = nn.ModuleList([LDENet(dim=D, layer_depth=LAYER_DEPTH, sigma=SIGMA).to(device)
                          for _ in range(N_STEPS)])
    attn = nn.Sequential(nn.ReLU(True), nn.Linear(N_STEPS*D, N_STEPS*2)).to(device)

    opt_F = optim.SGD(
        [{'params': n.downsample.parameters()} for n in nets] +
        [{'params': n.drift.parameters()}      for n in nets] +
        [{'params': attn.parameters()}],
        lr=LR, momentum=0.9, weight_decay=5e-4)
    opt_G = optim.SGD(
        [{'params': n.diffusion.parameters()} for n in nets],
        lr=LR2, momentum=0.9, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 训练
    for epoch in range(1, EPOCHS+1):
        for n in nets: n.train()
        total = 0.0
        for _ in range(ITER):
            opt_F.zero_grad()
            raw   = [nets[i](X_trains[i]) for i in range(N_STEPS)]
            means, sigmas, ml = fuse(raw, attn)
            tgts  = [Y_trains[i][:ml] for i in range(N_STEPS)]
            lf    = sum(nll_loss(tgts[i], means[i], sigmas[i]) for i in range(N_STEPS))
            lf.backward()
            nn.utils.clip_grad_norm_([p for g in opt_F.param_groups for p in g['params']], 5.0)
            opt_F.step(); total += lf.item()

            opt_G.zero_grad()
            pi   = [nets[i](X_trains[i], training_diffusion=True) for i in range(N_STEPS)]
            li   = sum(criterion(pi[i], torch.full_like(pi[i], 0)) for i in range(N_STEPS))
            poin = [2*torch.randn_like(X_trains[i])+X_trains[i]   for i in range(N_STEPS)]
            po   = [nets[i](poin[i], training_diffusion=True)      for i in range(N_STEPS)]
            lo   = sum(criterion(po[i], torch.full_like(po[i], 1)) for i in range(N_STEPS))
            (li+lo).backward()
            nn.utils.clip_grad_norm_([p for g in opt_G.param_groups for p in g['params']], 5.0)
            opt_G.step()

        if epoch % 20 == 0:
            print(f'  Epoch {epoch:3d} | Loss={total/ITER:.5f}')

    # 验证集残差
    for n in nets: n.eval()
    with torch.no_grad():
        raw_v = [nets[i](X_vals[i]) for i in range(N_STEPS)]
        means_v, _, ml_v = fuse(raw_v, attn)

    residuals = {}
    stats_rows = []
    inv = lambda arr: scaler.inverse_transform(arr.reshape(-1,1)).flatten()

    for i in range(N_STEPS):
        step  = f't+{i+1}'
        n_    = ml_v
        pred  = means_v[i].cpu().numpy()[:n_]
        true_ = Y_vals[i].cpu().numpy()[:n_]

        # 原始价格空间
        pred_p = inv(pred)
        true_p = inv(true_)
        resid  = true_p - pred_p           # 残差 = 真实 - 预测（价格空间）
        resid_sc = true_ - pred            # 标准化空间残差

        residuals[step] = resid

        # 统计量
        w_stat, w_p = stats.shapiro(resid_sc[:min(5000, len(resid_sc))])
        stats_rows.append({
            'dataset': name, 'step': step,
            'mean':    float(resid_sc.mean()),
            'std':     float(resid_sc.std()),
            'skew':    float(stats.skew(resid_sc)),
            'kurt':    float(stats.kurtosis(resid_sc)),
            'shapiro_W': float(w_stat),
            'shapiro_p': float(w_p),
            'MAE':  float(mean_absolute_error(true_p, pred_p)),
            'RMSE': float(np.sqrt(mean_squared_error(true_p, pred_p))),
        })
        print(f'  {step}  mean={resid_sc.mean():.5f}  std={resid_sc.std():.5f}  '
              f'skew={stats.skew(resid_sc):.3f}  kurt={stats.kurtosis(resid_sc):.3f}')

    # 保存残差
    fname = f'residuals_{name}.npz'
    save_dict = {k: v for k, v in residuals.items()}
    np.savez(fname, **save_dict)
    print(f'  [保存] {fname}')

    return residuals, stats_rows

# ============================================================
#  5. 主循环
# ============================================================
all_residuals = {}
all_stats     = []

for name, path in DATASETS.items():
    resids, srows = run_dataset(name, path)
    all_residuals[name] = resids
    all_stats.extend(srows)

# 保存统计表
df_stats = pd.DataFrame(all_stats)
df_stats.to_csv('residual_stats.csv', index=False, encoding='utf-8-sig')
print('\n[保存] residual_stats.csv')
print(df_stats.to_string(index=False))

# ============================================================
#  6. 可视化（4 × 4 面板：每行一个数据集，每列一种图表）
# ============================================================
STEP_SHOW = 't+1'   # 可改为 t+2/t+3/t+4
COLORS    = ['#3266ad', '#1a9e75', '#8e44ad', '#e67e22']
DS_NAMES  = list(DATASETS.keys())

fig = plt.figure(figsize=(20, 16))
fig.suptitle(f'LDE 验证集残差分析（步长 {STEP_SHOW}）', fontsize=16, y=0.98)
gs  = gridspec.GridSpec(4, 4, hspace=0.45, wspace=0.35)

for row, (name, color) in enumerate(zip(DS_NAMES, COLORS)):
    resid = all_residuals[name][STEP_SHOW]

    # ── 时序折线图 ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.plot(resid, color=color, lw=0.8, alpha=0.9)
    ax1.axhline(0, color='#999', lw=0.8, ls='--')
    ax1.set_title(f'{name} — 残差序列', fontsize=10)
    ax1.set_xlabel('验证集时间步', fontsize=8)
    ax1.set_ylabel('残差（价格）', fontsize=8)
    ax1.tick_params(labelsize=7)

    # ── 分布直方图 ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.hist(resid, bins=40, color=color, alpha=0.75, edgecolor='white', lw=0.4)
    mu, sig = resid.mean(), resid.std()
    x = np.linspace(mu-4*sig, mu+4*sig, 200)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, stats.norm.pdf(x, mu, sig), 'k--', lw=1.2, alpha=0.6, label='Normal')
    ax2_twin.set_yticks([])
    ax2.set_title(f'{name} — 残差分布', fontsize=10)
    ax2.set_xlabel('残差值', fontsize=8); ax2.tick_params(labelsize=7)

    # ── QQ 图 ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[row, 2])
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist='norm')
    ax3.scatter(osm, osr, color=color, s=4, alpha=0.6)
    ax3.plot(osm, slope*np.array(osm)+intercept, 'k--', lw=1.2, alpha=0.7)
    ax3.set_title(f'{name} — QQ 图', fontsize=10)
    ax3.set_xlabel('理论分位数', fontsize=8)
    ax3.set_ylabel('样本分位数', fontsize=8)
    ax3.tick_params(labelsize=7)

    # ── 箱线图（各步长） ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[row, 3])
    box_data  = [all_residuals[name][f't+{s+1}'] for s in range(N_STEPS)]
    bp = ax4.boxplot(box_data, labels=[f't+{s+1}' for s in range(N_STEPS)],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color='white', lw=2))
    for patch in bp['boxes']:
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax4.axhline(0, color='#999', lw=0.8, ls='--')
    ax4.set_title(f'{name} — 各步长箱线图', fontsize=10)
    ax4.set_xlabel('预测步长', fontsize=8)
    ax4.set_ylabel('残差（价格）', fontsize=8)
    ax4.tick_params(labelsize=7)

plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
print('\n[保存] residual_analysis.png')
plt.show()
