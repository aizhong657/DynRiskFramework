# ============================================================
#  LDE 参数敏感性分析
#  分析维度：
#    1. Lévy 稳定分布指数 alpha ∈ [1.1, 1.2, ..., 1.9]
#    2. DiffusionNet 隐层维度 hidden  ∈ [32, 64, 128, 256]
#    3. 训练步数            epochs   ∈ [50, 100, 150, 200]
#
#  固定参数（与 chapter52.py / model_comparison.py 一致）：
#    D=22, TAU=1, N_STEPS=4, layer_depth=25, sigma=0.5
#    lr=1e-4, lr2=0.01, momentum=0.9, weight_decay=5e-4
#    ITER=50（每 epoch 迭代次数）
#
#  评估指标：MSE / RMSE / MAE（原始价格空间，测试集）
#  输出：lde_sensitivity_results.csv
# ============================================================

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import warnings, itertools, os, sys
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import levy_stable
import torch
import torch.nn.functional as F
from torch import nn, optim

# ============================================================
#  0. 固定超参数
# ============================================================
DATA_PATH   = DATA_DIR / "sz50_index_data.csv"
N_STEPS     = 4
D, TAU      = 22, 1
LAYER_DEPTH = 25
SIGMA       = 0.5
LR, LR2     = 1e-4, 0.01
ITER        = 50
SEED        = 4

# ── 敏感性扫描范围 ──────────────────────────────────────────
ALPHAS      = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
HIDDEN_DIMS = [32, 64, 128, 256]    # DiffusionNet 隐层维度
EPOCHS_LIST = [50, 100, 150, 200]

# 基准值（与对比实验 chapter52.py 一致）
BASELINE_ALPHA  = 1.2
BASELINE_HIDDEN = 100   # chapter52 原始值
BASELINE_EPOCHS = 100

# ============================================================
#  1. 随机种子 & 设备
# ============================================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {device}')
print(f'[Python] {sys.version.split()[0]}  NumPy={np.__version__}  Pandas={pd.__version__}  Torch={torch.__version__}')

def check_env():
    if not os.path.exists(DATA_PATH):
        print(f'[Error] 数据文件不存在: {DATA_PATH}')
        sys.exit(1)
    try:
        head = pd.read_csv(DATA_PATH, nrows=5)
    except Exception as e:
        print(f'[Error] 读取数据文件失败: {e}')
        sys.exit(1)
    need = {'date', 'close'}
    cols_lower = {c.lower() for c in head.columns}
    if not need.issubset(cols_lower):
        print(f'[Error] 数据列需包含: date, close; 实际列: {list(head.columns)}')
        sys.exit(1)
    out_dir = os.path.dirname(__file__)
    try:
        test_path = os.path.join(out_dir, '.io_test.tmp')
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('ok')
        os.remove(test_path)
        print(f'[IO] 输出目录可写: {out_dir}')
    except Exception as e:
        print(f'[Error] 输出目录不可写: {e}')
        sys.exit(1)

check_env()

# ============================================================
#  2. 数据加载与预处理
# ============================================================
data = pd.read_csv(DATA_PATH)
data['date'] = pd.to_datetime(data['date'])

stk_data = pd.DataFrame({
    'Date':  data['date'].values,
    'Close': data['close'].values,
})

TRAIN_RATIO, VAL_RATIO = 0.70, 0.10
n = len(stk_data)
n_train = int(np.ceil(TRAIN_RATIO * n))
n_val   = int(np.ceil(VAL_RATIO   * n))
n_test  = n - n_train - n_val

training_set = stk_data.iloc[:n_train,              1:2].values
testing_set  = stk_data.iloc[n_train+n_val:,        1:2].values

scaler = StandardScaler()
train_scaled = scaler.fit_transform(training_set).flatten()
test_scaled  = scaler.transform(testing_set).flatten()

def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1,1)).flatten()

def make_df(dates, values):
    return pd.DataFrame({'Date': dates, 'closescale': values})

train_df = make_df(stk_data['Date'].values[:n_train], train_scaled)
test_df  = make_df(stk_data['Date'].values[n_train+n_val:n_train+n_val+n_test], test_scaled)

print(f'[Data] 训练: {len(train_scaled)}  测试: {len(test_scaled)}')

# ============================================================
#  3. 相空间重构（与 chapter52 完全一致）
# ============================================================
def PhaSpaRecon(df, tau, d, T):
    values = np.array(df)[:, 1].astype(float)
    n = len(values)
    width = n - (d-1)*tau - 1
    Xn1 = np.stack([values[i*tau: i*tau+width] for i in range(d)], axis=1)
    Yn1 = values[T + (d-1)*tau: T + (d-1)*tau + width]
    Xn = pd.DataFrame(Xn1)
    Yn = pd.DataFrame(Yn1, columns=[0])
    return pd.concat([Xn, Yn], axis=1)

def build_xy(df, tau, d, T, drop_tail=0):
    X = PhaSpaRecon(df, tau, d, T).values
    if drop_tail > 0:
        X = X[:-drop_tail]
    return X[:, :d].astype(np.float64), X[:, d].astype(np.float64)

def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float64)).float().to(device)

# 构造 PSR 数据（固定，不随参数变化）
x_trains, y_trains, x_tests, y_tests = [], [], [], []
for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr, ytr = build_xy(train_df, TAU, D, T, drop)
    xte, yte = build_xy(test_df,  TAU, D, T, drop)
    x_trains.append(xtr); y_trains.append(ytr)
    x_tests.append(xte);  y_tests.append(yte)

X_trains = [to_tensor(x) for x in x_trains]
Y_trains = [to_tensor(y) for y in y_trains]
X_tests  = [to_tensor(x) for x in x_tests]
Y_tests  = [to_tensor(y) for y in y_tests]

TAIL_CUTS = list(range(N_STEPS - 1, -1, -1))

# ============================================================
#  4. 模型定义（可参数化 DiffusionNet 隐层维度）
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
            nn.Linear(dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, 1),
        )
    def forward(self, t, x): return self.net(x)

class LDENet(nn.Module):
    """LDE（Lévy-SDE）网络，alpha / hidden 可配置"""
    def __init__(self, dim, layer_depth=25, sigma=0.5, alpha=1.2, diff_hidden=100):
        super().__init__()
        self.layer_depth = layer_depth
        self.sigma       = sigma
        self.delta_t     = 1.0 / layer_depth
        self.alpha       = alpha
        self.downsample  = nn.Linear(dim, dim)
        self.drift       = DriftNet(dim)
        self.diffusion   = DiffusionNet(dim, hidden=diff_hidden)

    def forward(self, x, training_diffusion=False):
        out = self.downsample(x)
        if not training_diffusion:
            t = 0.0
            diff_term = self.sigma * torch.sigmoid(self.diffusion(t, out))
            for i in range(self.layer_depth):
                t = float(i) / self.layer_depth
                noise = torch.from_numpy(
                    levy_stable.rvs(self.alpha, 0, size=out.shape[-1], scale=0.1)
                ).to(device=x.device, dtype=x.dtype).clamp(-10, 10)
                out = out + self.drift(t, out) * self.delta_t \
                          + diff_term * (self.delta_t ** (1 / self.alpha)) * noise
            return self.drift(t, out), out
        else:
            return self.diffusion(0.0, out.detach())

# ============================================================
#  5. 损失 & 前向公共逻辑
# ============================================================
def nll_loss(y, mean, sigma):
    sigma = sigma.clamp(min=1e-3)
    return torch.mean(torch.log(sigma) + (y - mean)**2 / (2*sigma**2))

def forward_all(nets_, training_diffusion=False):
    return [nets_[i](X_trains[i], training_diffusion=training_diffusion)
            for i in range(N_STEPS)]

def fuse_outputs(raw_outs, attn_):
    outs    = [raw_outs[i][1] for i in range(N_STEPS)]
    aligned = [outs[i][:outs[-1].shape[0] - TAIL_CUTS[i] if TAIL_CUTS[i] > 0
                        else outs[-1].shape[0]]
               for i in range(N_STEPS)]
    min_len = min(a.shape[0] for a in aligned)
    cat     = torch.cat([a[:min_len] for a in aligned], dim=1)
    final   = attn_(cat)
    means   = [final[:, i] for i in range(N_STEPS)]
    sigmas  = [F.softplus(final[:, N_STEPS+i]) + 1e-3 for i in range(N_STEPS)]
    return means, sigmas, min_len

# ============================================================
#  6. 单次训练 & 评估
# ============================================================
def train_and_eval(alpha, diff_hidden, epochs, run_label=''):
    setup_seed(SEED)
    nets_ = nn.ModuleList([
        LDENet(dim=D, layer_depth=LAYER_DEPTH, sigma=SIGMA,
               alpha=alpha, diff_hidden=diff_hidden).to(device)
        for _ in range(N_STEPS)
    ])
    attn_ = nn.Sequential(
        nn.ReLU(True),
        nn.Linear(N_STEPS * D, N_STEPS * 2),
    ).to(device)

    opt_F = optim.SGD(
        [{'params': n.downsample.parameters()} for n in nets_] +
        [{'params': n.drift.parameters()}      for n in nets_] +
        [{'params': attn_.parameters()}],
        lr=LR, momentum=0.9, weight_decay=5e-4
    )
    opt_G = optim.SGD(
        [{'params': n.diffusion.parameters()} for n in nets_],
        lr=LR2, momentum=0.9, weight_decay=5e-4
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        for n in nets_: n.train()
        total = 0.0
        for _ in range(ITER):
            opt_F.zero_grad()
            raw   = forward_all(nets_)
            means, sigmas, min_len = fuse_outputs(raw, attn_)
            tgts  = [Y_trains[i][:min_len] for i in range(N_STEPS)]
            lf    = sum(nll_loss(tgts[i], means[i], sigmas[i]) for i in range(N_STEPS))
            lf.backward()
            nn.utils.clip_grad_norm_([p for g in opt_F.param_groups for p in g['params']], 5.0)
            opt_F.step(); total += lf.item()

            opt_G.zero_grad()
            pi    = [nets_[i](X_trains[i], training_diffusion=True) for i in range(N_STEPS)]
            li    = sum(criterion(pi[i], torch.full_like(pi[i], 0)) for i in range(N_STEPS))
            po_in = [2*torch.randn_like(X_trains[i])+X_trains[i]    for i in range(N_STEPS)]
            po    = [nets_[i](po_in[i], training_diffusion=True)     for i in range(N_STEPS)]
            lo    = sum(criterion(po[i], torch.full_like(po[i], 1))  for i in range(N_STEPS))
            (li + lo).backward()
            nn.utils.clip_grad_norm_([p for g in opt_G.param_groups for p in g['params']], 5.0)
            opt_G.step()

        if epoch % 50 == 0:
            print(f'    {run_label} Epoch {epoch:3d} | Loss={total/ITER:.5f}')

    # 评估
    for n in nets_: n.eval()
    results = {}
    with torch.no_grad():
        raw_te = [nets_[i](X_tests[i]) for i in range(N_STEPS)]
        means_te, _, min_len_te = fuse_outputs(raw_te, attn_)
        tgts_te = [Y_tests[i][:min_len_te] for i in range(N_STEPS)]
        for i in range(N_STEPS):
            step     = i + 1
            fc_price = inv(means_te[i].cpu().numpy()[:min_len_te])
            gt_price = inv(tgts_te[i].cpu().numpy()[:min_len_te])
            mse  = mean_squared_error(gt_price, fc_price)
            results[f't+{step}'] = dict(
                MSE=mse, RMSE=np.sqrt(mse),
                MAE=mean_absolute_error(gt_price, fc_price)
            )
    return results

# ============================================================
#  7. 扫描 A：alpha（固定 hidden=100, epochs=100）
# ============================================================
print('\n' + '='*60)
print('  扫描 A：alpha（hidden=100, epochs=100）')
print('='*60)
records_alpha = []
for alpha in ALPHAS:
    label = f'α={alpha:.1f}'
    print(f'\n  [{label}]')
    res = train_and_eval(alpha, diff_hidden=100, epochs=100, run_label=label)
    for step_label, m in res.items():
        records_alpha.append({'alpha': alpha, 'step': step_label, **m})
    print(f'    MAE(t+1)={res["t+1"]["MAE"]:.5f}')

# ============================================================
#  8. 扫描 B：hidden（固定 alpha=1.2, epochs=100）
# ============================================================
print('\n' + '='*60)
print('  扫描 B：DiffusionNet 隐层维度（alpha=1.2, epochs=100）')
print('='*60)
records_hidden = []
for hidden in HIDDEN_DIMS:
    label = f'hidden={hidden}'
    print(f'\n  [{label}]')
    res = train_and_eval(alpha=1.2, diff_hidden=hidden, epochs=100, run_label=label)
    for step_label, m in res.items():
        records_hidden.append({'hidden': hidden, 'step': step_label, **m})
    print(f'    MAE(t+1)={res["t+1"]["MAE"]:.5f}')

# ============================================================
#  9. 扫描 C：epochs（固定 alpha=1.2, hidden=100）
# ============================================================
print('\n' + '='*60)
print('  扫描 C：训练步数（alpha=1.2, hidden=100）')
print('='*60)
records_epochs = []
for epochs in EPOCHS_LIST:
    label = f'epochs={epochs}'
    print(f'\n  [{label}]')
    res = train_and_eval(alpha=1.2, diff_hidden=100, epochs=epochs, run_label=label)
    for step_label, m in res.items():
        records_epochs.append({'epochs': epochs, 'step': step_label, **m})
    print(f'    MAE(t+1)={res["t+1"]["MAE"]:.5f}')

# ============================================================
#  10. 保存结果
# ============================================================
out_dir = os.path.dirname(__file__)
pd.DataFrame(records_alpha ).to_csv(os.path.join(out_dir, 'lde_sensitivity_alpha.csv'),  index=False, encoding='utf-8-sig')
pd.DataFrame(records_hidden).to_csv(os.path.join(out_dir, 'lde_sensitivity_hidden.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(records_epochs).to_csv(os.path.join(out_dir, 'lde_sensitivity_epochs.csv'), index=False, encoding='utf-8-sig')

print('\n[保存] lde_sensitivity_alpha.csv / lde_sensitivity_hidden.csv / lde_sensitivity_epochs.csv')

# 打印汇总透视表
print('\n===== alpha 敏感性（MAE，t+1）=====')
df_a = pd.DataFrame(records_alpha)
print(df_a[df_a['step']=='t+1'][['alpha','MAE']].to_string(index=False))

print('\n===== hidden 敏感性（MAE，t+1）=====')
df_h = pd.DataFrame(records_hidden)
print(df_h[df_h['step']=='t+1'][['hidden','MAE']].to_string(index=False))

print('\n===== epochs 敏感性（MAE，t+1）=====')
df_e = pd.DataFrame(records_epochs)
print(df_e[df_e['step']=='t+1'][['epochs','MAE']].to_string(index=False))
