# ============================================================
#  上证50指数多步预测 —— 五模型对比实验
#  模型：
#    ARIMA       —— 统计基线（自动定阶）
#    LSTM        —— 双层循环神经网络
#    SDE         —— 高斯噪声随机微分方程网络（α=2）
#    LDE         —— Lévy 噪声随机微分方程网络（α=1.2，chapter52）
#    LDE-CorrDim —— LDE + 关联维数 + LinearFusion（α=1.9，chapter518）
#  评估指标：MSE / RMSE / MAE / MAPE
#  数据划分：70% 训练 / 10% 验证 / 20% 测试
# ============================================================

# ── 标准库 ──────────────────────────────────────────────────
from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import os
import warnings
import argparse

warnings.filterwarnings('ignore')

# ── 第三方库 ────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import levy_stable
import torch
import torch.nn.functional as F
from torch import nn, optim
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# ============================================================
#  0. 超参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',    type=int,   default=100)
parser.add_argument('--n_steps',   type=int,   default=4,    help='预测步数')
parser.add_argument('--lr',        type=float, default=1e-4)
parser.add_argument('--lr2',       type=float, default=0.01)
parser.add_argument('--seed',      type=int,   default=4)
parser.add_argument('--gpu',       type=int,   default=0)
parser.add_argument('--data_path',     type=str, default=DATA_DIR / "sz50_index_data.csv")
parser.add_argument('--corr_dim_path', type=str, default=CORR_PATH,
                    help='关联维数特征文件路径（LDE-CorrDim 使用）')
parser.add_argument('--embed_dim_cd',  type=int, default=19,
                    help='LDE-CorrDim 的嵌入维数（chapter518 默认 19）')
parser.add_argument('--metric_space',  type=str, default='standardized',
                    choices=['price', 'log_return', 'standardized'],
                    help='评估空间：price（价格）、log_return（对数收益率）、standardized（标准化值）')
args = parser.parse_args()

# ── 随机种子 & 设备 ─────────────────────────────────────────
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {device}')

N_STEPS = args.n_steps

# ============================================================
#  1. 数据加载与预处理
# ============================================================
data = pd.read_csv(args.data_path)
data = data[['date', 'open', 'close', 'high', 'low', 'volume']]
data['date'] = pd.to_datetime(data['date'])

stk_data = pd.DataFrame({
    'Date':  data['date'].values,
    'Close': data['close'].values,
})

TRAIN_RATIO, VAL_RATIO = 0.70, 0.10
n      = len(stk_data)
n_train = int(np.ceil(TRAIN_RATIO * n))
n_val   = int(np.ceil(VAL_RATIO   * n))
n_test  = n - n_train - n_val

training_set   = stk_data.iloc[:n_train,              1:2].values
validation_set = stk_data.iloc[n_train:n_train+n_val, 1:2].values
testing_set    = stk_data.iloc[n_train+n_val:,        1:2].values

print(f'[Data] 训练: {training_set.shape}  验证: {validation_set.shape}  测试: {testing_set.shape}')

scaler = StandardScaler()
train_scaled = scaler.fit_transform(training_set).flatten()
val_scaled   = scaler.transform(validation_set).flatten()
test_scaled  = scaler.transform(testing_set).flatten()

def make_df(dates, values):
    return pd.DataFrame({'Date': dates, 'closescale': values})

train_df = make_df(stk_data['Date'].values[:n_train],                          train_scaled)
val_df   = make_df(stk_data['Date'].values[n_train:n_train+n_val],             val_scaled)
test_df  = make_df(stk_data['Date'].values[n_train+n_val:n_train+n_val+n_test], test_scaled)

# ============================================================
#  2. 相空间重构（LDE / SDE 使用）
# ============================================================
D, TAU = 22, 1   # 嵌入维数 & 时延（与 chapter52.py 一致）

def PhaSpaRecon(df, tau, d, T):
    values = np.array(df)[:, 1].astype(float)
    n = len(values)
    width = n - (d-1)*tau - 1
    Xn1 = np.stack([values[i*tau: i*tau+width] for i in range(d)], axis=1)
    Yn1 = values[T + (d-1)*tau: T + (d-1)*tau + width]
    Xn = pd.DataFrame(Xn1)
    Yn = pd.DataFrame(Yn1, columns=[0])
    X  = pd.concat([Xn, Yn], axis=1)
    return Xn, Yn, X

def build_xy(df, tau, d, T, drop_tail=0):
    _, _, X = PhaSpaRecon(df, tau, d, T)
    arr = X.values
    if drop_tail > 0:
        arr = arr[:-drop_tail]
    return arr[:, :d].astype(np.float64), arr[:, d].astype(np.float64)

x_trains, y_trains = [], []
x_vals,   y_vals   = [], []
x_tests,  y_tests  = [], []
for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr, ytr = build_xy(train_df, TAU, D, T, drop)
    xva, yva = build_xy(val_df,   TAU, D, T, drop)
    xte, yte = build_xy(test_df,  TAU, D, T, drop)
    x_trains.append(xtr); y_trains.append(ytr)
    x_vals.append(xva);   y_vals.append(yva)
    x_tests.append(xte);  y_tests.append(yte)

def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float64)).float().to(device)

X_trains = [to_tensor(x) for x in x_trains]
Y_trains = [to_tensor(y) for y in y_trains]
X_vals   = [to_tensor(x) for x in x_vals]
Y_vals   = [to_tensor(y) for y in y_vals]
X_tests  = [to_tensor(x) for x in x_tests]
Y_tests  = [to_tensor(y) for y in y_tests]

# ============================================================
#  3. 评估指标
# ============================================================
def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1,1)).flatten()

def log_return(curr, prev, eps=1e-12):
    curr = np.asarray(curr, dtype=np.float64)
    prev = np.asarray(prev, dtype=np.float64)
    return np.log((curr + eps) / (prev + eps))

def prev_series_from_psr(df, tau, d, T, drop_tail=0):
    values = np.array(df)[:, 1].astype(float)
    n = len(values)
    width = n - (d-1)*tau - 1
    prev_start = (d-1)*tau + T - 1
    prev = values[prev_start: prev_start + width]
    if drop_tail > 0:
        prev = prev[:-drop_tail]
    return prev.astype(np.float64)

prev_tests = [prev_series_from_psr(test_df, TAU, D, T, T-1) for T in range(1, N_STEPS + 1)]

def eval_metrics(y_true, y_pred, prev=None, label=''):
    if args.metric_space == 'log_return':
        if prev is None:
            raise ValueError('metric_space=log_return 需要 prev_price')
        min_len = min(len(y_true), len(y_pred), len(prev))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        prev   = prev[:min_len]
        gt   = log_return(y_true, prev)
        pred = log_return(y_pred, prev)
        return evaluate(gt, pred, label=label)
    return evaluate(y_true, y_pred, label=label)

def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-8
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate(y_true, y_pred, label=''):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mp   = mape(y_true, y_pred)
    if label:
        print(f'  {label:>15s} | MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mp:.2f}%')
    return dict(MSE=mse, RMSE=rmse, MAE=mae, MAPE=mp)

# ============================================================
#  4-A. 基线模型：ARIMA（每步独立拟合，auto 选阶 via AIC 网格）
# ============================================================
print('\n' + '='*60)
print('  ARIMA 基线（auto_arima 自动定阶）')
print('='*60)

# auto_arima：在训练集上自动搜索最优 (p,d,q)，仅搜索一次供所有步长复用
print('  [auto_arima] 正在搜索最优阶数...')
from pmdarima import auto_arima as _auto_arima
auto_model = _auto_arima(
    train_scaled,
    start_p=1, max_p=8,
    start_q=0, max_q=4,
    d=None,          # 自动差分阶数（ADF 检验）
    seasonal=False,
    information_criterion='aic',
    stepwise=True,   # 逐步搜索，速度快
    suppress_warnings=True,
    error_action='ignore',
)
best_order = auto_model.order
print(f'  [auto_arima] 最优阶数 ARIMA{best_order}')

def arima_one_shot(train_vals, test_len, step, order):
    """用选定阶数一次性预测，取第 step 步预测值对齐测试集。"""
    fit = ARIMA(train_vals, order=order).fit()
    fc  = fit.forecast(steps=test_len + step)
    return fc[step-1: step-1+test_len]

arima_results = {}
for i in range(N_STEPS):
    step      = i + 1
    yte       = y_tests[i]
    fc_scaled = arima_one_shot(train_scaled, len(yte), step, best_order)
    if args.metric_space == 'standardized':
        gt_sc   = yte[:len(fc_scaled)]
        pred_sc = fc_scaled[:len(gt_sc)]
        metrics = eval_metrics(gt_sc, pred_sc, label=f't+{step}')
    elif args.metric_space == 'price':
        fc_price  = inv(fc_scaled[:len(yte)])
        gt_price  = inv(yte[:len(fc_scaled)])
        metrics   = eval_metrics(gt_price, fc_price, label=f't+{step}')
    else:
        fc_price  = inv(fc_scaled[:len(yte)])
        gt_price  = inv(yte[:len(fc_scaled)])
        prev_price = inv(prev_tests[i][:len(fc_scaled)])
        metrics   = eval_metrics(gt_price, fc_price, prev=prev_price, label=f't+{step}')
    arima_results[f't+{step}'] = metrics

# ============================================================
#  4-B. LSTM（双层，seq_len=D，多步独立训练）
# ============================================================
print('\n' + '='*60)
print('  LSTM 预测（双层，seq_len=D=22）')
print('='*60)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)   # 取最后时间步


def make_lstm_data(scaled_vals, seq_len, step):
    X, Y = [], []
    for i in range(len(scaled_vals) - seq_len - step + 1):
        X.append(scaled_vals[i: i+seq_len])
        Y.append(scaled_vals[i+seq_len+step-1])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def make_lstm_data_with_prev(scaled_vals, seq_len, step):
    X, Y, P = [], [], []
    for i in range(len(scaled_vals) - seq_len - step + 1):
        X.append(scaled_vals[i: i+seq_len])
        tgt_idx = i + seq_len + step - 1
        Y.append(scaled_vals[tgt_idx])
        P.append(scaled_vals[tgt_idx - 1])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(P, dtype=np.float32)

SEQ_LEN_LSTM = D        # 与相空间重构嵌入维数一致
LSTM_EPOCHS  = args.epochs
LSTM_LR      = 1e-3
LSTM_BATCH   = 64

lstm_results = {}

for i in range(N_STEPS):
    step = i + 1
    # 构造数据
    X_tr, Y_tr = make_lstm_data(train_scaled, SEQ_LEN_LSTM, step)
    X_te, Y_te, P_te = make_lstm_data_with_prev(
        np.concatenate([train_scaled[-SEQ_LEN_LSTM:], test_scaled]),
        SEQ_LEN_LSTM, step
    )
    # 对齐测试集长度（与相空间数据一致）
    tgt_len = min(len(y_tests[i]), len(Y_te))
    X_te, Y_te, P_te = X_te[:tgt_len], Y_te[:tgt_len], P_te[:tgt_len]

    # DataLoader
    tr_ds  = torch.utils.data.TensorDataset(
        torch.tensor(X_tr).unsqueeze(-1).to(device),
        torch.tensor(Y_tr).to(device)
    )
    tr_dl  = torch.utils.data.DataLoader(tr_ds, batch_size=LSTM_BATCH, shuffle=True)

    model_lstm = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(device)
    opt_lstm   = optim.Adam(model_lstm.parameters(), lr=LSTM_LR)
    crit_lstm  = nn.MSELoss()

    for ep in range(1, LSTM_EPOCHS + 1):
        model_lstm.train()
        for xb, yb in tr_dl:
            opt_lstm.zero_grad()
            loss = crit_lstm(model_lstm(xb), yb)
            loss.backward()
            opt_lstm.step()

    model_lstm.eval()
    with torch.no_grad():
        X_te_t   = torch.tensor(X_te).unsqueeze(-1).to(device)
        pred_sc  = model_lstm(X_te_t).cpu().numpy()

    if args.metric_space == 'standardized':
        metrics = eval_metrics(Y_te[:tgt_len], pred_sc[:tgt_len], label=f't+{step}')
    elif args.metric_space == 'price':
        fc_price = inv(pred_sc[:tgt_len])
        gt_price = inv(Y_te[:tgt_len])
        metrics  = eval_metrics(gt_price, fc_price, label=f't+{step}')
    else:
        fc_price = inv(pred_sc[:tgt_len])
        gt_price = inv(Y_te[:tgt_len])
        prev_price = inv(P_te[:tgt_len])
        metrics  = eval_metrics(gt_price, fc_price, prev=prev_price, label=f't+{step}')
    lstm_results[f't+{step}'] = metrics

# ============================================================
#  4-C. SDE（高斯噪声，无 Lévy）
# ============================================================
print('\n' + '='*60)
print('  SDE（Gaussian 噪声）')
print('='*60)

ALPHA_GAUSS = 2.0   # α=2 → 高斯分布（SDE 基础版本）

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

class SDENetBase(nn.Module):
    """通用 SDE 网络：alpha 参数控制噪声类型（alpha=2 → 高斯；alpha<2 → Lévy）"""
    def __init__(self, dim, layer_depth=25, sigma=0.5, alpha=2.0):
        super().__init__()
        self.layer_depth = layer_depth
        self.sigma       = sigma
        self.delta_t     = 1.0 / layer_depth
        self.alpha       = alpha

        self.downsample = nn.Linear(dim, dim)
        self.drift      = DriftNet(dim)
        self.diffusion  = DiffusionNet(dim)

    def forward(self, x, training_diffusion=False):
        out = self.downsample(x)
        if not training_diffusion:
            t         = 0.0
            diff_term = self.sigma * torch.sigmoid(self.diffusion(t, out))
            for i in range(self.layer_depth):
                t = float(i) / self.layer_depth
                if self.alpha == 2.0:
                    noise = torch.randn_like(out)
                else:
                    noise = torch.from_numpy(
                        levy_stable.rvs(self.alpha, 0, size=out.shape[-1], scale=0.1)
                    ).to(device=x.device, dtype=x.dtype).clamp(-10, 10)
                out = out + self.drift(t, out) * self.delta_t \
                          + diff_term * (self.delta_t ** (1 / self.alpha)) * noise
            return self.drift(t, out), out
        else:
            return self.diffusion(0.0, out.detach())


def build_nets(alpha_val):
    nets_     = nn.ModuleList([
        SDENetBase(dim=D, layer_depth=25, sigma=0.5, alpha=alpha_val).to(device)
        for _ in range(N_STEPS)
    ])
    attn_     = nn.Sequential(
        nn.ReLU(True),
        nn.Linear(N_STEPS * D, N_STEPS * 2),
    ).to(device)
    return nets_, attn_


TAIL_CUTS = list(range(N_STEPS - 1, -1, -1))
ITER      = 50
ITER_TEST = 1

def forward_all(nets_, X_list, training_diffusion=False):
    return [nets_[i](X_list[i], training_diffusion=training_diffusion)
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

def nll_loss(y, mean, sigma):
    sigma = sigma.clamp(min=1e-3)
    return torch.mean(torch.log(sigma) + (y - mean)**2 / (2*sigma**2))

def mse_loss_(y, mean):
    return torch.mean((y - mean)**2)

def get_targets(Y_list, min_len):
    return [Y_list[i][:min_len] for i in range(N_STEPS)]


def train_model(nets_, attn_, alpha_label, epochs=args.epochs):
    """通用训练函数，SDE / LDE 共用"""
    opt_F = optim.SGD(
        [{'params': n.downsample.parameters()} for n in nets_] +
        [{'params': n.drift.parameters()}      for n in nets_] +
        [{'params': attn_.parameters()}],
        lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    opt_G = optim.SGD(
        [{'params': n.diffusion.parameters()} for n in nets_],
        lr=args.lr2, momentum=0.9, weight_decay=5e-4
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        for n in nets_: n.train()
        total = 0.0
        for _ in range(ITER):
            # Drift/Attention
            opt_F.zero_grad()
            raw   = forward_all(nets_, X_trains)
            means, sigmas, min_len = fuse_outputs(raw, attn_)
            tgts  = get_targets(Y_trains, min_len)
            lf    = sum(nll_loss(tgts[i], means[i], sigmas[i]) for i in range(N_STEPS))
            lf.backward(); nn.utils.clip_grad_norm_(
                [p for g in opt_F.param_groups for p in g['params']], 5.0)
            opt_F.step(); total += lf.item()

            # Diffusion
            opt_G.zero_grad()
            pi = [nets_[i](X_trains[i], training_diffusion=True) for i in range(N_STEPS)]
            li = sum(criterion(pi[i], torch.full_like(pi[i], 0)) for i in range(N_STEPS))
            po_inp = [2*torch.randn_like(X_trains[i])+X_trains[i] for i in range(N_STEPS)]
            po = [nets_[i](po_inp[i], training_diffusion=True)    for i in range(N_STEPS)]
            lo = sum(criterion(po[i], torch.full_like(po[i], 1))  for i in range(N_STEPS))
            (li + lo).backward(); nn.utils.clip_grad_norm_(
                [p for g in opt_G.param_groups for p in g['params']], 5.0)
            opt_G.step()

        if epoch % 20 == 0:
            print(f'  [{alpha_label}] Epoch {epoch:3d} | Loss={total/ITER:.5f}')

    # 测试集评估
    for n in nets_: n.eval()
    results = {}
    with torch.no_grad():
        raw   = forward_all(nets_, X_tests)
        means, _, min_len = fuse_outputs(raw, attn_)
        tgts  = get_targets(Y_tests, min_len)
        for i in range(N_STEPS):
            step     = i + 1
            if args.metric_space == 'standardized':
                gt_sc   = tgts[i].cpu().numpy()[:min_len]
                pred_sc = means[i].cpu().numpy()[:min_len]
                metrics = eval_metrics(gt_sc, pred_sc, label=f't+{step}')
            elif args.metric_space == 'price':
                fc_price = inv(means[i].cpu().numpy()[:min_len])
                gt_price = inv(tgts[i].cpu().numpy()[:min_len])
                metrics  = eval_metrics(gt_price, fc_price, label=f't+{step}')
            else:
                fc_price = inv(means[i].cpu().numpy()[:min_len])
                gt_price = inv(tgts[i].cpu().numpy()[:min_len])
                prev_price = inv(prev_tests[i][:min_len])
                metrics  = eval_metrics(gt_price, fc_price, prev=prev_price, label=f't+{step}')
            results[f't+{step}'] = metrics
    return results


# ── 训练 SDE（Gaussian, alpha=2）──────────────────────────────
sde_nets, sde_attn = build_nets(alpha_val=2.0)
sde_results = train_model(sde_nets, sde_attn, alpha_label='SDE-Gaussian')

# ============================================================
#  4-D. LDE（Lévy 噪声，alpha=1.2，与 chapter52.py 一致）
# ============================================================
print('\n' + '='*60)
print('  LDE（Lévy 噪声，α=1.2）')
print('='*60)

lde_nets, lde_attn = build_nets(alpha_val=1.2)
lde_results = train_model(lde_nets, lde_attn, alpha_label='LDE-Levy')

# ============================================================
#  4-E. LDE-CorrDim（chapter518：α=1.9 Lévy + 关联维数 + LinearFusion）
# ============================================================
print('\n' + '='*60)
print('  LDE-CorrDim（Lévy α=1.9 + 关联维数 + LinearFusion）')
print('='*60)

# ── 关联维数特征加载 ────────────────────────────────────────
D_CD  = args.embed_dim_cd   # 嵌入维数（chapter518 默认 19）
TAU_CD = 1
ALPHA_CD, BETA_CD = 1.9, 0
CORR_DIM = 10               # 关联维数列数（cd_1 … cd_10）

corr_dim_raw = pd.read_csv(args.corr_dim_path)
corr_dim_raw.iloc[:, 0] = pd.to_datetime(corr_dim_raw.iloc[:, 0])
corr_dim_raw.columns    = ['Date'] + [f'cd_{i}' for i in range(1, CORR_DIM + 1)]
CORR_DIM_COLS = [f'cd_{i}' for i in range(1, CORR_DIM + 1)]

def align_corr_dim(date_series, corr_df):
    df_dates = pd.DataFrame({'Date': pd.to_datetime(date_series)})
    corr_df  = corr_df.copy(); corr_df['Date'] = pd.to_datetime(corr_df['Date'])
    merged   = df_dates.merge(corr_df, on='Date', how='left')
    merged[CORR_DIM_COLS] = merged[CORR_DIM_COLS].ffill().fillna(0.0)
    return merged[CORR_DIM_COLS].values.astype(np.float64)

corr_train_full = align_corr_dim(stk_data['Date'].values[:n_train],                          corr_dim_raw)
corr_val_full   = align_corr_dim(stk_data['Date'].values[n_train:n_train+n_val],              corr_dim_raw)
corr_test_full  = align_corr_dim(stk_data['Date'].values[n_train+n_val:n_train+n_val+n_test], corr_dim_raw)

PSR_OFFSET_CD = (D_CD - 1) * TAU_CD

def slice_cd(cd_full, n_x, drop):
    start = PSR_OFFSET_CD
    end   = start + n_x + drop
    arr   = cd_full[start:end]
    if drop > 0:
        arr = arr[:-drop]
    return arr[:n_x]

# ── 相空间重构（D_CD 嵌入维数） ─────────────────────────────
x_trains_cd, y_trains_cd = [], []
x_vals_cd,   y_vals_cd   = [], []
x_tests_cd,  y_tests_cd  = [], []
cd_trains_t, cd_vals_t, cd_tests_t = [], [], []

for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr, ytr = build_xy(train_df, TAU_CD, D_CD, T, drop)
    xva, yva = build_xy(val_df,   TAU_CD, D_CD, T, drop)
    xte, yte = build_xy(test_df,  TAU_CD, D_CD, T, drop)
    x_trains_cd.append(xtr); y_trains_cd.append(ytr)
    x_vals_cd.append(xva);   y_vals_cd.append(yva)
    x_tests_cd.append(xte);  y_tests_cd.append(yte)
    cd_trains_t.append(to_tensor(slice_cd(corr_train_full, xtr.shape[0], drop)))
    cd_vals_t.append(  to_tensor(slice_cd(corr_val_full,   xva.shape[0], drop)))
    cd_tests_t.append( to_tensor(slice_cd(corr_test_full,  xte.shape[0], drop)))

X_trains_cd = [to_tensor(x) for x in x_trains_cd]
Y_trains_cd = [to_tensor(y) for y in y_trains_cd]
X_tests_cd  = [to_tensor(x) for x in x_tests_cd]
Y_tests_cd  = [to_tensor(y) for y in y_tests_cd]

prev_tests_cd = [prev_series_from_psr(test_df, TAU_CD, D_CD, T, T-1) for T in range(1, N_STEPS + 1)]

# ── LinearFusion 融合层（chapter518 原版） ──────────────────
class LinearFusion(nn.Module):
    """拼接 SDE 隐状态 + 关联维数 → ReLU → Linear → (N, N_STEPS*2)"""
    def __init__(self, n_steps, sde_dim, corr_dim, out_dim=None):
        super().__init__()
        in_dim  = n_steps * sde_dim + corr_dim
        out_dim = out_dim or n_steps * 2
        self.net = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim))

    def forward(self, sde_tokens, corr_feat):
        x = torch.cat([sde_tokens, corr_feat], dim=1)
        return self.net(x)

# ── SDENet（α=1.9 Lévy） ────────────────────────────────────
class SDENetCD(nn.Module):
    def __init__(self, dim, layer_depth=25, sigma=0.5):
        super().__init__()
        self.layer_depth = layer_depth
        self.sigma       = sigma
        self.delta_t     = 1.0 / layer_depth
        self.downsample  = nn.Linear(dim, dim)
        self.drift       = DriftNet(dim)
        self.diffusion   = DiffusionNet(dim)

    def forward(self, x, training_diffusion=False):
        out = self.downsample(x)
        if not training_diffusion:
            t         = 0.0
            diff_term = self.sigma * torch.sigmoid(self.diffusion(t, out))
            for i in range(self.layer_depth):
                t = float(i) / self.layer_depth
                noise = torch.from_numpy(
                    levy_stable.rvs(ALPHA_CD, BETA_CD, size=out.shape[-1], scale=0.1)
                ).to(device=x.device, dtype=x.dtype).clamp(-10, 10)
                out = out + self.drift(t, out) * self.delta_t \
                          + diff_term * (self.delta_t ** (1 / ALPHA_CD)) * noise
            return self.drift(t, out), out
        else:
            return self.diffusion(0.0, out.detach())

nets_cd = nn.ModuleList([SDENetCD(dim=D_CD).to(device) for _ in range(N_STEPS)])
linear_fusion = LinearFusion(N_STEPS, D_CD, CORR_DIM).to(device)

TAIL_CUTS_CD = list(range(N_STEPS - 1, -1, -1))

def fuse_cd(raw_outs, CD_list):
    outs    = [raw_outs[i][1] for i in range(N_STEPS)]
    aligned = [outs[i][:outs[-1].shape[0] - TAIL_CUTS_CD[i] if TAIL_CUTS_CD[i] > 0
                        else outs[-1].shape[0]]
               for i in range(N_STEPS)]
    min_len = min(a.shape[0] for a in aligned)
    cat     = torch.cat([a[:min_len] for a in aligned], dim=1)
    corr_feat = CD_list[-1][:min_len]
    final   = linear_fusion(cat, corr_feat)
    means   = [final[:, i]                          for i in range(N_STEPS)]
    sigmas  = [F.softplus(final[:, N_STEPS+i])+1e-3 for i in range(N_STEPS)]
    return means, sigmas, min_len

opt_F_cd = optim.SGD(
    [{'params': n.downsample.parameters()} for n in nets_cd] +
    [{'params': n.drift.parameters()}      for n in nets_cd] +
    [{'params': linear_fusion.parameters()}],
    lr=args.lr, momentum=0.9, weight_decay=5e-4
)
opt_G_cd = optim.SGD(
    [{'params': n.diffusion.parameters()} for n in nets_cd],
    lr=args.lr2, momentum=0.9, weight_decay=5e-4
)
criterion_cd = nn.BCEWithLogitsLoss()

for epoch in range(1, args.epochs + 1):
    for n in nets_cd: n.train()
    total_cd = 0.0
    for _ in range(ITER):
        # Drift / LinearFusion
        opt_F_cd.zero_grad()
        raw_cd = [nets_cd[i](X_trains_cd[i]) for i in range(N_STEPS)]
        means_cd, sigmas_cd, min_len_cd = fuse_cd(raw_cd, cd_trains_t)
        tgts_cd = [Y_trains_cd[i][:min_len_cd] for i in range(N_STEPS)]
        lf_cd   = sum(nll_loss(tgts_cd[i], means_cd[i], sigmas_cd[i]) for i in range(N_STEPS))
        lf_cd.backward()
        nn.utils.clip_grad_norm_([p for g in opt_F_cd.param_groups for p in g['params']], 5.0)
        opt_F_cd.step(); total_cd += lf_cd.item()

        # Diffusion 判别器
        opt_G_cd.zero_grad()
        pi_cd = [nets_cd[i](X_trains_cd[i], training_diffusion=True) for i in range(N_STEPS)]
        li_cd = sum(criterion_cd(pi_cd[i], torch.full_like(pi_cd[i], 0)) for i in range(N_STEPS))
        po_inp_cd = [2*torch.randn_like(X_trains_cd[i])+X_trains_cd[i] for i in range(N_STEPS)]
        po_cd     = [nets_cd[i](po_inp_cd[i], training_diffusion=True) for i in range(N_STEPS)]
        lo_cd = sum(criterion_cd(po_cd[i], torch.full_like(po_cd[i], 1)) for i in range(N_STEPS))
        (li_cd + lo_cd).backward()
        nn.utils.clip_grad_norm_([p for g in opt_G_cd.param_groups for p in g['params']], 5.0)
        opt_G_cd.step()

    if epoch % 20 == 0:
        print(f'  [LDE-CorrDim] Epoch {epoch:3d} | Loss={total_cd/ITER:.5f}')

# 测试集评估
for n in nets_cd: n.eval()
lde_cd_results = {}
with torch.no_grad():
    raw_cd_te = [nets_cd[i](X_tests_cd[i]) for i in range(N_STEPS)]
    means_cd_te, _, min_len_cd_te = fuse_cd(raw_cd_te, cd_tests_t)
    tgts_cd_te = [Y_tests_cd[i][:min_len_cd_te] for i in range(N_STEPS)]
    for i in range(N_STEPS):
        step     = i + 1
        if args.metric_space == 'standardized':
            gt_sc   = tgts_cd_te[i].cpu().numpy()[:min_len_cd_te]
            pred_sc = means_cd_te[i].cpu().numpy()[:min_len_cd_te]
            metrics = eval_metrics(gt_sc, pred_sc, label=f't+{step}')
        elif args.metric_space == 'price':
            fc_price = inv(means_cd_te[i].cpu().numpy()[:min_len_cd_te])
            gt_price = inv(tgts_cd_te[i].cpu().numpy()[:min_len_cd_te])
            metrics  = eval_metrics(gt_price, fc_price, label=f't+{step}')
        else:
            fc_price = inv(means_cd_te[i].cpu().numpy()[:min_len_cd_te])
            gt_price = inv(tgts_cd_te[i].cpu().numpy()[:min_len_cd_te])
            prev_price = inv(prev_tests_cd[i][:min_len_cd_te])
            metrics  = eval_metrics(gt_price, fc_price, prev=prev_price, label=f't+{step}')
        lde_cd_results[f't+{step}'] = metrics

# ============================================================
#  5. 汇总对比表
# ============================================================
print('\n\n' + '='*70)
if args.metric_space == 'log_return':
    print('  五模型对比结果（测试集，对数收益率）')
elif args.metric_space == 'standardized':
    print('  五模型对比结果（测试集，标准化空间）')
else:
    print('  五模型对比结果（测试集，原始价格空间）')
print('='*70)

models   = ['ARIMA', 'LSTM', 'SDE', 'LDE', 'LDE-CorrDim']
all_res  = [arima_results, lstm_results, sde_results, lde_results, lde_cd_results]
metrics_ = ['MSE', 'RMSE', 'MAE', 'MAPE']

step_labels = [f't+{i+1}' for i in range(N_STEPS)]

# 按步长打印
for step_label in step_labels:
    print(f'\n  ── 预测步长 {step_label} ──')
    header = f"{'模型':>14s}" + ''.join(f'  {m:>10s}' for m in metrics_)
    print('  ' + header)
    print('  ' + '-' * (16 + 12 * len(metrics_)))
    for model_name, res in zip(models, all_res):
        if step_label not in res:
            continue
        row = f"{model_name:>14s}"
        for m in metrics_:
            v = res[step_label][m]
            row += f'  {v:>10.4f}'
        print('  ' + row)

# 汇总保存为 CSV
records = []
for model_name, res in zip(models, all_res):
    for step_label in step_labels:
        if step_label not in res:
            continue
        row = {'Model': model_name, 'Step': step_label}
        row.update(res[step_label])
        records.append(row)

df_result = pd.DataFrame(records)
out_csv   = 'model_comparison_results.csv'
df_result.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f'\n[保存] 对比结果已写入 → {out_csv}')
print(df_result.to_string(index=False))
