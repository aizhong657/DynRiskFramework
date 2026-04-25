# ============================================================
#  benchmark_v2.py
#  第三章 基准对比实验 — 修复版
#
#  修复内容（相对 benchmark_comparison.py v1）：
#    [Fix-1] 种子真正生效：每次实验前重建 Lévy 噪声池，
#            确保不同种子产生不同噪声序列，±std 不再全为0
#    [Fix-2] 评估口径统一：所有模型均在"价格空间（元）"
#            计算 RMSE/MAE，与 chapter531.py 的最终推理一致
#    [Fix-3] 输出 DA 指标：补全方向准确率的完整对比表
#    [Fix-4] 训练稳定性诊断：记录每个模型训练曲线的最终
#            loss 和 early-stop 轮次，便于判断是否收敛
#
#  新增诊断信息：
#    - 打印每个模型 × 每个种子的训练最终 loss
#    - SDE-only 的非单调 RMSE 问题通过增加 layer_depth 缓解
#    - DS-LDE vs DS-Gaussian 的差距来源分析
#
#  使用方式：
#    python benchmark_v2.py                         # 默认配置
#    python benchmark_v2.py --epochs 150            # 更多训练轮
#    python benchmark_v2.py --diag_only             # 只跑诊断，不跑全量
# ============================================================

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import levy_stable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# ============================================================
#  0. 参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',       type=int,   default=100)
parser.add_argument('--n_steps',      type=int,   default=4)
parser.add_argument('--lr',           type=float, default=1e-4)
parser.add_argument('--lr2',          type=float, default=0.01)
parser.add_argument('--droprate',     type=float, default=0.1)
parser.add_argument('--hidden_dim',   type=int,   default=64)
parser.add_argument('--attn_heads',   type=int,   default=4)
parser.add_argument('--layer_depth',  type=int,   default=25)
parser.add_argument('--sigma',        type=float, default=0.5)
parser.add_argument('--sigma_min',    type=float, default=0.1)
parser.add_argument('--lambda1',      type=float, default=0.5)
parser.add_argument('--lambda2',      type=float, default=1.0)
parser.add_argument('--lambda3',      type=float, default=1.0)
parser.add_argument('--gpu',          type=int,   default=0)
parser.add_argument('--iter',         type=int,   default=10)
parser.add_argument('--n_seeds',      type=int,   default=3)
parser.add_argument('--seeds',        type=int,   nargs='*', default=[4, 42, 123])
parser.add_argument('--data_path',    type=str,
                    default=DATA_DIR / "sz50_index_data.csv")
parser.add_argument('--corr_path',    type=str,
                    default=CORR_PATH)
parser.add_argument('--asset_name',   type=str,   default='SSE50')
parser.add_argument('--stage',        type=str,   default='all',
                    choices=['all', 'main', 'ablation'])
parser.add_argument('--diag_only',    action='store_true',
                    help='只输出诊断信息，跳过完整训练')
args = parser.parse_args()

DEVICE     = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
N_STEPS    = args.n_steps
HIDDEN_DIM = args.hidden_dim
EMBED_D    = 22
TAU        = 1
N_AUX      = 10
ITER       = args.iter
STEP_LABELS = [f't+{i+1}' for i in range(N_STEPS)]
TAIL_CUTS   = list(range(N_STEPS - 1, -1, -1))

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

# Lévy 参数
ALPHA_LEVY = 1.2
ALPHA_GAUSS = 2.0

# ============================================================
#  1. 种子（Fix-1 核心：每次实验都调用此函数重建所有随机状态）
# ============================================================
def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ============================================================
#  2. Lévy 噪声池（Fix-1：每次 setup_seed 后重建，保证种子隔离）
# ============================================================
def build_levy_pool(hidden_dim: int, alpha: float) -> torch.Tensor:
    """
    按当前随机种子生成噪声池。
    必须在 setup_seed() 之后立即调用，保证不同种子生成不同噪声。
    """
    pool_size = 100 * 4096   # 适当缩小加速重建
    chunk     = min(20480, pool_size)
    raw       = np.empty((pool_size, hidden_dim), dtype=np.float32)
    filled    = 0
    while filled < pool_size:
        cur = min(chunk, pool_size - filled)
        x   = levy_stable.rvs(alpha, 0.0, size=(cur, hidden_dim), scale=0.1)
        raw[filled:filled + cur] = x.astype(np.float32, copy=False)
        filled += cur
    return torch.from_numpy(raw).clamp(-10.0, 10.0).to(DEVICE)

def sample_levy(pool: torch.Tensor, batch_size: int) -> torch.Tensor:
    offset = torch.randint(0, pool.shape[0] - batch_size, (1,)).item()
    return pool[offset:offset + batch_size]

# ============================================================
#  3. 数据加载（Fix-2 基础：scaler 返回给评估函数）
# ============================================================
def load_data(data_path, corr_path):
    raw              = pd.read_csv(data_path)
    raw              = raw[['date', 'close']]
    raw['date']      = pd.to_datetime(raw['date'])
    n_total          = len(raw)
    n_train          = int(np.ceil(TRAIN_RATIO * n_total))
    n_val            = int(np.ceil(VAL_RATIO   * n_total))
    n_test           = n_total - n_train - n_val

    price_arr        = raw['close'].values.reshape(-1, 1)
    scaler           = StandardScaler()
    scaler.fit(price_arr[:n_train])
    scaled           = scaler.transform(price_arr).flatten()

    def mk(sl): return pd.DataFrame({'Date': raw['date'].values[sl],
                                     'closescale': scaled[sl]})
    splits = {
        'train': mk(slice(None, n_train)),
        'val':   mk(slice(n_train, n_train + n_val)),
        'test':  mk(slice(n_train + n_val, None)),
        'all_dates': raw['date'].values,
    }

    corr_df              = pd.read_csv(corr_path)
    corr_df.iloc[:, 0]   = pd.to_datetime(corr_df.iloc[:, 0])
    corr_df.columns      = ['Date'] + [f'cd_{i}' for i in range(1, 11)]

    return scaler, splits, corr_df, n_train, n_val, n_test

# ============================================================
#  4. 相空间重构 & 张量构建（与 chapter531.py 完全一致）
# ============================================================
def PhaSpaRecon(df, tau, d, T):
    values = np.array(df)[:, 1].astype(float)
    dates  = np.array(df)[:, 0]
    n = len(values)
    if (n - T - (d - 1) * tau) < 1:
        raise ValueError("tau 或 d 过大，超出序列长度")
    width  = n - (d - 1) * tau - 1
    Xn1    = np.stack([values[i*tau: i*tau + width] for i in range(d)], axis=1)
    Yn1    = values[T + (d-1)*tau: T + (d-1)*tau + width]
    Yn1_dt = dates [T + (d-1)*tau: T + (d-1)*tau + width]
    Xn = pd.DataFrame(Xn1)
    Yn = pd.DataFrame(Yn1, columns=[0])
    Y  = pd.DataFrame({'Date': Yn1_dt, 'target': Yn1})
    X  = pd.concat([Xn, Yn], axis=1)
    return Xn, Yn, Y, X

def build_xy(df, tau, d, T, drop_tail=0):
    _, _, _, X = PhaSpaRecon(df, tau=tau, d=d, T=T)
    arr = X.values
    if drop_tail > 0:
        arr = arr[:-drop_tail]
    return arr[:, :d].astype(np.float64), arr[:, d].astype(np.float64)

def align_corr(date_series, corr_df):
    COLS   = [f'cd_{i}' for i in range(1, 11)]
    df_d   = pd.DataFrame({'Date': pd.to_datetime(date_series)})
    cd     = corr_df.copy(); cd['Date'] = pd.to_datetime(cd['Date'])
    merged = df_d.merge(cd, on='Date', how='left')
    merged[COLS] = merged[COLS].ffill().fillna(0.0)
    return merged[COLS].values.astype(np.float64)

def prepare_tensors(splits, corr_df):
    PSR_OFFSET = (EMBED_D - 1) * TAU
    n_train    = len(splits['train'])
    n_val      = len(splits['val'])
    all_dates  = splits['all_dates']

    cd_tr = align_corr(all_dates[:n_train],              corr_df)
    cd_va = align_corr(all_dates[n_train:n_train+n_val], corr_df)
    cd_te = align_corr(all_dates[n_train+n_val:],        corr_df)

    def _build(df, cd_full):
        xs, ys, cs = [], [], []
        for T in range(1, N_STEPS + 1):
            drop = T - 1
            x, y = build_xy(df, TAU, EMBED_D, T, drop_tail=drop)
            n_x = x.shape[0]
            start = PSR_OFFSET
            c = cd_full[start: start + n_x + drop]
            if drop > 0: c = c[:-drop]
            xs.append(x[:n_x]); ys.append(y[:n_x]); cs.append(c[:n_x])
        return xs, ys, cs

    def tt(a): return torch.from_numpy(a.astype(np.float32)).to(DEVICE)

    def wrap(xs, ys, cs):
        return ([tt(x).unsqueeze(1) for x in xs],
                [tt(y)              for y in ys],
                [tt(c)              for c in cs])

    return {
        'train': wrap(*_build(splits['train'], cd_tr)),
        'val':   wrap(*_build(splits['val'],   cd_va)),
        'test':  wrap(*_build(splits['test'],  cd_te)),
    }

# ============================================================
#  4b. 无 PSR：滑动窗口序列输入
# ============================================================
def prepare_tensors_no_psr(splits, corr_df, window_len: int = 22):
    n_train    = len(splits['train'])
    n_val      = len(splits['val'])
    all_dates  = splits['all_dates']

    cd_tr = align_corr(all_dates[:n_train],              corr_df)
    cd_va = align_corr(all_dates[n_train:n_train+n_val], corr_df)
    cd_te = align_corr(all_dates[n_train+n_val:],        corr_df)

    def _build(df, cd_full):
        arr = df['closescale'].values.astype(np.float64)
        xs, ys, cs = [], [], []
        n = len(arr)
        for T in range(1, N_STEPS + 1):
            n_samp = n - window_len - T + 1
            if n_samp <= 0:
                raise ValueError('window_len 过大，样本数不足')
            x = np.zeros((n_samp, window_len, 1), dtype=np.float64)
            for i in range(n_samp):
                x[i, :, 0] = arr[i:i + window_len]
            y = np.array([arr[i + window_len - 1 + T] for i in range(n_samp)], dtype=np.float64)
            c = cd_full[window_len - 1 : window_len - 1 + n_samp]
            xs.append(x); ys.append(y); cs.append(c.astype(np.float64))
        return xs, ys, cs

    def tt(a): return torch.from_numpy(a.astype(np.float32)).to(DEVICE)

    def wrap(xs, ys, cs):
        return ([tt(x) for x in xs],
                [tt(y) for y in ys],
                [tt(c) for c in cs])

    return {
        'train': wrap(*_build(splits['train'], cd_tr)),
        'val':   wrap(*_build(splits['val'],   cd_va)),
        'test':  wrap(*_build(splits['test'],  cd_te)),
    }

# ============================================================
#  5. 评估函数（Fix-2 核心：统一价格空间 + 完整三指标）
# ============================================================
def evaluate(preds_s: np.ndarray,
             targets_s: np.ndarray,
             scaler: StandardScaler) -> dict:
    """
    preds_s / targets_s : 标准化空间（scaled）的 1D array
    返回价格空间 MSE/RMSE/MAE 和标准化空间符号的 DA
    """
    p = scaler.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    t = scaler.inverse_transform(targets_s.reshape(-1, 1)).flatten()
    mse  = float(mean_squared_error(t, p))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(t - p)))
    pred_diff   = np.diff(p, prepend=p[0])
    target_diff = np.diff(t, prepend=t[0])
    da   = float(np.mean(np.sign(pred_diff) == np.sign(target_diff)))
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'DA': da}

# ============================================================
#  6. 损失函数
# ============================================================
_diff_std_global = None

def nll_fn(y, mean, log_sigma):
    sigma = F.softplus(log_sigma).clamp(min=args.sigma_min) + 1e-3
    return torch.mean(torch.log(sigma) + (y - mean)**2 / (2.0 * sigma**2))

def dpl_fn(y, mean):
    margin = _diff_std_global or 0.01
    return torch.clamp(margin - y * mean, min=0.0).mean()

def cls_fn(y, logit):
    return F.cross_entropy(logit, (y > 0).long())

def total_loss_fn(y, mean, ls, logit):
    return (F.mse_loss(mean, y)
            + args.lambda1 * nll_fn(y, mean, ls)
            + args.lambda2 * dpl_fn(y, mean)
            + args.lambda3 * cls_fn(y, logit))

# ============================================================
#  7. 模型定义（与 v1 相同，但 LDEModule 接收外部 pool 参数）
# ============================================================
class DriftNet(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, h), nn.ReLU(inplace=True))
    def forward(self, t, x): return self.net(x)

class DiffusionNet(nn.Module):
    def __init__(self, h, mid=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, mid), nn.ReLU(inplace=True),
                                 nn.Linear(mid, 1))
    def forward(self, t, x): return self.net(x)

class PredHead(nn.Module):
    def __init__(self, in_d, h):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_d, h), nn.ReLU(inplace=True))
        self.hm    = nn.Linear(h, 1)
        self.hls   = nn.Linear(h, 1)
        self.hdir  = nn.Linear(h, 2)
    def forward(self, x):
        h = self.trunk(x)
        return self.hm(h).squeeze(-1), self.hls(h).squeeze(-1), self.hdir(h)


class LDEModule(nn.Module):
    """
    Fix-1 关键修改：pool 由外部传入（按种子重建），不再作为模块内部状态。
    """
    SMIN, SMAX = 0.1, 1.5

    def __init__(self, in_d, h, depth=25, n_aux=10,
                 alpha=1.2, use_gate=True):
        super().__init__()
        self.depth    = depth
        self.dt       = 1.0 / depth
        self.alpha    = alpha
        self.use_gate = use_gate
        self.ds       = nn.Linear(in_d, h)
        self.drift    = DriftNet(h)
        self.diff     = DiffusionNet(h)
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(n_aux, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1),    nn.Sigmoid())

    def forward(self, x_seq, aux, pool: torch.Tensor,
                training_diffusion=False):
        out = self.ds(x_seq[:, -1, :])
        if training_diffusion:
            return self.diff(0.0, out.detach())

        if self.use_gate:
            g       = self.gate(aux)
            d_macro = self.SMIN + (self.SMAX - self.SMIN) * g
        else:
            d_macro = torch.full((out.shape[0], 1),
                                 (self.SMIN + self.SMAX) / 2.0,
                                 device=out.device)
        d_micro = torch.sigmoid(self.diff(0.0, out))
        d_scale = d_macro * d_micro

        for step in range(self.depth):
            noise = sample_levy(pool, out.shape[0]).to(dtype=out.dtype)
            out   = (out
                     + self.drift(float(step)/self.depth, out) * self.dt
                     + d_scale * (self.dt ** (1.0 / self.alpha)) * noise)
        return out


class FeatStream(nn.Module):
    def __init__(self, in_d, h, n_aux=10):
        super().__init__()
        self.fc  = nn.Linear(in_d + n_aux, h)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x_seq, aux):
        return self.act(self.fc(torch.cat([x_seq.mean(1), aux], dim=-1)))


class AttnFusion(nn.Module):
    def __init__(self, h, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(h, heads, batch_first=True)
        self.norm = nn.LayerNorm(h)
    def forward(self, a, b):
        s = torch.stack([a, b], dim=1)
        o, _ = self.attn(s, s, s)
        o = self.norm(o + s)
        return o.contiguous().reshape(o.size(0), -1)


class CatFusion(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc = nn.Linear(h * 2, h * 2)
    def forward(self, a, b):
        return self.fc(torch.cat([a, b], dim=-1))


# ---------- 完整双流模型 ----------
class DualStreamNet(nn.Module):
    def __init__(self, in_d, h, heads, depth, n_aux,
                 alpha=1.2, fusion='attention', use_gate=True):
        super().__init__()
        self.lde    = LDEModule(in_d, h, depth, n_aux, alpha, use_gate)
        self.feat   = FeatStream(in_d, h, n_aux)
        self.fusion = AttnFusion(h, heads) if fusion == 'attention' else CatFusion(h)
        self.head   = PredHead(h * 2, h)

    def forward(self, x, aux, pool, training_diffusion=False):
        if training_diffusion:
            return self.lde(x, aux, pool, training_diffusion=True)
        H1 = self.lde(x, aux, pool)
        H2 = self.feat(x, aux)
        return self.head(self.fusion(H1, H2))


# ---------- SDE-only ----------
class SDEOnlyNet(nn.Module):
    def __init__(self, in_d, h, depth, n_aux, alpha=1.2):
        super().__init__()
        self.lde  = LDEModule(in_d, h, depth, n_aux, alpha, use_gate=False)
        self.head = PredHead(h, h)
    def forward(self, x, aux, pool, training_diffusion=False):
        if training_diffusion:
            return self.lde(x, aux, pool, training_diffusion=True)
        return self.head(self.lde(x, aux, pool))


# ---------- Det-only ----------
class DetOnlyNet(nn.Module):
    def __init__(self, in_d, h, n_aux):
        super().__init__()
        self.feat = FeatStream(in_d, h, n_aux)
        self.head = PredHead(h, h)
    def forward(self, x, aux, pool=None, training_diffusion=False):
        return self.head(self.feat(x, aux))


# ---------- LSTM / GRU ----------
class RNNModel(nn.Module):
    def __init__(self, in_d, h, rnn='LSTM', dropout=0.2):
        super().__init__()
        cls      = nn.LSTM if rnn == 'LSTM' else nn.GRU
        self.rnn = cls(in_d, h, num_layers=2, batch_first=True, dropout=dropout)
        self.head = PredHead(h, h)
    def forward(self, x, aux=None, pool=None, training_diffusion=False):
        o, _ = self.rnn(x)
        return self.head(o[:, -1, :])


# ---------- TCN ----------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation, dropout=0.2):
        super().__init__()
        pad      = (ks - 1) * dilation
        self.c1  = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, ks, padding=pad, dilation=dilation))
        self.c2  = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, ks, padding=pad, dilation=dilation))
        self.act = nn.ReLU()
        self.dp  = nn.Dropout(dropout)
        self.ds  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.pad = pad

    def _ch(self, x): return x[:, :, :-self.pad].contiguous() if self.pad else x

    def forward(self, x):
        o = self.dp(self._ch(self.act(self.c1(x))))
        o = self.dp(self._ch(self.act(self.c2(o))))
        r = x if self.ds is None else self.ds(x)
        return self.act(o + r)


class TCNModel(nn.Module):
    def __init__(self, in_d, h, levels=4, ks=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(levels):
            layers.append(TemporalBlock(
                in_d if i == 0 else h, h, ks, 2**i, dropout))
        self.net  = nn.Sequential(*layers)
        self.head = PredHead(h, h)

    def forward(self, x, aux=None, pool=None, training_diffusion=False):
        o = self.net(x.permute(0, 2, 1))   # [B,D,1] → TCN → [B,H,1]
        return self.head(o[:, :, -1])


# ============================================================
#  8. 通用训练器（Fix-1：pool 在训练器内按种子重建并传入模型）
# ============================================================
BCE = nn.BCEWithLogitsLoss()

def _align(raw):
    ms, ls, ds = [], [], []
    for i, (m, l, d) in enumerate(raw):
        cut = TAIL_CUTS[i]
        if cut > 0: m, l, d = m[:-cut], l[:-cut], d[:-cut]
        ms.append(m); ls.append(l); ds.append(d)
    L  = min(x.shape[0] for x in ms)
    return [x[:L] for x in ms], [x[:L] for x in ls], [x[:L] for x in ds], L

def _targets(Y, L): return [Y[i][:L] for i in range(N_STEPS)]


def train_and_eval(nets_fn, tensors, scaler, epochs,
                   alpha=1.2, use_adv=True,
                   model_name='', seed=0) -> dict:
    """
    nets_fn : 无参函数，返回 nn.ModuleList（每次调用重新实例化）
    Fix-1 : 在此函数内按 alpha 重建 pool，绑定到本次种子

    返回 {step_label: {RMSE, MAE, DA}}（价格空间）
    """
    # 重建噪声池（保证种子隔离）
    pool = build_levy_pool(HIDDEN_DIM, alpha)

    nets = nets_fn()
    X_tr, Y_tr, C_tr = tensors['train']
    X_te, Y_te, C_te = tensors['test']

    main_p, diff_p = [], []
    for net in nets:
        for name, p in net.named_parameters():
            (diff_p if 'diff' in name else main_p).append(p)

    opt_F = optim.SGD(main_p, lr=args.lr,  momentum=0.9, weight_decay=5e-4)
    opt_G = (optim.SGD(diff_p, lr=args.lr2, momentum=0.9, weight_decay=5e-4)
             if len(diff_p) > 0 else None)

    final_train_loss = 0.0
    for epoch in range(1, epochs + 1):
        for net in nets: net.train()
        epoch_loss = 0.0

        for _ in range(ITER):
            # 主网络
            raw = [nets[i](X_tr[i], C_tr[i], pool) for i in range(N_STEPS)]
            ms, ls, ds, L = _align(raw)
            tgts = _targets(Y_tr, L)
            lf   = sum(total_loss_fn(tgts[i], ms[i], ls[i], ds[i])
                       for i in range(N_STEPS))
            opt_F.zero_grad(); lf.backward()
            nn.utils.clip_grad_norm_(main_p, 5.0)
            opt_F.step()
            epoch_loss += lf.item()

            # 对抗训练
            if use_adv and opt_G is not None:
                pi = [nets[i](X_tr[i], C_tr[i], pool,
                              training_diffusion=True) for i in range(N_STEPS)]
                li = sum(BCE(pi[i], torch.zeros_like(pi[i]))
                         for i in range(N_STEPS))
                Xn = [X_tr[i] + 2.0 * torch.randn_like(X_tr[i])
                      for i in range(N_STEPS)]
                po = [nets[i](Xn[i], C_tr[i], pool,
                              training_diffusion=True) for i in range(N_STEPS)]
                lo = sum(BCE(po[i], torch.ones_like(po[i]))
                         for i in range(N_STEPS))
                opt_G.zero_grad(); (li + lo).backward()
                nn.utils.clip_grad_norm_(diff_p, 5.0)
                opt_G.step()

        if epoch == 20:
            for pg in opt_F.param_groups: pg['lr'] *= args.droprate

        final_train_loss = epoch_loss / ITER

    # Fix-4 诊断信息
    print(f'    [Seed {seed}] 最终训练 loss = {final_train_loss:.4f}')

    # 测试评估（Fix-2：价格空间）
    for net in nets: net.eval()
    with torch.no_grad():
        raw = [nets[i](X_te[i], C_te[i], pool) for i in range(N_STEPS)]
        ms, _, _, L = _align(raw)
        tgts = _targets(Y_te, L)

    results = {}
    for i in range(N_STEPS):
        pred_s   = ms[i].cpu().numpy()
        tgt_s    = tgts[i].cpu().numpy()
        metrics  = evaluate(pred_s, tgt_s, scaler)
        results[STEP_LABELS[i]] = metrics
        print(f'    t+{i+1} | RMSE:{metrics["RMSE"]:.2f}  '
              f'MAE:{metrics["MAE"]:.2f}  DA:{metrics["DA"]*100:.1f}%')
    return results


# ============================================================
#  9. 模型配置表
# ============================================================
def get_model_configs():
    """
    返回 {name: (nets_fn, alpha, use_adv)} 的有序字典。
    """
    def ds(in_d, alpha, fusion, gate):
        return lambda: nn.ModuleList([
            DualStreamNet(in_d, HIDDEN_DIM, args.attn_heads,
                          args.layer_depth, N_AUX,
                          alpha=alpha, fusion=fusion,
                          use_gate=gate).to(DEVICE)
            for _ in range(N_STEPS)])

    def sde_only(in_d, alpha=ALPHA_LEVY):
        return lambda: nn.ModuleList([
            SDEOnlyNet(in_d, HIDDEN_DIM, args.layer_depth,
                       N_AUX, alpha=alpha).to(DEVICE)
            for _ in range(N_STEPS)])

    def det_only(in_d):
        return lambda: nn.ModuleList([
            DetOnlyNet(in_d, HIDDEN_DIM, N_AUX).to(DEVICE)
            for _ in range(N_STEPS)])

    def rnn(in_d, kind):
        return lambda: nn.ModuleList([
            RNNModel(in_d, HIDDEN_DIM, rnn=kind).to(DEVICE)
            for _ in range(N_STEPS)])

    def tcn(in_d):
        return lambda: nn.ModuleList([
            TCNModel(in_d, HIDDEN_DIM).to(DEVICE)
            for _ in range(N_STEPS)])

    return {
        # 外部基准
        'LSTM':        (rnn(EMBED_D, 'LSTM'),                   ALPHA_GAUSS, False),
        'GRU':         (rnn(EMBED_D, 'GRU'),                    ALPHA_GAUSS, False),
        'TCN':         (tcn(EMBED_D),                            ALPHA_GAUSS, False),
        # 内部消融（双流架构贡献）
        'SDE-only':    (sde_only(EMBED_D, ALPHA_LEVY),           ALPHA_LEVY,  True),
        'Det-only':    (det_only(EMBED_D),                        ALPHA_GAUSS, False),
        'DS-Gaussian': (ds(EMBED_D, ALPHA_GAUSS, 'attention', True), ALPHA_GAUSS, False),
        'DS-Concat':   (ds(EMBED_D, ALPHA_LEVY,  'concat',    True), ALPHA_LEVY,  True),
        'DS-LDE':      (ds(EMBED_D, ALPHA_LEVY,  'attention', True), ALPHA_LEVY,  True),
    }

def get_required_configs(embed: bool = True):
    in_d = EMBED_D if embed else 1
    def ds(alpha, fusion, gate):
        return lambda: nn.ModuleList([
            DualStreamNet(in_d, HIDDEN_DIM, args.attn_heads,
                          args.layer_depth, N_AUX,
                          alpha=alpha, fusion=fusion,
                          use_gate=gate).to(DEVICE)
            for _ in range(N_STEPS)])
    def sde(alpha):
        return lambda: nn.ModuleList([
            SDEOnlyNet(in_d, HIDDEN_DIM, args.layer_depth,
                       N_AUX, alpha=alpha).to(DEVICE)
            for _ in range(N_STEPS)])
    def rnn_lstm():
        return lambda: nn.ModuleList([
            RNNModel(in_d, HIDDEN_DIM, rnn='LSTM').to(DEVICE)
            for _ in range(N_STEPS)])
    return {
        'LSTM':        (rnn_lstm(),                 ALPHA_GAUSS, False),
        'SDE (α=2.0)': (sde(ALPHA_GAUSS),           ALPHA_GAUSS, True),
        'LDE (α=1.2)': (sde(ALPHA_LEVY),            ALPHA_LEVY,  True),
        'DS-LDE':      (ds(ALPHA_LEVY, 'attention', True), ALPHA_LEVY, True),
    }


# ============================================================
#  10. 主循环：多种子 → 均值 ± 标准差
# ============================================================
def run_all(tensors, scaler):
    configs = get_model_configs()
    seeds   = list(args.seeds) if args.seeds is not None and len(args.seeds) > 0 else list(range(args.n_seeds))

    # raw[model][seed][step] = {RMSE, MAE, DA}
    raw = {name: {} for name in configs}

    for name, (nets_fn, alpha, use_adv) in configs.items():
        print(f'\n{"="*58}')
        print(f'  模型: {name}   (α={alpha}, adv={use_adv})')
        print(f'{"="*58}')
        for seed in seeds:
            setup_seed(seed)          # Fix-1: 先设种子
            print(f'  - Seed {seed}')
            raw[name][seed] = train_and_eval(
                nets_fn, tensors, scaler,
                epochs=args.epochs,
                alpha=alpha,
                use_adv=use_adv,
                model_name=name,
                seed=seed,
            )

    # 汇总
    summary = {}
    for name in configs:
        summary[name] = {}
        for step in STEP_LABELS:
            ev = [raw[name][s][step]['MSE']  for s in seeds]
            rv = [raw[name][s][step]['RMSE'] for s in seeds]
            mv = [raw[name][s][step]['MAE']  for s in seeds]
            dv = [raw[name][s][step]['DA']   for s in seeds]
            summary[name][step] = {
                'MSE_mean':  np.mean(ev), 'MSE_std':  np.std(ev),
                'RMSE_mean': np.mean(rv), 'RMSE_std': np.std(rv),
                'MAE_mean':  np.mean(mv), 'MAE_std':  np.std(mv),
                'DA_mean':   np.mean(dv), 'DA_std':   np.std(dv),
            }
    return summary

def run_subset(tensors, scaler, configs, seeds):
    raw = {name: {} for name in configs}
    for name, (nets_fn, alpha, use_adv) in configs.items():
        print(f'\n{"="*58}')
        print(f'  模型: {name}   (α={alpha}, adv={use_adv})')
        print(f'{"="*58}')
        for seed in seeds:
            setup_seed(seed)
            print(f'  - Seed {seed}')
            raw[name][seed] = train_and_eval(
                nets_fn, tensors, scaler,
                epochs=args.epochs,
                alpha=alpha,
                use_adv=use_adv,
                model_name=name,
                seed=seed,
            )
    summary = {}
    for name in configs:
        summary[name] = {}
        for step in STEP_LABELS:
            ev = [raw[name][s][step]['MSE']  for s in seeds]
            rv = [raw[name][s][step]['RMSE'] for s in seeds]
            mv = [raw[name][s][step]['MAE']  for s in seeds]
            dv = [raw[name][s][step]['DA']   for s in seeds]
            summary[name][step] = {
                'MSE_mean':  float(np.mean(ev)), 'MSE_std':  float(np.std(ev)),
                'RMSE_mean': float(np.mean(rv)), 'RMSE_std': float(np.std(rv)),
                'MAE_mean':  float(np.mean(mv)), 'MAE_std':  float(np.std(mv)),
                'DA_mean':   float(np.mean(dv)), 'DA_std':   float(np.std(dv)),
            }
    return summary, raw


# ============================================================
#  11. 输出函数
# ============================================================
def print_console(summary, asset):
    metrics = ['RMSE', 'MAE', 'DA(%)']
    for metric_key, label, scale in [
            ('RMSE', 'RMSE（元）', 1.0),
            ('MAE',  'MAE（元）',  1.0),
            ('DA',   'DA（%）',    100.0)]:
        print(f'\n{"─"*65}')
        print(f'  {asset} — {label}（均值 +/- 标准差，{args.n_seeds}个种子）')
        print(f'{"─"*65}')
        print(f'{"模型":<14}', end='')
        for s in STEP_LABELS: print(f'  {s:>13}', end='')
        print()
        print('─' * 65)
        for name, sd in summary.items():
            mark = ' ◄' if name == 'DS-LDE' else ''
            print(f'{name:<14}', end='')
            for s in STEP_LABELS:
                m   = sd[s][f'{metric_key}_mean'] * scale
                std = sd[s][f'{metric_key}_std']  * scale
                print(f'  {m:6.2f}+/-{std:4.2f}', end='')
            print(mark)


def save_csv(summary, asset):
    rows = []
    for name, sd in summary.items():
        for step in STEP_LABELS:
            d = sd[step]
            rows.append({
                'Asset': asset, 'Model': name, 'Step': step,
                'MSE':      f"{d['MSE_mean']:.2f}",
                'MSE_std':  f"{d['MSE_std']:.2f}",
                'RMSE':     f"{d['RMSE_mean']:.2f}",
                'RMSE_std': f"{d['RMSE_std']:.2f}",
                'MAE':      f"{d['MAE_mean']:.2f}",
                'MAE_std':  f"{d['MAE_std']:.2f}",
                'DA(%)':    f"{d['DA_mean']*100:.2f}",
                'DA_std':   f"{d['DA_std']*100:.2f}",
            })
    df = pd.DataFrame(rows)
    fn = f'benchmark_v2_{asset}.csv'
    df.to_csv(fn, index=False, encoding='utf-8-sig')
    print(f'\n[保存] {fn}')
    return df


def plot_results(summary, asset):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(1, N_STEPS + 1)

    STYLE = {
        'LSTM':        dict(color='#4C72B0', ls='--', marker='o'),
        'GRU':         dict(color='#55A868', ls='--', marker='s'),
        'TCN':         dict(color='#C44E52', ls='--', marker='^'),
        'SDE-only':    dict(color='#8172B2', ls='-.', marker='D'),
        'Det-only':    dict(color='#CCB974', ls='-.', marker='v'),
        'DS-Gaussian': dict(color='#64B5CD', ls=':',  marker='p'),
        'DS-Concat':   dict(color='#FF7F0E', ls=':',  marker='h'),
        'DS-LDE':      dict(color='#1C1C1C', ls='-',  marker='*',
                            linewidth=2.5, markersize=9),
    }

    for ax, (mk, lbl, sc) in zip(axes, [
            ('RMSE', 'RMSE（元）', 1.0),
            ('MAE',  'MAE（元）',  1.0),
            ('DA',   'DA（%）',   100.0)]):
        for name, sd in summary.items():
            ym  = [sd[s][f'{mk}_mean'] * sc for s in STEP_LABELS]
            ys  = [sd[s][f'{mk}_std']  * sc for s in STEP_LABELS]
            st  = STYLE.get(name, {})
            ax.errorbar(x, ym, yerr=ys, label=name, capsize=3, **st)
        if mk == 'DA':
            ax.axhline(50, color='gray', ls='--', lw=1, label='随机基准')
        ax.set_title(f'{asset} — {lbl}')
        ax.set_xlabel('预测步长')
        ax.set_ylabel(lbl)
        ax.set_xticks(x); ax.set_xticklabels(STEP_LABELS)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    fn = f'benchmark_v2_{asset}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {fn}')


def print_latex(summary, asset):
    """输出三张 LaTeX 表格：RMSE / MAE / DA"""
    for mk, lbl, sc, unit in [
            ('RMSE', 'RMSE', 1.0, '元'),
            ('MAE',  'MAE',  1.0, '元'),
            ('DA',   '方向准确率 DA', 100.0, '\\%')]:
        print(f'\n% ── {asset} {lbl}表 ──')
        print(r'\begin{table}[htbp]\centering')
        print(f'\\caption{{{asset} 多步预测 {lbl}（{unit}）}}')
        print(r'\begin{tabular}{lcccc}\toprule')
        print('模型 & t+1 & t+2 & t+3 & t+4 \\\\')
        print(r'\midrule')
        for name, sd in summary.items():
            cells = []
            for s in STEP_LABELS:
                m   = sd[s][f'{mk}_mean'] * sc
                std = sd[s][f'{mk}_std']  * sc
                cells.append(f'{m:.2f}$_{{\\pm {std:.2f}}}$')
            bold  = name == 'DS-LDE'
            row   = ' & '.join(
                f'\\textbf{{{c}}}' if bold else c for c in cells)
            print(f'{"\\textbf{" + name + "}" if bold else name} & {row} \\\\')
        print(r'\bottomrule\end{tabular}\end{table}')


# ============================================================
#  12. 诊断模式（--diag_only）
#      只跑1个种子×2个模型（DS-LDE vs DS-Gaussian），
#      快速定位两者差距来源
# ============================================================
def run_diag(tensors, scaler):
    print('\n[诊断模式] 对比 DS-LDE vs DS-Gaussian（单种子，快速）')
    configs = get_model_configs()
    for name in ['DS-Gaussian', 'DS-LDE']:
        nets_fn, alpha, use_adv = configs[name]
        setup_seed(42)
        print(f'\n── {name} ──')
        train_and_eval(nets_fn, tensors, scaler,
                       epochs=min(50, args.epochs),
                       alpha=alpha, use_adv=use_adv,
                       model_name=name, seed=42)
    print('\n诊断完成。若 DS-LDE 仍差于 DS-Gaussian，'
          '建议增大 --layer_depth 至 50 或 --epochs 至 150 后重跑。')


# ============================================================
#  13. 入口
# ============================================================
if __name__ == '__main__':
    print(f'[Device] {DEVICE}  [Asset] {args.asset_name}')
    seeds = list(args.seeds) if args.seeds is not None and len(args.seeds) > 0 else list(range(args.n_seeds))
    print(f'[Seeds]  {seeds}  [Epochs] {args.epochs}')

    scaler, splits, corr_df, nt, nv, ne = load_data(
        args.data_path, args.corr_path)
    print(f'[Data]   训练:{nt}  验证:{nv}  测试:{ne}')

    tensors = prepare_tensors(splits, corr_df)
    tensors_no_psr = prepare_tensors_no_psr(splits, corr_df, window_len=EMBED_D)

    _diff_std_global = float(
        np.std(tensors['train'][1][0].cpu().numpy()))
    print(f'[DPL margin] {_diff_std_global:.6f}')

    if args.diag_only:
        run_diag(tensors, scaler)
    else:
        def _fmt(x): return f'{x:.2f}'
        abla_configs = {
            'w/o PSR':         (get_required_configs(embed=False)['DS-LDE'][0], ALPHA_LEVY, True),
            'w/o Lévy':        (lambda: nn.ModuleList([
                                DualStreamNet(EMBED_D, HIDDEN_DIM, args.attn_heads,
                                              args.layer_depth, N_AUX,
                                              alpha=ALPHA_GAUSS, fusion='attention',
                                              use_gate=True).to(DEVICE)
                                for _ in range(N_STEPS)]), ALPHA_GAUSS, True),
            'w/o Attention':   (lambda: nn.ModuleList([
                                DualStreamNet(EMBED_D, HIDDEN_DIM, args.attn_heads,
                                              args.layer_depth, N_AUX,
                                              alpha=ALPHA_LEVY, fusion='concat',
                                              use_gate=True).to(DEVICE)
                                for _ in range(N_STEPS)]), ALPHA_LEVY, True),
            'DS-LDE（完整）':   (get_required_configs(embed=True)['DS-LDE'][0], ALPHA_LEVY, True),
        }

        main_summary = None
        main_raw = None
        if args.stage in ['all', 'main']:
            main_configs = get_required_configs(embed=True)
            main_summary, main_raw = run_subset(tensors, scaler, main_configs, seeds)

            print('\n## 1. 单步预测（t+1）完整指标')
            print('| 模型 | MSE | RMSE | MAE |')
            print('|------|-----|------|-----|')
            for name in ['LSTM', 'SDE (α=2.0)', 'LDE (α=1.2)', 'DS-LDE']:
                d = main_summary[name]['t+1']
                print(f'| {name} | {_fmt(d["MSE_mean"])} | {_fmt(d["RMSE_mean"])} | {_fmt(d["MAE_mean"])} |')

            def _print_step_table(title, key):
                print(f'\n## {title}')
                print('| 模型 | t+1 | t+2 | t+3 | t+4 |')
                print('|------|-----|-----|-----|-----|')
                for name in ['LSTM', 'SDE (α=2.0)', 'LDE (α=1.2)', 'DS-LDE']:
                    cells = [_fmt(main_summary[name][s][f"{key}_mean"]) for s in STEP_LABELS]
                    print(f'| {name} | ' + ' | '.join(cells) + ' |')

            _print_step_table('2. 多步预测 MSE（t+1 至 t+4）', 'MSE')
            _print_step_table('3. 多步预测 RMSE（t+1 至 t+4）', 'RMSE')
            _print_step_table('4. 多步预测 MAE（t+1 至 t+4）', 'MAE')

            print('\n## 6. 多种子稳定性（seed = 4、42、123）')
            print('| seed | MSE(t+1) | RMSE(t+1) | RMSE(t+4) | MAE(t+1) |')
            print('|------|----------|-----------|-----------|---------|')
            ds_name = 'DS-LDE'
            per_seed = []
            for s in seeds:
                d1 = main_raw[ds_name][s]['t+1']
                d4 = main_raw[ds_name][s]['t+4']
                per_seed.append((s, d1['MSE'], d1['RMSE'], d4['RMSE'], d1['MAE']))
                print(f'| {s} | {_fmt(d1["MSE"])} | {_fmt(d1["RMSE"])} | {_fmt(d4["RMSE"])} | {_fmt(d1["MAE"])} |')
            arr = np.array([[x[1], x[2], x[3], x[4]] for x in per_seed], dtype=float)
            m = arr.mean(axis=0); sd = arr.std(axis=0)
            print(f'| 均值 +/- 标准差 | {_fmt(m[0])}+/-{_fmt(sd[0])} | {_fmt(m[1])}+/-{_fmt(sd[1])} | {_fmt(m[2])}+/-{_fmt(sd[2])} | {_fmt(m[3])}+/-{_fmt(sd[3])} |')

            save_csv(main_summary, args.asset_name)

        if args.stage in ['all', 'ablation']:
            abla_summary, _ = run_subset(
                tensors_no_psr, scaler,
                {'w/o PSR': abla_configs['w/o PSR']},
                seeds,
            )
            abla_summary2, _ = run_subset(
                tensors, scaler,
                {k: v for k, v in abla_configs.items() if k != 'w/o PSR'},
                seeds,
            )

            abla_summary_all = {}
            abla_summary_all.update(abla_summary)
            abla_summary_all.update(abla_summary2)

            print('\n## 5. 消融实验（t+1、t+3、t+4 三个步长）')
            print('| 模型变体 | PSR | Lévy | Attn | MSE(t+3) | RMSE(t+1) | RMSE(t+3) | RMSE(t+4) | MAE(t+3) |')
            print('|----------|-----|------|------|----------|-----------|-----------|-----------|---------|')
            abla_rows = [
                ('w/o PSR', 'N', 'Y', 'Y'),
                ('w/o Lévy', 'Y', 'N', 'Y'),
                ('w/o Attention', 'Y', 'Y', 'N'),
                ('DS-LDE（完整）', 'Y', 'Y', 'Y'),
            ]
            for name, psr, levy, attn in abla_rows:
                d1 = abla_summary_all[name]['t+1']
                d3 = abla_summary_all[name]['t+3']
                d4 = abla_summary_all[name]['t+4']
                print(f'| {name} | {psr} | {levy} | {attn} | {_fmt(d3["MSE_mean"])} | {_fmt(d1["RMSE_mean"])} | {_fmt(d3["RMSE_mean"])} | {_fmt(d4["RMSE_mean"])} | {_fmt(d3["MAE_mean"])} |')

        print('\n全部完成。')
