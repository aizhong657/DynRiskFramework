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
parser.add_argument('--n_seeds',      type=int,   default=3)
parser.add_argument('--data_path',    type=str,
                    default=DATA_DIR / "sz50_index_data.csv")
parser.add_argument('--corr_path',    type=str,
                    default=CORR_PATH)
parser.add_argument('--asset_name',   type=str,   default='SSE50')
parser.add_argument('--diag_only',    action='store_true',
                    help='只输出诊断信息，跳过完整训练')
args = parser.parse_args()

DEVICE     = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
N_STEPS    = args.n_steps
HIDDEN_DIM = args.hidden_dim
D          = 22
TAU        = 1
N_AUX      = 10
ITER       = 50
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
    PSR_OFFSET = (D - 1) * TAU
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
            x, y = build_xy(df, TAU, D, T, drop_tail=drop)
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
#  5. 评估函数
#     统一在价格空间（元）计算 MSE / RMSE / MAE
#     所有模型共用同一个 scaler.inverse_transform 路径
# ============================================================
def evaluate(preds_s: np.ndarray,
             targets_s: np.ndarray,
             scaler: StandardScaler) -> dict:
    """
    preds_s   : 标准化空间的预测值，shape [N]
    targets_s : 标准化空间的真实值，shape [N]
    返回价格空间（元）的 MSE / RMSE / MAE
    """
    p    = scaler.inverse_transform(preds_s.reshape(-1,  1)).flatten()
    t    = scaler.inverse_transform(targets_s.reshape(-1, 1)).flatten()
    mse  = float(mean_squared_error(t, p))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(t - p)))
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

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

    返回 {step_label: {MSE, RMSE, MAE}}（价格空间，元）
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

    batch_size = 256
    final_train_loss = 0.0
    for epoch in range(1, epochs + 1):
        for net in nets: net.train()
        with torch.no_grad():
            raw_full = [nets[i](X_tr[i], C_tr[i], pool) for i in range(N_STEPS)]
            _, _, _, L = _align(raw_full)

        perm = torch.randperm(L, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, L, batch_size):
            idx = perm[start:start + batch_size]
            Xb = [X_tr[i][idx] for i in range(N_STEPS)]
            Yb = [Y_tr[i][idx] for i in range(N_STEPS)]
            Cb = [C_tr[i][idx] for i in range(N_STEPS)]

            raw = [nets[i](Xb[i], Cb[i], pool) for i in range(N_STEPS)]
            ms, ls, ds, Lb = _align(raw)
            tgts = _targets(Yb, Lb)
            lf = sum(total_loss_fn(tgts[i], ms[i], ls[i], ds[i])
                     for i in range(N_STEPS))
            opt_F.zero_grad()
            lf.backward()
            nn.utils.clip_grad_norm_(main_p, 5.0)
            opt_F.step()
            epoch_loss += lf.item()
            n_batches += 1

            if use_adv and opt_G is not None:
                pi = [nets[i](Xb[i], Cb[i], pool, training_diffusion=True)
                      for i in range(N_STEPS)]
                li = sum(BCE(pi[i], torch.zeros_like(pi[i]))
                         for i in range(N_STEPS))
                Xn = [Xb[i] + 2.0 * torch.randn_like(Xb[i])
                      for i in range(N_STEPS)]
                po = [nets[i](Xn[i], Cb[i], pool, training_diffusion=True)
                      for i in range(N_STEPS)]
                lo = sum(BCE(po[i], torch.ones_like(po[i]))
                         for i in range(N_STEPS))
                opt_G.zero_grad()
                (li + lo).backward()
                nn.utils.clip_grad_norm_(diff_p, 5.0)
                opt_G.step()

        if epoch == 20:
            for pg in opt_F.param_groups: pg['lr'] *= args.droprate

        final_train_loss = epoch_loss / max(1, n_batches)

    # Fix-4 诊断信息
    print(f'    [Seed {seed}] 最终训练 loss = {final_train_loss:.4f}')

    # 测试评估（价格空间：MSE / RMSE / MAE）
    for net in nets: net.eval()
    with torch.no_grad():
        raw = [nets[i](X_te[i], C_te[i], pool) for i in range(N_STEPS)]
        ms, _, _, L = _align(raw)
        tgts = _targets(Y_te, L)

    results = {}
    for i in range(N_STEPS):
        pred_s  = ms[i].cpu().numpy()
        tgt_s   = tgts[i].cpu().numpy()
        metrics = evaluate(pred_s, tgt_s, scaler)
        results[STEP_LABELS[i]] = metrics
        print(f'    t+{i+1} | MSE:{metrics["MSE"]:8.2f}  '
              f'RMSE:{metrics["RMSE"]:.2f}  MAE:{metrics["MAE"]:.2f}')
    return results


# ============================================================
#  9. 模型配置表
# ============================================================
def get_model_configs():
    """
    返回 {name: (nets_fn, alpha, use_adv)} 的有序字典。
    """
    def ds(alpha, fusion, gate):
        return lambda: nn.ModuleList([
            DualStreamNet(D, HIDDEN_DIM, args.attn_heads,
                          args.layer_depth, N_AUX,
                          alpha=alpha, fusion=fusion,
                          use_gate=gate).to(DEVICE)
            for _ in range(N_STEPS)])

    def sde_only(alpha=ALPHA_LEVY):
        return lambda: nn.ModuleList([
            SDEOnlyNet(D, HIDDEN_DIM, args.layer_depth,
                       N_AUX, alpha=alpha).to(DEVICE)
            for _ in range(N_STEPS)])

    def det_only():
        return lambda: nn.ModuleList([
            DetOnlyNet(D, HIDDEN_DIM, N_AUX).to(DEVICE)
            for _ in range(N_STEPS)])

    def rnn(kind):
        return lambda: nn.ModuleList([
            RNNModel(D, HIDDEN_DIM, rnn=kind).to(DEVICE)
            for _ in range(N_STEPS)])

    def tcn():
        return lambda: nn.ModuleList([
            TCNModel(D, HIDDEN_DIM).to(DEVICE)
            for _ in range(N_STEPS)])

    return {
        # 外部基准
        'LSTM':        (rnn('LSTM'),                   ALPHA_GAUSS, False),
        'GRU':         (rnn('GRU'),                    ALPHA_GAUSS, False),
        'TCN':         (tcn(),                          ALPHA_GAUSS, False),
        # 内部消融（双流架构贡献）
        'SDE-only':    (sde_only(ALPHA_LEVY),           ALPHA_LEVY,  True),
        'Det-only':    (det_only(),                     ALPHA_GAUSS, False),
        'DS-Gaussian': (ds(ALPHA_GAUSS, 'attention', True), ALPHA_GAUSS, False),
        'DS-Concat':   (ds(ALPHA_LEVY,  'concat',    True), ALPHA_LEVY,  True),
        'DS-LDE':      (ds(ALPHA_LEVY,  'attention', True), ALPHA_LEVY,  True),
    }


# ============================================================
#  10. 主循环：多种子 → 均值 ± 标准差
# ============================================================
def run_all(tensors, scaler):
    configs = get_model_configs()
    seeds   = list(range(args.n_seeds))

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

    # 汇总：均值 ± 标准差
    summary = {}
    for name in configs:
        summary[name] = {}
        for step in STEP_LABELS:
            sv = [raw[name][s][step]['MSE']  for s in seeds]
            rv = [raw[name][s][step]['RMSE'] for s in seeds]
            mv = [raw[name][s][step]['MAE']  for s in seeds]
            summary[name][step] = {
                'MSE_mean':  np.mean(sv), 'MSE_std':  np.std(sv),
                'RMSE_mean': np.mean(rv), 'RMSE_std': np.std(rv),
                'MAE_mean':  np.mean(mv), 'MAE_std':  np.std(mv),
            }
    return summary


# ============================================================
#  11. 输出函数
# ============================================================
def print_console(summary, asset):
    for metric_key, label in [
            ('MSE',  'MSE（元²）'),
            ('RMSE', 'RMSE（元）'),
            ('MAE',  'MAE（元）')]:
        print(f'\n{"─"*70}')
        print(f'  {asset} — {label}（均值 ± 标准差，{args.n_seeds}个种子）')
        print(f'{"─"*70}')
        print(f'{"模型":<14}', end='')
        for s in STEP_LABELS: print(f'  {s:>14}', end='')
        print()
        print('─' * 70)
        for name, sd in summary.items():
            mark = ' ◄' if name == 'DS-LDE' else ''
            print(f'{name:<14}', end='')
            for s in STEP_LABELS:
                m   = sd[s][f'{metric_key}_mean']
                std = sd[s][f'{metric_key}_std']
                print(f'  {m:8.2f}±{std:5.2f}', end='')
            print(mark)


def save_csv(summary, asset):
    rows = []
    for name, sd in summary.items():
        for step in STEP_LABELS:
            d = sd[step]
            rows.append({
                'Asset':    asset,
                'Model':    name,
                'Step':     step,
                'MSE':      f"{d['MSE_mean']:.4f}",
                'MSE_std':  f"{d['MSE_std']:.4f}",
                'RMSE':     f"{d['RMSE_mean']:.2f}",
                'RMSE_std': f"{d['RMSE_std']:.2f}",
                'MAE':      f"{d['MAE_mean']:.2f}",
                'MAE_std':  f"{d['MAE_std']:.2f}",
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

    for ax, (mk, lbl) in zip(axes, [
            ('MSE',  'MSE（元²）'),
            ('RMSE', 'RMSE（元）'),
            ('MAE',  'MAE（元）')]):
        for name, sd in summary.items():
            ym = [sd[s][f'{mk}_mean'] for s in STEP_LABELS]
            ys = [sd[s][f'{mk}_std']  for s in STEP_LABELS]
            st = STYLE.get(name, {})
            ax.errorbar(x, ym, yerr=ys, label=name, capsize=3, **st)
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
    """输出三张 LaTeX 表格：MSE / RMSE / MAE，可直接粘贴进论文"""
    for mk, lbl, unit in [
            ('MSE',  'MSE',  '元^2'),
            ('RMSE', 'RMSE', '元'),
            ('MAE',  'MAE',  '元')]:
        print(f'\n% ── {asset} {lbl}表 ──')
        print(r'\begin{table}[htbp]')
        print(r'\centering')
        print(f'\\caption{{{asset} 多步预测 {lbl}（{unit}）'
              f'，均值$_{{\\pm\\text{{std}}}}$，{args.n_seeds}个随机种子}}')
        print(r'\begin{tabular}{lcccc}')
        print(r'\toprule')
        print(r'模型 & $t+1$ & $t+2$ & $t+3$ & $t+4$ \\')
        print(r'\midrule')
        # 找最优值（每列最小值）用于加粗
        best = {}
        for s in STEP_LABELS:
            best[s] = min(summary[n][s][f'{mk}_mean'] for n in summary)
        for name, sd in summary.items():
            cells = []
            for s in STEP_LABELS:
                m   = sd[s][f'{mk}_mean']
                std = sd[s][f'{mk}_std']
                cell = f'{m:.2f}$_{{\\pm {std:.2f}}}$'
                if abs(m - best[s]) < 1e-6:
                    cell = f'\\textbf{{{cell}}}'
                cells.append(cell)
            row  = ' & '.join(cells)
            name_tex = f'\\textbf{{{name}}}' if name == 'DS-LDE' else name
            print(f'{name_tex} & {row} \\\\')
        print(r'\bottomrule')
        print(r'\end{tabular}')
        print(r'\end{table}')


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
    print(f'[Seeds]  {list(range(args.n_seeds))}  [Epochs] {args.epochs}')

    scaler, splits, corr_df, nt, nv, ne = load_data(
        args.data_path, args.corr_path)
    print(f'[Data]   训练:{nt}  验证:{nv}  测试:{ne}')

    tensors = prepare_tensors(splits, corr_df)

    _diff_std_global = float(
        np.std(tensors['train'][1][0].cpu().numpy()))
    print(f'[DPL margin] {_diff_std_global:.6f}')

    if args.diag_only:
        run_diag(tensors, scaler)
    else:
        summary = run_all(tensors, scaler)
        print_console(summary, args.asset_name)
        save_csv(summary, args.asset_name)
        plot_results(summary, args.asset_name)
        print_latex(summary, args.asset_name)
        print('\n全部完成。')
