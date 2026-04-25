# ============================================================
#  实验框架：对比实验 / 消融实验 / 稳健性分析
#
#  用法：
#    # 对比实验（单资产）
#    python experiment_runner.py --mode baseline --asset sz50
#
#    # 消融实验
#    python experiment_runner.py --mode ablation --asset sz50
#
#    # 多随机种子稳健性
#    python experiment_runner.py --mode robustness --asset sz50
#
#    # 多资产稳健性
#    python experiment_runner.py --mode multiasset
#
#    # 全部跑完，输出汇总表
#    python experiment_runner.py --mode all
#
#  输出文件：
#    results_baseline.csv   对比实验汇总
#    results_ablation.csv   消融实验汇总
#    results_robustness.csv 多种子稳健性汇总
#    results_multiasset.csv 多资产汇总
# ============================================================

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import argparse
import warnings
warnings.filterwarnings('ignore')

import os
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import levy_stable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# ============================================================
#  0. 全局参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--mode',    type=str, default='all',
                    choices=['baseline', 'ablation', 'robustness', 'multiasset', 'all'])
parser.add_argument('--asset',   type=str, default='sz50',
                    help='单资产模式下的资产名，对应 ASSET_FILES 的 key')
parser.add_argument('--epochs',  type=int, default=100)
parser.add_argument('--n_steps', type=int, default=4)
parser.add_argument('--gpu',     type=int, default=0)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {device}')

# ---------------------------------------------------------------
#  资产文件路径配置
#  key: 资产名（与论文表3-1对应）
#  value: (价格CSV路径, 关联维数CSV路径)
#  按你的实际路径修改
# ---------------------------------------------------------------
ASSET_FILES = {
    'sz50':     (DATA_DIR / "sz50_index_data.csv",     CORR_PATH),
    'csi300':   (DATA_DIR / "hs300_index_data.csv",   CORR_PATH),
    'petro':    (DATA_DIR / "cnpc_data.csv",         CORR_PATH),
    'zhaoshang':(DATA_DIR / "cmb_data.csv",         CORR_PATH),
}

# 超参数（与主模型保持一致）
HIDDEN_DIM  = 64
ATTN_HEADS  = 4
LAYER_DEPTH = 25
N_STEPS     = args.n_steps
EPOCHS      = args.epochs
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
ALPHA_LEVY  = 1.2
D_EMBED     = 22        # 嵌入维数 m*，与论文3.1.2节一致，原代码为23请核对
TAU_EMBED   = 5         # 时延 τ*，与论文3.1.2节一致
N_AUX       = 10        # 关联维数向量维度
SEEDS       = [4, 42, 123, 2024, 7]   # 稳健性分析用的随机种子

# Lévy 噪声池（全局）
LEVY_POOL: torch.Tensor = None
LEVY_POOL_ALPHA = None

def build_levy_pool(hidden_dim: int, alpha: float):
    global LEVY_POOL, LEVY_POOL_ALPHA
    if LEVY_POOL is not None and LEVY_POOL_ALPHA == alpha:
        return
    pool_size = 200 * 4096
    chunk = 40960
    raw = np.empty((pool_size, hidden_dim), dtype=np.float32)
    filled = 0
    while filled < pool_size:
        cur = min(chunk, pool_size - filled)
        x = levy_stable.rvs(alpha, 0.0, size=(cur, hidden_dim), scale=0.1)
        raw[filled:filled+cur] = x.astype(np.float32, copy=False)
        filled += cur
    LEVY_POOL = torch.from_numpy(raw).clamp(-10.0, 10.0).to(device)
    LEVY_POOL_ALPHA = alpha

def sample_levy(batch_size: int, hidden_dim: int) -> torch.Tensor:
    offset = torch.randint(0, LEVY_POOL.shape[0] - batch_size, (1,)).item()
    return LEVY_POOL[offset: offset + batch_size]

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ============================================================
#  1. 数据加载（统一入口，所有模型共用）
# ============================================================
CORR_DIM_COLS = [
    'corr_dim_raw',
    'corr_dim_scaled',
    'lyap_raw',
    'lyap_scaled',
    'ret_raw',
    'ret_scaled',
    'macd_raw',
    'macd_scaled',
    'rsi_raw',
    'rsi_scaled',
]

def load_data(asset_key: str):
    """
    返回：
      train_df, val_df, test_df  : 标准化收益率 DataFrame（Date + ret_scaled）
      scaler                     : 已 fit 的 StandardScaler
      corr_dim_raw               : 关联维数 DataFrame
      ret_dates                  : 收益率序列对应的日期数组
      n_train, n_val, n_test     : 各集样本量
    """
    price_path, corr_path = ASSET_FILES[asset_key]
    data      = pd.read_csv(price_path)
    data['date'] = pd.to_datetime(data['date'])
    close_all = data['close'].values.astype(float)
    dates_all = data['date'].values

    # 对数收益率
    ret_all   = np.diff(np.log(close_all)).reshape(-1, 1)
    ret_dates = dates_all[1:]

    n_total = len(ret_all)
    n_train = int(np.ceil(TRAIN_RATIO * n_total))
    n_val   = int(np.ceil(VAL_RATIO   * n_total))
    n_test  = n_total - n_train - n_val

    scaler = StandardScaler()
    train_sc = scaler.fit_transform(ret_all[:n_train]).flatten()
    val_sc   = scaler.transform(ret_all[n_train:n_train+n_val]).flatten()
    test_sc  = scaler.transform(ret_all[n_train+n_val:]).flatten()

    def make_df(dates, vals):
        return pd.DataFrame({'Date': dates, 'ret_scaled': vals})

    train_df = make_df(ret_dates[:n_train],              train_sc)
    val_df   = make_df(ret_dates[n_train:n_train+n_val], val_sc)
    test_df  = make_df(ret_dates[n_train+n_val:],        test_sc)

    corr_dim_raw = pd.read_csv(corr_path)
    if 'Date' not in corr_dim_raw.columns:
        if 'date' in corr_dim_raw.columns:
            corr_dim_raw['Date'] = pd.to_datetime(corr_dim_raw['date'])
        else:
            corr_dim_raw['Date'] = pd.to_datetime(corr_dim_raw.iloc[:, 0])
    corr_dim_raw['Date'] = pd.to_datetime(corr_dim_raw['Date'])
    missing = [c for c in CORR_DIM_COLS if c not in corr_dim_raw.columns]
    if missing:
        raise ValueError(f'corr_dim 文件缺少列: {missing}')
    corr_dim_raw = corr_dim_raw[['Date'] + CORR_DIM_COLS]

    return (train_df, val_df, test_df, scaler,
            corr_dim_raw, ret_dates, n_train, n_val, n_test)


def align_corr_dim(date_series, corr_df):
    df_dates = pd.DataFrame({'Date': pd.to_datetime(date_series)})
    corr_df  = corr_df.copy()
    corr_df['Date'] = pd.to_datetime(corr_df['Date'])
    merged   = df_dates.merge(corr_df, on='Date', how='left')
    merged[CORR_DIM_COLS] = merged[CORR_DIM_COLS].ffill().fillna(0.0)
    return merged[CORR_DIM_COLS].values.astype(np.float64)

# ============================================================
#  2. PSR（相空间重构）& 滑动窗口 两种输入模式
# ============================================================
def psr_build_xy(df, tau: int, d: int, T: int, drop_tail: int = 0):
    """Takens 延迟嵌入，输出 X [N, d]，y [N]"""
    values = np.array(df)[:, 1].astype(float)
    n      = len(values)
    width  = n - (d - 1) * tau - T
    if width < 1:
        raise ValueError("tau/d/T 过大，无法构造样本")
    Xn = np.stack([values[i*tau: i*tau+width] for i in range(d)], axis=1)
    Yn = values[T + (d-1)*tau: T + (d-1)*tau + width]
    arr = np.concatenate([Xn, Yn.reshape(-1, 1)], axis=1)
    if drop_tail > 0:
        arr = arr[:-drop_tail]
    return arr[:, :d].astype(np.float64), arr[:, d].astype(np.float64)


def sliding_build_xy(df, window: int, T: int, drop_tail: int = 0):
    """
    标准滑动窗口：X [N, window]，y [N]
    消融实验 w/o PSR 使用，window=D_EMBED 保持公平性
    """
    values = np.array(df)[:, 1].astype(float)
    n      = len(values)
    rows_x, rows_y = [], []
    for i in range(window, n - T + 1):
        rows_x.append(values[i - window: i])
        rows_y.append(values[i + T - 1])
    X = np.array(rows_x, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)
    if drop_tail > 0:
        X, y = X[:-drop_tail], y[:-drop_tail]
    return X, y


def prepare_tensors(train_df, val_df, test_df,
                    corr_dim_raw, ret_dates,
                    n_train, n_val,
                    use_psr: bool = True):
    """
    准备 X / Y / C 张量列表（长度 = N_STEPS）。
    use_psr=True  → Takens 嵌入（DS-LDE、SDE、LDE 使用）
    use_psr=False → 滑动窗口（w/o PSR 消融变体使用）
    """
    PSR_OFFSET = (D_EMBED - 1) * TAU_EMBED

    X_tr, Y_tr, C_tr = [], [], []
    X_va, Y_va, C_va = [], [], []
    X_te, Y_te, C_te = [], [], []

    corr_tr = align_corr_dim(ret_dates[:n_train],              corr_dim_raw)
    corr_va = align_corr_dim(ret_dates[n_train:n_train+n_val], corr_dim_raw)
    corr_te = align_corr_dim(ret_dates[n_train+n_val:],        corr_dim_raw)

    def slice_cd(cd_full, n_x, drop):
        start = PSR_OFFSET if use_psr else D_EMBED
        arr   = cd_full[start: start + n_x + drop]
        if drop > 0:
            arr = arr[:-drop]
        return arr[:n_x]

    def tt(arr):
        return torch.from_numpy(arr.astype(np.float32)).to(device)

    # PSR 输出形状：[N, D]，unsqueeze(1) → [N, 1, D]
    # seq_len=1 是有意为之：D 维 PSR 向量已将时序结构编码在特征维度内。
    # LDEModule  取 x_seq[:, -1, :] = 完整 PSR 向量，做 E-M 随机积分
    # FeatureStream 用 GRU 以 D 维特征处理长度为1的序列，输出隐状态
    # 两路计算路径不同，梯度解耦，注意力可以学到有意义的权重分配

    for T in range(1, N_STEPS + 1):
        drop = T - 1
        if use_psr:
            xtr, ytr = psr_build_xy(train_df, TAU_EMBED, D_EMBED, T, drop)
            xva, yva = psr_build_xy(val_df,   TAU_EMBED, D_EMBED, T, drop)
            xte, yte = psr_build_xy(test_df,  TAU_EMBED, D_EMBED, T, drop)
        else:
            xtr, ytr = sliding_build_xy(train_df, D_EMBED, T, drop)
            xva, yva = sliding_build_xy(val_df,   D_EMBED, T, drop)
            xte, yte = sliding_build_xy(test_df,  D_EMBED, T, drop)

        X_tr.append(tt(xtr).unsqueeze(1))
        Y_tr.append(tt(ytr))
        C_tr.append(tt(slice_cd(corr_tr, xtr.shape[0], drop)))

        X_va.append(tt(xva).unsqueeze(1))
        Y_va.append(tt(yva))
        C_va.append(tt(slice_cd(corr_va, xva.shape[0], drop)))

        X_te.append(tt(xte).unsqueeze(1))
        Y_te.append(tt(yte))
        C_te.append(tt(slice_cd(corr_te, xte.shape[0], drop)))

    return (X_tr, Y_tr, C_tr), (X_va, Y_va, C_va), (X_te, Y_te, C_te)

# ============================================================
#  3. 模型定义
# ============================================================

# ---- 3a. DS-LDE 子模块（与主脚本一致）----

class DriftNet(nn.Module):
    def __init__(self, h): super().__init__(); self.net = nn.Sequential(nn.Linear(h,h), nn.ReLU(True))
    def forward(self, t, x): return self.net(x)

class DiffusionNet(nn.Module):
    def __init__(self, h, m=100): super().__init__(); self.net = nn.Sequential(nn.Linear(h,m), nn.ReLU(True), nn.Linear(m,1))
    def forward(self, t, x): return self.net(x)

class LDEModule(nn.Module):
    SIGMA_MIN, SIGMA_MAX = 0.1, 1.5
    def __init__(self, input_dim, hidden_dim, layer_depth=25, sigma=0.5,
                 n_aux=10, ablate_d2=False, alpha=1.2):
        super().__init__()
        self.layer_depth = layer_depth
        self.delta_t     = 1.0 / layer_depth
        self.alpha       = alpha
        self.ablate_d2   = ablate_d2
        self.downsample  = nn.Linear(input_dim, hidden_dim)
        self.drift       = DriftNet(hidden_dim)
        self.diffusion   = DiffusionNet(hidden_dim)
        self.d2_gate     = nn.Sequential(
            nn.Linear(n_aux, 16), nn.ReLU(True), nn.Linear(16, 1), nn.Sigmoid())
        if ablate_d2:
            for p in self.d2_gate.parameters(): p.requires_grad = False

    def forward(self, x_seq, aux_feat, training_diffusion=False):
        out = self.downsample(x_seq[:, -1, :])
        if training_diffusion:
            return self.diffusion(0.0, out.detach())
        if self.ablate_d2:
            diff_macro = torch.full_like(aux_feat[:, :1],
                                         (self.SIGMA_MIN + self.SIGMA_MAX) / 2.0)
        else:
            gate       = self.d2_gate(aux_feat)
            diff_macro = self.SIGMA_MIN + (self.SIGMA_MAX - self.SIGMA_MIN) * gate
        diff_scale = diff_macro * torch.sigmoid(self.diffusion(0.0, out))
        for step in range(self.layer_depth):
            if self.alpha >= 1.999:
                noise = (0.1 * (2.0 ** 0.5)) * torch.randn_like(out)
            else:
                noise = sample_levy(out.shape[0], out.shape[1]).to(dtype=out.dtype)
            out   = (out + self.drift(float(step)/self.layer_depth, out) * self.delta_t
                     + diff_scale * (self.delta_t ** (1.0/self.alpha)) * noise)
        return out

class FeatureStream(nn.Module):
    """
    确定性分支：GRU 编码全序列时序结构 + 关联维数辅助特征。

    与 LDEModule 的输入视图差异：
      LDEModule : x_seq[:, -1, :]  → 局部状态（最后时步）→ E-M 随机积分
      FeatureStream : GRU(x_seq)   → 全局时序上下文     → 确定性编码

    x_seq 形状为 [B, 1, D]（单时步 PSR 嵌入向量）。
    GRU 以 D 维特征为输入，hidden=hidden_dim，输出最后隐状态。
    再与 aux_feat 拼接后过线性层。
    """
    def __init__(self, input_dim, hidden_dim, n_aux=10):
        super().__init__()
        # GRU 编码时序：输入维度=D，隐层=hidden_dim
        self.gru    = nn.GRU(input_dim, hidden_dim,
                             num_layers=1, batch_first=True)
        # 拼接辅助特征后映射
        self.linear = nn.Linear(hidden_dim + n_aux, hidden_dim)
        self.norm   = nn.LayerNorm(hidden_dim)
        self.act    = nn.ReLU(True)

    def forward(self, x_seq, aux_feat):
        # x_seq: [B, 1, D] → GRU → h_last: [B, hidden_dim]
        _, h_n  = self.gru(x_seq)
        h_last  = h_n[-1]                                    # [B, hidden_dim]
        inp     = torch.cat([h_last, aux_feat], dim=-1)      # [B, hidden_dim+n_aux]
        return self.act(self.norm(self.linear(inp)))          # [B, hidden_dim]


class AttentionFusion(nn.Module):
    """
    双流自注意力融合。

    两路特征在进入注意力前分别做 LayerNorm，
    确保量纲对齐，避免随机分支的 Lévy 噪声尺度
    压制确定性分支的梯度信号。
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm_lde  = nn.LayerNorm(hidden_dim)
        self.norm_feat = nn.LayerNorm(hidden_dim)
        self.attn      = nn.MultiheadAttention(hidden_dim, num_heads,
                                               batch_first=True)
        self.norm_out  = nn.LayerNorm(hidden_dim)

    def forward(self, H_lde, H_feat):
        # 输入归一化，消除两路量纲差异
        H_lde  = self.norm_lde(H_lde)
        H_feat = self.norm_feat(H_feat)
        seq    = torch.stack([H_lde, H_feat], dim=1)   # [B, 2, D_h]
        out, _ = self.attn(seq, seq, seq)
        out    = self.norm_out(out + seq)               # 残差
        return out.reshape(out.size(0), -1)             # [B, 2*D_h]


class LinearFusion(nn.Module):
    """消融变体 w/o Attention：直接拼接 + 线性层（不含归一化）"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

    def forward(self, H_lde, H_feat):
        return F.relu(self.fc(torch.cat([H_lde, H_feat], dim=-1)))

class PredictionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.trunk          = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(True))
        self.head_mean      = nn.Linear(hidden_dim, 1)
        self.head_log_sigma = nn.Linear(hidden_dim, 1)
        self.head_dir       = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        h = self.trunk(x)
        return self.head_mean(h).squeeze(-1), self.head_log_sigma(h).squeeze(-1), self.head_dir(h)


# ---- 3b. DS-LDE 完整模型（支持消融配置）----

class DualStreamSDENet(nn.Module):
    """
    cfg 字典控制消融变体：
      use_psr       : True=PSR输入  False=滑动窗口（在 prepare_tensors 层控制）
      use_levy      : True=α-Lévy  False=高斯噪声（α=2）
      use_attention : True=MHA融合 False=线性拼接融合
      use_d2_gate   : True=D₂门控  False=固定扩散尺度均值
    """
    def __init__(self, input_dim, hidden_dim=64, num_heads=4,
                 layer_depth=25, sigma=0.5, n_aux=10, cfg=None):
        super().__init__()
        cfg = cfg or {}
        alpha      = ALPHA_LEVY if cfg.get('use_levy', True) else 2.0
        ablate_d2  = not cfg.get('use_d2_gate', True)
        use_attn   = cfg.get('use_attention', True)
        self.lde   = LDEModule(input_dim, hidden_dim, layer_depth, sigma,
                               n_aux, ablate_d2=ablate_d2, alpha=alpha)
        self.feat  = FeatureStream(input_dim, hidden_dim, n_aux)
        self.fuse  = (AttentionFusion(hidden_dim, num_heads)
                      if use_attn else LinearFusion(hidden_dim))
        self.head  = PredictionHead(hidden_dim)

    def forward(self, x_seq, aux_feat, training_diffusion=False):
        if training_diffusion:
            return self.lde(x_seq, aux_feat, training_diffusion=True)
        H_lde  = self.lde(x_seq, aux_feat)
        H_feat = self.feat(x_seq, aux_feat)
        fused  = self.fuse(H_lde, H_feat)
        return self.head(fused)


# ---- 3c. LSTM 基准模型 ----

class LSTMBaseline(nn.Module):
    """双层 LSTM，输入 [B, 1, D] → 预测标量均值（无概率头）"""
    def __init__(self, input_dim, hidden_dim=64, n_aux=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.2)
        # 辅助特征拼接后接预测头
        self.fc   = nn.Linear(hidden_dim + n_aux, 1)

    def forward(self, x_seq, aux_feat, **kwargs):
        out, _ = self.lstm(x_seq)
        h_last = out[:, -1, :]                          # [B, hidden_dim]
        fused  = torch.cat([h_last, aux_feat], dim=-1)  # [B, hidden_dim+n_aux]
        mean   = self.fc(fused).squeeze(-1)
        # 补齐 log_sigma 和 dir_logit 接口，保持与 SDE 模型一致
        log_sigma = torch.zeros_like(mean)
        dir_logit = torch.zeros(mean.size(0), 2, device=mean.device)
        return mean, log_sigma, dir_logit

# ============================================================
#  4. 训练 / 评估工具
# ============================================================

SIGMA_MIN = 0.1
ITER      = 50

def make_nets(model_type: str, cfg: dict = None):
    """为 N_STEPS 个预测步长各实例化一个独立网络"""
    nets_list = []
    for _ in range(N_STEPS):
        if model_type == 'lstm':
            net = LSTMBaseline(D_EMBED, HIDDEN_DIM, N_AUX)
        else:  # ds_lde / sde / lde / ablation variants
            net = DualStreamSDENet(D_EMBED, HIDDEN_DIM, ATTN_HEADS,
                                   LAYER_DEPTH, sigma=0.5,
                                   n_aux=N_AUX, cfg=cfg)
        nets_list.append(net.to(device))
    return nn.ModuleList(nets_list)


def make_optimizers(nets, model_type):
    """
    SDE 系列：双优化器（主网络 SGD + 判别器 SGD）
    LSTM：单 Adam 优化器

    FeatureStream 使用 3× 主学习率，原因：
      LDEModule 的随机分支因 Lévy 噪声注入，隐状态方差天然较大，
      梯度信号较强；FeatureStream 的确定性分支梯度相对平稳但偏小，
      提高学习率使两路特征在收敛速度上对齐，
      注意力模块才能学到有意义的权重分配。
    """
    if model_type == 'lstm':
        opt_F = optim.Adam(nets.parameters(), lr=1e-3)
        opt_G = None
    else:
        main_params = (
            [{'params': net.lde.downsample.parameters(), 'lr': 1e-4} for net in nets] +
            [{'params': net.lde.drift.parameters(),      'lr': 1e-4} for net in nets] +
            [{'params': net.feat.parameters(),           'lr': 3e-4} for net in nets] +  # 3×
            [{'params': net.fuse.parameters(),           'lr': 1e-4} for net in nets] +
            [{'params': net.head.parameters(),           'lr': 1e-4} for net in nets]
        )
        diff_params = [{'params': net.lde.diffusion.parameters()} for net in nets]
        opt_F = optim.SGD(main_params, lr=1e-4, momentum=0.9, weight_decay=5e-4)
        opt_G = optim.SGD(diff_params, lr=0.01, momentum=0.9, weight_decay=5e-4)
    return opt_F, opt_G


def _diff_std_val(Y_trains):
    return float(np.std(Y_trains[0].cpu().numpy()))

TAIL_CUTS = list(range(N_STEPS - 1, -1, -1))

def align_outputs(raw):
    means, lss, dls = [], [], []
    for i, (m, ls, dl) in enumerate(raw):
        cut = TAIL_CUTS[i]
        if cut > 0: m, ls, dl = m[:-cut], ls[:-cut], dl[:-cut]
        means.append(m); lss.append(ls); dls.append(dl)
    min_len = min(x.shape[0] for x in means)
    return ([x[:min_len] for x in means],
            [x[:min_len] for x in lss],
            [x[:min_len] for x in dls],
            min_len)

def get_targets(Y_list, min_len):
    return [Y_list[i][:min_len] for i in range(N_STEPS)]

def compute_losses(nets, X_list, C_list, Y_list, opt_F, opt_G,
                   model_type, diff_std):
    """单次前向+反向，返回损失标量"""
    # 主网络
    opt_F.zero_grad()
    raw     = [nets[i](X_list[i], C_list[i]) for i in range(N_STEPS)]
    means, log_sigmas, dir_logits, min_len = align_outputs(raw)
    targets = get_targets(Y_list, min_len)

    loss_f = torch.tensor(0.0, device=device)
    for i in range(N_STEPS):
        y, mu, ls, dl = targets[i], means[i], log_sigmas[i], dir_logits[i]
        sigma  = F.softplus(ls).clamp(min=SIGMA_MIN) + 1e-3
        l_mse  = F.mse_loss(mu, y)
        l_nll  = (torch.log(sigma) + (y - mu)**2 / (2*sigma**2)).mean()
        l_dpl  = torch.clamp(diff_std - y * mu, min=0.0).mean()
        l_cls  = F.cross_entropy(dl, (y > 0).long())
        loss_f = loss_f + l_mse + 0.5*l_nll + 1.0*l_dpl + 1.0*l_cls

    loss_f.backward()
    nn.utils.clip_grad_norm_(
        [p for g in opt_F.param_groups for p in g['params']], max_norm=5.0)
    opt_F.step()

    # 对抗训练（仅 SDE 系列）
    if opt_G is not None:
        opt_G.zero_grad()
        real_out = [nets[i](X_list[i], C_list[i], training_diffusion=True)
                    for i in range(N_STEPS)]
        X_noisy  = [X_list[i] + 2.0 * torch.randn_like(X_list[i])
                    for i in range(N_STEPS)]
        fake_out = [nets[i](X_noisy[i], C_list[i], training_diffusion=True)
                    for i in range(N_STEPS)]
        bce = nn.BCEWithLogitsLoss()
        l_adv = (sum(bce(r, torch.zeros_like(r)) for r in real_out) +
                 sum(bce(f, torch.ones_like(f))  for f in fake_out))
        l_adv.backward()
        nn.utils.clip_grad_norm_(
            [p for g in opt_G.param_groups for p in g['params']], max_norm=5.0)
        opt_G.step()

    return loss_f.item()


@torch.no_grad()
def evaluate(nets, X_list, C_list, Y_list, scaler):
    """
    返回各步长的 RMSE / MAE / DA（收益率量纲）
    """
    for net in nets: net.eval()
    raw     = [nets[i](X_list[i], C_list[i]) for i in range(N_STEPS)]
    means, _, _, min_len = align_outputs(raw)
    targets = get_targets(Y_list, min_len)

    results = {}
    for i in range(N_STEPS):
        pred_sc = means[i].cpu().numpy()
        true_sc = targets[i].cpu().numpy()
        # 还原至收益率量纲
        pred_r  = scaler.inverse_transform(pred_sc.reshape(-1,1)).flatten()
        true_r  = scaler.inverse_transform(true_sc.reshape(-1,1)).flatten()
        rmse    = float(np.sqrt(mean_squared_error(true_r, pred_r)))
        mae     = float(np.mean(np.abs(true_r - pred_r)))
        da      = float(np.mean(np.sign(true_r) == np.sign(pred_r)))
        results[f't+{i+1}'] = {'RMSE': rmse, 'MAE': mae, 'DA': da}

    for net in nets: net.train()
    return results


def train_and_eval(model_type: str, cfg: dict,
                   data_tuple: tuple, scaler,
                   seed: int = 4,
                   use_psr: bool = True,
                   verbose: bool = False):
    """
    完整训练一个模型，返回测试集评估结果字典。

    data_tuple = (train_dfs, val_dfs, test_dfs, corr_dim_raw,
                  ret_dates, n_train, n_val)
    """
    setup_seed(seed)
    (train_df, val_df, test_df, _scaler,
     corr_dim_raw, ret_dates, n_train, n_val, _n_test) = data_tuple

    (X_tr, Y_tr, C_tr), (X_va, Y_va, C_va), (X_te, Y_te, C_te) = \
        prepare_tensors(train_df, val_df, test_df, corr_dim_raw,
                        ret_dates, n_train, n_val, use_psr=use_psr)

    if model_type != 'lstm':
        use_levy = cfg.get('use_levy', True)
        alpha = ALPHA_LEVY if use_levy else 2.0
        if use_levy:
            build_levy_pool(HIDDEN_DIM, alpha=alpha)

    nets   = make_nets(model_type, cfg)
    opt_F, opt_G = make_optimizers(nets, model_type)
    diff_std = _diff_std_val(Y_tr)

    best_val_rmse = float('inf')
    best_state    = None
    patience, patience_limit = 0, 15   # 早停

    for epoch in range(1, EPOCHS + 1):
        for net in nets: net.train()
        for _ in range(ITER):
            compute_losses(nets, X_tr, C_tr, Y_tr, opt_F, opt_G,
                           model_type, diff_std)

        # 学习率衰减（第20个 epoch）：各组按自身 lr × 0.1
        if epoch == 20:
            for pg in opt_F.param_groups:
                pg['lr'] = pg['lr'] * 0.1

        # 验证集早停
        val_res   = evaluate(nets, X_va, C_va, Y_va, scaler)
        val_rmse  = np.mean([val_res[k]['RMSE'] for k in val_res])
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state    = copy.deepcopy(nets.state_dict())
            patience      = 0
        else:
            patience += 1
        if patience >= patience_limit:
            if verbose:
                print(f'  早停 @ epoch {epoch}，最佳验证 RMSE={best_val_rmse:.6f}')
            break

        if verbose and epoch % 10 == 0:
            te_res = evaluate(nets, X_te, C_te, Y_te, scaler)
            print(f'  Epoch {epoch:3d} | val_RMSE={val_rmse:.5f} | '
                  f'test t+1 RMSE={te_res["t+1"]["RMSE"]:.5f} '
                  f'DA={te_res["t+1"]["DA"]:.3f}')

    # 载入最佳权重后做最终评估
    nets.load_state_dict(best_state)
    return evaluate(nets, X_te, C_te, Y_te, scaler)


# ============================================================
#  5. 对比实验
# ============================================================
# 模型配置表（model_type, cfg, use_psr, 论文名称）
BASELINE_CONFIGS = [
    ('lstm',   {},                                                  True,  'LSTM'),
    ('ds_lde', {'use_levy': False, 'use_attention': True,
                'use_d2_gate': True},                              True,  'SDE'),
    ('ds_lde', {'use_levy': True,  'use_attention': False,
                'use_d2_gate': False},                             True,  'LDE'),
    ('ds_lde', {'use_levy': True,  'use_attention': True,
                'use_d2_gate': True},                              True,  'DS-LDE'),
]

# 消融配置表
ABLATION_CONFIGS = [
    ('ds_lde', {'use_levy': True, 'use_attention': True,
                'use_d2_gate': True},   True,  'DS-LDE (完整)'),
    ('ds_lde', {'use_levy': True, 'use_attention': True,
                'use_d2_gate': True},   False, 'w/o PSR'),
    ('ds_lde', {'use_levy': False,'use_attention': True,
                'use_d2_gate': True},   True,  'w/o Lévy'),
    ('ds_lde', {'use_levy': True, 'use_attention': False,
                'use_d2_gate': True},   True,  'w/o Attention'),
    ('ds_lde', {'use_levy': True, 'use_attention': True,
                'use_d2_gate': False},  True,  'w/o D₂门控'),
]


def run_experiment(configs, asset_key, seeds, tag, verbose=False):
    """
    跑一组实验配置，对给定资产和种子列表求均值±标准差。
    返回 DataFrame。
    """
    print(f'\n{"="*60}')
    print(f'[{tag}] 资产={asset_key}  种子数={len(seeds)}')
    print(f'{"="*60}')

    data_tuple = load_data(asset_key)
    scaler     = data_tuple[3]

    rows = []
    for model_type, cfg, use_psr, name in configs:
        seed_results = {f't+{i+1}': {'RMSE':[], 'MAE':[], 'DA':[]}
                        for i in range(N_STEPS)}
        for seed in seeds:
            print(f'  {name} | seed={seed} ...', flush=True)
            res = train_and_eval(model_type, cfg, data_tuple, scaler,
                                 seed=seed, use_psr=use_psr, verbose=verbose)
            for step_key, metrics in res.items():
                for m, v in metrics.items():
                    seed_results[step_key][m].append(v)

        # 汇总均值 ± 标准差
        for step_key in seed_results:
            row = {'Model': name, 'Step': step_key}
            for m in ['RMSE', 'MAE', 'DA']:
                vals = seed_results[step_key][m]
                row[f'{m}_mean'] = np.mean(vals)
                row[f'{m}_std']  = np.std(vals)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_table(df, metric='RMSE'):
    """打印论文风格的对比表（均值±标准差）"""
    steps = [f't+{i+1}' for i in range(N_STEPS)]
    models = df['Model'].unique()
    header = f"{'模型':<20}" + ''.join(f'{s:>18}' for s in steps)
    print(f'\n--- {metric} (均值 ± 标准差，收益率量纲) ---')
    print(header)
    print('-' * (20 + 18 * N_STEPS))
    for model in models:
        sub = df[df['Model'] == model]
        vals = []
        for s in steps:
            row = sub[sub['Step'] == s].iloc[0]
            vals.append(f'{row[f"{metric}_mean"]:.5f}±{row[f"{metric}_std"]:.5f}')
        print(f'{model:<20}' + ''.join(f'{v:>18}' for v in vals))


# ============================================================
#  6. 主入口
# ============================================================

def run_baseline(asset_key):
    seeds  = SEEDS[:3]   # 对比实验用3个种子
    df     = run_experiment(BASELINE_CONFIGS, asset_key, seeds,
                            tag='对比实验', verbose=True)
    fname  = f'results_baseline_{asset_key}.csv'
    df.to_csv(fname, index=False, encoding='utf-8-sig')
    print(f'\n[保存] {fname}')
    for m in ['RMSE', 'MAE', 'DA']:
        print_table(df, m)
    return df


def run_ablation(asset_key):
    seeds  = SEEDS[:3]
    df     = run_experiment(ABLATION_CONFIGS, asset_key, seeds,
                            tag='消融实验', verbose=True)
    fname  = f'results_ablation_{asset_key}.csv'
    df.to_csv(fname, index=False, encoding='utf-8-sig')
    print(f'\n[保存] {fname}')
    for m in ['RMSE', 'MAE', 'DA']:
        print_table(df, m)
    return df


def run_robustness(asset_key):
    """5个种子，仅跑 DS-LDE，考察结果稳定性"""
    configs = [c for c in BASELINE_CONFIGS if c[3] == 'DS-LDE']
    df      = run_experiment(configs, asset_key, SEEDS,
                             tag='稳健性（多种子）', verbose=False)
    fname   = f'results_robustness_{asset_key}.csv'
    df.to_csv(fname, index=False, encoding='utf-8-sig')
    print(f'\n[保存] {fname}')
    print_table(df, 'RMSE')
    return df


def run_multiasset():
    """在所有资产上跑 DS-LDE（3个种子），汇总跨资产性能"""
    configs  = [c for c in BASELINE_CONFIGS if c[3] == 'DS-LDE']
    seeds    = SEEDS[:3]
    all_rows = []
    for asset_key in ASSET_FILES:
        print(f'\n>>> 资产: {asset_key}')
        try:
            data_tuple = load_data(asset_key)
        except FileNotFoundError as e:
            print(f'  跳过（文件未找到）: {e}')
            continue
        scaler = data_tuple[3]
        for model_type, cfg, use_psr, name in configs:
            seed_results = {f't+{i+1}': {'RMSE':[], 'MAE':[], 'DA':[]}
                            for i in range(N_STEPS)}
            for seed in seeds:
                res = train_and_eval(model_type, cfg, data_tuple, scaler,
                                     seed=seed, use_psr=use_psr)
                for step_key, metrics in res.items():
                    for m, v in metrics.items():
                        seed_results[step_key][m].append(v)
            for step_key in seed_results:
                row = {'Asset': asset_key, 'Model': name, 'Step': step_key}
                for m in ['RMSE', 'MAE', 'DA']:
                    vals = seed_results[step_key][m]
                    row[f'{m}_mean'] = np.mean(vals)
                    row[f'{m}_std']  = np.std(vals)
                all_rows.append(row)

    df    = pd.DataFrame(all_rows)
    fname = 'results_multiasset.csv'
    df.to_csv(fname, index=False, encoding='utf-8-sig')
    print(f'\n[保存] {fname}')
    # 打印跨资产 t+1 RMSE 汇总
    sub = df[df['Step'] == 't+1'][['Asset','RMSE_mean','RMSE_std','DA_mean']]
    print('\n--- 跨资产 t+1 RMSE 汇总 ---')
    print(sub.to_string(index=False))
    return df


if __name__ == '__main__':
    mode = args.mode
    asset = args.asset if args.asset in ASSET_FILES else 'sz50'

    if mode == 'baseline':
        run_baseline(asset)
    elif mode == 'ablation':
        run_ablation(asset)
    elif mode == 'robustness':
        run_robustness(asset)
    elif mode == 'multiasset':
        run_multiasset()
    elif mode == 'all':
        print('\n>>> 阶段1：对比实验')
        df_b = run_baseline(asset)
        print('\n>>> 阶段2：消融实验')
        df_a = run_ablation(asset)
        print('\n>>> 阶段3：多种子稳健性')
        df_r = run_robustness(asset)
        print('\n>>> 阶段4：多资产')
        df_m = run_multiasset()
        print('\n全部实验完成。')
                      