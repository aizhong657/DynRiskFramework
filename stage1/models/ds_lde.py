# ============================================================
#  DS-LSDE 主模型
#  双流融合神经随机微分方程（Dual-Stream Lévy-SDE）
#
#  架构说明：
#    上路（随机流）  ：LDEModule — Lévy 驱动 SDE（α=1.2），
#                      Euler-Maruyama 积分，输出隐状态 H_lde
#    下路（特征流）  ：FeatureStream — mean pooling + 关联维数
#                      + 动力学辅助特征 → H_feature
#    融合层         ：AttentionFusion — MultiheadAttention
#                      Self-Attention，残差 + LayerNorm
#    解码器         ：PredictionHead — 共享 trunk 三头输出
#                      mean / log_sigma / dir_logit
#
#    解码器采用高斯参数化 (μ̂, σ̂)，厚尾建模由两层分担：
#      · LDEModule 的 Lévy 积分（隐状态层跳跃捕捉）
#      · 第四章 ETGPD-Transformer GPD 尾部（残差层）
#
#  损失函数：
#    L_total = MSE + λ1·NLL(高斯) + λ2·DPL(Hinge) + λ3·CE(方向分类)
#
#  输出（第16节）：
#    sde_bridge_features.npz
#      ├── resid_z    : 一步预测残差 ε̂_t（滚动z-score）
#      ├── sigma_z    : 条件波动率 σ̂_t（价格空间，滚动z-score）
#      ├── delta_mu_z : 均值变化速率 Δμ̂_t（滚动z-score）
#      ├── resid_raw  : 残差原始值（标准化空间）
#      └── sigma_raw  : 波动率原始值（价格空间）
# ============================================================

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import levy_stable
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchsde  # noqa: F401

# ============================================================
#  1. 超参数
# ============================================================
parser = argparse.ArgumentParser(description='DS-LSDE 主模型')
parser.add_argument('--epochs',         type=int,   default=100)
parser.add_argument('--n_steps',        type=int,   default=4)
parser.add_argument('--lr',             type=float, default=1e-4)
parser.add_argument('--lr2',            type=float, default=0.01)
parser.add_argument('--droprate',       type=float, default=0.1)
parser.add_argument('--decreasing_lr',  default=[20], nargs='+')
parser.add_argument('--decreasing_lr2', default=[],   nargs='+')
parser.add_argument('--hidden_dim',     type=int,   default=64)
parser.add_argument('--attn_heads',     type=int,   default=4)
parser.add_argument('--layer_depth',    type=int,   default=25)
parser.add_argument('--sigma',          type=float, default=0.5)
parser.add_argument('--lambda1',        type=float, default=0.5,
                    help='高斯NLL权重')
parser.add_argument('--lambda2',        type=float, default=1.0,
                    help='DPL方向惩罚权重')
parser.add_argument('--lambda3',        type=float, default=1.0,
                    help='方向分类CE权重')
parser.add_argument('--sigma_min',      type=float, default=0.1)
parser.add_argument('--run_omega_search', action='store_true', default=False)
parser.add_argument('--ablate_d2_gate', action='store_true', default=False)
parser.add_argument('--gpu',            type=int,   default=0)
parser.add_argument('--seed',           type=int,   default=4)
parser.add_argument('--bridge_path',    type=str,   default='sde_bridge_features.npz')
args = parser.parse_args()

# ============================================================
#  2. 随机种子 & 设备
# ============================================================
def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {device}')

# ============================================================
#  3. Lévy 稳定分布参数 & 全局噪声池
#
#  α=1.2：比 α=1.5/1.9 更厚的尾部，更强的跳跃特征，
#          适合金融危机期的极端价格跳跃建模。
#  噪声池：启动时一次性预生成，训练中随机切片复用，
#          消除每步调用 scipy 的开销（速度提升约100-200x）。
# ============================================================
ALPHA_LEVY     = 1.2
BETA_LEVY      = 0.0
LEVY_POOL_SIZE = 200 * 4096
LEVY_POOL: torch.Tensor = None


def build_levy_pool(hidden_dim: int, device: torch.device) -> None:
    global LEVY_POOL
    print(f'[噪声池] 预生成 Lévy 噪声 α={ALPHA_LEVY}，'
          f'大小={LEVY_POOL_SIZE}×{hidden_dim}…', flush=True)
    chunk = min(40960, LEVY_POOL_SIZE)
    raw = np.empty((LEVY_POOL_SIZE, hidden_dim), dtype=np.float32)
    filled = 0
    while filled < LEVY_POOL_SIZE:
        cur = min(chunk, LEVY_POOL_SIZE - filled)
        x = levy_stable.rvs(
            ALPHA_LEVY, BETA_LEVY,
            size=(cur, hidden_dim),
            scale=0.1,
        )
        raw[filled:filled + cur] = x.astype(np.float32, copy=False)
        filled += cur
    LEVY_POOL = torch.from_numpy(raw).clamp(-10.0, 10.0).to(device)
    print(f'[噪声池] 完成，内存≈{LEVY_POOL.numel()*4/1024/1024:.1f} MB')


def sample_levy_noise(batch_size: int, hidden_dim: int) -> torch.Tensor:
    pool_len = LEVY_POOL.shape[0]
    offset   = torch.randint(0, pool_len - batch_size, (1,)).item()
    return LEVY_POOL[offset : offset + batch_size]


# ============================================================
#  4. 数据加载与预处理
# ============================================================
data = pd.read_csv(DATA_DIR / "sz50_index_data.csv")
data = data[['date', 'code', 'open', 'close', 'high', 'low', 'volume']]
data['date'] = pd.to_datetime(data['date'])

stk_data = pd.DataFrame({
    'Date':  data['date'].values,
    'Open':  data['open'].values,
    'High':  data['high'].values,
    'Low':   data['low'].values,
    'Close': data['close'].values,
})

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

# ---------------------------------------------------------------
#  构造对数收益率序列（与论文式3-1一致）
#  r_t = ln(P_t / P_{t-1})，长度比价格序列少一行
# ---------------------------------------------------------------
close_all = stk_data['Close'].values.astype(float)
ret_all   = np.diff(np.log(close_all)).reshape(-1, 1)   # shape: (N-1, 1)
ret_dates = stk_data['Date'].values[1:]                 # 对齐日期（从第2个交易日起）

n_total = len(ret_all)
n_train = int(np.ceil(TRAIN_RATIO * n_total))
n_val   = int(np.ceil(VAL_RATIO   * n_total))
n_test  = n_total - n_train - n_val

ret_train = ret_all[:n_train]
ret_val   = ret_all[n_train:n_train + n_val]
ret_test  = ret_all[n_train + n_val:]

# StandardScaler 仅在训练集上 fit，防止数据泄露
scaler = StandardScaler()
train_scaled = scaler.fit_transform(ret_train).flatten()   # 标准化收益率
val_scaled   = scaler.transform(ret_val).flatten()
test_scaled  = scaler.transform(ret_test).flatten()

print(f'[Data] 对数收益率序列总长度: {n_total}')
print(f'       训练集:{ret_train.shape}  验证集:{ret_val.shape}  测试集:{ret_test.shape}')
print(f'[Scaler] 收益率均值={scaler.mean_[0]:.6f}  标准差={scaler.scale_[0]:.6f}')

# 覆盖率统计（保留，用于论文附录参考）
OMEGA = 0.3
test_scaled_ret = test_scaled
coverage = float(np.mean(np.abs(test_scaled_ret) < OMEGA))
print(f"上证50测试集覆盖率（|r̃| < {OMEGA}）: {coverage*100:.1f}%")

def make_date_value_df(dates, values):
    return pd.DataFrame({'Date': dates, 'ret_scaled': values})

train_df = make_date_value_df(ret_dates[:n_train],              train_scaled)
val_df   = make_date_value_df(ret_dates[n_train:n_train+n_val], val_scaled)
test_df  = make_date_value_df(ret_dates[n_train+n_val:],        test_scaled)

# ============================================================
#  5. 相空间重构
# ============================================================
def PhaSpaRecon(df, tau: int, d: int, T: int):
    values = np.array(df)[:, 1].astype(float)
    dates  = np.array(df)[:, 0]
    n = len(values)
    if (n - T - (d - 1) * tau) < 1:
        raise ValueError("tau 或 d 过大，超出序列长度")
    width    = n - (d - 1) * tau - 1
    Xn1      = np.stack([values[i*tau : i*tau + width] for i in range(d)], axis=1)
    Yn1      = values[T + (d-1)*tau : T + (d-1)*tau + width]
    Yn1_date = dates [T + (d-1)*tau : T + (d-1)*tau + width]
    Xn = pd.DataFrame(Xn1)
    Yn = pd.DataFrame(Yn1, columns=[0])
    Y  = pd.DataFrame({'Date': Yn1_date, 'target': Yn1})
    X  = pd.concat([Xn, Yn], axis=1)
    return Xn, Yn, Y, X

def build_xy(df, tau, d, T, drop_tail=0):
    _, _, _, X = PhaSpaRecon(df, tau=tau, d=d, T=T)
    arr = X.values
    if drop_tail > 0:
        arr = arr[:-drop_tail]
    return arr[:, :d].astype(np.float64), arr[:, d].astype(np.float64)

D, TAU     = 22, 1
N_STEPS    = args.n_steps
PSR_OFFSET = (D - 1) * TAU

x_trains, y_trains = [], []
x_vals,   y_vals   = [], []
x_tests,  y_tests  = [], []

for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr, ytr = build_xy(train_df, TAU, D, T, drop_tail=drop)
    xva, yva = build_xy(val_df,   TAU, D, T, drop_tail=drop)
    xte, yte = build_xy(test_df,  TAU, D, T, drop_tail=drop)
    x_trains.append(xtr); y_trains.append(ytr)
    x_vals.append(xva);   y_vals.append(yva)
    x_tests.append(xte);  y_tests.append(yte)
    print(f'T={T}  x_train:{xtr.shape}  x_val:{xva.shape}  x_test:{xte.shape}')

# ============================================================
#  6. 关联维数数据加载与对齐
# ============================================================
corr_dim_raw = pd.read_csv(CORR_PATH)
corr_dim_raw.iloc[:, 0] = pd.to_datetime(corr_dim_raw.iloc[:, 0])
corr_dim_raw.columns = ['Date'] + [f'cd_{i}' for i in range(1, 11)]
CORR_DIM_COLS = [f'cd_{i}' for i in range(1, 11)]
CORR_DIM = len(CORR_DIM_COLS)   # = 10
N_AUX    = CORR_DIM

def align_corr_dim(date_series, corr_df):
    df_dates = pd.DataFrame({'Date': pd.to_datetime(date_series)})
    corr_df  = corr_df.copy()
    corr_df['Date'] = pd.to_datetime(corr_df['Date'])
    merged   = df_dates.merge(corr_df, on='Date', how='left')
    merged[CORR_DIM_COLS] = merged[CORR_DIM_COLS].ffill().fillna(0.0)
    return merged[CORR_DIM_COLS].values.astype(np.float64)

# 注意：ret_dates 已经是从第2个交易日起，与 n_train/n_val/n_test 对应
corr_train_full = align_corr_dim(ret_dates[:n_train],              corr_dim_raw)
corr_val_full   = align_corr_dim(ret_dates[n_train:n_train+n_val], corr_dim_raw)
corr_test_full  = align_corr_dim(ret_dates[n_train+n_val:],        corr_dim_raw)

def slice_cd(cd_full, n_x, drop):
    start = PSR_OFFSET
    end   = start + n_x + drop
    arr   = cd_full[start:end]
    if drop > 0:
        arr = arr[:-drop]
    return arr[:n_x]

cd_trains, cd_vals, cd_tests = [], [], []
for T in range(1, N_STEPS + 1):
    drop = T - 1
    cd_trains.append(slice_cd(corr_train_full, x_trains[T-1].shape[0], drop))
    cd_vals.append(  slice_cd(corr_val_full,   x_vals[T-1].shape[0],   drop))
    cd_tests.append( slice_cd(corr_test_full,  x_tests[T-1].shape[0],  drop))

# ============================================================
#  7. 数据转 Tensor
#
#  x_seq : [N, D] → unsqueeze(1) → [N, 1, D]
#  aux   : [N, CORR_DIM=10]
# ============================================================
def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).to(device)

X_trains = [to_tensor(x).unsqueeze(1) for x in x_trains]   # [N, 1, D]
Y_trains = [to_tensor(y)              for y in y_trains]    # [N]
C_trains = [to_tensor(c)              for c in cd_trains]   # [N, 10]

X_vals   = [to_tensor(x).unsqueeze(1) for x in x_vals]
Y_vals   = [to_tensor(y)              for y in y_vals]
C_vals   = [to_tensor(c)              for c in cd_vals]

X_tests  = [to_tensor(x).unsqueeze(1) for x in x_tests]
Y_tests  = [to_tensor(y)              for y in y_tests]
C_tests  = [to_tensor(c)              for c in cd_tests]

print(f'[Tensor] X_trains[0]:{X_trains[0].shape}  '
      f'Y_trains[0]:{Y_trains[0].shape}  '
      f'C_trains[0]:{C_trains[0].shape}')

# ============================================================
#  8. 模型子模块
# ============================================================

class DriftNet(nn.Module):
    """f(t, x)：线性 + ReLU，输入输出均为 [B, hidden_dim]。"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionNet(nn.Module):
    """g(t, x)：输出 logit（不含 Sigmoid），输入输出 [B, hidden_dim] → [B, 1]。"""
    def __init__(self, hidden_dim: int, mid_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, 1),
        )
    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LDEModule(nn.Module):
    """
    上路随机流：Lévy 驱动 SDE（α=1.2）+ D₂条件门控

    数据流：
      x_seq [B, T, D] → 取最后时步 → downsample → Euler-Maruyama → H_lde [B, D_h]

    噪声从全局池切片，每步形状 [B, hidden_dim]。

    扩散尺度由两路调制信号相乘决定：
      · 宏观门控 g_t  : D₂等辅助特征 → d2_gate(两层MLP+Sigmoid) → [B,1] ∈ (0,1)
                        diff_macro = σ_min + (σ_max - σ_min) * g_t
                        高D₂(市场噪声主导) → g_t偏大 → 扩散尺度偏大 → 预测区间增宽
                        低D₂(集体行为主导) → g_t偏小 → 扩散受抑 → 模型倚重漂移项
      · 微观调制      : DiffusionNet(h_t) → Sigmoid → 基于隐状态的逐步微调

    最终 Euler-Maruyama 更新：
      diff_scale = diff_macro * σ(g̃(h_t))
      h_{t+Δt} = h_t + f(t,h_t)·Δt + diff_scale·Δt^{1/α}·η_t

    超参数：
      sigma_min = 0.1  （低D₂时扩散下界）
      sigma_max = 1.5  （高D₂时扩散上界）
    """
    SIGMA_MIN = 0.1
    SIGMA_MAX = 1.5

    def __init__(self, input_dim: int, hidden_dim: int,
                 layer_depth: int = 25, sigma: float = 0.5,
                 n_aux: int = 10):
        super().__init__()
        self.layer_depth = layer_depth
        self.sigma       = sigma
        self.delta_t     = 1.0 / layer_depth
        self.downsample  = nn.Linear(input_dim, hidden_dim)
        self.drift       = DriftNet(hidden_dim)
        self.diffusion   = DiffusionNet(hidden_dim)

        # D₂条件门控子网络
        # 输入：n_aux维辅助特征（含关联维数D₂及其他动力学指标）
        # 输出：标量门控系数 g_t ∈ (0,1)，用于线性插值扩散尺度上下界
        # 对应论文式(3-9a)：g_t = Sigmoid(W₂·ReLU(W₁·X_aux + b₁) + b₂)
        self.d2_gate = nn.Sequential(
            nn.Linear(n_aux, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        if args.ablate_d2_gate:
            for p in self.d2_gate.parameters():
                p.requires_grad = False

    def _levy_noise(self, shape, device, dtype) -> torch.Tensor:
        return sample_levy_noise(shape[0], shape[1]).to(dtype=dtype)

    def forward(self, x_seq: torch.Tensor,
                aux_feat: torch.Tensor,
                training_diffusion: bool = False) -> torch.Tensor:
        x_last = x_seq[:, -1, :]
        out    = self.downsample(x_last)

        if training_diffusion:
            return self.diffusion(0.0, out.detach())

        # 宏观门控：D₂驱动的扩散尺度线性插值
        # g_t ∈ (0,1)；高D₂→g_t大→扩散强；低D₂→g_t小→扩散弱
        if args.ablate_d2_gate:
            diff_macro = torch.full_like(
                aux_feat[:, :1],
                fill_value=(self.SIGMA_MIN + self.SIGMA_MAX) / 2.0,
            )
        else:
            gate       = self.d2_gate(aux_feat)                           # [B, 1]
            diff_macro = self.SIGMA_MIN + (self.SIGMA_MAX - self.SIGMA_MIN) * gate  # [B, 1]

        # 微观调制：基于当前隐状态的逐步扩散系数
        diff_micro   = torch.sigmoid(self.diffusion(0.0, out))          # [B, 1]

        # 联合扩散尺度：宏观（市场机制）× 微观（隐状态）
        diff_scale   = diff_macro * diff_micro                          # [B, 1]

        for step in range(self.layer_depth):
            t          = float(step) / self.layer_depth
            levy_noise = self._levy_noise(out.shape, out.device, out.dtype)
            out = (out
                   + self.drift(t, out) * self.delta_t
                   + diff_scale * (self.delta_t ** (1.0 / ALPHA_LEVY)) * levy_noise)
        return out   # [B, hidden_dim]


class FeatureStream(nn.Module):
    """
    下路确定性分支：全序列 mean pooling + 关联维数辅助特征。

    数据流：
      x_seq [B,T,D] → mean(T) → [B,D]
      cat aux [B,n_aux] → Linear+ReLU → H_feature [B,D_h]
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_aux: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim + n_aux, hidden_dim)
        self.act    = nn.ReLU(inplace=True)

    def forward(self, x_seq: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
        x_mean = x_seq.mean(dim=1)
        fused  = torch.cat([x_mean, aux_feat], dim=-1)
        return self.act(self.linear(fused))


class AttentionFusion(nn.Module):
    """
    MultiheadAttention Self-Attention 融合模块。

    数据流：
      H_lde [B,D_h] + H_feature [B,D_h]
        → stack → [B,2,D_h]
        → MHA(Q=K=V) → 残差+LayerNorm
        → reshape → [B, 2*D_h]

    注意力动态调节两路信任度：
      平稳期：两路权重相近
      危机期（D2 骤降时特征流含强信号）：权重自动向特征流倾斜
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f'hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除')
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H_lde: torch.Tensor, H_feature: torch.Tensor) -> torch.Tensor:
        seq      = torch.cat([H_lde.unsqueeze(1),
                               H_feature.unsqueeze(1)], dim=1)  # [B,2,D_h]
        attn_out, _ = self.attn(seq, seq, seq)
        attn_out    = self.norm(attn_out + seq)
        return attn_out.contiguous().reshape(attn_out.size(0), -1)  # [B,2*D_h]


class PredictionHead(nn.Module):
    """
    共享 trunk + 三头输出（高斯参数化）：
      mean      [B]    条件均值，线性输出
      log_sigma [B]    条件 log_sigma
      dir_logit [B,2]  方向分类 logits（0=下跌, 1=上涨）

    不使用 Student-t 自由度参数。厚尾建模由 Lévy 过程和
    第四章 GPD 分层承担，解码器专注均值路径估计。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        in_dim = 2 * hidden_dim
        self.trunk          = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                            nn.ReLU(inplace=True))
        self.head_mean      = nn.Linear(hidden_dim, 1)
        self.head_log_sigma = nn.Linear(hidden_dim, 1)
        self.head_dir       = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor):
        h         = self.trunk(x)
        mean      = self.head_mean(h).squeeze(-1)
        log_sigma = self.head_log_sigma(h).squeeze(-1)
        dir_logit = self.head_dir(h)
        return mean, log_sigma, dir_logit


class DualStreamSDENet(nn.Module):
    """
    完整双流融合模型（DS-LSDE）

    数据流：
      x_seq [B,T,D]  aux [B,n_aux]
         |                  |  |
      LDEModule         FeatureStream
      (Lévy α=1.2          (mean pool + 关联维数)
       + D₂门控)
         |                  |
         └──AttentionFusion──┘
                  |
            PredictionHead
            (mean, log_sigma, dir_logit)

    关键变化：aux_feat 同时送入 LDEModule（宏观门控）
    和 FeatureStream（确定性编码），实现D₂对扩散项
    的显式条件调制。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_heads: int = 4, layer_depth: int = 25,
                 sigma: float = 0.5, n_aux: int = 10):
        super().__init__()
        # n_aux 同时传给 LDEModule（用于 d2_gate）和 FeatureStream
        self.lde            = LDEModule(input_dim, hidden_dim,
                                        layer_depth, sigma, n_aux=n_aux)
        self.feature_stream = FeatureStream(input_dim, hidden_dim, n_aux=n_aux)
        self.attn_fusion    = AttentionFusion(hidden_dim, num_heads)
        self.pred_head      = PredictionHead(hidden_dim)

    def forward(self, x_seq: torch.Tensor, aux_feat: torch.Tensor,
                training_diffusion: bool = False):
        if training_diffusion:
            # 对抗训练阶段仍只需要扩散网络的 logit 输出
            return self.lde(x_seq, aux_feat, training_diffusion=True)
        # 随机流：含 D₂ 宏观门控的 Lévy 积分
        H_lde     = self.lde(x_seq, aux_feat, training_diffusion=False)
        # 确定性流：均值池化 + 辅助特征编码
        H_feature = self.feature_stream(x_seq, aux_feat)
        fused     = self.attn_fusion(H_lde, H_feature)
        return self.pred_head(fused)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
#  9. 实例化模型 & 优化器
# ============================================================
HIDDEN_DIM  = args.hidden_dim
ATTN_HEADS  = args.attn_heads
LAYER_DEPTH = args.layer_depth
SIGMA       = args.sigma
INPUT_DIM   = D
ITER        = 50
ITER_TEST   = 1

nets = nn.ModuleList([
    DualStreamSDENet(
        input_dim   = INPUT_DIM,
        hidden_dim  = HIDDEN_DIM,
        num_heads   = ATTN_HEADS,
        layer_depth = LAYER_DEPTH,
        sigma       = SIGMA,
        n_aux       = N_AUX,
    ).to(device)
    for _ in range(N_STEPS)
])

print(f'[Model] 每个 DualStreamSDENet 参数量: {count_parameters(nets[0]):,}')

build_levy_pool(hidden_dim=HIDDEN_DIM, device=device)

optimizer_F = optim.SGD(
    [{'params': net.lde.downsample.parameters()}    for net in nets] +
    [{'params': net.lde.drift.parameters()}         for net in nets] +
    [{'params': net.feature_stream.parameters()}    for net in nets] +
    [{'params': net.attn_fusion.parameters()}       for net in nets] +
    [{'params': net.pred_head.parameters()}         for net in nets],
    lr=args.lr, momentum=0.9, weight_decay=5e-4,
)
optimizer_G = optim.SGD(
    [{'params': net.lde.diffusion.parameters()} for net in nets],
    lr=args.lr2, momentum=0.9, weight_decay=5e-4,
)

criterion  = nn.BCEWithLogitsLoss()
real_label = 0.0
fake_label = 1.0
TAIL_CUTS  = list(range(N_STEPS - 1, -1, -1))

def omega_grid_search():
    raise SystemExit('mean_scale 已移除：Ω 网格搜索已禁用。')

# ============================================================
#  10. 损失函数
#
#  联合损失：L = MSE + λ1·NLL + λ2·DPL + λ3·CE
#
#  NLL  : 高斯负对数似然，优化 (μ̂, σ̂) 联合估计
#  DPL  : Hinge 方向惩罚，软约束连续值头的方向
#  CE   : 交叉熵，专门优化分类头方向（梯度最强）
#  margin = 训练集差分标准差（自适应 Margin）
# ============================================================
_diff_std = float(np.std(y_trains[0]))
print(f'[损失] DPL margin = {_diff_std:.6f}')
print(f'[损失] λ1(NLL)={args.lambda1}  λ2(DPL)={args.lambda2}  λ3(CE)={args.lambda3}')


def nll_loss(y: torch.Tensor, mean: torch.Tensor,
             log_sigma: torch.Tensor) -> torch.Tensor:
    sigma = F.softplus(log_sigma).clamp(min=args.sigma_min) + 1e-3
    return torch.mean(torch.log(sigma) + (y - mean) ** 2 / (2.0 * sigma ** 2))

def mse_loss_fn(y: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return torch.mean((y - mean) ** 2)

def dpl_loss(y: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return torch.clamp(_diff_std - y * mean, min=0.0).mean()

def cls_loss(y: torch.Tensor, dir_logit: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(dir_logit, (y > 0).long())

def combined_loss(y, mean, log_sigma, dir_logit):
    l_mse = mse_loss_fn(y, mean)
    l_nll = nll_loss(y, mean, log_sigma)
    l_dpl = dpl_loss(y, mean)
    l_cls = cls_loss(y, dir_logit)
    total = l_mse + args.lambda1*l_nll + args.lambda2*l_dpl + args.lambda3*l_cls
    return total, l_mse, l_nll, l_dpl, l_cls

# ============================================================
#  11. 前向传播公共逻辑
# ============================================================
def forward_all(nets, X_list, C_list, training_diffusion=False):
    return [nets[i](X_list[i], C_list[i],
                    training_diffusion=training_diffusion)
            for i in range(N_STEPS)]

def align_outputs(raw_outs):
    clipped_means, clipped_ls, clipped_dl = [], [], []
    for i in range(N_STEPS):
        mean_i, ls_i, dl_i = raw_outs[i]
        cut = TAIL_CUTS[i]
        if cut > 0:
            mean_i = mean_i[:-cut]
            ls_i   = ls_i[:-cut]
            dl_i   = dl_i[:-cut]
        clipped_means.append(mean_i)
        clipped_ls.append(ls_i)
        clipped_dl.append(dl_i)
    min_len    = min(m.shape[0] for m in clipped_means)
    means      = [m[:min_len]  for m in clipped_means]
    log_sigmas = [ls[:min_len] for ls in clipped_ls]
    dir_logits = [dl[:min_len] for dl in clipped_dl]
    return means, log_sigmas, dir_logits, min_len

def get_targets(Y_list, min_len):
    return [Y_list[i][:min_len] for i in range(N_STEPS)]

if args.run_omega_search:
    omega_grid_search()
    raise SystemExit(0)

# ============================================================
#  12. 训练 & 测试
# ============================================================
def train_epoch(epoch: int):
    for net in nets: net.train()
    total_loss = total_mse = total_nll = total_dpl = total_cls = 0.0
    total_loss_in = total_loss_out = 0.0

    for _ in range(ITER):
        optimizer_F.zero_grad()
        raw    = forward_all(nets, X_trains, C_trains, training_diffusion=False)
        means, log_sigmas, dir_logits, min_len = align_outputs(raw)
        targets = get_targets(Y_trains, min_len)

        loss_parts = [combined_loss(targets[i], means[i],
                                    log_sigmas[i], dir_logits[i])
                      for i in range(N_STEPS)]
        loss_f   = sum(p[0] for p in loss_parts)
        loss_mse = sum(p[1] for p in loss_parts)
        loss_nll = sum(p[2] for p in loss_parts)
        loss_dpl = sum(p[3] for p in loss_parts)
        loss_cls = sum(p[4] for p in loss_parts)

        loss_f.backward()
        nn.utils.clip_grad_norm_(
            [p for g in optimizer_F.param_groups for p in g['params']], max_norm=5.0)
        optimizer_F.step()
        total_loss += loss_f.item();  total_mse += loss_mse.item()
        total_nll  += loss_nll.item(); total_dpl += loss_dpl.item()
        total_cls  += loss_cls.item()

        optimizer_G.zero_grad()
        preds_in    = [nets[i](X_trains[i], C_trains[i], training_diffusion=True)
                       for i in range(N_STEPS)]
        labels_real = [torch.full_like(preds_in[i],  real_label) for i in range(N_STEPS)]
        loss_in     = sum(criterion(preds_in[i], labels_real[i]) for i in range(N_STEPS))
        X_noisy     = [X_trains[i] + 2.0 * torch.randn_like(X_trains[i])
                       for i in range(N_STEPS)]
        preds_out   = [nets[i](X_noisy[i], C_trains[i], training_diffusion=True)
                       for i in range(N_STEPS)]
        labels_fake = [torch.full_like(preds_out[i], fake_label) for i in range(N_STEPS)]
        loss_out    = sum(criterion(preds_out[i], labels_fake[i]) for i in range(N_STEPS))
        (loss_in + loss_out).backward()
        nn.utils.clip_grad_norm_(
            [p for g in optimizer_G.param_groups for p in g['params']], max_norm=5.0)
        optimizer_G.step()
        total_loss_in  += loss_in.item()
        total_loss_out += loss_out.item()

    with torch.no_grad():
        raw    = forward_all(nets, X_trains, C_trains, training_diffusion=False)
        means, _, dir_logits, min_len = align_outputs(raw)
        targets = get_targets(Y_trains, min_len)
        mse_list, da_list = [], []
        for i in range(N_STEPS):
            pred_np = means[i].cpu().numpy()
            if not np.isfinite(pred_np).all():
                raise RuntimeError(
                    f'Epoch {epoch}, step t+{i+1}: NaN/Inf，请降低 lr={args.lr}')
            tgt_np = targets[i].cpu().numpy()
            mse_list.append(mean_squared_error(tgt_np, pred_np))
            da_list.append(np.mean(np.sign(pred_np) == np.sign(tgt_np)))

    da_str = '  '.join(f't+{i+1}:{da_list[i]:.3f}' for i in range(N_STEPS))
    print(f'[Train] Epoch {epoch:3d} | '
          f'Loss:{total_loss/ITER:.4f}  MSE:{total_mse/ITER:.4f}  '
          f'NLL:{total_nll/ITER:.4f}  DPL:{total_dpl/ITER:.4f}  CE:{total_cls/ITER:.4f}')
    print(f'         DA: {da_str}')
    return tuple(mse_list)


def test_epoch(epoch: int):
    for net in nets: net.eval()
    with torch.no_grad():
        raw    = forward_all(nets, X_tests, C_tests, training_diffusion=False)
        means, log_sigmas, dir_logits, min_len = align_outputs(raw)
        targets = get_targets(Y_tests, min_len)
        # 逐步长计算，均在标准化收益率空间
        mse_list  = [mean_squared_error(targets[i].cpu().numpy(),
                                        means[i].cpu().numpy())
                     for i in range(N_STEPS)]
        rmse_list = [m ** 0.5 for m in mse_list]
        mae_list  = [np.mean(np.abs(targets[i].cpu().numpy() -
                                    means[i].cpu().numpy()))
                     for i in range(N_STEPS)]
        da_list   = [np.mean(np.sign(means[i].cpu().numpy()) ==
                             np.sign(targets[i].cpu().numpy()))
                     for i in range(N_STEPS)]
    rmse_str = '  '.join(f't+{i+1}:{rmse_list[i]:.6f}' for i in range(N_STEPS))
    mae_str  = '  '.join(f't+{i+1}:{mae_list[i]:.6f}'  for i in range(N_STEPS))
    da_str   = '  '.join(f't+{i+1}:{da_list[i]:.3f}'   for i in range(N_STEPS))
    print(f'[ Test] Epoch {epoch:3d} | RMSE(标准化): {rmse_str}')
    print(f'                       MAE(标准化):  {mae_str}')
    print(f'                       DA:           {da_str}')
    return tuple(mse_list)

# ============================================================
#  13. 训练主循环
# ============================================================
train_losses, test_losses = [], []
for epoch in range(1, args.epochs + 1):
    train_losses.append(train_epoch(epoch))
    test_losses.append(test_epoch(epoch))
    if epoch in [int(e) for e in args.decreasing_lr]:
        for pg in optimizer_F.param_groups: pg['lr'] *= args.droprate
    if epoch in [int(e) for e in args.decreasing_lr2]:
        for pg in optimizer_G.param_groups: pg['lr'] *= args.droprate

print('\n===== DS-LSDE 最终测试 MSE（标准化空间）=====')
step_labels = [f't+{i+1}' for i in range(N_STEPS)]
for i, label in enumerate(step_labels):
    print(f'  {label}: {test_losses[-1][i]:.6f}')

# ============================================================
#  14. 验证集残差 & 测试集最终推理
# ============================================================
def to_np(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()

def to_sigma(log_sigma_t: torch.Tensor) -> np.ndarray:
    return (F.softplus(log_sigma_t).clamp(min=args.sigma_min) + 1e-3).cpu().numpy()

def inv_ret(arr: np.ndarray) -> np.ndarray:
    """将标准化收益率还原为原始对数收益率（无量纲）"""
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

for net in nets: net.eval()

with torch.no_grad():
    raw_val = forward_all(nets, X_vals, C_vals, training_diffusion=False)
    val_means, val_log_sigmas, _, val_min_len = align_outputs(raw_val)
    val_targets = get_targets(Y_vals, val_min_len)

val_residuals = [to_np(val_targets[i]) - to_np(val_means[i]) for i in range(N_STEPS)]
print(f'\n[验证集残差] 各步长残差长度: {[len(r) for r in val_residuals]}')

with torch.no_grad():
    raw_test = forward_all(nets, X_tests, C_tests, training_diffusion=False)
    final_means, final_log_sigmas, _, min_len = align_outputs(raw_test)
    final_targets = get_targets(Y_tests, min_len)

# 还原至原始对数收益率空间（无量纲）
test_true_ret  = [inv_ret(to_np(final_targets[i]))    for i in range(N_STEPS)]
test_mean_ret  = [inv_ret(to_np(final_means[i]))       for i in range(N_STEPS)]
# σ̂ 还原至收益率量纲：Softplus(log_σ) × scaler.scale_
test_sigma_ret = [to_sigma(final_log_sigmas[i]) * scaler.scale_[0]
                  for i in range(N_STEPS)]
# 验证集残差还原至收益率量纲
val_resid_ret  = [r * scaler.scale_[0] for r in val_residuals]

# ---------------------------------------------------------------
#  打印各步长指标（收益率空间，无量纲）
# ---------------------------------------------------------------
print('\n===== DS-LSDE 测试集指标（对数收益率空间，无量纲）=====')
for i in range(N_STEPS):
    rmse = np.sqrt(np.mean((test_true_ret[i] - test_mean_ret[i])**2))
    mae  = np.mean(np.abs(test_true_ret[i] - test_mean_ret[i]))
    da   = np.mean(np.sign(test_true_ret[i]) == np.sign(test_mean_ret[i]))
    print(f'  t+{i+1}: RMSE={rmse:.6f}  MAE={mae:.6f}  DA={da:.4f}')

# ============================================================
#  15. 各步长结果保存（对数收益率空间）
# ============================================================
for i in range(N_STEPS):
    fname = f'ds_lsde_{step_labels[i]}.npz'
    np.savez(fname,
             test_true  = test_true_ret[i],
             test_mean  = test_mean_ret[i],
             test_sigma = test_sigma_ret[i],
             val_resid  = val_resid_ret[i])
    print(f'[保存] {fname}  '
          f'test_true:{test_true_ret[i].shape}  '
          f'val_resid:{val_resid_ret[i].shape}')

# ============================================================
#  16. 桥接信号提取 → 供第四章 ETGPD-Transformer 使用
#
#  所有信号均在对数收益率量纲下计算（无量纲），与论文式3-1一致。
#
#  ε̂_t  (resid_z)   : 一步预测残差 r_t - μ̂_{t+1}（收益率量纲）
#                      第四章 GPD 尾部建模的直接对象
#  σ̂_t  (sigma_z)   : 条件波动率（收益率量纲 = Softplus(log_σ)×scaler.scale_）
#                      危机期系统性抬升，增强尾部风险感知
#  Δμ̂_t (delta_mu_z): 均值预测一阶差分（收益率量纲，动量代理变量）
#
#  三个信号均经滚动 z-score 标准化（窗口60），并入后 33→36 维。
# ============================================================
def rolling_zscore(arr: np.ndarray, window: int = 60) -> np.ndarray:
    s  = pd.Series(arr)
    mu = s.rolling(window, min_periods=10).mean().fillna(0.0)
    sd = s.rolling(window, min_periods=10).std().fillna(1.0).replace(0.0, 1.0)
    return ((s - mu) / sd).clip(-4, 4).values


def extract_sde_features(
    final_means:      list,
    final_log_sigmas: list,
    final_targets:    list,
    scaler:           StandardScaler,
    step:             int = 0,
    zscore_window:    int = 60,
) -> dict:
    """
    提取三个桥接信号。step=0 使用 t+1 步预测（默认）。

    所有信号均在对数收益率量纲下计算：
      ε̂_t  = r_t - μ̂_{t+1}         （收益率预测残差，无量纲）
      σ̂_t  = Softplus(log_σ̂) × scale （条件波动率，收益率量纲）
      Δμ̂_t = μ̂_{t+1} - μ̂_t          （均值预测一阶差分，收益率量纲）
    """
    mu_np  = to_np(final_means[step])
    ls_np  = to_np(final_log_sigmas[step])
    tgt_np = to_np(final_targets[step])

    # 残差在标准化空间计算后还原至收益率量纲
    resid_std   = tgt_np - mu_np                                    # 标准化空间
    resid       = resid_std * scaler.scale_[0]                      # 收益率量纲

    # 条件波动率还原至收益率量纲
    sigma_std   = (F.softplus(torch.tensor(ls_np)).numpy()
                   .clip(min=args.sigma_min) + 1e-3)
    sigma_ret   = sigma_std * scaler.scale_[0]                      # 收益率量纲

    # 均值预测还原至收益率量纲，再取一阶差分
    mu_ret      = inv_ret(mu_np)
    delta_mu    = np.diff(mu_ret, prepend=mu_ret[0])                # 收益率量纲

    resid_z    = rolling_zscore(resid,    window=zscore_window)
    sigma_z    = rolling_zscore(sigma_ret, window=zscore_window)
    delta_mu_z = rolling_zscore(delta_mu, window=zscore_window)

    print(f'[Bridge] 残差范围（收益率）:   [{resid.min():.6f}, {resid.max():.6f}]')
    print(f'[Bridge] 波动率范围（收益率）: [{sigma_ret.min():.6f}, {sigma_ret.max():.6f}]')
    print(f'[Bridge] Δμ̂ 范围（收益率）:  [{delta_mu.min():.6f}, {delta_mu.max():.6f}]')

    return {
        'resid_z':    resid_z,
        'sigma_z':    sigma_z,
        'delta_mu_z': delta_mu_z,
        'resid_raw':  resid,
        'sigma_raw':  sigma_ret,
    }


print('\n► Step 16: 提取桥接信号 → 第四章特征矩阵')
sde_feats = extract_sde_features(
    final_means      = final_means,
    final_log_sigmas = final_log_sigmas,
    final_targets    = final_targets,
    scaler           = scaler,
    step             = 0,
    zscore_window    = 60,
)

np.savez(
    args.bridge_path,
    resid_z    = sde_feats['resid_z'],
    sigma_z    = sde_feats['sigma_z'],
    delta_mu_z = sde_feats['delta_mu_z'],
    resid_raw  = sde_feats['resid_raw'],
    sigma_raw  = sde_feats['sigma_raw'],
)
print(f'[Bridge] 已保存 → {args.bridge_path}')
print(f'         样本数: {len(sde_feats["resid_z"])}')
print(f'         并入第四章后特征维度: 33 → 36')
print('\n全部完成。')
