# ============================================================
#  上证50指数多步预测 —— SDENet（随机微分方程神经网络）
#  重构版本：通用SDENet类 + 参数前置 + 代码精简
# ============================================================

# ── 标准库 ──────────────────────────────────────────────────
from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import argparse
import os

# ── 第三方库 ────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import levy_stable
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchsde  # noqa: F401（保留，部分环境需要）

# ============================================================
#  1. 超参数（argparse 最先定义，全局可用）
# ============================================================
parser = argparse.ArgumentParser(description='SDENet Training for SSE50')
parser.add_argument('--epochs',        type=int,   default=100)
parser.add_argument('--n_steps',       type=int,   default=4,    help='预测步数（SDENet 个数），即同时预测 t+1 … t+n_steps')
parser.add_argument('--lr',            type=float, default=1e-4, help='drift/attention 学习率')
parser.add_argument('--lr2',           type=float, default=0.01, help='diffusion 学习率')
parser.add_argument('--droprate',      type=float, default=0.1,  help='学习率衰减系数')
parser.add_argument('--decreasing_lr', default=[20], nargs='+',  help='drift lr 衰减 epoch')
parser.add_argument('--decreasing_lr2',default=[],  nargs='+',   help='diffusion lr 衰减 epoch')
parser.add_argument('--gpu',           type=int,   default=0)
parser.add_argument('--seed',          type=float, default=4)
args = parser.parse_args([])

# ============================================================
#  2. 随机种子 & 设备
# ============================================================
def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(int(args.seed))
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {device}')

# ============================================================
#  3. 数据加载与预处理
# ============================================================
data = pd.read_csv(DATA_DIR / "sz50_index_data.csv")
data = data[['date', 'code', 'open', 'close', 'high', 'low', 'volume']]
data['date'] = pd.to_datetime(data['date'])

# 整理为统一列名的 DataFrame
stk_data = pd.DataFrame({
    'Date':  data['date'].values,
    'Open':  data['open'].values,
    'High':  data['high'].values,
    'Low':   data['low'].values,
    'Close': data['close'].values,
})

# 按 70/10/20 划分训练/验证/测试集（仅取收盘价）
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
n_train = int(np.ceil(TRAIN_RATIO * len(stk_data)))
n_val   = int(np.ceil(VAL_RATIO   * len(stk_data)))
n_test  = len(stk_data) - n_train - n_val

training_set   = stk_data.iloc[:n_train,              4:5].values
validation_set = stk_data.iloc[n_train:n_train+n_val, 4:5].values
testing_set    = stk_data.iloc[n_train+n_val:,        4:5].values
print(f'[Data] 训练集: {training_set.shape}  验证集: {validation_set.shape}  测试集: {testing_set.shape}')

# 标准化（只用训练集 fit）
scaler = StandardScaler()
train_scaled = scaler.fit_transform(training_set).flatten()
val_scaled   = scaler.transform(validation_set).flatten()
test_scaled  = scaler.transform(testing_set).flatten()

# 拼接日期列，供相空间重构使用
def make_date_value_df(dates, values):
    return pd.DataFrame({'Date': dates, 'closescale': values})

train_df = make_date_value_df(stk_data['Date'].values[:n_train],                         train_scaled)
val_df   = make_date_value_df(stk_data['Date'].values[n_train:n_train+n_val],             val_scaled)
test_df  = make_date_value_df(stk_data['Date'].values[n_train+n_val:n_train+n_val+n_test], test_scaled)

# ============================================================
#  3b. 关联维数数据加载与对齐
# ============================================================
# 文件第1列为时间，后10列为关联维数特征，共11列
corr_dim_raw = pd.read_csv(CORR_PATH)
corr_dim_raw.iloc[:, 0] = pd.to_datetime(corr_dim_raw.iloc[:, 0])
corr_dim_raw.columns = ['Date'] + [f'cd_{i}' for i in range(1, 11)]
CORR_DIM_COLS = [f'cd_{i}' for i in range(1, 11)]
CORR_DIM = len(CORR_DIM_COLS)   # = 10，后续网络用到

def align_corr_dim(date_series, corr_df):
    """
    按日期左连接，把关联维数对齐到指定日期序列。
    找不到的日期用前向填充（ffill），再用0填充剩余NaN。
    返回 numpy array，shape (n, 10)。
    """
    df_dates = pd.DataFrame({'Date': pd.to_datetime(date_series)})
    corr_df = corr_df.copy()
    corr_df['Date'] = pd.to_datetime(corr_df['Date'])
    merged = df_dates.merge(corr_df, on='Date', how='left')
    merged[CORR_DIM_COLS] = merged[CORR_DIM_COLS].ffill().fillna(0.0)
    return merged[CORR_DIM_COLS].values.astype(np.float64)

# 对训练/验证/测试集各自对齐（行数与 stk_data 分段一致）
corr_train_full = align_corr_dim(stk_data['Date'].values[:n_train],                          corr_dim_raw)
corr_val_full   = align_corr_dim(stk_data['Date'].values[n_train:n_train+n_val],              corr_dim_raw)
corr_test_full  = align_corr_dim(stk_data['Date'].values[n_train+n_val:n_train+n_val+n_test], corr_dim_raw)
print(f'[CorrDim] train:{corr_train_full.shape}  val:{corr_val_full.shape}  test:{corr_test_full.shape}')

# ============================================================
#  4. 相空间重构
# ============================================================
def PhaSpaRecon(df, tau: int, d: int, T: int):
    """
    参数
    ----
    df  : 含 [Date, closescale] 两列的 DataFrame
    tau : 时延
    d   : 嵌入维数
    T   : 预测步数
    返回
    ----
    Xn  : 输入特征矩阵 (DataFrame)
    Yn  : 目标值 (DataFrame)
    Y   : [日期, 目标值] (DataFrame)
    X   : [特征, 目标值] (DataFrame)
    """
    values = np.array(df)[:, 1].astype(float)
    dates  = np.array(df)[:, 0]
    n = len(values)

    if (n - T - (d - 1) * tau) < 1:
        raise ValueError("tau 或 d 过大，超出序列长度")

    width = n - (d - 1) * tau - 1
    Xn1 = np.stack([values[i*tau : i*tau + width] for i in range(d)], axis=1)  # (width, d)
    Yn1      = values[T + (d-1)*tau : T + (d-1)*tau + width]
    Yn1_date = dates [T + (d-1)*tau : T + (d-1)*tau + width]

    Xn = pd.DataFrame(Xn1)
    Yn = pd.DataFrame(Yn1, columns=[0])
    Y  = pd.DataFrame({'Date': Yn1_date, 'target': Yn1})
    X  = pd.concat([Xn, Yn], axis=1)
    return Xn, Yn, Y, X


def build_xy(df, tau, d, T, drop_tail=0):
    """相空间重构后提取 numpy 数组，可选删除末尾若干行以对齐多步预测。"""
    _, _, _, X = PhaSpaRecon(df, tau=tau, d=d, T=T)
    arr = X.values
    if drop_tail > 0:
        arr = arr[:-drop_tail]
    return arr[:, :d].astype(np.float64), arr[:, d].astype(np.float64)


# 嵌入维数 & 时延
D, TAU = 22, 1

# 构建 T=1..N_STEPS 的训练/验证/测试数据（对齐：T=k 需删去末尾 k-1 行）
N_STEPS = args.n_steps   # ← 通过 --n_steps 控制，默认 4
x_trains, y_trains = [], []
x_vals,   y_vals   = [], []
x_tests,  y_tests  = [], []
# 关联维数：与相空间重构对齐，头部偏移 (d-1)*tau，尾部再删去 drop_tail 行
# 相空间重构后第一个有效样本对应原序列第 (d-1)*tau 个时间点
PSR_OFFSET = (D - 1) * TAU   # 相空间重构头部偏移量（固定）
cd_trains, cd_vals, cd_tests = [], [], []

def slice_cd(cd_full, n_x, drop):
    """将 corr_dim 全量数组按相空间重构偏移量和 drop_tail 对齐后截取，确保与 x 等长。"""
    start = PSR_OFFSET
    end   = start + n_x + drop    # drop 行已被 build_xy 裁掉，原始末端需加回
    arr   = cd_full[start:end]
    if drop > 0:
        arr = arr[:-drop]
    return arr[:n_x]              # 保险截断，确保与 x 完全等长

for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr, ytr = build_xy(train_df, TAU, D, T, drop_tail=drop)
    xva, yva = build_xy(val_df,   TAU, D, T, drop_tail=drop)
    xte, yte = build_xy(test_df,  TAU, D, T, drop_tail=drop)
    x_trains.append(xtr); y_trains.append(ytr)
    x_vals.append(xva);   y_vals.append(yva)
    x_tests.append(xte);  y_tests.append(yte)
    cd_trains.append(slice_cd(corr_train_full, xtr.shape[0], drop))
    cd_vals.append(  slice_cd(corr_val_full,   xva.shape[0], drop))
    cd_tests.append( slice_cd(corr_test_full,  xte.shape[0], drop))
    print(f'T={T}  x_train:{xtr.shape}  cd_train:{cd_trains[-1].shape}  '
          f'x_val:{xva.shape}  x_test:{xte.shape}')

# 时间轴（以年为单位，252 交易日/年）
ts_     = np.arange(len(y_trains[0])) / 252
ts_ext_ = np.arange(len(y_trains[0]), len(y_trains[0]) + len(y_tests[0])) / 252

# ============================================================
#  5. 通用 SDENet（Lévy 噪声驱动的随机微分方程网络）
# ============================================================
ALPHA, BETA = 1.2, 0   # Lévy 稳定分布参数

class DriftNet(nn.Module):
    """漂移项网络 f(t, x)：线性 + ReLU"""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))

    def forward(self, t, x):
        return self.net(x)


class DiffusionNet(nn.Module):
    """扩散项网络 g(t, x)：线性 + ReLU + 线性，输出 logit（不含 Sigmoid）
    - training_diffusion=True  → 直接送入 BCEWithLogitsLoss（内置数值稳定 Sigmoid）
    - training_diffusion=False → 手动 Sigmoid 后用于 SDE 扩散缩放"""
    def __init__(self, dim: int, hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, t, x):
        return self.net(x)


class SDENet(nn.Module):
    """
    通用随机微分方程网络
    
    参数
    ----
    dim         : 输入/状态维度（= 嵌入维数 d）
    layer_depth : 离散化步数（Euler-Maruyama 步数）
    sigma       : 扩散项缩放系数
    """
    def __init__(self, dim: int, layer_depth: int = 25, sigma: float = 0.5):
        super().__init__()
        self.layer_depth = layer_depth
        self.sigma   = sigma
        self.delta_t = 1.0 / layer_depth

        self.downsample = nn.Linear(dim, dim)
        self.drift      = DriftNet(dim)
        self.diffusion  = DiffusionNet(dim)

    def forward(self, x, training_diffusion: bool = False):
        out = self.downsample(x)

        if not training_diffusion:
            # ── 前向 SDE 积分（Euler-Maruyama + Lévy 噪声）──
            t = 0.0
            # diffusion 输出 logit，手动 Sigmoid 后才用于扩散项缩放
            diff_term = self.sigma * torch.sigmoid(self.diffusion(t, out))
            for i in range(self.layer_depth):
                t = float(i) / self.layer_depth
                levy_noise = torch.from_numpy(
                    levy_stable.rvs(ALPHA, BETA, size=out.shape[-1], scale=0.1)
                ).to(device=x.device, dtype=x.dtype)
                # clamp 防止 Lévy 重尾采样的极大值炸穿梯度
                levy_noise = levy_noise.clamp(-10.0, 10.0)
                out = out + self.drift(t, out) * self.delta_t \
                          + diff_term * (self.delta_t ** (1 / ALPHA)) * levy_noise
            drift_out = self.drift(t, out)
            return drift_out, out   # (漂移量, 隐状态)

        else:
            # ── 训练扩散项（判别器模式）──
            t = 0.0
            return self.diffusion(t, out.detach())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
#  6. 数据转 Tensor
# ============================================================
def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float64)).float()

X_trains = [to_tensor(x).to(device) for x in x_trains]
Y_trains = [to_tensor(y).to(device) for y in y_trains]
X_vals   = [to_tensor(x).to(device) for x in x_vals]
Y_vals   = [to_tensor(y).to(device) for y in y_vals]
X_tests  = [to_tensor(x).to(device) for x in x_tests]
Y_tests  = [to_tensor(y).to(device) for y in y_tests]

# 关联维数 Tensor（shape: (n_samples, 10)，与对应的 X 行数严格对齐）
CD_trains = [to_tensor(c).to(device) for c in cd_trains]
CD_vals   = [to_tensor(c).to(device) for c in cd_vals]
CD_tests  = [to_tensor(c).to(device) for c in cd_tests]

batch_sizes = [y.shape[0] for y in y_trains]

# ============================================================
#  7. 模型、优化器、损失函数
# ============================================================
N_STEPS     = args.n_steps   # 已在数据构建时赋值，此处保持一致
SEQ_LEN     = D              # 序列长度 = 嵌入维数
ITER        = 50             # 每 epoch 的迭代次数（训练）
ITER_TEST   = 1

nets = nn.ModuleList([SDENet(dim=SEQ_LEN, layer_depth=25).to(device) for _ in range(N_STEPS)])

# ============================================================
#  多头注意力融合模块
#  输入：SDE 输出拼接 (N, N_STEPS*SEQ_LEN) + 关联维数 (N, CORR_DIM)
#  输出：(N, N_STEPS*2)  即 [mean_1..mean_4, sigma_1..sigma_4]
#
#  设计：
#    1. 将每个步长的 SDE 隐状态投影为 d_model 维 token
#    2. 关联维数作为额外 context token 一同送入多头自注意力
#    3. 取前 N_STEPS 个 token 展平后线性映射到 N_STEPS*2 维输出
# ============================================================
class MultiHeadFusionAttention(nn.Module):
    """
    多头注意力融合层

    参数
    ----
    n_steps   : 预测步数（= SDE token 数）
    sde_dim   : 每个 SDE 网络输出的隐状态维度（= SEQ_LEN）
    corr_dim  : 关联维数特征维度（= 10）
    d_model   : 注意力内部维度，需能被 n_heads 整除
    n_heads   : 注意力头数
    out_dim   : 最终输出维度（= n_steps * 2，前半均值后半 logσ）
    """
    def __init__(self, n_steps: int, sde_dim: int, corr_dim: int,
                 d_model: int = 64, n_heads: int = 4, out_dim: int = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        out_dim = out_dim or n_steps * 2

        # 各 token 的投影层
        self.sde_proj  = nn.Linear(sde_dim,  d_model)
        self.corr_proj = nn.Linear(corr_dim, d_model)

        # 多头自注意力（batch_first=True：输入 shape (N, seq, d_model)）
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        # 取前 n_steps 个 token 展平后输出
        self.out_proj = nn.Linear(n_steps * d_model, out_dim)
        self.n_steps  = n_steps
        self.d_model  = d_model

    def forward(self, sde_tokens: torch.Tensor, corr_feat: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        sde_tokens : (N, n_steps * sde_dim)  — 各步 SDE 隐状态拼接
        corr_feat  : (N, corr_dim)            — 关联维数特征

        返回
        ----
        out        : (N, out_dim)             — 均值 + log-sigma 拼接
        """
        N = sde_tokens.shape[0]
        # 拆分为 n_steps 个 token，各自投影到 d_model
        sde_split = sde_tokens.view(N, self.n_steps, -1)          # (N, n_steps, sde_dim)
        sde_emb   = self.sde_proj(sde_split)                       # (N, n_steps, d_model)

        # 关联维数作为第 n_steps+1 个 context token
        corr_emb  = self.corr_proj(corr_feat).unsqueeze(1)         # (N, 1, d_model)

        # 拼接：[SDE token × n_steps | corr token × 1]
        tokens    = torch.cat([sde_emb, corr_emb], dim=1)         # (N, n_steps+1, d_model)

        # 多头自注意力 + 残差 + LayerNorm
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens      = self.norm(tokens + attn_out)

        # 只取前 n_steps 个 token，展平后线性映射到输出
        sde_out = tokens[:, :self.n_steps, :].reshape(N, -1)      # (N, n_steps * d_model)
        return self.out_proj(sde_out)                              # (N, out_dim)


attention = MultiHeadFusionAttention(
    n_steps  = N_STEPS,
    sde_dim  = SEQ_LEN,
    corr_dim = CORR_DIM,
    d_model  = 64,
    n_heads  = 4,
    out_dim  = N_STEPS * 2,
).to(device)

print(f'[Model] 每个 SDENet 参数量: {count_parameters(nets[0]):,}')

# 优化器：分两组，drift/attention 与 diffusion 分开
optimizer_F = optim.SGD(
    [{'params': net.downsample.parameters()} for net in nets] +
    [{'params': net.drift.parameters()}      for net in nets] +
    [{'params': attention.parameters()}],
    lr=args.lr, momentum=0.9, weight_decay=5e-4,
)
optimizer_G = optim.SGD(
    [{'params': net.diffusion.parameters()} for net in nets],
    lr=args.lr2, momentum=0.9, weight_decay=5e-4,
)

criterion = nn.BCEWithLogitsLoss()  # 内置 Sigmoid，数值稳定，避免边界值触发 CUDA assertion
real_label, fake_label = 0, 1


def nll_loss(y, mean, sigma):
    """数值稳定的负对数似然损失（高斯）
    对 sigma 做 clamp 防止除零和 log(0)；等价于标准 NLL 但避免梯度爆炸。"""
    sigma = sigma.clamp(min=1e-3)          # 防止 sigma 趋零导致 1/sigma² → ∞
    return torch.mean(torch.log(sigma) + (y - mean) ** 2 / (2 * sigma ** 2))

def mse_loss(y, mean):
    return torch.mean((y - mean) ** 2)


# ============================================================
#  8. 前向传播公共逻辑（train / test 共用）
# ============================================================
# 对齐切片：第 i 个网络（T=i+1）需去掉末尾 (N_STEPS-1-i) 行，最后一个不切
# 例如 N_STEPS=4: [3, 2, 1, 0]；N_STEPS=3: [2, 1, 0]
TAIL_CUTS = list(range(N_STEPS - 1, -1, -1))

def forward_all(nets, X_list, training_diffusion=False):
    """对 4 个网络分别前向，返回 (drifts, outs) 或 diffusion 输出列表"""
    results = [nets[i](X_list[i], training_diffusion=training_diffusion)
               for i in range(N_STEPS)]
    return results

def fuse_outputs(raw_outs, CD_list):
    """
    raw_outs : [(Drift_i, Out_i), ...]        — N_STEPS 个 SDE 网络输出
    CD_list  : [Tensor(n_i, corr_dim), ...]   — N_STEPS 个关联维数 Tensor

    流程
    ----
    1. 取各步 SDE 隐状态 Out_i，按 TAIL_CUTS 对齐到最短长度 min_len
    2. 同样对 CD_list 中最后一个网络对应的关联维数截到 min_len
       （各步长的 corr_dim 行数相同，取 CD_list[-1] 代表对齐后的行索引）
    3. 拼接 SDE 输出 (N, N_STEPS*SEQ_LEN) 送入多头注意力融合模块，
       同时传入关联维数 (N, CORR_DIM)
    4. 输出 (N, N_STEPS*2)，前半为均值，后半经 softplus 为方差
    """
    outs = [raw_outs[i][1] for i in range(N_STEPS)]                  # Out_i: (n_i, SEQ_LEN)

    # SDE 隐状态对齐（与原逻辑一致）
    aligned = [outs[i][:outs[-1].shape[0] - TAIL_CUTS[i]]
               for i in range(N_STEPS)]
    min_len = min(a.shape[0] for a in aligned)
    cat = torch.cat([a[:min_len] for a in aligned], dim=1)            # (min_len, N_STEPS*SEQ_LEN)

    # 关联维数对齐：各步 CD 行数与对应 X 等长；取最后一步（最短）的前 min_len 行
    # CD_list[-1] 对应 T=N_STEPS 步，行数最少，再截 min_len 保险
    corr_feat = CD_list[-1][:min_len]                                 # (min_len, CORR_DIM)

    # 多头注意力融合：SDE token × N_STEPS + corr context token × 1 → (min_len, N_STEPS*2)
    final  = attention(cat, corr_feat)                                # (min_len, N_STEPS*2)
    means  = [final[:, i]                          for i in range(N_STEPS)]
    sigmas = [F.softplus(final[:, N_STEPS + i]) + 1e-3 for i in range(N_STEPS)]
    return means, sigmas, min_len


def get_targets(Y_list, min_len):
    """对齐目标序列"""
    return [Y_list[i][:min_len] for i in range(N_STEPS)]


# ============================================================
#  9. 训练 & 测试函数
# ============================================================
def train_epoch(epoch):
    for net in nets: net.train()
    total_loss = total_loss_in = total_loss_out = 0.0

    for _ in range(ITER):
        # ── Drift / Attention 更新 ──────────────────────────
        optimizer_F.zero_grad()
        raw = forward_all(nets, X_trains)
        means, sigmas, min_len = fuse_outputs(raw, CD_trains)
        targets = get_targets(Y_trains, min_len)

        loss_f = sum(nll_loss(targets[i], means[i], sigmas[i]) for i in range(N_STEPS))
        loss_f.backward()
        nn.utils.clip_grad_norm_(
            [p for g in optimizer_F.param_groups for p in g['params']], max_norm=5.0
        )
        optimizer_F.step()
        total_loss += loss_f.item()

        # ── Diffusion（判别器）更新 ─────────────────────────
        # 用 torch.full_like 让 label shape 自动与 pred 对齐，避免 BCELoss shape mismatch
        optimizer_G.zero_grad()

        # 真实输入 → label=0
        preds_in    = [nets[i](X_trains[i], training_diffusion=True) for i in range(N_STEPS)]
        labels_real = [torch.full_like(preds_in[i], real_label)       for i in range(N_STEPS)]
        loss_in     = sum(criterion(preds_in[i], labels_real[i])       for i in range(N_STEPS))

        # 噪声输入 → label=1
        inputs_out  = [2 * torch.randn_like(X_trains[i]) + X_trains[i] for i in range(N_STEPS)]
        preds_out   = [nets[i](inputs_out[i], training_diffusion=True)  for i in range(N_STEPS)]
        labels_fake = [torch.full_like(preds_out[i], fake_label)        for i in range(N_STEPS)]
        loss_out    = sum(criterion(preds_out[i], labels_fake[i])        for i in range(N_STEPS))

        # 两个 loss 合并后一次 backward，避免计算图被提前释放
        (loss_in + loss_out).backward()
        nn.utils.clip_grad_norm_(
            [p for g in optimizer_G.param_groups for p in g['params']], max_norm=5.0
        )
        optimizer_G.step()
        total_loss_in  += loss_in.item()
        total_loss_out += loss_out.item()

    # 计算本 epoch 训练 MSE（带 NaN 检测）
    with torch.no_grad():
        raw = forward_all(nets, X_trains)
        means, _, min_len = fuse_outputs(raw, CD_trains)
        targets = get_targets(Y_trains, min_len)
        mse_list = []
        for i in range(N_STEPS):
            pred_np = means[i].cpu().detach().numpy()
            if not np.isfinite(pred_np).all():
                raise RuntimeError(
                    f'Epoch {epoch}, step t+{i+1}: 预测值出现 NaN/Inf，训练已发散。'
                    f'建议降低学习率（当前 lr={args.lr}）或减小 sigma。'
                )
            mse_list.append(mean_squared_error(targets[i].cpu().numpy(), pred_np))

    print(f'[Train] Epoch {epoch:3d} | '
          f'Loss: {total_loss/ITER:.6f} | '
          f'Loss_in: {total_loss_in/ITER:.6f} | '
          f'Loss_out: {total_loss_out/ITER:.6f}')
    return tuple(mse_list)


def test_epoch(epoch):
    for net in nets: net.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(ITER_TEST):
            raw = forward_all(nets, X_tests)
            means, sigmas, min_len = fuse_outputs(raw, CD_tests)
            targets = get_targets(Y_tests, min_len)

            loss = sum(mse_loss(targets[i], means[i]) for i in range(N_STEPS))
            total_loss += loss.item()

        mse_list = [mean_squared_error(targets[i].cpu().numpy(),
                                       means[i].cpu().detach().numpy())
                    for i in range(N_STEPS)]

    print(f'[ Test] Epoch {epoch:3d} | RMSE: {(total_loss/ITER_TEST)**0.5:.6f}')
    return tuple(mse_list)


# ============================================================
#  10. 训练主循环
# ============================================================
train_losses = []
test_losses  = []

for epoch in range(1, args.epochs + 1):
    train_losses.append(train_epoch(epoch))
    test_losses.append(test_epoch(epoch))

    # 学习率衰减
    if epoch in [int(e) for e in args.decreasing_lr]:
        for pg in optimizer_F.param_groups:
            pg['lr'] *= args.droprate

    if epoch in [int(e) for e in args.decreasing_lr2]:
        for pg in optimizer_G.param_groups:
            pg['lr'] *= args.droprate

# ============================================================
#  11. 最终结果打印
# ============================================================
print('\n===== 最终测试 MSE（标准化空间）=====')
step_labels = [f't+{i+1}' for i in range(N_STEPS)]
for i, label in enumerate(step_labels):
    print(f'  {label}: {test_losses[-1][i]:.6f}')

# ============================================================
#  12. 收集验证集残差（用于残差 bootstrap）
# ============================================================
for net in nets: net.eval()
with torch.no_grad():
    raw_val            = forward_all(nets, X_vals)
    val_means, val_sigmas, val_min_len = fuse_outputs(raw_val, CD_vals)
    val_targets        = get_targets(Y_vals, val_min_len)

def to_np(t): return t.cpu().detach().numpy()

# 残差在标准化空间计算，后续 bootstrap 也在标准化空间操作
val_residuals = [
    to_np(val_targets[i]) - to_np(val_means[i])   # shape: (n_val_aligned,)
    for i in range(N_STEPS)
]
print(f'\n[验证集残差] 各步长残差长度: {[len(r) for r in val_residuals]}')

# ============================================================
#  13. 获取测试集最终推理结果
# ============================================================
with torch.no_grad():
    raw_test                            = forward_all(nets, X_tests)
    final_means, final_sigmas, min_len  = fuse_outputs(raw_test, CD_tests)
    final_targets                       = get_targets(Y_tests, min_len)

# 把测试集真实值和预测参数都反标准化到原始价格空间
def inv(arr):
    """把标准化空间的 1D 数组还原到原始价格空间"""
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

test_true_price  = [inv(to_np(final_targets[i])) for i in range(N_STEPS)]
test_mean_price  = [inv(to_np(final_means[i]))   for i in range(N_STEPS)]
# sigma 在原始空间的等效值：σ_orig = σ_scaled × scaler.scale_
test_sigma_price = [to_np(final_sigmas[i]) * scaler.scale_[0] for i in range(N_STEPS)]
val_resid_price  = [r * scaler.scale_[0]  for r in val_residuals]

# ============================================================
#  14. 保存实验所需数据（.npz）
# ============================================================
# 每个步长保存为独立文件：data_t+1.npz, data_t+2.npz, ...
# 各文件包含四个数组：
#   test_true   : 测试集真实价格          shape (n_test_days,)
#   test_mean   : 预测均值（原价格空间）  shape (n_test_days,)
#   test_sigma  : 预测标准差（原价格空间）shape (n_test_days,)
#   val_resid   : 验证集残差（原价格空间）shape (n_val_days,)

for i in range(N_STEPS):
    fname = f'ADE3_data_{step_labels[i]}.npz'
    np.savez(
        fname,
        test_true  = test_true_price[i],
        test_mean  = test_mean_price[i],
        test_sigma = test_sigma_price[i],
        val_resid  = val_resid_price[i],
    )
    print(f'[保存] {fname}  '
          f'test_true:{test_true_price[i].shape}  '
          f'val_resid:{val_resid_price[i].shape}')

print('\n全部数据已保存，可在实验脚本中用以下方式加载：')
print("  d = np.load('data_t+1.npz')")
print("  test_true, test_mean, test_sigma, val_resid = "
      "d['test_true'], d['test_mean'], d['test_sigma'], d['val_resid']")
