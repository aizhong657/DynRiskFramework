# ============================================================
#  LDENet（随机微分方程神经网络）
#  重构版本：通用LDENet类 + 参数前置 + 代码精简
#  新增：在 DriftNet 里，forward 从 return self.net(x) 改为 return self.net(x) + x
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
parser = argparse.ArgumentParser(description='LDENet Training for SSE50')
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
for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr, ytr = build_xy(train_df, TAU, D, T, drop_tail=drop)
    xva, yva = build_xy(val_df,   TAU, D, T, drop_tail=drop)
    xte, yte = build_xy(test_df,  TAU, D, T, drop_tail=drop)
    x_trains.append(xtr); y_trains.append(ytr)
    x_vals.append(xva);   y_vals.append(yva)
    x_tests.append(xte);  y_tests.append(yte)
    print(f'T={T}  x_train:{xtr.shape}  x_val:{xva.shape}  x_test:{xte.shape}')

# 时间轴（以年为单位，252 交易日/年）
ts_     = np.arange(len(y_trains[0])) / 252
ts_ext_ = np.arange(len(y_trains[0]), len(y_trains[0]) + len(y_tests[0])) / 252

# ============================================================
#  5. 通用 SDENet（Lévy 噪声驱动的随机微分方程网络）
# ============================================================
ALPHA, BETA = 1.2, 0   # Lévy 稳定分布参数

class DriftNet(nn.Module):
    """漂移项网络 f(t, x)：线性 + ReLU + 残差连接"""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))

    def forward(self, t, x):
        return self.net(x) + x  # 残差连接：输入输出维度相同，直接相加


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

batch_sizes = [y.shape[0] for y in y_trains]

# ============================================================
#  7. 模型、优化器、损失函数
# ============================================================
N_STEPS     = args.n_steps   # 已在数据构建时赋值，此处保持一致
SEQ_LEN     = D              # 序列长度 = 嵌入维数
ITER        = 50             # 每 epoch 的迭代次数（训练）
ITER_TEST   = 1

nets = nn.ModuleList([SDENet(dim=SEQ_LEN, layer_depth=25).to(device) for _ in range(N_STEPS)])

# Attention 融合层：将 4 个网络输出拼接后映射到 (均值×4 + sigma×4)
attention = nn.Sequential(
    nn.ReLU(inplace=True),
    nn.Linear(N_STEPS * SEQ_LEN, N_STEPS * 2),   # 输出：mean1~4, sigma1~4
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

def fuse_outputs(raw_outs):
    """
    raw_outs: [(Drift_i, Out_i), ...]
    对齐后拼接，经 attention 得到 [mean×4, sigma×4]
    """
    outs = [raw_outs[i][1] for i in range(N_STEPS)]                 # Out1~Out4
    aligned = [outs[i][:outs[-1].shape[0] - TAIL_CUTS[i] if TAIL_CUTS[i] > 0
                       else outs[-1].shape[0]]
               for i in range(N_STEPS)]
    # 统一长度为最短的那个（即去掉尾部后的 Out4 长度）
    min_len = min(a.shape[0] for a in aligned)
    cat = torch.cat([a[:min_len] for a in aligned], dim=1)           # (N, 4*SEQ_LEN)
    final = attention(cat)                                            # (N, 8)
    means  = [final[:, i]                        for i in range(N_STEPS)]
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
        means, sigmas, min_len = fuse_outputs(raw)
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
        means, _, min_len = fuse_outputs(raw)
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
            means, sigmas, min_len = fuse_outputs(raw)
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
    val_means, val_sigmas, val_min_len = fuse_outputs(raw_val)
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
    raw_test                          = forward_all(nets, X_tests)
    final_means, final_sigmas, min_len = fuse_outputs(raw_test)
    final_targets                      = get_targets(Y_tests, min_len)

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
    fname = f'LDE_data_{step_labels[i]}.npz'
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
