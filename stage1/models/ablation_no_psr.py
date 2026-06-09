# ============================================================
#  消融实验 A：w/o PSR（移除相空间重构）
#  改动说明：
#    - 不使用 PhaSpaRecon，改用定长滑动窗口（window_size = D = 22）
#      构建输入特征，维度与原版保持一致（输入仍为 D 列），
#      因此 SDENet / MultiHeadFusionAttention 结构完全不变。
#    - 关联维数同样按滑动窗口末端日期对齐，无 PSR 头部偏移。
#    - 其余（SDE、多头Attention、训练循环）与 chapter52_new.py 完全相同。
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
import torchsde  # noqa: F401

# ============================================================
#  1. 超参数
# ============================================================
parser = argparse.ArgumentParser(description='Ablation: w/o PSR')
parser.add_argument('--epochs',        type=int,   default=100)
parser.add_argument('--n_steps',       type=int,   default=4)
parser.add_argument('--lr',            type=float, default=1e-4)
parser.add_argument('--lr2',           type=float, default=0.01)
parser.add_argument('--droprate',      type=float, default=0.1)
parser.add_argument('--decreasing_lr', default=[20], nargs='+')
parser.add_argument('--decreasing_lr2',default=[],   nargs='+')
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

stk_data = pd.DataFrame({
    'Date':  data['date'],
    'Open':  data['open'].values,
    'High':  data['high'].values,
    'Low':   data['low'].values,
    'Close': data['close'].values,
})

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
n_train = int(np.ceil(TRAIN_RATIO * len(stk_data)))
n_val   = int(np.ceil(VAL_RATIO   * len(stk_data)))
n_test  = len(stk_data) - n_train - n_val

training_set   = stk_data.iloc[:n_train,              4:5].values
validation_set = stk_data.iloc[n_train:n_train+n_val, 4:5].values
testing_set    = stk_data.iloc[n_train+n_val:,        4:5].values
print(f'[Data] 训练集: {training_set.shape}  验证集: {validation_set.shape}  测试集: {testing_set.shape}')

scaler = StandardScaler()
train_scaled = scaler.fit_transform(training_set).flatten()
val_scaled   = scaler.transform(validation_set).flatten()
test_scaled  = scaler.transform(testing_set).flatten()

# ============================================================
#  3b. 关联维数数据加载
# ============================================================
corr_dim_raw = pd.read_csv(CORR_PATH)
corr_dim_raw.iloc[:, 0] = pd.to_datetime(corr_dim_raw.iloc[:, 0])
corr_dim_raw.columns = ['Date'] + [f'cd_{i}' for i in range(1, 11)]
CORR_DIM_COLS = [f'cd_{i}' for i in range(1, 11)]
CORR_DIM = len(CORR_DIM_COLS)

def align_corr_dim(date_series, corr_df):
    df_dates = pd.DataFrame({'Date': pd.to_datetime(date_series)})
    # 确保 corr_df 的 'Date' 列也是日期时间类型
    corr_df = corr_df.copy()
    corr_df['Date'] = pd.to_datetime(corr_df['Date'])
    merged = df_dates.merge(corr_df, on='Date', how='left')
    merged[CORR_DIM_COLS] = merged[CORR_DIM_COLS].ffill().fillna(0.0)
    return merged[CORR_DIM_COLS].values.astype(np.float64)

corr_train_full = align_corr_dim(stk_data['Date'].values[:n_train],                          corr_dim_raw)
corr_val_full   = align_corr_dim(stk_data['Date'].values[n_train:n_train+n_val],              corr_dim_raw)
corr_test_full  = align_corr_dim(stk_data['Date'].values[n_train+n_val:n_train+n_val+n_test], corr_dim_raw)

# ============================================================
#  4. [w/o PSR] 滑动窗口构建（替代相空间重构）
#
#  原版 PSR(tau=1, d=D, T=k) 的输入 X[t] = [close[t], close[t-1], …, close[t-D+1]]
#  滑动窗口以同样方式取最近 D 个时间步作为输入，目标为 t+T 时刻的收盘价。
#  二者在 tau=1 时等价于相同的时间窗口，PSR 的头部偏移 (D-1)*TAU 在此处同样存在，
#  但不依赖相空间嵌入理论，仅作朴素滑动窗口特征。
# ============================================================
D   = 22   # 窗口宽度，与原版嵌入维数保持一致
TAU = 1    # 保留，用于关联维数偏移计算的对齐

N_STEPS = args.n_steps

def build_sliding_window(values, window: int, T: int, drop_tail: int = 0):
    """
    滑动窗口特征构建（不做相空间嵌入）。
    X[i] = values[i : i+window]   （最近 window 个值）
    y[i] = values[i + window + T - 1]  （T 步后的目标）

    drop_tail : 末尾删去行数，用于多步预测对齐（与原版 build_xy 语义相同）
    """
    n = len(values)
    max_i = n - window - T + 1       # 可用样本数（含 drop_tail 部分）
    if max_i <= 0:
        raise ValueError("序列太短，无法构建滑动窗口")
    X_list, y_list = [], []
    for i in range(max_i):
        X_list.append(values[i : i + window])
        y_list.append(values[i + window + T - 1])
    X = np.array(X_list, dtype=np.float64)   # (max_i, window)
    y = np.array(y_list, dtype=np.float64)   # (max_i,)
    if drop_tail > 0:
        X = X[:-drop_tail]
        y = y[:-drop_tail]
    return X, y

# 关联维数对齐：窗口末端索引 = i + window - 1，对应原序列第 (window-1) 起的样本
# 等同于 PSR 的 PSR_OFFSET = (D-1)*TAU，此处 offset = D - 1
SLIDE_OFFSET = D - 1

def slice_cd_slide(cd_full, n_x, drop):
    """滑动窗口版关联维数对齐，语义与 PSR 版 slice_cd 相同。"""
    start = SLIDE_OFFSET
    end   = start + n_x + drop
    arr   = cd_full[start:end]
    if drop > 0:
        arr = arr[:-drop]
    return arr[:n_x]

x_trains, y_trains = [], []
x_vals,   y_vals   = [], []
x_tests,  y_tests  = [], []
cd_trains, cd_vals, cd_tests = [], [], []

for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr, ytr = build_sliding_window(train_scaled, D, T, drop_tail=drop)
    xva, yva = build_sliding_window(val_scaled,   D, T, drop_tail=drop)
    xte, yte = build_sliding_window(test_scaled,  D, T, drop_tail=drop)
    x_trains.append(xtr); y_trains.append(ytr)
    x_vals.append(xva);   y_vals.append(yva)
    x_tests.append(xte);  y_tests.append(yte)
    cd_trains.append(slice_cd_slide(corr_train_full, xtr.shape[0], drop))
    cd_vals.append(  slice_cd_slide(corr_val_full,   xva.shape[0], drop))
    cd_tests.append( slice_cd_slide(corr_test_full,  xte.shape[0], drop))
    print(f'[Slide] T={T}  x_train:{xtr.shape}  cd_train:{cd_trains[-1].shape}  '
          f'x_val:{xva.shape}  x_test:{xte.shape}')

ts_     = np.arange(len(y_trains[0])) / 252
ts_ext_ = np.arange(len(y_trains[0]), len(y_trains[0]) + len(y_tests[0])) / 252

# ============================================================
#  5. SDENet（与原版完全相同）
# ============================================================
ALPHA, BETA = 1.2, 0

class DriftNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True))
    def forward(self, t, x):
        return self.net(x)

class DiffusionNet(nn.Module):
    def __init__(self, dim: int, hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    def forward(self, t, x):
        return self.net(x)

class SDENet(nn.Module):
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
            t = 0.0
            diff_term = self.sigma * torch.sigmoid(self.diffusion(t, out))
            for i in range(self.layer_depth):
                t = float(i) / self.layer_depth
                levy_noise = torch.from_numpy(
                    levy_stable.rvs(ALPHA, BETA, size=out.shape[-1], scale=0.1)
                ).to(device=x.device, dtype=x.dtype)
                levy_noise = levy_noise.clamp(-10.0, 10.0)
                out = out + self.drift(t, out) * self.delta_t \
                          + diff_term * (self.delta_t ** (1 / ALPHA)) * levy_noise
            drift_out = self.drift(t, out)
            return drift_out, out
        else:
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
CD_trains = [to_tensor(c).to(device) for c in cd_trains]
CD_vals   = [to_tensor(c).to(device) for c in cd_vals]
CD_tests  = [to_tensor(c).to(device) for c in cd_tests]

batch_sizes = [y.shape[0] for y in y_trains]

# ============================================================
#  7. 模型、优化器、损失函数（与原版完全相同）
# ============================================================
SEQ_LEN   = D
ITER      = 50
ITER_TEST = 1

nets = nn.ModuleList([SDENet(dim=SEQ_LEN, layer_depth=25).to(device) for _ in range(N_STEPS)])

class MultiHeadFusionAttention(nn.Module):
    def __init__(self, n_steps, sde_dim, corr_dim, d_model=64, n_heads=4, out_dim=None):
        super().__init__()
        assert d_model % n_heads == 0
        out_dim = out_dim or n_steps * 2
        self.sde_proj  = nn.Linear(sde_dim,  d_model)
        self.corr_proj = nn.Linear(corr_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=0.1, batch_first=True)
        self.norm     = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(n_steps * d_model, out_dim)
        self.n_steps  = n_steps
        self.d_model  = d_model

    def forward(self, sde_tokens, corr_feat):
        N = sde_tokens.shape[0]
        sde_split = sde_tokens.view(N, self.n_steps, -1)
        sde_emb   = self.sde_proj(sde_split)
        corr_emb  = self.corr_proj(corr_feat).unsqueeze(1)
        tokens    = torch.cat([sde_emb, corr_emb], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens      = self.norm(tokens + attn_out)
        sde_out = tokens[:, :self.n_steps, :].reshape(N, -1)
        return self.out_proj(sde_out)

attention = MultiHeadFusionAttention(
    n_steps=N_STEPS, sde_dim=SEQ_LEN, corr_dim=CORR_DIM,
    d_model=64, n_heads=4, out_dim=N_STEPS * 2,
).to(device)

print(f'[Model] 每个 SDENet 参数量: {count_parameters(nets[0]):,}')

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

criterion = nn.BCEWithLogitsLoss()
real_label, fake_label = 0, 1

def nll_loss(y, mean, sigma):
    sigma = sigma.clamp(min=1e-3)
    return torch.mean(torch.log(sigma) + (y - mean) ** 2 / (2 * sigma ** 2))

def mse_loss(y, mean):
    return torch.mean((y - mean) ** 2)

# ============================================================
#  8. 前向传播（与原版相同）
# ============================================================
TAIL_CUTS = list(range(N_STEPS - 1, -1, -1))

def forward_all(nets, X_list, training_diffusion=False):
    return [nets[i](X_list[i], training_diffusion=training_diffusion)
            for i in range(N_STEPS)]

def fuse_outputs(raw_outs, CD_list):
    outs    = [raw_outs[i][1] for i in range(N_STEPS)]
    aligned = [outs[i][:outs[-1].shape[0] - TAIL_CUTS[i]] for i in range(N_STEPS)]
    min_len = min(a.shape[0] for a in aligned)
    cat     = torch.cat([a[:min_len] for a in aligned], dim=1)
    corr_feat = CD_list[-1][:min_len]
    final   = attention(cat, corr_feat)
    means   = [final[:, i]                          for i in range(N_STEPS)]
    sigmas  = [F.softplus(final[:, N_STEPS + i]) + 1e-3 for i in range(N_STEPS)]
    return means, sigmas, min_len

def get_targets(Y_list, min_len):
    return [Y_list[i][:min_len] for i in range(N_STEPS)]

# ============================================================
#  9. 训练 & 测试（与原版相同）
# ============================================================
def train_epoch(epoch):
    for net in nets: net.train()
    total_loss = total_loss_in = total_loss_out = 0.0
    for _ in range(ITER):
        optimizer_F.zero_grad()
        raw = forward_all(nets, X_trains)
        means, sigmas, min_len = fuse_outputs(raw, CD_trains)
        targets = get_targets(Y_trains, min_len)
        loss_f = sum(nll_loss(targets[i], means[i], sigmas[i]) for i in range(N_STEPS))
        loss_f.backward()
        nn.utils.clip_grad_norm_(
            [p for g in optimizer_F.param_groups for p in g['params']], max_norm=5.0)
        optimizer_F.step()
        total_loss += loss_f.item()

        optimizer_G.zero_grad()
        preds_in    = [nets[i](X_trains[i], training_diffusion=True) for i in range(N_STEPS)]
        labels_real = [torch.full_like(preds_in[i], real_label)      for i in range(N_STEPS)]
        loss_in     = sum(criterion(preds_in[i], labels_real[i])      for i in range(N_STEPS))
        inputs_out  = [2 * torch.randn_like(X_trains[i]) + X_trains[i] for i in range(N_STEPS)]
        preds_out   = [nets[i](inputs_out[i], training_diffusion=True)  for i in range(N_STEPS)]
        labels_fake = [torch.full_like(preds_out[i], fake_label)        for i in range(N_STEPS)]
        loss_out    = sum(criterion(preds_out[i], labels_fake[i])       for i in range(N_STEPS))
        (loss_in + loss_out).backward()
        nn.utils.clip_grad_norm_(
            [p for g in optimizer_G.param_groups for p in g['params']], max_norm=5.0)
        optimizer_G.step()
        total_loss_in  += loss_in.item()
        total_loss_out += loss_out.item()

    with torch.no_grad():
        raw = forward_all(nets, X_trains)
        means, _, min_len = fuse_outputs(raw, CD_trains)
        targets = get_targets(Y_trains, min_len)
        mse_list = []
        for i in range(N_STEPS):
            pred_np = means[i].cpu().detach().numpy()
            if not np.isfinite(pred_np).all():
                raise RuntimeError(
                    f'Epoch {epoch}, step t+{i+1}: NaN/Inf，请降低 lr={args.lr}')
            mse_list.append(mean_squared_error(targets[i].cpu().numpy(), pred_np))

    print(f'[Train/NoPSR] Epoch {epoch:3d} | '
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
    print(f'[ Test/NoPSR] Epoch {epoch:3d} | RMSE: {(total_loss/ITER_TEST)**0.5:.6f}')
    return tuple(mse_list)

# ============================================================
#  10. 训练主循环
# ============================================================
train_losses, test_losses = [], []
for epoch in range(1, args.epochs + 1):
    train_losses.append(train_epoch(epoch))
    test_losses.append(test_epoch(epoch))
    if epoch in [int(e) for e in args.decreasing_lr]:
        for pg in optimizer_F.param_groups: pg['lr'] *= args.droprate
    if epoch in [int(e) for e in args.decreasing_lr2]:
        for pg in optimizer_G.param_groups: pg['lr'] *= args.droprate

# ============================================================
#  11. 最终结果打印
# ============================================================
print('\n===== [w/o PSR] 最终测试 MSE（标准化空间）=====')
step_labels = [f't+{i+1}' for i in range(N_STEPS)]
for i, label in enumerate(step_labels):
    print(f'  {label}: {test_losses[-1][i]:.6f}')

# ============================================================
#  12-13. 验证集残差 & 测试集最终推理
# ============================================================
def to_np(t): return t.cpu().detach().numpy()

for net in nets: net.eval()
with torch.no_grad():
    raw_val = forward_all(nets, X_vals)
    val_means, val_sigmas, val_min_len = fuse_outputs(raw_val, CD_vals)
    val_targets = get_targets(Y_vals, val_min_len)

val_residuals = [to_np(val_targets[i]) - to_np(val_means[i]) for i in range(N_STEPS)]
print(f'\n[验证集残差] 各步长残差长度: {[len(r) for r in val_residuals]}')

with torch.no_grad():
    raw_test = forward_all(nets, X_tests)
    final_means, final_sigmas, min_len = fuse_outputs(raw_test, CD_tests)
    final_targets = get_targets(Y_tests, min_len)

def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

test_true_price  = [inv(to_np(final_targets[i])) for i in range(N_STEPS)]
test_mean_price  = [inv(to_np(final_means[i]))   for i in range(N_STEPS)]
test_sigma_price = [to_np(final_sigmas[i]) * scaler.scale_[0] for i in range(N_STEPS)]
val_resid_price  = [r * scaler.scale_[0] for r in val_residuals]

# ============================================================
#  14. 保存
# ============================================================
for i in range(N_STEPS):
    fname = f'ablation_no_psr_{step_labels[i]}.npz'
    np.savez(fname,
             test_true  = test_true_price[i],
             test_mean  = test_mean_price[i],
             test_sigma = test_sigma_price[i],
             val_resid  = val_resid_price[i])
    print(f'[保存] {fname}  test_true:{test_true_price[i].shape}  val_resid:{val_resid_price[i].shape}')
