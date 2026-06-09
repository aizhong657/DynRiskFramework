# ============================================================
#  上证50指数多步预测 —— 双流融合架构（Dual-Stream Fusion LDENet）
#
#  相比原版 LDENet 的改动：
#    1. 新增 corr_dim 输入（滚动窗口 G-P 算法计算关联维数）
#    2. 上路 LDEModule：输入 x_seq [B, T, D]，Euler-Maruyama + Lévy 噪声
#    3. 下路 FeatureStream：全序列 mean pooling 后拼 corr_dim → Linear
#    4. 注意力融合：MultiheadAttention Self-Attention（序列长度=2）
#    5. 输出双头：mean + log_sigma → 与原版 NLL 损失完全兼容
#    6. N_STEPS 个独立模型实例，保存格式与原版一致（.npz）
# ============================================================

# ── 标准库 ──────────────────────────────────────────────────
from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import argparse
import warnings
warnings.filterwarnings('ignore')

# ── 第三方库 ────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch import nn, optim
import math
from scipy.stats import levy_stable   # 仅用于启动时预生成噪声池，训练中不再调用


# ============================================================
#  1. 超参数（argparse 最先定义，全局可用）
# ============================================================
parser = argparse.ArgumentParser(description='Dual-Stream LDENet for SSE50')
parser.add_argument('--epochs',         type=int,   default=100)
parser.add_argument('--n_steps',        type=int,   default=4,    help='预测步数，即同时预测 t+1…t+n_steps')
parser.add_argument('--lr',             type=float, default=1e-4, help='主网络（drift/feature/attn/head）学习率')
parser.add_argument('--lr2',            type=float, default=0.01, help='diffusion 判别器学习率')
parser.add_argument('--droprate',       type=float, default=0.1,  help='学习率衰减系数')
parser.add_argument('--decreasing_lr',  default=[20], nargs='+',  help='主网络 lr 衰减 epoch')
parser.add_argument('--decreasing_lr2', default=[],   nargs='+',  help='diffusion lr 衰减 epoch')
parser.add_argument('--hidden_dim',     type=int,   default=64,   help='上/下路统一隐层维度')
parser.add_argument('--attn_heads',     type=int,   default=4,    help='MultiheadAttention 头数')
parser.add_argument('--layer_depth',    type=int,   default=25,   help='Euler-Maruyama 离散步数')
parser.add_argument('--sigma',          type=float, default=0.5,  help='扩散项缩放系数')
parser.add_argument('--dir_weight',     type=float, default=1.0,  help='方向损失权重（建议1.0~2.0）')
parser.add_argument('--nll_weight',     type=float, default=0.5,  help='Student-t NLL 损失权重（建议0.1~1.0）')
parser.add_argument('--sigma_min',      type=float, default=0.1,  help='sigma 下界（防止NLL坍缩，建议0.05~0.2）')
parser.add_argument('--nu_init',        type=float, default=5.0,  help='Student-t 自由度初始值（>2 保证方差存在，建议2~30）')
parser.add_argument('--corr_dim_path',  type=str,   default='corr_dim_scaled.csv',
                                                              help='预计算关联维数 CSV 路径（由 compute_corr_dim.py 生成）')
parser.add_argument('--gpu',            type=int,   default=0)
parser.add_argument('--seed',           type=int,   default=4)
args = parser.parse_args([])


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
# ============================================================
ALPHA_LEVY = 1.5   # 稳定指数（α=2 退化为高斯）
BETA_LEVY  = 0.0   # 偏斜参数（0 = 对称）

# ── 噪声池设计 ───────────────────────────────────────────────
# 问题根源：levy_stable.rvs() 是 scipy 纯Python实现，每次调用都很慢。
# 训练时每个forward需要 layer_depth 次采样，batch × hidden_dim 个值，
# 100 epoch × ITER × N_STEPS × 25步 = 数十万次调用，严重拖慢速度。
#
# 解决方案：启动时一次性预生成足够大的噪声张量存到GPU，
# 训练时随机取偏移量切片复用，完全消除 scipy 调用开销。
#
# 池大小 = POOL_MULTIPLIER × 最大batch × hidden_dim
# POOL_MULTIPLIER=200 意味着约200轮不重复，统计特性足够随机。
LEVY_POOL_SIZE   = 200 * 4096   # 约80万个样本，覆盖多轮训练
LEVY_POOL: torch.Tensor = None  # 启动后由 build_levy_pool() 填充


def build_levy_pool(hidden_dim: int, device: torch.device) -> None:
    """
    一次性从 α-stable 分布采样并存到 GPU。
    在模型实例化之后、训练开始之前调用一次即可。

    噪声池形状：[LEVY_POOL_SIZE, hidden_dim]
    训练时用随机偏移量切片取 [B, hidden_dim] 的块，避免重复调用 scipy。
    """
    global LEVY_POOL
    print(f'[噪声池] 预生成 Lévy 噪声，大小={LEVY_POOL_SIZE}×{hidden_dim}…', flush=True)
    raw = levy_stable.rvs(
        ALPHA_LEVY, BETA_LEVY,
        size=(LEVY_POOL_SIZE, hidden_dim),
        scale=0.1,
    ).astype(np.float32)
    LEVY_POOL = torch.from_numpy(raw).clamp(-10.0, 10.0).to(device)
    print(f'[噪声池] 完成，已移至 {device}，'
          f'内存占用 ≈ {LEVY_POOL.numel() * 4 / 1024 / 1024:.1f} MB')


def sample_levy_noise(batch_size: int, hidden_dim: int) -> torch.Tensor:
    """
    从噪声池随机切片，返回 [batch_size, hidden_dim] 的 Lévy 噪声。
    使用随机偏移保证每次取到的块位置不同，避免周期性重复。
    """
    pool_len = LEVY_POOL.shape[0]
    # 随机起始偏移，保证切片不越界
    offset = torch.randint(0, pool_len - batch_size, (1,)).item()
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

# 70 / 10 / 20 划分，仅取收盘价
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
n_total = len(stk_data)
n_train = int(np.ceil(TRAIN_RATIO * n_total))
n_val   = int(np.ceil(VAL_RATIO   * n_total))
n_test  = n_total - n_train - n_val

training_set   = stk_data.iloc[:n_train,              4:5].values
validation_set = stk_data.iloc[n_train:n_train+n_val, 4:5].values
testing_set    = stk_data.iloc[n_train+n_val:,        4:5].values
print(f'[Data] 训练集: {training_set.shape}  验证集: {validation_set.shape}  测试集: {testing_set.shape}')

# 标准化（只用训练集 fit）
scaler = StandardScaler()
train_scaled = scaler.fit_transform(training_set).flatten()
val_scaled   = scaler.transform(validation_set).flatten()
test_scaled  = scaler.transform(testing_set).flatten()

def make_date_value_df(dates, values):
    return pd.DataFrame({'Date': dates, 'closescale': values})

train_df = make_date_value_df(stk_data['Date'].values[:n_train],                          train_scaled)
val_df   = make_date_value_df(stk_data['Date'].values[n_train:n_train+n_val],             val_scaled)
test_df  = make_date_value_df(stk_data['Date'].values[n_train+n_val:n_train+n_val+n_test], test_scaled)


# ============================================================
#  5. 相空间重构
# ============================================================
def PhaSpaRecon(df, tau: int, d: int, T: int):
    """
    参数
    ────
    df  : 含 [Date, closescale] 的 DataFrame
    tau : 时延
    d   : 嵌入维数
    T   : 预测步数
    返回
    ────
    Xn, Yn, Y, X（与原版接口完全一致）
    """
    values = np.array(df)[:, 1].astype(float)
    dates  = np.array(df)[:, 0]
    n = len(values)

    if (n - T - (d - 1) * tau) < 1:
        raise ValueError("tau 或 d 过大，超出序列长度")

    width = n - (d - 1) * tau - 1
    Xn1      = np.stack([values[i*tau : i*tau + width] for i in range(d)], axis=1)
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


def price_to_return(x_price: np.ndarray) -> np.ndarray:
    """
    将价格水平矩阵转换为收益率矩阵。

    输入 x_price : [N, D]，每列为相空间重构的价格水平
                   col 0 = price(t-(D-1)*tau)  最旧
                   col D-1 = price(t)           最新

    转换方式：对相邻列做差分，得到 D-1 个收益率
      ret[:, i] = price[:, i+1] - price[:, i]   i = 0..D-2

    输出 x_ret   : [N, D-1]，每列为相邻时刻收益率

    说明：
      - 收益率与 y_diff 的相关性远高于价格水平（诊断显示 0.073 vs 0.045）
      - 差分后去掉了价格水平的趋势成分，剩下真正的动量信息
      - 维度从 D 变为 D-1（= 22），需同步更新模型 input_dim
    """
    return np.diff(x_price, axis=1)   # [N, D-1]


D, TAU    = 22, 1           # 嵌入维数、时延（与原版一致）
D_PRICE   = D + 1
D_RET     = D               # 收益率特征维度（= 22），作为模型 input_dim
N_STEPS   = args.n_steps    # 预测步数

# ============================================================
#  数据构建：x 改为收益率矩阵，y 保持差分目标
#
#  x_price [N, D]   → price_to_return → x_ret [N, D-1=22]
#  y_prev  = x_price[:, -1]（最新时刻价格，仅用于还原绝对价格）
#  y_diff  = y_abs - y_prev = price(t+k) - price(t)
#
#  为什么收益率能改善方向准确率：
#    价格水平 与 y_diff 相关系数 ≈ 0.04（诊断结果）
#    相邻差分（收益率）与 y_diff 相关系数 ≈ 0.07（高一倍）
#    更重要的是：收益率序列包含动量信息——
#      近期连续上涨 → 收益率多为正 → y_diff 倾向正
#      近期连续下跌 → 收益率多为负 → y_diff 倾向负
#    这是模型能够学习的真实方向信号。
# ============================================================
x_trains, y_trains, y_abs_trains, y_prev_trains = [], [], [], []
x_vals,   y_vals,   y_abs_vals,   y_prev_vals   = [], [], [], []
x_tests,  y_tests,  y_abs_tests,  y_prev_tests  = [], [], [], []

for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr_price, ytr_abs = build_xy(train_df, TAU, D_PRICE, T, drop_tail=drop)
    xva_price, yva_abs = build_xy(val_df,   TAU, D_PRICE, T, drop_tail=drop)
    xte_price, yte_abs = build_xy(test_df,  TAU, D_PRICE, T, drop_tail=drop)

    # y_prev：t 时刻价格（x最后一列），用于差分目标和还原绝对价格
    y_prev_tr = xtr_price[:, -1]
    y_prev_va = xva_price[:, -1]
    y_prev_te = xte_price[:, -1]

    # 差分目标：price(t+k) - price(t)
    ytr_diff = ytr_abs - y_prev_tr
    yva_diff = yva_abs - y_prev_va
    yte_diff = yte_abs - y_prev_te

    # x 转为收益率矩阵 [N, D-1=22]
    xtr_ret = price_to_return(xtr_price)
    xva_ret = price_to_return(xva_price)
    xte_ret = price_to_return(xte_price)

    x_trains.append(xtr_ret);   y_trains.append(ytr_diff)
    y_abs_trains.append(ytr_abs); y_prev_trains.append(y_prev_tr)

    x_vals.append(xva_ret);     y_vals.append(yva_diff)
    y_abs_vals.append(yva_abs);   y_prev_vals.append(y_prev_va)

    x_tests.append(xte_ret);    y_tests.append(yte_diff)
    y_abs_tests.append(yte_abs);  y_prev_tests.append(y_prev_te)

    # 打印诊断信息：收益率特征与y_diff的相关性
    max_corr = max(abs(np.corrcoef(xtr_ret[:, col], ytr_diff)[0, 1])
                   for col in range(D_RET))
    print(f'T={T}  x_ret:{xtr_ret.shape}  '
          f'y_diff: mean={ytr_diff.mean():.4f} std={ytr_diff.std():.4f}  '
          f'上涨比={np.mean(ytr_diff>0):.3f}  '
          f'x与y_diff最大相关={max_corr:.4f}')


# ============================================================
#  6. 读取预计算的关联维数（由 compute_corr_dim.py 生成）
# ============================================================
import os

corr_csv_path = args.corr_dim_path
if not os.path.exists(corr_csv_path):
    raise FileNotFoundError(
        f'找不到关联维数文件：{corr_csv_path}\n'
        f'请先运行：python compute_corr_dim.py'
    )

corr_df = pd.read_csv(corr_csv_path, parse_dates=['date'])

# 行数校验：必须与原始价格序列完全对齐
n_total_price = n_train + n_val + n_test
if len(corr_df) != n_total_price:
    raise ValueError(
        f'辅助特征 CSV 行数 ({len(corr_df)}) '
        f'与价格序列总长度 ({n_total_price}) 不匹配。\n'
        f'请用相同数据集重新运行 compute_corr_dim.py。'
    )

# 读取 5 个标准化列：corr_dim / lyap / ret / macd / rsi
# 形状均为 (n_total,)，后续按 train/val/test 切分再与相空间重构对齐
FEAT_COLS = ['corr_dim_scaled', 'lyap_scaled', 'ret_scaled',
             'macd_scaled',     'rsi_scaled']
N_AUX     = len(FEAT_COLS)    # = 5，供 FeatureStream 使用

# 检查列是否存在（防止用旧版 CSV 运行新主程序）
missing = [c for c in FEAT_COLS if c not in corr_df.columns]
if missing:
    raise ValueError(
        f'CSV 中缺少以下列：{missing}\n'
        f'请重新运行 compute_corr_dim.py 生成包含全部指标的新 CSV。'
    )

# full_aux: (n_total, 5)，每列为一个标准化辅助特征
full_aux = corr_df[FEAT_COLS].values.astype(np.float64)

# 按 train / val / test 切分 → 各为 (n_segment, 5)
train_aux_full = full_aux[:n_train]
val_aux_full   = full_aux[n_train : n_train + n_val]
test_aux_full  = full_aux[n_train + n_val:]
print(f'[辅助特征] 已从 {corr_csv_path} 读取，列：{FEAT_COLS}')
print(f'[辅助特征] train:{train_aux_full.shape}  '
      f'val:{val_aux_full.shape}  test:{test_aux_full.shape}')


# ============================================================
#  7. 辅助特征与相空间重构对齐
#
#  PhaSpaRecon 内部截取导致输出长度 < 输入长度，需对辅助特征
#  做相同的索引截取，使其行数与 x_trains[i] 完全一致。
#
#  对齐规则（与原 corr_dim 一致，现扩展到 5 列）：
#    start     = (d-1)*tau        相空间重构丢弃的前缀行数
#    width     = n_series - (d-1)*tau - 1
#    end       = start + width - (T-1)
# ============================================================
def align_aux(aux_arr: np.ndarray, n_series: int,
              d: int, tau: int, T: int) -> np.ndarray:
    """
    将辅助特征矩阵对齐到相空间重构后的样本数。

    参数
    ────
    aux_arr  : 辅助特征，shape (n_series, n_feat) 或 (n_series,)
    n_series : 对应的价格序列长度
    d, tau   : 嵌入维数、时延
    T        : 预测步数（drop_tail = T-1）

    返回
    ────
    对齐后的数组，shape (n_samples, n_feat) 或 (n_samples,)
    """
    drop_tail = T - 1
    start     = (d - 1) * tau
    width     = n_series - (d - 1) * tau - 1
    end       = start + width - drop_tail
    return aux_arr[start : end]


# 对每个步长 T 分别对齐，结果形状 (n_samples, 5)
aux_trains, aux_vals, aux_tests = [], [], []
for T in range(1, N_STEPS + 1):
    at = align_aux(train_aux_full, len(train_scaled), D_PRICE, TAU, T)
    av = align_aux(val_aux_full,   len(val_scaled),   D_PRICE, TAU, T)
    ae = align_aux(test_aux_full,  len(test_scaled),  D_PRICE, TAU, T)
    aux_trains.append(at)
    aux_vals.append(av)
    aux_tests.append(ae)
    assert len(at) == len(x_trains[T-1]), (
        f'T={T} aux_train 行数 {len(at)} ≠ x_train 行数 {len(x_trains[T-1])}'
    )
    print(f'T={T}  aux_train:{at.shape}  aux_val:{av.shape}  aux_test:{ae.shape}')


# ============================================================
#  8. 数据转 Tensor
#
#  x_seq：[N, D_RET=22] → unsqueeze → [N, 1, 22]
#          input_dim = D_RET = 22（收益率，比原价格矩阵少1列）
#  y_diff：[N]   差分目标
#  y_prev：[N]   t 时刻价格水平，仅推理阶段用于还原绝对价格
#  aux：   [N, 5] 5 个辅助特征
# ============================================================
def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).to(device)

# x_ret [N, D_RET] → [N, 1, D_RET]
X_trains    = [to_tensor(x).unsqueeze(1) for x in x_trains]     # [N, 1, 22]
Y_trains    = [to_tensor(y)              for y in y_trains]      # [N]
C_trains    = [to_tensor(a)              for a in aux_trains]    # [N, 5]
Yp_trains   = [to_tensor(p)             for p in y_prev_trains] # [N] t时刻价格

X_vals      = [to_tensor(x).unsqueeze(1) for x in x_vals]
Y_vals      = [to_tensor(y)              for y in y_vals]
C_vals      = [to_tensor(a)              for a in aux_vals]
Yp_vals     = [to_tensor(p)             for p in y_prev_vals]

X_tests     = [to_tensor(x).unsqueeze(1) for x in x_tests]
Y_tests     = [to_tensor(y)              for y in y_tests]
C_tests     = [to_tensor(a)              for a in aux_tests]
Yp_tests    = [to_tensor(p)             for p in y_prev_tests]

print(f'\n[Tensor] X_trains[0]:{X_trains[0].shape}  '
      f'Y_trains[0](diff):{Y_trains[0].shape}  '
      f'C_trains[0]:{C_trains[0].shape}  '
      f'Yp_trains[0]:{Yp_trains[0].shape}')
print(f'[输入维度] input_dim = D_RET = {D_RET}（收益率特征，原 D_PRICE={D_PRICE} 价格水平 - 1）')

# ── 绝对价格目标（仅用于最终还原，不参与训练损失）──────────────
Y_abs_trains  = [to_tensor(y) for y in y_abs_trains]
Y_abs_vals    = [to_tensor(y) for y in y_abs_vals]
Y_abs_tests   = [to_tensor(y) for y in y_abs_tests]


# ============================================================
#  9. 模型子模块定义
# ============================================================

# ── 9.1 漂移项网络 ───────────────────────────────────────────
class DriftNet(nn.Module):
    """
    f(t, x)：线性 + ReLU，输入输出均为 [B, hidden_dim]。
    参数 t 保留接口（时变扩展用），当前实现为时间无关。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 9.2 扩散项网络 ───────────────────────────────────────────
class DiffusionNet(nn.Module):
    """
    g(t, x)：输出 logit（不含 Sigmoid）。
      • training_diffusion=False → 手动 Sigmoid × sigma，作为扩散缩放
      • training_diffusion=True  → 直接送入 BCEWithLogitsLoss（数值稳定）
    输入：[B, hidden_dim]，输出：[B, 1]
    """
    def __init__(self, hidden_dim: int, mid_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 9.3 上路：LDE 模块（随机流）────────────────────────────
class LDEModule(nn.Module):
    """
    Lévy-Driven SDE 模块（上路随机流）

    数据流：
      x_seq [B, T, D]
        → 取最后时间步 x_last [B, D]
        → downsample Linear [B, hidden_dim]
        → Euler-Maruyama 积分（layer_depth 步，每步叠加 Lévy 噪声）
        → H_lde [B, hidden_dim]

    training_diffusion=True 时：
      直接对 downsample 输出（detach）跑 DiffusionNet，返回 logit [B, 1]
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 layer_depth: int = 25, sigma: float = 0.5):
        super().__init__()
        self.layer_depth = layer_depth
        self.sigma       = sigma
        self.delta_t     = 1.0 / layer_depth

        self.downsample = nn.Linear(input_dim, hidden_dim)
        self.drift      = DriftNet(hidden_dim)
        self.diffusion  = DiffusionNet(hidden_dim)

    def _levy_noise(self, shape, device, dtype) -> torch.Tensor:
        """
        从全局噪声池切片取噪声，形状 [B, hidden_dim]。
        相比原来每步调用 levy_stable.rvs()，速度提升约 100-200x。
        噪声池在训练开始前由 build_levy_pool() 一次性生成并存到 GPU。
        """
        batch_size, hidden_dim = shape[0], shape[1]
        noise = sample_levy_noise(batch_size, hidden_dim)
        # 噪声池已在GPU上，直接返回（dtype转换保险起见保留）
        return noise.to(dtype=dtype)

    def forward(self, x_seq: torch.Tensor,
                training_diffusion: bool = False) -> torch.Tensor:
        """
        x_seq : [B, T, input_dim]
        返回
        ────
        False → H_lde [B, hidden_dim]
        True  → diffusion logit [B, 1]
        """
        # 取最后时间步并降维：[B, input_dim] → [B, hidden_dim]
        x_last = x_seq[:, -1, :]        # [B, input_dim]
        out    = self.downsample(x_last) # [B, hidden_dim]

        if training_diffusion:
            # 判别器模式：detach 防止梯度回流到 downsample
            return self.diffusion(0.0, out.detach())   # [B, 1]

        # 扩散缩放系数（在积分前固定）：[B, 1]，广播到 [B, hidden_dim]
        diff_scale = self.sigma * torch.sigmoid(self.diffusion(0.0, out))

        # Euler-Maruyama 积分
        for step in range(self.layer_depth):
            t          = float(step) / self.layer_depth
            levy_noise = self._levy_noise(out.shape, out.device, out.dtype)
            # x_{t+Δt} = x_t + f(t,x)·Δt + g(t,x)·Δt^{1/α}·dL
            out = (out
                   + self.drift(t, out) * self.delta_t
                   + diff_scale * (self.delta_t ** (1.0 / ALPHA_LEVY)) * levy_noise)

        return out   # H_lde: [B, hidden_dim]


# ── 9.4 下路：特征流（全序列 mean pooling + 5 个辅助特征）────────
class FeatureStream(nn.Module):
    """
    特征流模块（下路）

    数据流：
      x_seq  [B, T, input_dim]
        → mean pooling over T  → x_mean   [B, input_dim]
        → cat aux_feat [B, 5]  → [B, input_dim + n_aux]
        → Linear + ReLU        → H_feature [B, hidden_dim]

    5 个辅助特征（均已标准化）：
      corr_dim  G-P 关联维数
      lyap      局部李雅普诺夫指数
      ret       对数收益率
      macd      MACD 柱（DIF - Signal）
      rsi       RSI(14)

    mean pooling 保留全序列的历史信息，弥补上路只取最后时间步的局限。
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_aux: int = 5):
        super().__init__()
        # 拼接后维度：input_dim（mean pooling）+ n_aux（辅助特征）
        self.linear = nn.Linear(input_dim + n_aux, hidden_dim)
        self.act    = nn.ReLU(inplace=True)

    def forward(self, x_seq: torch.Tensor,
                aux_feat: torch.Tensor) -> torch.Tensor:
        """
        x_seq    : [B, T, input_dim]
        aux_feat : [B, n_aux]  5 个标准化辅助特征
        返回      : H_feature [B, hidden_dim]
        """
        # 对 T 维度做均值池化：[B, T, D] → [B, D]
        x_mean = x_seq.mean(dim=1)                           # [B, input_dim]

        # 拼接辅助特征：[B, D] cat [B, n_aux] → [B, D + n_aux]
        fused = torch.cat([x_mean, aux_feat], dim=-1)        # [B, input_dim + n_aux]

        # 线性映射到 hidden_dim
        return self.act(self.linear(fused))                   # [B, hidden_dim]


# ── 9.5 注意力融合 ───────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    MultiheadAttention Self-Attention 融合模块

    数据流：
      H_lde [B, D_h] + H_feature [B, D_h]
        → stack → seq [B, 2, D_h]
        → MultiheadAttention (Q=K=V=seq)
        → 残差 + LayerNorm
        → reshape → fused_flat [B, 2 * D_h]
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f'hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除'
        )
        # batch_first=True：I/O 形状均为 [B, seq_len, embed_dim]
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H_lde: torch.Tensor,
                H_feature: torch.Tensor) -> torch.Tensor:
        """
        H_lde, H_feature : [B, hidden_dim]
        返回               : fused_flat [B, 2 * hidden_dim]
        """
        # 组合为长度=2 的序列：[B, 2, hidden_dim]
        seq = torch.cat([H_lde.unsqueeze(1), H_feature.unsqueeze(1)], dim=1)

        # Self-Attention
        attn_out, _ = self.attn(seq, seq, seq)   # [B, 2, hidden_dim]

        # 残差 + LayerNorm（稳定训练，防止梯度消失）
        attn_out = self.norm(attn_out + seq)

        # 展平为 [B, 2 * hidden_dim]
        return attn_out.contiguous().reshape(attn_out.size(0), -1)


# ── 9.6 全局输出头（n+1 路输入，n 个 mean + n 个独立 log_sigma）──
class GlobalPredictionHead(nn.Module):
    """
    全局预测头：接收 n 个 SDE 主干的 fused 输出 + 1 个全局 FeatureStream 输出，
    共 n+1 路，输出 n 个独立均值 + n 个独立 log_sigma。

    数据流：
      fused_list  : list of [B, 2×hidden]，长度 n（来自 n 个 SDE 主干）
      H_feat_glob : [B, hidden]（来自全局独立 FeatureStream）
        → cat → [B, (2n+1)×hidden]
        → trunk Linear+ReLU → [B, hidden]
        → head_means:  n 个独立 Linear(hidden→1) → means     list of [B]
        → head_sigmas: n 个独立 Linear(hidden→1) → log_sigmas list of [B]

    设计说明：
      • 全局 FeatureStream 提供一个与步长无关的辅助特征视角，
        补充各 SDE 主干只看单步长的局限
      • n 个独立 log_sigma 使每个步长的不确定性估计各自收敛，
        短期预测和长期预测的不确定性天然不同，不宜强制共享
      • trunk 做跨步长 + 跨来源的特征交互
    """
    def __init__(self, hidden_dim: int, n_steps: int, mean_scale: float = 0.3):
        super().__init__()
        self.n_steps    = n_steps
        self.mean_scale = mean_scale
        # 输入：n 个 fused（各 2×hidden）+ 1 个 H_feat_glob（hidden）
        in_dim = 2 * hidden_dim * n_steps + hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # n 个独立均值头
        self.head_means  = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_steps)
        ])
        # n 个独立 log_sigma 头
        self.head_sigmas = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_steps)
        ])

    def forward(self, fused_list: list, H_feat_glob: torch.Tensor):
        """
        参数
        ────
        fused_list  : list of [B, 2×hidden]，长度 n
        H_feat_glob : [B, hidden]

        返回
        ────
        means      : list of [B]，长度 n
        log_sigmas : list of [B]，长度 n（各步长独立）
        """
        # concat n 个 fused + 全局特征：[B, (2n+1)×hidden]
        x = torch.cat(fused_list + [H_feat_glob], dim=-1)
        h = self.trunk(x)                                       # [B, hidden]

        means = [
            (torch.tanh(self.head_means[i](h)) * self.mean_scale).squeeze(-1)
            for i in range(self.n_steps)
        ]
        log_sigmas = [
            self.head_sigmas[i](h).squeeze(-1)
            for i in range(self.n_steps)
        ]
        return means, log_sigmas


# ── 9.7 全局独立 FeatureStream ───────────────────────────────
# 与各 SDE 内部的 FeatureStream 结构相同，但独立实例化，
# 固定使用第 1 个步长（T=1）的输入，只跑一次前向，
# 为 GlobalPredictionHead 提供与步长无关的辅助特征视角。
# （实例化在第 11 节，此处仅做说明）


# ============================================================
#  10. 单步双流融合主干（不含预测头）
# ============================================================
class DualStreamSDENet(nn.Module):
    """
    单步双流融合主干：LDEModule + FeatureStream + AttentionFusion。

    forward 推理模式返回 fused [B, 2×hidden]，供 GlobalPredictionHead 使用。
    内部的 FeatureStream 仍保留，与全局 FeatureStream 独立，
    各自捕捉对应步长的局部特征。

    数据流：
      x_seq [B,T,D]
        ├─ [LDEModule]      → H_lde     [B, hidden]
        └─ [FeatureStream]  → H_feature [B, hidden]
                ↓
        [AttentionFusion]   → fused [B, 2×hidden]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_heads: int = 4, layer_depth: int = 25, sigma: float = 0.5):
        super().__init__()
        self.lde            = LDEModule(input_dim, hidden_dim, layer_depth, sigma)
        self.feature_stream = FeatureStream(input_dim, hidden_dim, n_aux=N_AUX)
        self.attn_fusion    = AttentionFusion(hidden_dim, num_heads)

    def forward(self, x_seq: torch.Tensor, corr_dim: torch.Tensor,
                training_diffusion: bool = False):
        """
        返回
        ────
        training_diffusion=True  → diffusion logit [B, 1]
        training_diffusion=False → fused [B, 2×hidden]
        """
        if training_diffusion:
            return self.lde(x_seq, training_diffusion=True)
        H_lde     = self.lde(x_seq, training_diffusion=False)
        H_feature = self.feature_stream(x_seq, corr_dim)
        fused     = self.attn_fusion(H_lde, H_feature)
        return fused


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
#  11. 实例化 N_STEPS 个模型 & 优化器
# ============================================================
HIDDEN_DIM  = args.hidden_dim
ATTN_HEADS  = args.attn_heads
LAYER_DEPTH = args.layer_depth
SIGMA       = args.sigma
SEQ_LEN     = D_RET          # 输入维度 = 收益率特征数 = D-1 = 22
ITER        = 50
ITER_TEST   = 1

nets = nn.ModuleList([
    DualStreamSDENet(
        input_dim=SEQ_LEN,
        hidden_dim=HIDDEN_DIM,
        num_heads=ATTN_HEADS,
        layer_depth=LAYER_DEPTH,
        sigma=SIGMA,
    ).to(device)
    for _ in range(N_STEPS)
])

# 全局独立 FeatureStream：固定使用 T=1（第 0 个步长）的输入，
# 为 GlobalPredictionHead 提供与步长无关的辅助特征视角
global_feature_stream = FeatureStream(
    input_dim=SEQ_LEN,
    hidden_dim=HIDDEN_DIM,
    n_aux=N_AUX,
).to(device)

# 全局预测头：n+1 路输入，n 个 mean + n 个独立 log_sigma
global_head = GlobalPredictionHead(
    hidden_dim=HIDDEN_DIM,
    n_steps=N_STEPS,
    mean_scale=0.3,
).to(device)

print(f'[Model] 每个 DualStreamSDENet 主干参数量: {count_parameters(nets[0]):,}')
print(f'[Model] GlobalFeatureStream 参数量: {count_parameters(global_feature_stream):,}')
print(f'[Model] GlobalPredictionHead 参数量: {count_parameters(global_head):,}')

# ── 可学习 Student-t 自由度 nu（n 个独立，与 n 个 log_sigma 对应）──
_log_nu_init = math.log(math.expm1(max(args.nu_init - 2.0, 0.01)))
log_nus = nn.ParameterList([
    nn.Parameter(torch.tensor(_log_nu_init, device=device))
    for _ in range(N_STEPS)
])
print(f'[Model] Student-t 初始自由度 nu ≈ {args.nu_init:.1f}（各步长独立可学习）')

# ── 预生成全局 Lévy 噪声池（训练前执行一次，消除 scipy 逐步调用开销）──
# 必须在模型实例化（确定 hidden_dim）之后、训练循环之前调用
build_levy_pool(hidden_dim=HIDDEN_DIM, device=device)

# ── 分两组优化器 ─────────────────────────────────────────────
# optimizer_F：主网络（LDE的downsample+drift, feature_stream, attn_fusion, pred_head）
# optimizer_G：扩散判别器（LDE的diffusion）
optimizer_F = optim.SGD(
    [{'params': net.lde.downsample.parameters()}          for net in nets] +
    [{'params': net.lde.drift.parameters()}               for net in nets] +
    [{'params': net.feature_stream.parameters()}          for net in nets] +
    [{'params': net.attn_fusion.parameters()}             for net in nets] +
    [{'params': global_feature_stream.parameters()}]                       +
    [{'params': global_head.parameters()}]                                 +
    [{'params': [log_nus[i]]}                             for i in range(N_STEPS)],
    lr=args.lr, momentum=0.9, weight_decay=5e-4,
)
optimizer_G = optim.SGD(
    [{'params': net.lde.diffusion.parameters()} for net in nets],
    lr=args.lr2, momentum=0.9, weight_decay=5e-4,
)

criterion  = nn.BCEWithLogitsLoss()   # 内置 Sigmoid，数值稳定
real_label = 0.0                      # 真实输入 label=0
fake_label = 1.0                      # 噪声输入 label=1

# 对齐切片：第 i 个网络（T=i+1）末尾需去掉 (N_STEPS-1-i) 行
# 例如 N_STEPS=4: [3, 2, 1, 0]
TAIL_CUTS = list(range(N_STEPS - 1, -1, -1))


# ============================================================
#  12. 损失函数
#
#  联合损失公式：
#    Loss = MSE(y_diff, mean_diff) + λ · DPL(y_diff, mean_diff)
#
#  DPL 实现：Hinge 方向损失（可微）
#
#    DPL = mean( max(0, margin - y_diff · mean_diff) )
#
#    • 乘积 > margin  → 方向正确且置信充足，惩罚 = 0
#    • 乘积 ∈ (0, margin) → 方向正确但置信不足，施加软惩罚
#    • 乘积 ≤ 0       → 方向错误，惩罚 = margin - 乘积（越错越大）
#
#  ⚠️  为什么原来的 torch.where 硬判断无效：
#    torch.where(wrong, full_like(penalty), zeros_like)
#    选到 penalty 常数的那些位置，其对 mean_diff 的梯度 = 0。
#    无论 λ 多大，方向错误样本都无法通过 DPL 给 mean_diff
#    任何梯度信号，模型感知不到"方向错了要往哪改"。
#
#  Hinge 损失的梯度（乘积 < margin 时）：
#    ∂DPL/∂mean_diff = -y_diff / batch_size
#    方向：推动 mean_diff 与 y_diff 同号（即方向正确）
#    幅度：与真实变化幅度成比例（真实大涨/大跌时梯度更强）
#
#  margin = _diff_std 的含义：
#    不仅要方向对，还要求置信度达到一个标准差量级，
#    防止模型用趋近于 0 的预测值（"我不知道"策略）规避惩罚。
#
#  λ（dir_weight）调参建议：
#    1.0  → 方向与精度并重（推荐起点）
#    2.0  → 更强调方向，DA 仍低时使用
#    0.5  → 更强调精度，MSE 偏高时回退
# ============================================================

# ============================================================
#  12. 损失函数
#
#  联合损失公式：
#    Loss = MSE(y_diff, mean_diff)
#          + λ_hinge · DPL(y_diff, mean_diff)
#          + λ_nll   · StudentT_NLL(y_diff, mean_diff, sigma, nu)
#
#  各项分工：
#    MSE          → 保证预测幅度准确
#    DPL(Hinge)   → 约束方向（软约束，有梯度）
#    StudentT NLL → 训练 log_sigma 头，同时与 Lévy 厚尾假设一致
#
#  Student-t NLL 说明：
#    原代码使用高斯 NLL，但模型底层是 Lévy 过程（厚尾），
#    高斯似然（轻尾）与之统计矛盾。Student-t 分布具有厚尾性质
#    （自由度 nu 越小尾越厚），且有封闭形式 PDF，计算可行。
#    nu 作为可学习参数，由数据自适应决定尾部厚度。
#    nu → ∞ 时退化为高斯，nu=1 时为 Cauchy 分布。
#
#  DPL Hinge 说明：
#    margin = _diff_std：不仅要方向对，还要求置信度达到一个标准差量级，
#    防止模型用趋近于 0 的预测值规避惩罚。
#    梯度：∂DPL/∂mean_diff = -y_diff（乘积 < margin 时），
#    推动 mean_diff 与 y_diff 同号。
# ============================================================

_diff_std = float(np.std(y_trains[0]))
print(f'[损失] 差分值标准差(_diff_std) = {_diff_std:.6f}')
print(f'[损失] Hinge margin = _diff_std = {_diff_std:.6f}')
print(f'[损失] dir_weight(λ_hinge)={args.dir_weight}  nll_weight(λ_nll)={args.nll_weight}')


def dpl_loss(y_diff: torch.Tensor,
             mean_diff: torch.Tensor) -> torch.Tensor:
    """
    Hinge 方向损失
    DPL = mean( max(0, margin - y_diff · mean_diff) )
    梯度：∂DPL/∂mean_diff = -y_diff（乘积 < margin 时）
    """
    margin  = float(_diff_std)
    product = y_diff * mean_diff
    hinge   = torch.clamp(margin - product, min=0.0)
    return hinge.mean()


def student_t_nll(y: torch.Tensor,
                  mean: torch.Tensor,
                  log_sigma: torch.Tensor,
                  log_nu: nn.Parameter) -> torch.Tensor:
    """
    Student-t 负对数似然（厚尾，与 Lévy 过程假设一致）

    分布：t_nu( (y - mean) / sigma ) / sigma
    NLL = log(sigma)
        + 0.5*(nu+1) * log(1 + (y-mean)^2 / (nu * sigma^2))
        - log( Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2)) )

    参数
    ────
    y         : 真实差分目标 [B]
    mean      : 预测均值 [B]
    log_sigma : 预测 log_sigma [B]（head_log_sigma 输出）
    log_nu    : 可学习自由度参数（标量），nu = softplus(log_nu) + 2

    说明
    ────
    nu = softplus(log_nu) + 2 保证 nu > 2（方差存在）
    sigma = softplus(log_sigma) + sigma_min 保证严格正
    常数项 log_gamma 不影响梯度方向，但保留使 NLL 可比较
    """
    nu    = F.softplus(log_nu) + 2.0                              # 标量，nu > 2
    sigma = F.softplus(log_sigma).clamp(min=args.sigma_min) + 1e-3  # [B]

    residual   = (y - mean) / sigma                               # [B]
    log_sigma_term = torch.log(sigma)                             # [B]
    core_term  = 0.5 * (nu + 1.0) * torch.log(
        1.0 + residual ** 2 / nu
    )                                                             # [B]
    # 对数归一化常数（标量，随 nu 变化参与梯度）
    log_norm = (torch.lgamma((nu + 1.0) / 2.0)
                - torch.lgamma(nu / 2.0)
                - 0.5 * torch.log(nu * torch.tensor(math.pi, device=y.device)))

    nll = log_sigma_term + core_term - log_norm
    return nll.mean()


def combined_loss(y_diff: torch.Tensor,
                  mean_diff: torch.Tensor,
                  log_sigma: torch.Tensor,
                  log_nu: nn.Parameter) -> tuple:
    """
    联合损失：MSE + λ_hinge·DPL + λ_nll·StudentT_NLL

    log_sigma 和 log_nu 均为各步长独立参数。
    返回 (total_loss, loss_mse, loss_dpl, loss_nll) 四元组
    """
    loss_mse  = mse_loss_fn(y_diff, mean_diff)
    loss_dpl  = dpl_loss(y_diff, mean_diff)
    loss_nll  = student_t_nll(y_diff, mean_diff, log_sigma, log_nu)
    total     = (loss_mse
                 + args.dir_weight * loss_dpl
                 + args.nll_weight * loss_nll)
    return total, loss_mse, loss_dpl, loss_nll


def mse_loss_fn(y: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    return torch.mean((y - mean) ** 2)


# ============================================================
#  13. 前向传播公共逻辑（train / val / test 共用）
# ============================================================
def forward_all(nets, X_list, C_list, training_diffusion=False):
    """
    对 N_STEPS 个主干网络分别前向。

    返回
    ────
    training_diffusion=False → list of fused [B, 2×hidden]，长度 N_STEPS
    training_diffusion=True  → list of diffusion logit [B, 1]，长度 N_STEPS
    """
    return [
        nets[i](X_list[i], C_list[i], training_diffusion=training_diffusion)
        for i in range(N_STEPS)
    ]


def forward_global(nets, X_list, C_list):
    """
    n 个主干前向 + 全局 FeatureStream + 全局预测头。

    全局 FeatureStream 的输入 = 所有步长截尾对齐后的 x_seq 和 aux
    各自在步长维度做平均（mean pooling over N_STEPS），
    为 GlobalPredictionHead 提供与单一步长无关的全局辅助特征视角。

    返回
    ────
    means      : list of [min_len]，长度 N_STEPS
    log_sigmas : list of [min_len]，长度 N_STEPS（各步长独立）
    min_len    : int
    """
    # n 个主干输出 fused [N_i, 2×hidden]
    fused_list_raw = [
        nets[i](X_list[i], C_list[i], training_diffusion=False)
        for i in range(N_STEPS)
    ]

    # 截尾对齐
    fused_clipped = []
    x_clipped     = []
    c_clipped     = []
    for i in range(N_STEPS):
        cut = TAIL_CUTS[i]
        f   = fused_list_raw[i][:-cut] if cut > 0 else fused_list_raw[i]
        x   = X_list[i][:-cut]         if cut > 0 else X_list[i]
        c   = C_list[i][:-cut]         if cut > 0 else C_list[i]
        fused_clipped.append(f)
        x_clipped.append(x)
        c_clipped.append(c)

    min_len       = min(f.shape[0] for f in fused_clipped)
    fused_aligned = [f[:min_len] for f in fused_clipped]  # list of [min_len, 2×hidden]

    # 全局 FeatureStream 输入：N_STEPS 个步长的 x 和 c 在步长维度平均
    # x_clipped[i]: [N_i, 1, D_RET]，截到 min_len 后 stack → [N_STEPS, min_len, 1, D_RET]
    x_mean = torch.stack([x[:min_len] for x in x_clipped], dim=0).mean(dim=0)  # [min_len, 1, D_RET]
    c_mean = torch.stack([c[:min_len] for c in c_clipped], dim=0).mean(dim=0)  # [min_len, N_AUX]
    H_feat_glob = global_feature_stream(x_mean, c_mean)                         # [min_len, hidden]

    # 全局预测头：n 个 mean + n 个 log_sigma
    means, log_sigmas = global_head(fused_aligned, H_feat_glob)
    return means, log_sigmas, min_len


def align_outputs(raw_outs):
    """保留兼容接口（仅用于 diffusion 判别器训练，raw_outs 为 logit list）"""
    return raw_outs


def get_targets(Y_list, min_len):
    """对齐目标序列到 min_len。"""
    return [Y_list[i][:min_len] for i in range(N_STEPS)]


# ============================================================
#  14. 训练 & 测试函数
# ============================================================
def train_epoch(epoch: int):
    for net in nets: net.train()
    global_feature_stream.train()
    global_head.train()
    total_loss = total_mse = total_dpl = total_nll = 0.0
    total_loss_in = total_loss_out = 0.0

    for _ in range(ITER):
        # ── 主网络 + 全局头更新 ──────────────────────────────
        optimizer_F.zero_grad()
        means, log_sigmas, min_len = forward_global(nets, X_trains, C_trains)
        targets = get_targets(Y_trains, min_len)

        loss_parts = [
            combined_loss(targets[i], means[i], log_sigmas[i], log_nus[i])
            for i in range(N_STEPS)
        ]
        loss_f       = sum(p[0] for p in loss_parts)
        loss_mse     = sum(p[1] for p in loss_parts)
        loss_dpl     = sum(p[2] for p in loss_parts)
        loss_nll_val = sum(p[3] for p in loss_parts)

        loss_f.backward()
        nn.utils.clip_grad_norm_(
            [p for g in optimizer_F.param_groups for p in g['params']],
            max_norm=5.0,
        )
        optimizer_F.step()
        total_loss += loss_f.item()
        total_mse  += loss_mse.item()
        total_dpl  += loss_dpl.item()
        total_nll  += loss_nll_val.item()

        # ── 扩散判别器更新 ────────────────────────────────────
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
            [p for g in optimizer_G.param_groups for p in g['params']],
            max_norm=5.0,
        )
        optimizer_G.step()
        total_loss_in  += loss_in.item()
        total_loss_out += loss_out.item()

    # ── 训练指标统计 ─────────────────────────────────────────
    with torch.no_grad():
        means, _, min_len = forward_global(nets, X_trains, C_trains)
        targets = get_targets(Y_trains, min_len)
        mse_list = []
        da_list  = []
        nu_list  = []
        for i in range(N_STEPS):
            pred_np = means[i].cpu().numpy()
            if not np.isfinite(pred_np).all():
                raise RuntimeError(
                    f'Epoch {epoch}, step t+{i+1}: 预测值出现 NaN/Inf，训练已发散。'
                    f'建议降低 lr（当前={args.lr}）或减小 sigma。'
                )
            tgt_np = targets[i].cpu().numpy()
            mse_list.append(mean_squared_error(tgt_np, pred_np))
            da_list.append(np.mean(np.sign(pred_np) == np.sign(tgt_np)))
            nu_list.append((F.softplus(log_nus[i]) + 2.0).item())

    da_str = '  '.join(f't+{i+1}:{da_list[i]:.3f}'  for i in range(N_STEPS))
    nu_str = '  '.join(f't+{i+1}:{nu_list[i]:.2f}'  for i in range(N_STEPS))
    print(f'[Train] Epoch {epoch:3d} | '
          f'Loss:{total_loss/ITER:.4f}  '
          f'MSE:{total_mse/ITER:.4f}  '
          f'DPL:{total_dpl/ITER:.4f}  '
          f'NLL:{total_nll/ITER:.4f}')
    print(f'         DA:{da_str}')
    print(f'         nu:{nu_str}')
    return tuple(mse_list)


def test_epoch(epoch: int):
    for net in nets: net.eval()
    global_feature_stream.eval()
    global_head.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(ITER_TEST):
            means, log_sigmas, min_len = forward_global(nets, X_tests, C_tests)
            targets    = get_targets(Y_tests, min_len)
            loss       = sum(mse_loss_fn(targets[i], means[i]) for i in range(N_STEPS))
            total_loss += loss.item()

        mse_list = []
        da_list  = []
        for i in range(N_STEPS):
            pred_np = means[i].cpu().numpy()
            tgt_np  = targets[i].cpu().numpy()
            mse_list.append(mean_squared_error(tgt_np, pred_np))
            da_list.append(np.mean(np.sign(pred_np) == np.sign(tgt_np)))

    da_str = '  '.join(f't+{i+1}:{da_list[i]:.3f}' for i in range(N_STEPS))
    print(f'[ Test] Epoch {epoch:3d} | RMSE:{(total_loss/ITER_TEST)**0.5:.6f} | DA:{da_str}')
    return tuple(mse_list)


# ============================================================
#  15. 训练主循环
# ============================================================
train_losses = []
test_losses  = []

for epoch in range(1, args.epochs + 1):
    train_losses.append(train_epoch(epoch))
    test_losses.append(test_epoch(epoch))

    # 主网络学习率衰减
    if epoch in [int(e) for e in args.decreasing_lr]:
        for pg in optimizer_F.param_groups:
            pg['lr'] *= args.droprate

    # diffusion 学习率衰减
    if epoch in [int(e) for e in args.decreasing_lr2]:
        for pg in optimizer_G.param_groups:
            pg['lr'] *= args.droprate

# ── 最终结果打印 ─────────────────────────────────────────────
step_labels = [f't+{i+1}' for i in range(N_STEPS)]
print('\n===== 最终测试 MSE（标准化空间）=====')
for i, label in enumerate(step_labels):
    print(f'  {label}: {test_losses[-1][i]:.6f}')


# ============================================================
#  16. 验证集残差收集
# ============================================================
for net in nets: net.eval()

with torch.no_grad():
    val_means, val_log_sigmas, val_min_len = forward_global(nets, X_vals, C_vals)
    val_targets = get_targets(Y_vals, val_min_len)
    val_y_prevs = get_targets(Yp_vals, val_min_len)

# 残差在差分空间计算（差分空间更稳定，bootstrap 在差分空间操作）
val_residuals = [
    val_targets[i].cpu().numpy() - val_means[i].cpu().numpy()
    for i in range(N_STEPS)
]
print(f'\n[验证集残差] 各步长残差长度: {[len(r) for r in val_residuals]}')


# ============================================================
#  17. 测试集最终推理
# ============================================================
with torch.no_grad():
    final_means, final_log_sigmas, test_min_len = forward_global(nets, X_tests, C_tests)
    final_targets = get_targets(Y_tests, test_min_len)
    final_y_prevs = get_targets(Yp_tests, test_min_len)
    final_abs_tgt = get_targets(Y_abs_tests, test_min_len)

def to_np(t: torch.Tensor) -> np.ndarray:
    return t.cpu().numpy()

def to_sigma(log_sigma_tensor: torch.Tensor) -> np.ndarray:
    # sigma 经 softplus + sigma_min 约束，与 Student-t NLL 训练时一致
    return (F.softplus(log_sigma_tensor).clamp(min=args.sigma_min) + 1e-3).cpu().numpy()

def inv(arr: np.ndarray) -> np.ndarray:
    """标准化空间 → 原始价格空间"""
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

# ── 还原绝对价格 ─────────────────────────────────────────────
# 差分预测：mean_diff = 预测的 price(t+k) - price(t)（标准化空间）
# 还原：price_pred(t+k) = price(t) + mean_diff（标准化空间），再 inv() 到原始空间
# 真实值直接用 y_abs（绝对价格），保证评估指标与原版一致
test_true_price  = [inv(to_np(final_abs_tgt[i])) for i in range(N_STEPS)]
test_mean_price  = [inv(to_np(final_y_prevs[i]) + to_np(final_means[i]))
                    for i in range(N_STEPS)]

# sigma 在差分空间，还原到原始价格空间：σ_orig = σ_scaled × scaler.scale_[0]
test_sigma_price = [to_sigma(final_log_sigmas[i]) * scaler.scale_[0]
                    for i in range(N_STEPS)]

# 残差还原到原始空间
val_resid_price  = [val_residuals[i] * scaler.scale_[0]
                    for i in range(N_STEPS)]


# ============================================================
#  18. 保存实验数据（.npz，格式与原版完全一致）
#
#  每个步长保存一个文件：data_t+1.npz, data_t+2.npz, ...
#  各文件包含：
#    test_true  : 测试集真实价格          shape (n_test_days,)
#    test_mean  : 预测均值（原价格空间）  shape (n_test_days,)
#    test_sigma : 预测标准差（原价格空间）shape (n_test_days,)
#    val_resid  : 验证集残差（原价格空间）shape (n_val_days,)
# ============================================================
for i in range(N_STEPS):
    fname = f'ADE2_data_{step_labels[i]}.npz'
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
