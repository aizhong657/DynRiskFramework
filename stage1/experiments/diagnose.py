# ============================================================
#  诊断脚本：排查方向准确率卡在50%的根本原因
#
#  运行方式：python diagnose.py
#  依赖：与主程序相同，无需 GPU，直接 CPU 运行
#
#  检查项：
#    1. y_diff 统计：均值是否≈0，上涨/下跌比例是否均衡
#    2. y_prev 索引验证：x最后一列是否真的对应 t 时刻
#    3. 预测值分布：模型是否在输出接近0的值（规避惩罚）
#    4. Hinge 梯度验证：dpl_loss 对 mean_diff 是否有非零梯度
#    5. 数据泄露检查：x 特征与 y_diff 的相关性
# ============================================================
from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

# ── 复用主程序的参数 ─────────────────────────────────────────
DATA_PATH   = DATA_DIR / "sz50_index_data.csv"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
D, TAU      = 22, 1
N_STEPS     = 4
_diff_std_placeholder = None   # 在数据加载后赋值

# ============================================================
#  数据加载（与主程序完全一致）
# ============================================================
data = pd.read_csv(DATA_PATH)
data = data[['date', 'close']]
data['date'] = pd.to_datetime(data['date'])
close = data['close'].values.reshape(-1, 1)

n_total = len(close)
n_train = int(np.ceil(TRAIN_RATIO * n_total))
n_val   = int(np.ceil(VAL_RATIO   * n_total))

scaler = StandardScaler()
train_scaled = scaler.fit_transform(close[:n_train]).flatten()
val_scaled   = scaler.transform(close[n_train:n_train+n_val]).flatten()
test_scaled  = scaler.transform(close[n_train+n_val:]).flatten()

def make_df(dates, values):
    return pd.DataFrame({'Date': dates, 'closescale': values})

train_df = make_df(data['date'].values[:n_train], train_scaled)

def PhaSpaRecon(df, tau, d, T):
    values = np.array(df)[:, 1].astype(float)
    n      = len(values)
    width  = n - (d - 1) * tau - 1
    Xn1    = np.stack([values[i*tau : i*tau + width] for i in range(d)], axis=1)
    Yn1    = values[T + (d-1)*tau : T + (d-1)*tau + width]
    Xn     = pd.DataFrame(Xn1)
    Yn     = pd.DataFrame(Yn1, columns=[0])
    X      = pd.concat([Xn, Yn], axis=1)
    return Xn, Yn, None, X

def build_xy(df, tau, d, T, drop_tail=0):
    _, _, _, X = PhaSpaRecon(df, tau=tau, d=d, T=T)
    arr = X.values
    if drop_tail > 0:
        arr = arr[:-drop_tail]
    return arr[:, :d].astype(np.float64), arr[:, d].astype(np.float64)

SEP = '=' * 60

# ============================================================
#  检查1：y_prev 索引验证
#
#  PhaSpaRecon 构造规则：
#    X[:,  0] = values[0      : width]         → 对应时刻 t = 0..width-1
#    X[:,  1] = values[tau    : tau+width]
#    X[:, -1] = values[(d-1)*tau : (d-1)*tau+width]  → 对应时刻 t = (d-1)*tau
#    Y        = values[T+(d-1)*tau : T+(d-1)*tau+width]  → 对应时刻 t+T
#
#  正确的 y_prev（预测基准点）应该是 Y 对应的 t 时刻，即 X[:, -1]
#  但 X[:, -1] 对应的是 values[(d-1)*tau]，而不是 values[T+(d-1)*tau-T] = values[(d-1)*tau]
#  ——等等，实际上是同一个时刻吗？需要逐行验证。
# ============================================================
print(f'\n{SEP}')
print('检查1：y_prev 索引验证（x最后一列是否对应预测基准时刻）')
print(SEP)

raw_values = train_scaled  # 原始序列

for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr, ytr_abs = build_xy(train_df, TAU, D, T, drop_tail=drop)

    # X[:, -1] 来自 values[(d-1)*tau : (d-1)*tau + width][0:len-drop]
    # Y 来自 values[T+(d-1)*tau : ...][0:len-drop]
    # 两者的时间偏移差 = T
    # 所以 x_last[i] = values[(d-1)*tau + i]
    #      y_abs[i]  = values[T + (d-1)*tau + i]
    # y_diff = y_abs - x_last = values[T+(d-1)*tau+i] - values[(d-1)*tau+i]
    #        = price(t+T) - price(t)   ← 这才是正确的 T 步差分！

    x_last   = xtr[:, -1]             # 第 d-1 列
    y_diff   = ytr_abs - x_last

    # 用原始序列手工验证前5行
    start = (D - 1) * TAU             # = 22
    errors = []
    for i in range(min(5, len(y_diff))):
        expected_ydiff = raw_values[start + T + i] - raw_values[start + i]
        actual_ydiff   = y_diff[i]
        errors.append(abs(expected_ydiff - actual_ydiff))

    max_err = max(errors)
    status  = 'OK' if max_err < 1e-10 else f'ERROR max_err={max_err:.2e}'
    print(f'  T={T}: y_diff = price(t+{T}) - price(t)  验证: {status}')
    print(f'         y_diff统计: mean={y_diff.mean():.5f}  std={y_diff.std():.5f}  '
          f'上涨比例={np.mean(y_diff > 0):.3f}')

# ============================================================
#  检查2：y_diff 统计 —— 均值是否≈0，上涨/下跌是否均衡
# ============================================================
print(f'\n{SEP}')
print('检查2：y_diff 分布统计')
print(SEP)

for T in range(1, N_STEPS + 1):
    drop = T - 1
    xtr, ytr_abs = build_xy(train_df, TAU, D, T, drop_tail=drop)
    y_diff = ytr_abs - xtr[:, -1]

    pos  = np.mean(y_diff > 0)
    neg  = np.mean(y_diff < 0)
    zero = np.mean(y_diff == 0)
    print(f'  T={T}: 上涨={pos:.3f}  下跌={neg:.3f}  零={zero:.4f}  '
          f'mean={y_diff.mean():.5f}  std={y_diff.std():.5f}')
    if abs(y_diff.mean()) > 0.1 * y_diff.std():
        print(f'    ⚠ 均值偏离0较大，naive预测"总是输出正数"可以获得{pos:.1%}的DA')

# ============================================================
#  检查3：x 特征与 y_diff 的皮尔逊相关系数
#
#  如果所有列与 y_diff 相关性都很低（< 0.05），
#  说明模型从 x 里学不到方向信号，需要添加收益率特征到 x
# ============================================================
print(f'\n{SEP}')
print('检查3：x特征列与y_diff的皮尔逊相关系数')
print(SEP)

xtr, ytr_abs = build_xy(train_df, TAU, D, 1, drop_tail=0)
y_diff = ytr_abs - xtr[:, -1]

corrs = []
for col in range(D):
    c = np.corrcoef(xtr[:, col], y_diff)[0, 1]
    corrs.append(c)

print(f'  x[:,0]（最旧时刻）相关系数: {corrs[0]:.4f}')
print(f'  x[:,-1]（最新时刻）相关系数: {corrs[-1]:.4f}')
print(f'  所有列绝对相关系数: max={max(abs(c) for c in corrs):.4f}  '
      f'mean={np.mean([abs(c) for c in corrs]):.4f}')

# 价格差分列与y_diff的相关性（x的一阶差分 ≈ 收益率信号）
x_ret = np.diff(xtr, axis=1)   # [N, D-1]，逐列差分 ≈ 相邻时刻收益率
corrs_ret = [np.corrcoef(x_ret[:, col], y_diff)[0, 1] for col in range(D-1)]
print(f'  x相邻差分列与y_diff: max={max(abs(c) for c in corrs_ret):.4f}  '
      f'mean={np.mean([abs(c) for c in corrs_ret]):.4f}')
print()
if max(abs(c) for c in corrs) < 0.05:
    print('  ⚠ x特征与y_diff相关性极低，模型从价格水平里无法提取方向信号')
    print('    建议：在x里加入收益率序列（ret特征）作为输入')
else:
    print('  x特征与y_diff存在一定相关性，信号可用')

# ============================================================
#  检查4：Hinge梯度验证 —— dpl_loss对mean_diff是否有非零梯度
# ============================================================
print(f'\n{SEP}')
print('检查4：Hinge DPL梯度验证')
print(SEP)

xtr, ytr_abs = build_xy(train_df, TAU, D, 1, drop_tail=0)
y_diff_np = ytr_abs - xtr[:, -1]
_diff_std  = float(np.std(y_diff_np))

def dpl_loss_test(y_diff, mean_diff):
    margin  = _diff_std
    product = y_diff * mean_diff
    hinge   = torch.clamp(margin - product, min=0.0)
    return hinge.mean()

# 模拟3种预测场景
scenarios = [
    ('全部预测为0（规避惩罚）',   torch.zeros(100)),
    ('随机预测（DA≈50%）',        torch.randn(100) * _diff_std * 0.1),
    ('正确方向但幅度不足',         torch.tensor(y_diff_np[:100]).float() * 0.01),
]

y_true = torch.tensor(y_diff_np[:100]).float()

for name, mean_pred in scenarios:
    mean_pred = mean_pred.requires_grad_(True)
    loss = dpl_loss_test(y_true, mean_pred)
    loss.backward()
    grad = mean_pred.grad
    grad_norm   = grad.norm().item()
    grad_nonzero = (grad.abs() > 1e-8).float().mean().item()
    da = (torch.sign(mean_pred.detach()) == torch.sign(y_true)).float().mean().item()
    print(f'  [{name}]')
    print(f'    DPL={loss.item():.4f}  梯度L2={grad_norm:.4f}  '
          f'非零梯度比例={grad_nonzero:.3f}  当前DA={da:.3f}')

# ============================================================
#  检查5：输出接近0的问题 —— 模型是否在用"输出趋近0"绕过惩罚
#
#  如果 mean_diff ≈ 0（绝对值很小），则：
#    product = y_diff * mean_diff ≈ 0
#    hinge = clamp(margin - 0, 0) = margin  ← 每个样本都有惩罚！
#  这说明输出趋近0并不能绕过惩罚，反而是最大惩罚状态。
#  模型应该会主动让 mean_diff 与 y_diff 同号以降低惩罚。
# ============================================================
print(f'\n{SEP}')
print('检查5：输出趋近0能否规避Hinge惩罚')
print(SEP)

y_true_small = torch.tensor(y_diff_np[:100]).float()
for scale in [0.001, 0.01, 0.1, 1.0]:
    mean_zero_like = torch.zeros(100)
    mean_small     = y_true_small * scale   # 正确方向但幅度小
    loss_zero  = dpl_loss_test(y_true_small, mean_zero_like).item()
    loss_small = dpl_loss_test(y_true_small, mean_small).item()
    print(f'  mean_diff≈0时DPL={loss_zero:.4f}  '
          f'mean_diff=y_diff×{scale}时DPL={loss_small:.4f}  '
          f'→ {"输出0更优（有问题）" if loss_zero < loss_small else "正确方向更优（正常）"}')

print(f'\n{SEP}')
print('诊断完成。请根据以上结果定位问题所在。')
print(SEP)
