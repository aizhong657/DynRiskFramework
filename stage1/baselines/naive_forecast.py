# ============================================================
#  基准模型：Naive（历史均值预测）
#
#  预测策略：对每个测试时间点 t，用 [train + val + test[:t]]
#            的历史均值作为 t+1 ~ t+N_STEPS 所有步长的预测值。
#  这是最简单的基准，用于衡量其他模型是否有实质提升。
#
#  数据划分、StandardScaler 与 chapter517.py 完全一致。
#  输出：naive_t+1.npz ~ naive_t+{N_STEPS}.npz
#        每个文件含 test_true / test_mean（价格原始空间）
# ============================================================

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
#  USER CONFIG —— 只需修改这里
# ============================================================
DATA_PATH     = DATA_DIR / "sz50_index_data.csv"
N_STEPS       = 4       # ← 预测步长，与 chapter517.py 保持一致
OUTPUT_PREFIX = 'naive'

# ============================================================
#  1. 数据加载与预处理（与 chapter517.py 完全一致）
# ============================================================
data = pd.read_csv(DATA_PATH)
data = data[['date', 'close']]
data['date'] = pd.to_datetime(data['date'])

close = data['close'].values.reshape(-1, 1)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
n_total = len(close)
n_train = int(np.ceil(TRAIN_RATIO * n_total))
n_val   = int(np.ceil(VAL_RATIO   * n_total))
n_test  = n_total - n_train - n_val

scaler = StandardScaler()
train_scaled = scaler.fit_transform(close[:n_train]).flatten()
val_scaled   = scaler.transform(close[n_train:n_train+n_val]).flatten()
test_scaled  = scaler.transform(close[n_train+n_val:]).flatten()

print(f'[Data] 训练:{len(train_scaled)}  验证:{len(val_scaled)}  测试:{len(test_scaled)}')

# ============================================================
#  2. Naive 滚动均值预测
#
#  t 时刻的预测 = mean(train + val + test[:t])
#  对 t+1 ~ t+N_STEPS 步预测值相同（均值不随步长变化）
# ============================================================
n_valid = n_test - N_STEPS + 1   # 有效预测数，与其他模型对齐

history_base = np.concatenate([train_scaled, val_scaled])

preds_scaled = np.zeros((n_valid, N_STEPS), dtype=np.float64)

for t in range(n_valid):
    history    = np.concatenate([history_base, test_scaled[:t]])
    hist_mean  = float(np.mean(history))
    # 均值预测：所有步长相同
    preds_scaled[t, :] = hist_mean

# ============================================================
#  3. 反标准化并保存
# ============================================================
def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

for s in range(N_STEPS):
    true_scaled = test_scaled[s : s + n_valid]
    mean_price  = inv(preds_scaled[:, s])
    true_price  = inv(true_scaled)
    min_len     = min(len(true_price), len(mean_price))

    fname = f'{OUTPUT_PREFIX}_t+{s+1}.npz'
    np.savez(fname,
             test_true  = true_price[:min_len],
             test_mean  = mean_price[:min_len],
             test_sigma = np.zeros(min_len),
             val_resid  = np.zeros(len(val_scaled)))
    print(f'[保存] {fname}  长度:{min_len}')

print('\n[Naive] 全部完成。')