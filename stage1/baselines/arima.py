# ============================================================
#  基准模型 A：ARIMA 多步预测
#
#  数据划分、StandardScaler 与 chapter517.py 完全一致。
#  预测策略：对每个测试样本，用训练+验证历史拟合 ARIMA，
#            一次性预测 t+1 ~ t+N_STEPS 步（直接多步预测）。
#  输出：arima_t+1.npz ~ arima_t+{N_STEPS}.npz
#        每个文件含 test_true / test_mean（价格原始空间）
# ============================================================

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

# ============================================================
#  USER CONFIG —— 只需修改这里
# ============================================================
DATA_PATH   = DATA_DIR / "sz50_index_data.csv"
N_STEPS     = 4          # ← 预测步长，与 chapter517.py 保持一致
ARIMA_ORDER = (23, 1, 0)  # (p, d, q)，可按需调整
OUTPUT_PREFIX = 'arima'  # 输出文件前缀

# ============================================================
#  1. 数据加载与预处理（与 chapter517.py 完全一致）
# ============================================================
data = pd.read_csv(DATA_PATH)
data = data[['date', 'code', 'open', 'close', 'high', 'low', 'volume']]
data['date'] = pd.to_datetime(data['date'])

stk_data = pd.DataFrame({
    'Date':  data['date'],
    'Close': data['close'].values,
})

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
n_total = len(stk_data)
n_train = int(np.ceil(TRAIN_RATIO * n_total))
n_val   = int(np.ceil(VAL_RATIO   * n_total))
n_test  = n_total - n_train - n_val

training_set   = stk_data['Close'].values[:n_train].reshape(-1, 1)
validation_set = stk_data['Close'].values[n_train:n_train+n_val].reshape(-1, 1)
testing_set    = stk_data['Close'].values[n_train+n_val:].reshape(-1, 1)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(training_set).flatten()
val_scaled   = scaler.transform(validation_set).flatten()
test_scaled  = scaler.transform(testing_set).flatten()

print(f'[Data] 训练:{len(train_scaled)}  验证:{len(val_scaled)}  测试:{len(test_scaled)}')

# ============================================================
#  2. ARIMA 滚动预测
#
#  对每个测试时间点 t，用 [train + val + test[:t]] 作为历史，
#  一次 forecast N_STEPS 步，得到 t+1 ~ t+N_STEPS 的预测值。
# ============================================================
history_base = np.concatenate([train_scaled, val_scaled])  # 训练+验证作为初始历史

# 存储每步的预测序列，长度对齐到 n_test - N_STEPS + 1
n_valid = n_test - N_STEPS + 1   # 可对齐的有效预测数量

# preds[s] = 第 s+1 步预测序列（标准化空间），长度 n_valid
preds_scaled = [[] for _ in range(N_STEPS)]

print(f'[ARIMA] 开始滚动预测，共 {n_valid} 个测试窗口 ...')
for t in tqdm(range(n_valid)):
    history = np.concatenate([history_base, test_scaled[:t]])
    try:
        model = ARIMA(history, order=ARIMA_ORDER)
        result = model.fit()
        fc = result.forecast(steps=N_STEPS)  # 长度 N_STEPS
    except Exception as e:
        print(f'  警告: t={t} 拟合失败({e})，用 NaN 填充')
        fc = np.full(N_STEPS, np.nan)
    for s in range(N_STEPS):
        preds_scaled[s].append(float(fc[s]))

preds_scaled = [np.array(p) for p in preds_scaled]

# ============================================================
#  3. 反标准化至价格原始空间
# ============================================================
def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

# 真实价格：测试集第 s+1 步对应 test_scaled[s : s+n_valid]
# 即对步长 s（0-indexed），真实值是 test_scaled[s], test_scaled[s+1], ...
for s in range(N_STEPS):
    true_scaled = test_scaled[s : s + n_valid]          # 长度 n_valid
    mean_price  = inv(preds_scaled[s])
    true_price  = inv(true_scaled)

    # 对齐：取最短（理论上相等，保险起见）
    min_len = min(len(true_price), len(mean_price))
    true_price = true_price[:min_len]
    mean_price = mean_price[:min_len]

    fname = f'{OUTPUT_PREFIX}_t+{s+1}.npz'
    np.savez(fname,
             test_true  = true_price,
             test_mean  = mean_price,
             test_sigma = np.zeros_like(mean_price),  # ARIMA 无不确定性估计，填0
             val_resid  = np.zeros(len(val_scaled)))   # 占位，对比报告不使用
    print(f'[保存] {fname}  长度:{min_len}')

print('\n[ARIMA] 全部完成。')