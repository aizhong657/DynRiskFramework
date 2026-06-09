# ============================================================
#  基准模型 B：LSTM 多步预测
#
#  数据划分、StandardScaler 与 chapter517.py 完全一致。
#  结构：单层 LSTM → 全连接，直接输出 N_STEPS 步预测（seq2scalar）。
#  输出：lstm_t+1.npz ~ lstm_t+{N_STEPS}.npz
#        每个文件含 test_true / test_mean（价格原始空间）
# ============================================================

from config import DATA_DIR, OUTPUT_DIR, CORR_PATH
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
#  USER CONFIG —— 只需修改这里
# ============================================================
DATA_PATH     = DATA_DIR / "sz50_index_data.csv"
N_STEPS       = 4     # ← 预测步长，与 chapter517.py 保持一致
SEQ_LEN       = 22    # 输入窗口长度（与 chapter517.py 的嵌入维数 D 保持一致）
HIDDEN_SIZE   = 64
NUM_LAYERS    = 1
EPOCHS        = 100
BATCH_SIZE    = 64
LR            = 1e-3
SEED          = 42
OUTPUT_PREFIX = 'lstm'

# ============================================================
#  1. 数据加载与预处理（与 chapter517.py 完全一致）
# ============================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {device}')

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
#  2. 构建滑窗数据集
#
#  X[i] = scaled[i : i+SEQ_LEN]         形状 (SEQ_LEN,)
#  Y[i] = scaled[i+SEQ_LEN : i+SEQ_LEN+N_STEPS]  形状 (N_STEPS,)
# ============================================================
def make_dataset(series, seq_len, n_steps):
    X, Y = [], []
    for i in range(len(series) - seq_len - n_steps + 1):
        X.append(series[i : i + seq_len])
        Y.append(series[i + seq_len : i + seq_len + n_steps])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# 训练集：仅用 train_scaled
X_tr, Y_tr = make_dataset(train_scaled, SEQ_LEN, N_STEPS)
# 验证集：用 train 末尾 + val（保证滑窗不越界）
full_tv    = np.concatenate([train_scaled, val_scaled])
X_va_full, Y_va_full = make_dataset(full_tv, SEQ_LEN, N_STEPS)
X_va = X_va_full[len(X_tr):]   # 只取验证区间部分
Y_va = Y_va_full[len(Y_tr):]

print(f'[Dataset] X_train:{X_tr.shape}  X_val:{X_va.shape}')

def to_loader(X, Y, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(-1),   # (N, SEQ_LEN, 1)
                       torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = to_loader(X_tr, Y_tr, shuffle=True)
val_loader   = to_loader(X_va, Y_va, shuffle=False)

# ============================================================
#  3. 模型定义
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, out_steps=N_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc   = nn.Linear(hidden_size, out_steps)

    def forward(self, x):
        out, _ = self.lstm(x)          # (N, SEQ_LEN, hidden)
        return self.fc(out[:, -1, :])  # (N, N_STEPS)

model = LSTMModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
print(f'[Model] 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

# ============================================================
#  4. 训练
# ============================================================
best_val_loss = float('inf')
best_state    = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tr_loss += loss.item() * len(xb)
    tr_loss /= len(X_tr)

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            va_loss += criterion(model(xb), yb).item() * len(xb)
    va_loss /= max(len(X_va), 1)

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(f'Epoch {epoch:3d}/{EPOCHS}  train_loss:{tr_loss:.6f}  val_loss:{va_loss:.6f}')

model.load_state_dict(best_state)
print(f'[训练完成] best_val_loss={best_val_loss:.6f}')

# ============================================================
#  5. 测试集推理
#
#  测试窗口：以 [train + val + test[:t]] 末尾 SEQ_LEN 点为输入，
#  预测 t+1 ~ t+N_STEPS。有效预测数 = n_test - N_STEPS + 1。
# ============================================================
model.eval()
full_history = np.concatenate([train_scaled, val_scaled, test_scaled])
n_valid      = n_test - N_STEPS + 1   # 有效预测样本数
offset       = n_train + n_val         # 测试集起始索引（在 full_history 中）

preds_scaled = np.zeros((n_valid, N_STEPS), dtype=np.float32)

with torch.no_grad():
    for t in range(n_valid):
        idx   = offset + t
        xb    = full_history[idx - SEQ_LEN : idx].astype(np.float32)
        xb    = torch.from_numpy(xb).unsqueeze(0).unsqueeze(-1).to(device)  # (1,SEQ_LEN,1)
        preds_scaled[t] = model(xb).cpu().numpy()

# ============================================================
#  6. 反标准化并保存（每步单独存一个 .npz）
# ============================================================
def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

for s in range(N_STEPS):
    true_scaled = test_scaled[s : s + n_valid]
    mean_price  = inv(preds_scaled[:, s])
    true_price  = inv(true_scaled)

    min_len    = min(len(true_price), len(mean_price))
    true_price = true_price[:min_len]
    mean_price = mean_price[:min_len]

    fname = f'{OUTPUT_PREFIX}_t+{s+1}.npz'
    np.savez(fname,
             test_true  = true_price,
             test_mean  = mean_price,
             test_sigma = np.zeros_like(mean_price),
             val_resid  = np.zeros(len(val_scaled)))
    print(f'[保存] {fname}  长度:{min_len}')

print('\n[LSTM] 全部完成。')
