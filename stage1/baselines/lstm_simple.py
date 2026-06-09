# ============================================================
#  基准模型：Simple LSTM（简单版，区别于 baseline_lstm.py）
#
#  与 baseline_lstm.py 的区别：
#    - 输入：原始收盘价滑窗（不使用相空间重构特征）
#    - 窗口长度：SEQ_LEN = 20（常规回望窗口，非 PSR 嵌入维数）
#    - 结构：双层 LSTM（更贴近文献中"简单 LSTM"基准的常见设置）
#    - 无 Levy 噪声、无 SDE、无关联维数辅助特征
#
#  数据划分、StandardScaler 与 chapter517.py 完全一致。
#  输出：simple_lstm_t+1.npz ~ simple_lstm_t+{N_STEPS}.npz
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
SEQ_LEN       = 20    # 回望窗口（简单 LSTM 常规设置，区别于 PSR 嵌入维数）
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2     # 双层 LSTM
DROPOUT       = 0.2
EPOCHS        = 100
BATCH_SIZE    = 64
LR            = 1e-3
SEED          = 42
OUTPUT_PREFIX = 'simple_lstm'

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
# ============================================================
def make_dataset(series, seq_len, n_steps):
    X, Y = [], []
    for i in range(len(series) - seq_len - n_steps + 1):
        X.append(series[i : i + seq_len])
        Y.append(series[i + seq_len : i + seq_len + n_steps])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X_tr, Y_tr = make_dataset(train_scaled, SEQ_LEN, N_STEPS)
full_tv     = np.concatenate([train_scaled, val_scaled])
X_va_full, Y_va_full = make_dataset(full_tv, SEQ_LEN, N_STEPS)
X_va = X_va_full[len(X_tr):]
Y_va = Y_va_full[len(Y_tr):]

print(f'[Dataset] X_train:{X_tr.shape}  X_val:{X_va.shape}')

def to_loader(X, Y, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(-1),
                       torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = to_loader(X_tr, Y_tr, shuffle=True)
val_loader   = to_loader(X_va, Y_va, shuffle=False)

# ============================================================
#  3. 模型定义（双层 LSTM）
# ============================================================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, out_steps=N_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, out_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

model     = SimpleLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
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
    scheduler.step()

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
        print(f'Epoch {epoch:3d}/{EPOCHS}  train:{tr_loss:.6f}  val:{va_loss:.6f}  '
              f'lr:{scheduler.get_last_lr()[0]:.2e}')

model.load_state_dict(best_state)
print(f'[训练完成] best_val_loss={best_val_loss:.6f}')

# ============================================================
#  5. 测试集推理
# ============================================================
model.eval()
full_history = np.concatenate([train_scaled, val_scaled, test_scaled])
n_valid      = n_test - N_STEPS + 1
offset       = n_train + n_val

preds_scaled = np.zeros((n_valid, N_STEPS), dtype=np.float32)

with torch.no_grad():
    for t in range(n_valid):
        idx = offset + t
        xb  = full_history[idx - SEQ_LEN : idx].astype(np.float32)
        xb  = torch.from_numpy(xb).unsqueeze(0).unsqueeze(-1).to(device)
        preds_scaled[t] = model(xb).cpu().numpy()

# ============================================================
#  6. 反标准化并保存
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

print('\n[Simple LSTM] 全部完成。')