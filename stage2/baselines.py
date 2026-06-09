
"""
第四章 风险度量对比基准模型
======================================================================
实现知识库第 9 节滚动回测对比所需的 5 个基准：

  1. NormalParametric   正态参数法     VaR = μ + σ·z_α
  2. HistoricalSim      历史模拟法(HS) VaR = -分位数(returns, 1-α)
  3. EVTPeaksOverThresh EVT/POT(GPD)   超阈值 GPD 极值外推
  4. LSTMVaR            LSTM 分位回归   Pinball Loss 直接预测 VaR
  5. TCNVaR             TCN 分位回归    因果膨胀卷积 + Pinball Loss

统一接口
--------
每个基准实现 rolling_backtest(returns, ...) -> (all_ret, all_var, all_es)，
窗口设置（train=800 / val=100 / step=50）与 ETGPD-Transformer 完全一致，
从而保证 BacktestSuite 的 Kupiec/Christoffersen/DQ/ESR 检验可直接复用。

注意
----
* 统计型基准（1–3）只需收益率序列，不使用 33 维特征，符合论文设定。
* 深度型基准（4–5）使用收益率滞后窗口作为输入（不接 33 维动力学特征，
  这正是对比的意义：证明动力学特征 + GPD 后验校正的增量价值）。
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

_Z = {0.95: 1.6448536269514722, 0.99: 2.326347874040841}
_SQRT2PI = math.sqrt(2 * math.pi)


def _z_alpha(conf: float) -> float:
    from scipy.stats import norm
    return _Z.get(conf, float(norm.ppf(conf)))


# ══════════════════════════════════════════════════════════════════
# 1. 正态参数法
# ══════════════════════════════════════════════════════════════════
def normal_parametric(train_ret: np.ndarray, conf: float) -> Tuple[float, float]:
    """返回 (VaR, ES)，损失 = -收益率。"""
    mu  = float(np.mean(-train_ret))
    sd  = float(np.std(-train_ret))
    z   = _z_alpha(conf)
    var = mu + sd * z
    es  = mu + sd * math.exp(-0.5 * z * z) / (_SQRT2PI * (1 - conf))
    return var, es


# ══════════════════════════════════════════════════════════════════
# 2. 历史模拟法 (HS)
# ══════════════════════════════════════════════════════════════════
def historical_sim(train_ret: np.ndarray, conf: float) -> Tuple[float, float]:
    losses = -train_ret
    var = float(np.quantile(losses, conf))
    tail = losses[losses >= var]
    es  = float(tail.mean()) if len(tail) else var
    return var, es


# ══════════════════════════════════════════════════════════════════
# 3. EVT / Peaks-Over-Threshold (GPD)
# ══════════════════════════════════════════════════════════════════
def evt_pot(train_ret: np.ndarray, conf: float,
            u_quantile: float = 0.90) -> Tuple[float, float]:
    from scipy.stats import genpareto
    losses = -train_ret
    u = float(np.quantile(losses, u_quantile))
    exc = losses[losses > u] - u
    if len(exc) < 10:
        return historical_sim(train_ret, conf)
    try:
        xi, _, beta = genpareto.fit(exc, floc=0)
        xi = float(np.clip(xi, -0.1, 0.5))
        beta = float(max(beta, 1e-6))
    except Exception:
        return historical_sim(train_ret, conf)
    nu = len(exc) / len(losses)          # 超阈比例
    # 标准 POT VaR 公式
    if abs(xi) > 1e-6:
        var = u + (beta / xi) * (((1 - conf) / nu) ** (-xi) - 1)
    else:
        var = u + beta * (-math.log((1 - conf) / nu))
    # GPD 解析 ES：ES = (VaR + β - ξu)/(1-ξ)
    es = (var + beta - xi * u) / max(1 - xi, 0.1)
    return float(var), float(es)


# ══════════════════════════════════════════════════════════════════
# 深度型基准的公共部件
# ══════════════════════════════════════════════════════════════════
class _SeqDataset(Dataset):
    """用收益率滞后窗口预测下一日损失。"""
    def __init__(self, returns: np.ndarray, seq_len: int = 60):
        self.r = returns.astype(np.float32)
        self.L = seq_len

    def __len__(self):
        return max(0, len(self.r) - self.L)

    def __getitem__(self, i):
        x = self.r[i: i + self.L]                  # [L]
        y = self.r[i + self.L]                     # 下一日收益率
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor(y)


def _pinball(pred_var: torch.Tensor, y: torch.Tensor,
             conf: float) -> torch.Tensor:
    """对损失序列 (=-y) 的 conf 分位 Pinball Loss。"""
    loss = -y
    err  = loss - pred_var
    return torch.mean(torch.maximum(conf * err, (conf - 1) * err))


class _LSTMNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d):
        super().__init__()
        pad = (k - 1) * d
        self.conv = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=d)
        self.pad = pad
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        if self.pad:
            y = y[:, :, :-self.pad]            # 因果裁剪
        return self.relu(y)


class _TCNNet(nn.Module):
    def __init__(self, ch: int = 32, k: int = 3):
        super().__init__()
        self.b1 = _TCNBlock(1,  ch, k, 1)
        self.b2 = _TCNBlock(ch, ch, k, 2)
        self.b3 = _TCNBlock(ch, ch, k, 4)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):                      # x: [B, L, 1]
        h = x.transpose(1, 2)                  # [B, 1, L]
        h = self.b3(self.b2(self.b1(h)))       # [B, ch, L]
        return self.head(h[:, :, -1]).squeeze(-1)


def _train_deep_var(net: nn.Module, train_ret: np.ndarray,
                    conf: float, seq_len: int = 60,
                    epochs: int = 40, lr: float = 1e-3) -> nn.Module:
    ds = _SeqDataset(train_ret, seq_len)
    if len(ds) < 16:
        return net
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            var = net(x)
            loss = _pinball(var, y, conf)
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
    return net


def _deep_predict(net: nn.Module, hist_ret: np.ndarray,
                  seq_len: int, conf: float) -> Tuple[float, float]:
    """用最近 seq_len 日预测下一日 VaR；ES 用历史超额均值近似。"""
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(
            hist_ret[-seq_len:].astype(np.float32)
        ).reshape(1, seq_len, 1)
        var = float(net(x).item())
    # ES 近似：训练尾部损失均值与 VaR 的比例外推
    losses = -hist_ret
    tail = losses[losses >= np.quantile(losses, conf)]
    ratio = (tail.mean() / np.quantile(losses, conf)) if len(tail) else 1.15
    return var, var * float(max(ratio, 1.0))


# ══════════════════════════════════════════════════════════════════
# 统一滚动回测驱动
# ══════════════════════════════════════════════════════════════════
def rolling_backtest_baseline(
    method: str,
    returns: np.ndarray,
    train_size: int = 800,
    val_size: int = 100,
    test_step: int = 50,
    seq_len: int = 60,
    conf: float = 0.99,
    epochs: int = 40,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    method ∈ {"normal", "hs", "evt", "lstm", "tcn"}

    返回 (all_ret, all_var, all_es)，长度、对齐方式与
    trainer.rolling_backtest 完全一致，可直接喂给 BacktestSuite。
    """
    method = method.lower()
    N = len(returns)
    all_ret, all_var, all_es = [], [], []
    pointer = train_size
    fold = 0

    while pointer + test_step <= N:
        fold += 1
        train_ret = returns[:pointer]
        test_ret  = returns[pointer: pointer + test_step]

        if method in ("normal", "hs", "evt"):
            # 静态型：每折用训练集估一组参数，应用于整个 test_step
            if method == "normal":
                var, es = normal_parametric(train_ret, conf)
            elif method == "hs":
                var, es = historical_sim(train_ret, conf)
            else:
                var, es = evt_pot(train_ret, conf)
            var_seq = np.full(test_step, var)
            es_seq  = np.full(test_step, es)

        else:
            # 深度型：训练后逐日滚动预测
            net = _LSTMNet() if method == "lstm" else _TCNNet()
            net = _train_deep_var(net, train_ret, conf,
                                  seq_len=seq_len, epochs=epochs)
            var_seq, es_seq = [], []
            for t in range(test_step):
                hist = returns[: pointer + t]
                if len(hist) < seq_len:
                    v, e = historical_sim(train_ret, conf)
                else:
                    v, e = _deep_predict(net, hist, seq_len, conf)
                var_seq.append(v)
                es_seq.append(e)
            var_seq = np.array(var_seq)
            es_seq  = np.array(es_seq)

        all_ret.extend(test_ret.tolist())
        all_var.extend(var_seq.tolist())
        all_es.extend(es_seq.tolist())

        if verbose:
            viol = float(np.mean(-test_ret > var_seq))
            print(f"  [{method}] Fold {fold}: "
                  f"VaR≈{np.mean(var_seq):.4f}  违例={viol:.1%}")

        pointer += test_step

    return np.array(all_ret), np.array(all_var), np.array(all_es)


# 便捷别名
ALL_BASELINES = ["normal", "hs", "evt", "lstm", "tcn"]
BASELINE_LABELS = {
    "normal": "正态参数法",
    "hs":     "历史模拟法(HS)",
    "evt":    "EVT",
    "lstm":   "LSTM",
    "tcn":    "TCN",
}

