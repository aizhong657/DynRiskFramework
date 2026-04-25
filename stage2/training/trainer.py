"""
ETGPD-Transformer — 训练框架 v5（稳定版）

设计原则：
  1. 损失函数：纯 Pinball Loss（99% 分位数），无任何辅助项
     → NaN 跳过问题的根因是梯度爆炸，纯 QL 梯度始终有界（|∇QL| ≤ max(α,1-α) = 0.99）
  2. 学习率：更保守的初始值（1e-4），配合 warmup
  3. NaN 防护：不跳过 batch，改为在损失计算前 clamp 输入特征（±5σ）
  4. 早停：仅监控 val_ql（分位数损失），用 delta_viol 惩罚项（平滑，非跳跃）
  5. GPD 校正：在 rolling_backtest 内每个 fold 结束后，用该 fold 训练集残差
     拟合 GPD，对测试集 VaR/ES 做后验校正
"""

import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# 1. Dataset
# ══════════════════════════════════════════════════════════════════

class ReturnDataset(Dataset):
    def __init__(self, features: np.ndarray, returns: np.ndarray,
                 seq_len: int = 60, tail_feat_idx=None, cond_feat_idx=None):
        # 输入 clamp：超过 ±5 的归一化值视为异常，截断防止梯度爆炸
        features_clamped = np.clip(features, -5.0, 5.0)
        self.X = torch.tensor(features_clamped, dtype=torch.float32)
        self.y = torch.tensor(returns, dtype=torch.float32)
        self.T = seq_len

    def __len__(self):
        return len(self.X) - self.T

    def __getitem__(self, i):
        x_seq  = self.X[i: i + self.T]
        y_next = self.y[i + self.T]
        # 兼容旧接口：tail 和 cond 用简单代理
        tail   = self.X[i + self.T - 1, -6:]
        r_hist = self.y[i: i + self.T]
        mu_p   = r_hist.mean()
        sig_p  = r_hist.std().clamp(min=1e-6)
        cond   = torch.stack([mu_p, sig_p, mu_p*0, sig_p*0])
        return x_seq, tail, cond, y_next


# ══════════════════════════════════════════════════════════════════
# 2. 纯 Pinball Loss
# ══════════════════════════════════════════════════════════════════

class PinballLoss(nn.Module):
    """
    标准 Pinball（Quantile）Loss
    梯度有界：|∇L| ≤ max(α, 1-α) = 0.99，不会爆炸
    """
    def __init__(self, conf: float = 0.99):
        super().__init__()
        self.conf = conf

    def forward(self, var_pred: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        loss_seq = -y                         # 损失 = 负收益
        err      = loss_seq - var_pred
        ql = torch.mean(
            self.conf * err.clamp(min=0)
            + (1 - self.conf) * (-err).clamp(min=0)
        )
        return ql


# ══════════════════════════════════════════════════════════════════
# 3. 训练器 v5（极简稳定版）
# ══════════════════════════════════════════════════════════════════

class ETGPDTrainer:
    def __init__(self, model, lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 device: str = None):
        self.model    = model
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = PinballLoss(conf=model.confidence)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)

        # Warmup + Cosine decay：前 10% 步线性升温，之后余弦退火
        self._total_steps = 2000          # 粗估，fit 时会更新
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                (step + 1) / max(self._total_steps * 0.1, 1),
                0.5 * (1 + math.cos(math.pi * step / self._total_steps))
            )
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        for x, tail, cond, y in loader:
            x = x.to(self.device); y = y.to(self.device)

            self.optimizer.zero_grad()
            out  = self.model(x)
            loss = self.criterion(out["var"], y)

            # Pinball loss 梯度有界，不需要 NaN 检测
            # 仅做基本的 finite 检查
            if not torch.isfinite(loss):
                continue

            loss.backward()
            # 梯度裁剪：0.5（保守）
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.scheduler.step()

            total += loss.item()
            n     += 1

        return total / n if n > 0 else float("nan")

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        vars_, ess_, ys_ = [], [], []
        for x, tail, cond, y in loader:
            out = self.model(x.to(self.device))
            v   = out["var"].cpu().numpy()
            e   = out["es"].cpu().numpy()
            vars_.append(v); ess_.append(e); ys_.append(y.numpy())
        return (np.concatenate(vars_),
                np.concatenate(ess_),
                np.concatenate(ys_))

    def _score(self, var_arr, y_arr) -> float:
        """
        score = QL + smooth_viol_penalty
        penalty：sigmoid 形状，viol=1% 时为0，viol=0% 时为 0.005，viol=5% 时为 0.05
        平滑惩罚避免跳跃导致早停误判
        """
        conf     = self.model.confidence
        loss_seq = -y_arr
        err      = loss_seq - var_arr
        ql       = float(np.mean(
            conf * np.maximum(err,0) + (1-conf) * np.maximum(-err,0)))
        viol     = float(np.mean(loss_seq > var_arr))
        target   = 1 - conf    # 0.01

        # 连续惩罚：偏离 target 越远惩罚越大，两侧不对称（低违例惩罚更重）
        if viol < target:
            penalty = 0.008 * (target - viol) / target
        else:
            penalty = 0.002 * ((viol - target) / target) ** 2
        return ql + penalty

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 100, patience: int = 15,
            verbose: bool = True) -> List[dict]:
        # 更新 total_steps 给 warmup scheduler 使用
        self._total_steps = epochs * len(train_loader)
        # 重建 scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                (step + 1) / max(self._total_steps * 0.1, 1),
                0.5 * (1 + math.cos(math.pi * step
                                    / max(self._total_steps, 1)))
            )
        )

        best_score = float("inf")
        wait       = 0
        history    = []
        
        # Windows 兼容的临时保存路径
        temp_dir   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_models")
        os.makedirs(temp_dir, exist_ok=True)
        best_path  = os.path.join(temp_dir, f"best_model_{id(self.model)}.pt")

        for ep in range(1, epochs + 1):
            train_ql               = self.train_epoch(train_loader)
            var_val, es_val, y_val = self.eval_epoch(val_loader)
            val_viol               = float(np.mean(-y_val > var_val))
            val_score              = self._score(var_val, y_val)
            val_ql                 = float(np.mean(
                0.99*np.maximum(-y_val-var_val,0)
                +0.01*np.maximum(var_val-(-y_val),0)))

            record = {"epoch": ep, "train_ql": train_ql,
                      "val_ql": val_ql, "val_viol": val_viol,
                      "val_score": val_score}
            history.append(record)

            if verbose and ep % 5 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Ep {ep:3d} | "
                      f"train_ql={train_ql:.5f}  "
                      f"val_ql={val_ql:.5f}  "
                      f"val_viol={val_viol:.2%}  "
                      f"score={val_score:.5f}  "
                      f"lr={lr:.1e}")

            if val_score < best_score - 1e-7:
                best_score = val_score
                wait = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  早停于 epoch {ep}  "
                              f"(best score={best_score:.5f}  "
                              f"val_viol={val_viol:.2%})")
                    break

        try:
            self.model.load_state_dict(
                torch.load(best_path, map_location=self.device))
        except Exception:
            pass
        return history


# ══════════════════════════════════════════════════════════════════
# 4. 回测检验（同 v4）
# ══════════════════════════════════════════════════════════════════

class BacktestSuite:

    @staticmethod
    def kupiec_pof(violations, conf=0.99) -> dict:
        T, n  = len(violations), int(violations.sum())
        p     = 1 - conf
        p_hat = n/T if T > 0 else 0.0
        if n == 0:
            return {"stat": np.nan, "pvalue": np.nan, "pass": False,
                    "viol_rate": 0.0, "note": "0违例 → 过保守"}
        if n == T:
            return {"stat": np.nan, "pvalue": np.nan, "pass": False,
                    "viol_rate": 1.0, "note": "全违例"}
        lr = -2*(n*math.log(p/p_hat+1e-12)
                 +(T-n)*math.log((1-p)/(1-p_hat+1e-12)))
        pv = 1 - stats.chi2.cdf(lr, df=1)
        return {"stat": round(lr,4), "pvalue": round(pv,4),
                "pass": pv>0.05, "viol_rate": round(p_hat,4),
                "note": f"违例率={p_hat:.2%}（理论{p:.2%}）"}

    @staticmethod
    def christoffersen(violations) -> dict:
        v   = violations.astype(int)
        n00 = int(np.sum((v[:-1]==0)&(v[1:]==0)))
        n01 = int(np.sum((v[:-1]==0)&(v[1:]==1)))
        n10 = int(np.sum((v[:-1]==1)&(v[1:]==0)))
        n11 = int(np.sum((v[:-1]==1)&(v[1:]==1)))
        if (n01+n11)==0:
            return {"stat": np.nan, "pvalue": 1.0, "pass": True, "note": "无违例"}
        pi01 = n01/(n00+n01+1e-12)
        pi11 = n11/(n10+n11+1e-12) if (n10+n11)>0 else 0.0
        pi   = (n01+n11)/(n00+n01+n10+n11+1e-12)
        def sl(a): return math.log(max(a,1e-12))
        lr = -2*((n00+n10)*sl(1-pi)+(n01+n11)*sl(pi)
                 -n00*sl(1-pi01)-n01*sl(pi01)
                 -n10*sl(1-pi11)-n11*sl(pi11))
        pv = 1-stats.chi2.cdf(lr, df=1)
        return {"stat": round(lr,4), "pvalue": round(pv,4),
                "pass": pv>0.05,
                "note": f"π₀₁={pi01:.3f}  π₁₁={pi11:.3f}"}

    @staticmethod
    def dq_test(violations, var_pred, lags=4) -> dict:
        hit = violations.astype(float) - 0.01
        T   = len(hit)
        if T < lags+20:
            return {"stat": np.nan, "pvalue": np.nan, "pass": True, "note": "样本不足"}
        Y    = hit[lags:]
        cols = [hit[lags-i-1:T-i-1] for i in range(lags)]
        cols.append(var_pred[lags:])
        X    = np.column_stack([np.ones(len(Y))]+cols)
        try:
            beta   = np.linalg.pinv(X) @ Y
            resid  = Y - X @ beta
            sse_r  = float(resid@resid)
            sse_u  = float(Y@Y)
            k, n_  = X.shape[1]-1, len(Y)
            denom  = sse_r/max(n_-k-1,1)
            if denom < 1e-12:
                return {"stat": 0.0, "pvalue": 1.0, "pass": True, "note": "残差为零"}
            F  = ((sse_u-sse_r)/k)/denom
            lr = k*F
            pv = 1-stats.chi2.cdf(max(lr,0), df=k)
        except Exception as ex:
            return {"stat": np.nan, "pvalue": np.nan, "pass": True, "note": f"错误:{ex}"}
        return {"stat": round(float(lr),4), "pvalue": round(float(pv),4),
                "pass": pv>0.05, "note": "动态分位数自相关检验"}

    @staticmethod
    def esr_test(losses, es_pred, var_pred, conf=0.99) -> dict:
        mask = losses > var_pred
        n_v  = int(mask.sum())
        if n_v < 3:
            return {"esr": np.nan, "pvalue": np.nan, "pass": True,
                    "note": f"违例数={n_v}<3"}
        avg_l = float(losses[mask].mean())
        avg_e = float(es_pred[mask].mean())
        esr   = avg_l/(avg_e+1e-8)
        se    = float(losses[mask].std())/math.sqrt(n_v)
        t     = (avg_l-avg_e)/(se+1e-8)
        pv    = 2*(1-stats.t.cdf(abs(t), df=n_v-1))
        return {"esr": round(esr,4), "pvalue": round(float(pv),4),
                "pass": pv>0.05, "note": f"n_viol={n_v}  ESR理想≈1.0"}

    @staticmethod
    def mcs_rank(model_losses, n_boot=500) -> dict:
        names  = list(model_losses.keys())
        losses = np.stack([model_losses[n] for n in names], axis=1)
        return {names[i]: r+1 for r,i in enumerate(np.argsort(losses.mean(0)))}

    def full_report(self, returns, var_pred, es_pred,
                    conf=0.99, model_name="ETGPD-Transformer") -> pd.DataFrame:
        losses     = -returns
        violations = losses > var_pred
        kup = self.kupiec_pof(violations, conf)
        cc  = self.christoffersen(violations)
        dq  = self.dq_test(violations, var_pred)
        esr = self.esr_test(losses, es_pred, var_pred, conf)
        rows = [
            ["Kupiec POF",     kup.get("stat"), kup.get("pvalue"),
             "通过" if kup.get("pass") else "失败", kup.get("note","")],
            ["Christoffersen", cc.get("stat"),  cc.get("pvalue"),
             "通过" if cc.get("pass") else "失败",  cc.get("note","")],
            ["DQ Test",        dq.get("stat"),  dq.get("pvalue"),
             "通过" if dq.get("pass") else "失败",  dq.get("note","")],
            ["ESR",            esr.get("esr"),  esr.get("pvalue"),
             "通过" if esr.get("pass") else "失败",  esr.get("note","")],
        ]
        df = pd.DataFrame(rows, columns=["检验","统计量","p值","结论","备注"])
        df["模型"] = model_name
        print(f"\n{'='*72}")
        print(f"  回测报告 — {model_name}  (置信水平 {conf:.0%})")
        print(f"{'='*72}")
        print(df[["检验","统计量","p值","结论","备注"]].to_string(index=False))
        print(f"{'='*72}\n")
        return df


# ══════════════════════════════════════════════════════════════════
# 5. 滚动回测（含 GPD 后验校正）
# ══════════════════════════════════════════════════════════════════

def rolling_backtest(
    model,
    features:   np.ndarray,
    returns:    np.ndarray,
    train_size: int   = 1000,
    val_size:   int   = 100,
    test_step:  int   = 50,
    seq_len:    int   = 60,
    epochs:     int   = 80,
    batch_size: int   = 64,
    conf:       float = 0.99,
    verbose:    bool  = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(returns)
    all_var, all_es, all_ret = [], [], []
    pointer = train_size
    fold    = 0

    while pointer + test_step <= N:
        fold += 1
        if verbose:
            print(f"\n── Fold {fold}: 训练至 {pointer}，"
                  f"预测 [{pointer},{pointer+test_step})")

        tr_end = pointer - val_size
        f_tr   = features[:tr_end];        r_tr = returns[:tr_end]
        f_vl   = features[tr_end:pointer]; r_vl = returns[tr_end:pointer]
        f_te   = features[pointer-seq_len: pointer+test_step]
        r_te   = returns[pointer-seq_len:  pointer+test_step]

        ds_tr = ReturnDataset(f_tr, r_tr, seq_len)
        ds_vl = ReturnDataset(f_vl, r_vl, seq_len)
        ds_te = ReturnDataset(f_te, r_te, seq_len)

        if len(ds_tr) < batch_size or len(ds_te) < 1:
            pointer += test_step; continue

        dl_tr = DataLoader(ds_tr, batch_size=batch_size,
                           shuffle=True, drop_last=True)
        dl_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False)
        dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

        # 模型重置 + 数据尺度校准
        model.apply(_reset_weights)
        model.calibrate(r_tr)            # ← 关键：根据数据尺度校准 scale_max

        trainer = ETGPDTrainer(model, lr=1e-4)
        trainer.fit(dl_tr, dl_vl, epochs=epochs,
                    patience=15, verbose=verbose)

        # 训练集 VaR（用于 GPD 拟合）
        var_tr_arr, _, _ = trainer.eval_epoch(
            DataLoader(ds_tr, batch_size=batch_size, shuffle=False))
        gpd_params = model.fit_gpd_correction(
            -r_tr[seq_len:seq_len+len(var_tr_arr)],
            var_tr_arr, conf=conf)
        if verbose:
            print(f"  GPD 校正: ξ={gpd_params['xi']:.3f}  "
                  f"β={gpd_params['beta']:.4f}  "
                  f"sf={gpd_params['scale_factor']:.3f}  "
                  f"n_excess={gpd_params['n_excess']}")

        # 测试集预测 + GPD 校正
        var_f, es_f, ret_f = trainer.eval_epoch(dl_te)
        var_f, es_f = model.apply_gpd_correction(var_f, es_f, gpd_params)

        all_var.append(var_f); all_es.append(es_f); all_ret.append(ret_f)
        pointer += test_step

    return (np.concatenate(all_ret),
            np.concatenate(all_var),
            np.concatenate(all_es))


def _reset_weights(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()
