"""
ETGPD-Transformer — 核心模型 v5（稳定版）

架构重新设计原则：
  训练阶段：纯 Pinball Loss 驱动，输出单一 VaR 预测
  推断阶段：用训练好的残差序列拟合 GPD，做尾部校正（后验修正）

  彻底消除 v1-v4 的所有尝试性融合架构，回到最简可靠的设计：
  1. Transformer → 直接回归 VaR（单头输出，sigmoid 缩放到合理范围）
  2. GPD 只在推断期用历史残差拟合，不参与训练梯度
  3. 没有 var_floor、没有 softplus 无界输出、没有复杂融合层

  为什么这样可行：
  - Pinball loss 在统计上是分位数回归的充分条件（Koenker 1978）
  - GPD 后验校正是 FHS（Filtered Historical Simulation）的标准做法
  - 解耦训练和尾部建模，各自稳定
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_Z99 = 2.326347874040841
_SQRT2PI = math.sqrt(2 * math.pi)


# ══════════════════════════════════════════════════════════════════
# 稀疏多头注意力
# ══════════════════════════════════════════════════════════════════

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, top_k=10, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.top_k   = top_k
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        scale = math.sqrt(self.d_head)
        Q = self.W_q(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        K = self.W_k(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        V = self.W_v(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / scale
        k = min(self.top_k, T)
        topk_vals, _ = scores.topk(k, dim=-1)
        scores = scores.masked_fill(scores < topk_vals[...,-1:], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        self.last_attn_weights = attn.detach()
        out = torch.matmul(self.drop(attn), V)
        return self.W_o(out.transpose(1,2).contiguous().view(B,T,self.d_model))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, top_k=10, dropout=0.1):
        super().__init__()
        self.attn  = SparseMultiHeadAttention(d_model, n_heads, top_k, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.attn(x, mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()
                        * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:,:x.size(1)])


# ══════════════════════════════════════════════════════════════════
# ETGPD-Transformer v5：纯分位数回归训练
# ══════════════════════════════════════════════════════════════════

class ETGPDTransformer(nn.Module):
    """
    训练阶段：
      Transformer → h_last → VaR_head → 直接输出 VaR

    VaR_head 设计：
      两路输出加权平均
      - 路径1（均值路）：Linear → mean，表示条件均值
      - 路径2（分散路）：Linear → scale（>0），乘以固定分位数系数
      VaR = mean + scale * z_99

      mean 和 scale 都有 LayerNorm 约束，防止爆炸
      scale 用 sigmoid * 2σ_data 初始化，确保量级合理

    推断阶段（fit_gpd_correction）：
      收集训练集残差 r_t - VaR_t（负值 = 超出）
      用超出部分拟合 GPD，输出校正后的 VaR 和 ES
    """

    def __init__(
        self,
        input_dim:     int   = 33,
        tail_feat_dim: int   = 6,
        d_model:       int   = 64,
        n_heads:       int   = 4,
        n_layers:      int   = 4,
        d_ff:          int   = 256,
        top_k:         int   = 10,
        seq_len:       int   = 60,
        dropout:       float = 0.1,
        confidence:    float = 0.99,
    ):
        super().__init__()
        self.confidence = confidence
        self.input_dim  = input_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, seq_len+10, dropout)
        self.encoder    = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, top_k, dropout)
            for _ in range(n_layers)])

        # ── VaR 输出头（稳定设计）──────────────────────────────
        # 均值头：无界，但 LayerNorm 约束量级
        self.mean_head  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1))

        # 尺度头：sigmoid 输出 (0,1)，再缩放到 [0, scale_max]
        # scale_max 在首次前向传播时根据数据自动校准
        self.scale_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid())

        self._scale_max = 1.0          # 将在 calibrate() 中设置
        self._calibrated = False

        # 注意力权重（供 SHAP 热图使用）
        self._last_attn = None
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def calibrate(self, sample_returns: np.ndarray):
        """
        根据训练数据校准 scale_max
        scale_max = 3 × std(returns)（99% VaR 的合理上界）
        确保 VaR 头输出量级与真实数据匹配
        """
        std = float(np.std(sample_returns))
        self._scale_max = max(std * 3.0, 1e-3)
        self._calibrated = True

    def forward(self, x, tail_feats=None, cond_feats=None):
        B, T, _ = x.shape

        h = self.pos_enc(self.input_proj(x))
        for layer in self.encoder:
            h = layer(h)
        h_last = h[:, -1, :]                     # [B, d_model]

        self._last_attn = self.encoder[-1].attn.last_attn_weights

        mu    = self.mean_head(h_last).squeeze(-1)            # [B]
        scale = self.scale_head(h_last).squeeze(-1)           # [B] ∈ (0,1)
        scale = scale * self._scale_max                       # [B] ∈ (0, scale_max)

        # VaR = mu + scale × z_99
        var_pred = mu + scale * _Z99                          # [B]

        # ES = mu + scale × φ(z_99)/(1-conf)
        phi_z  = math.exp(-0.5 * _Z99**2) / _SQRT2PI
        es_pred = mu + scale * phi_z / (1 - self.confidence)  # [B]

        return {
            "mu":    mu,
            "scale": scale,
            "var":   var_pred,
            "es":    es_pred,
            "attn":  self._last_attn,
            # 兼容旧接口
            "sigma": scale,
            "xi":    torch.zeros_like(mu),
            "beta":  torch.ones_like(mu) * 0.01,
            "alpha": torch.ones_like(mu) * 0.5,
        }

    # ────────────────────────────────────────────────────────────
    # GPD 后验校正（推断期，不涉及梯度）
    # ────────────────────────────────────────────────────────────

    def fit_gpd_correction(
        self,
        train_losses: np.ndarray,
        train_var:    np.ndarray,
        conf:         float = 0.99,
    ) -> dict:
        """
        用训练集残差拟合 GPD（FHS 方法）

        参数
        ----
        train_losses : -returns（损失序列）
        train_var    : 训练集上的 VaR 预测

        返回
        ----
        {"xi": float, "beta": float, "u": float, "scale_factor": float}
        """
        from scipy.stats import genpareto
        residuals = train_losses - train_var          # 超出量（正值=穿透）
        u         = np.percentile(residuals, 90)      # 90th 百分位为阈值
        excess    = residuals[residuals > u] - u
        excess    = excess[excess > 0]

        if len(excess) < 10:
            return {"xi": 0.1, "beta": float(np.std(train_losses)) * 0.1,
                    "u": float(u), "scale_factor": 1.0, "n_excess": 0}

        try:
            xi, loc, beta = genpareto.fit(excess, floc=0)
            xi   = float(np.clip(xi,   -0.1, 0.5))
            beta = float(np.clip(beta,  1e-6, None))
        except Exception:
            xi, beta = 0.1, float(np.std(excess))

        # 计算 GPD VaR 校正因子
        p_u   = len(excess) / len(residuals)
        if xi != 0:
            var_gpd_exc = beta / xi * ((((1 - conf) / p_u) ** (-xi)) - 1)
        else:
            var_gpd_exc = beta * (-math.log((1 - conf) / p_u))
        var_gpd_exc = max(var_gpd_exc, 0.0)

        # scale_factor：GPD VaR 超出 / 正态假设超出
        normal_exc = float(np.std(train_losses)) * (_Z99 - 1.645)
        scale_factor = (var_gpd_exc / normal_exc) if normal_exc > 0 else 1.0
        scale_factor = float(np.clip(scale_factor, 0.8, 2.0))

        return {
            "xi":           xi,
            "beta":         beta,
            "u":            float(u),
            "scale_factor": scale_factor,
            "n_excess":     len(excess),
        }

    def apply_gpd_correction(
        self,
        var_pred:  np.ndarray,
        es_pred:   np.ndarray,
        gpd_params: dict,
    ) -> tuple:
        """
        将 GPD 校正应用到测试集预测

        校正公式：
          VaR_corrected = VaR_pred × scale_factor
          ES_corrected  = VaR_corrected × (1/(1-ξ) + β/((1-ξ)×VaR))
        """
        sf  = gpd_params.get("scale_factor", 1.0)
        xi  = gpd_params.get("xi", 0.1)
        var_corr = var_pred * sf
        # ES 用 GPD 解析公式
        denom    = max(1.0 - xi, 0.1)
        es_corr  = var_corr / denom + gpd_params.get("beta", 0.01) / denom
        es_corr  = np.maximum(es_corr, var_corr * 1.05)
        return var_corr, es_corr
