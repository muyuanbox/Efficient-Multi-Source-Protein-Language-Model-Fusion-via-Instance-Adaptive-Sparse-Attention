"""
Target-Conditioned JMMD-HSIC Transfer Loss
标签核条件迁移损失

P_i = A_i @ B_i  低秩线性投影 (d_i -> r -> d_feat)
  - 提取原始 embedding 里最主要、最可迁移的低维方向
  - 不是随便学一个巨大的非线性映射

Source: p_i = P_i(e_i)   低秩投影后的 source 表征
Target: z_i = adapter_i(e_i)  adapter 输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 配置 =====================
TRANSFER_CFG = {
    "sigma_x": "auto",
    "sigma_y": 1.0,
    "lambda_disc": 0.1,
    "proj_rank": 64,       # 低秩投影的秩 r, r << d_i
}

@torch.no_grad()
def _median_sigma(X, Y=None):
    """
    Median heuristic: sigma = median of pairwise distances.
    X: (B, D), Y: (B, D) optional.
    If Y is given, compute on the union [X; Y].
    """
    if Y is not None:
        Z = torch.cat([X, Y], dim=0)       # (2B, D)
    else:
        Z = X
    
    # 成对距离的平方
    ZZ = Z.pow(2).sum(-1, keepdim=True)     # (N, 1)
    dist_sq = ZZ + ZZ.t() - 2.0 * Z @ Z.t()  # (N, N)
    
    # 取上三角（排除对角线的 0）
    mask = torch.triu(torch.ones_like(dist_sq, dtype=torch.bool), diagonal=1)
    dists = dist_sq[mask].clamp(min=1e-10).sqrt()
    
    sigma = dists.median()
    return sigma.clamp(min=1e-5)            # 防止退化为 0

# ===================== 低秩投影器 =====================

class LowRankProjector(nn.Module):
    """
    P_i: R^{d_i} -> R^{d_feat}
    P_i = A_i @ B_i,  A_i ∈ R^{d_feat x r}, B_i ∈ R^{r x d_i}
    等价于 x -> A_i (B_i x),  瓶颈维度 r << d_i
    """
    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        r = min(rank, d_in, d_out)
        self.B = nn.Linear(d_in, r, bias=False)    # R^{d_i} -> R^r
        self.A = nn.Linear(r, d_out, bias=False)    # R^r -> R^{d_feat}
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.B.weight)
        nn.init.orthogonal_(self.A.weight)

    def forward(self, x):
        return self.A(self.B(x))                 # (B, d_feat)


def build_projectors(num_concat, d_feat, rank):
    """为每个 PLM segment 构建一个低秩投影器"""
    return nn.ModuleList([
        LowRankProjector(d_i, d_feat, rank) for d_i in num_concat
    ])


# ===================== 核函数 =====================

def _rbf_kernel(X, Y, sigma):
    """Gaussian RBF kernel.  X: (B, D), Y: (B, D) -> (B, B)"""
    XX = X.pow(2).sum(-1, keepdim=True)
    YY = Y.pow(2).sum(-1, keepdim=True)
    dist_sq = XX + YY.t() - 2.0 * X @ Y.t()
    return (-dist_sq / (2.0 * sigma ** 2)).exp()


def _label_kernel(y, is_cls, sigma_y):
    if is_cls:
        return (y.unsqueeze(0) == y.unsqueeze(1)).float()
    if sigma_y is None or sigma_y == "auto":
        sigma_y = _median_sigma(y.unsqueeze(-1))
    diff_sq = (y.unsqueeze(0) - y.unsqueeze(1)).pow(2)
    return (-diff_sq / (2.0 * sigma_y ** 2)).exp()

# ===================== HSIC =====================

def _center_kernel(K):
    row_mean = K.mean(dim=1, keepdim=True)
    col_mean = K.mean(dim=0, keepdim=True)
    return K - row_mean - col_mean + K.mean()


def _hsic(K_x, K_y):
    B = K_x.size(0)
    return (_center_kernel(K_x) * _center_kernel(K_y)).sum() / ((B - 1) ** 2)


# ===================== 单专家 JMMD =====================

def _expert_jmmd(p_i, z_i, K_y, sigma_x):
    # sigma_x 此时传入 "auto" 或 None 表示自动计算
    if sigma_x is None or sigma_x == "auto":
        sigma_x = _median_sigma(p_i, z_i)
    
    K_pp = _rbf_kernel(p_i, p_i, sigma_x)
    K_zz = _rbf_kernel(z_i, z_i, sigma_x)
    K_pz = _rbf_kernel(p_i, z_i, sigma_x)
    return ((K_pp + K_zz - 2.0 * K_pz) * K_y).mean()


# ===================== 主入口 =====================

def jmmd_hsic_transfer_loss(segs, feats, projectors, y, beta, is_cls, cfg=None):
    """
    Args:
        segs:       list[Tensor(B, D_i)]      BN 后原始 PLM embedding
        feats:      list[Tensor(B, d_feat)]    adapter 输出
        projectors: nn.ModuleList              低秩投影器 P_i = A_i B_i
        y:          (B,) 标签
        beta:       (B, M) gate 权重
        is_cls:     bool
        cfg:        dict
    """
    cfg = {**TRANSFER_CFG, **(cfg or {})}
    sigma_x, sigma_y, lam_disc = cfg["sigma_x"], cfg["sigma_y"], cfg["lambda_disc"]
    M = len(segs)

    beta_bar = beta.detach().mean(dim=0)           # (M,)
    K_y = _label_kernel(y, is_cls, sigma_y)

    l_transfer = torch.tensor(0.0, device=y.device)

    for i in range(M):
        e_i = segs[i].detach()                     # 不回传到 BN
        p_i = projectors[i](e_i)                   # (B, d_feat), 梯度流向 projector
        z_i = feats[i]                             # 梯度流向 adapter

        l_jmmd_i = _expert_jmmd(p_i, z_i, K_y, sigma_x)
        sig_z = _median_sigma(z_i)
        K_z = _rbf_kernel(z_i, z_i, sig_z)
        l_disc_i = -_hsic(K_z, K_y)

        l_transfer = l_transfer + beta_bar[i] * (l_jmmd_i + lam_disc * l_disc_i)

    return l_transfer