"""
Expert Representation Orthogonality Loss
专家表示正交性损失

L_ortho = ||C C^T - I||_F^2

C ∈ R^{M × d_feat}，第 i 行 c_i 是专家 i 的原型向量：
  c_i = normalize( sum_b w_{b,i} * normalize(feat_{b,i}) )

其中 w_{b,i} 是 gate 对样本 b 分配给专家 i 的权重（detach，不传梯度给 gate）。
"""

import torch
import torch.nn.functional as F


def expert_orthogonality_loss(feats, gate_weights):
    """
    Args:
        feats:        list[Tensor(B, d_feat)]  adapter 输出, len = M
        gate_weights: Tensor(B, M)             gate 权重 (会被 detach)

    Returns:
        l_ortho: scalar tensor, 梯度只流向 feats (adapter 参数)
    """
    M = len(feats)
    w = gate_weights # (B, M)

    prototypes = []
    for i in range(M):
        h_norm = F.normalize(feats[i], dim=-1)      # (B, d_feat)
        wi = w[:, i]                                  # (B,)

        # 防止权重全零: 回退到均匀权重
        wi_sum = wi.sum()
        if wi_sum < 1e-8:
            wi = torch.ones_like(wi)
            wi_sum = wi.sum()

        wi = wi / wi_sum                              # 归一化为概率
        c_i = (wi.unsqueeze(-1) * h_norm).sum(dim=0)  # (d_feat,)
        c_i = F.normalize(c_i, dim=0)                 # 单位长度
        prototypes.append(c_i)

    C = torch.stack(prototypes, dim=0)    # (M, d_feat)
    gram = C @ C.t()                      # (M, M)
    I = torch.eye(M, device=gram.device, dtype=gram.dtype)

    l_ortho = (gram - I).pow(2).sum()
    return l_ortho

def diversity_loss_from_probs(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p_bar = p.mean(dim=0)                  # (M,)
    p_bar = p_bar.clamp(min=eps)
    entropy = -(p_bar * p_bar.log()).sum()  # H(p_bar)
    return -entropy                         # 最小化 -H ≡ 最大化 H
