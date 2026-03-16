import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import (gumbel_softmax_logits, add_gaussian_noise, exploration_loss_entropy, 
init_linear_kaiming_relu, init_linear_xavier, init_linear_small_normal, init_norm, init_orthogonal)

#序列投影器
class SequenceProjector(nn.Module):
    """
    输入: x_onehot (B, 21, L)
    输出: z (B, project_dim)
    这里复用你原来的 Conv1d 结构，只是在最后加一层 Linear 投到 project_dim。
    """
    def __init__(
        self,
        project_dim: int,
        input_dimension: int,
        output_dimension: int,
        hidden_dimension: int,
        dropout_rate: float,
        num_models: int,
        num_concat: list,
        topk: int | None = None,    # 新增
        gumbel_tau: float = 1.0, 
        use_gumbel: bool = True,
    ):
        super().__init__()
        self.topk = topk
        self.num_models = num_models
        self.num_concat = num_concat

        #Gumbel采样参数
        self.gumbel_tau = gumbel_tau
        self.use_gumbel = use_gumbel


        #超参数
        # 在 SequenceProjector 中内嵌一个 EmbeddingNetwork
        self.pred_head = EmbeddingNetwork(
            input_dimension=input_dimension,      
            output_dimension=output_dimension,
            hidden_dimension=hidden_dimension,
            dropout_rate=dropout_rate,
            num_concat=num_concat,
            topk=topk,
        )        

        #self.router = Router(project_dim=project_dim, num_models=num_models)

        

    def forward(self, protein, config, epoch, training=False):
            
        if training:    
            predictions, mask, aux = self.pred_head(protein, config, epoch, training)
            return predictions, mask, aux
        else:           
            predictions, mask, aux = self.pred_head(protein, config, epoch, training)
            return predictions, mask

"""
#路由器，用于选择哪个模型来处理输入的蛋白质序列
class Router(nn.Module):
    def __init__(self, project_dim, num_models,vocab_size=21, pad_index=20, emb_dim=32):
        super().__init__()
        self.pad_index = pad_index
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_index)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, project_dim),
            nn.LayerNorm(project_dim),
            nn.ReLU(),
        )
        # G_p: 标准策略网络
        self.fc_p = nn.Linear(project_dim, num_models)
        # G_n: 噪声方差网络
        self.fc_n = nn.Linear(project_dim, num_models)

        self._init_weights()

    def _init_weights(self):
        # 0) Embedding：常用初始化 + PAD 行置 0（避免 padding 产生路由偏置）
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if self.embed.padding_idx is not None:
            with torch.no_grad():
                self.embed.weight[self.embed.padding_idx].zero_()

        # 1) MLP：Linear 用 Kaiming，BN 用 (gamma=1, beta=0)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 2) head：小尺度高斯，保证初始 logits 很小 -> softmax 接近均匀
        nn.init.normal_(self.fc_p.weight, mean=0.0, std=1e-3)
        if self.fc_p.bias is not None:
            nn.init.zeros_(self.fc_p.bias)
        nn.init.normal_(self.fc_n.weight, mean=0.0, std=1e-3)
        if self.fc_n.bias is not None:
            nn.init.zeros_(self.fc_n.bias)

    def forward(self, x_idx):
        mask = (x_idx != self.pad_index)                 # (B, L)
        e = self.embed(x_idx)                            # (B, L, emb_dim)

        # masked max pooling (也可改 masked mean)
        neg_inf = torch.finfo(e.dtype).min
        e = e.masked_fill(~mask.unsqueeze(-1), neg_inf)  # (B, L, emb_dim)
        pooled = e.max(dim=1).values                     # (B, emb_dim)

        h = self.mlp(pooled)                             # (B, project_dim)

        logit_p = self.fc_p(h)                          # (B, num_models)
        logit_n = self.fc_n(h)                          # (B, num_models)

        return logit_p, logit_n

"""

class PerModelHead(nn.Module):
    """每个模型一个 head：既输出 logits，也输出 semantic 表示（用于 α 相似度更稳）"""
    def __init__(self, d_feat: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_feat, hidden)
        self.act = nn.ReLU()
        self.dp = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x, return_sem: bool = False):
        h = self.dp(self.act(self.fc1(x)))      # semantic: (B, hidden)
        y = self.fc2(h)                         # logits/reg: (B, out_dim)
        if return_sem:
            return y, h
        return y

# Model architecture based on https://www.biorxiv.org/content/10.1101/2023.12.13.571462v1 (https://github.com/RSchmirler/data-repo_plm-finetune-eval/blob/main/notebooks/embedding/Embedding_Predictor_Training.ipynb)
class EmbeddingNetwork(nn.Module): #预测头 (Prediction Head)
    def __init__(self, input_dimension, output_dimension, hidden_dimension, dropout_rate, num_concat, topk):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.dropout_rate = dropout_rate
        self.topk = topk
        self.num_models = len(num_concat)
        self.normalize = nn.BatchNorm1d(self.input_dimension)
        self.fully_connected = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Linear(self.hidden_dimension, self.output_dimension)
        self.num_concat = num_concat

        from transfer_loss import build_projectors
        proj_rank = 64 
        self.projectors = build_projectors(num_concat, self.hidden_dimension, proj_rank)
        
        #β attention 相关设计
        self.beta_temp = 1.0  # 例如 1.0



        #α attention 相关设计
        self.alpha_temp = 0.5
        # 1) per-model adapter: dim_i -> d_feat
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_i, self.hidden_dimension),
                nn.LayerNorm(self.hidden_dimension),
                nn.ReLU(),
            )
            for dim_i in num_concat
        ])

        # 2) per-model classifier heads g_j
        self.heads = nn.ModuleList([
            PerModelHead(self.hidden_dimension, self.hidden_dimension, self.output_dimension, dropout=0.2)
            for _ in range(self.num_models)
        ])



        # 3) Agile α：WF/WO，把 feature 与 output/semantic 投到同一 demb 后 cosine
        # WF: (d_feat -> d_emb)
        self.WF = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        # WO: 用 semantic 更稳（head_hidden -> d_emb）
        self.WO = nn.Linear(self.hidden_dimension, self.hidden_dimension)

        # 4) Agile β（inter-domain）：query/key 注意力
        # 论文里 query 用 concat(features) 线性映射；key 用 WF(share)
        self.WQ = nn.Linear(self.num_models * self.hidden_dimension, self.hidden_dimension)
        self._initialize_weights()

    def _initialize_weights(self):
        # 1) adapters: Linear -> LN -> ReLU
        for ad in self.adapters:
            for m in ad.modules():
                if isinstance(m, nn.Linear):
                    init_linear_kaiming_relu(m)
                elif isinstance(m, nn.LayerNorm):
                    init_norm(m)

        # 2) per-model heads
        for head in self.heads:
            init_linear_kaiming_relu(head.fc1)
            # 输出层：更保守，避免初期 logits/回归值过大
            # 分类可用 1e-2；回归建议 1e-3（你是 GB1 回归的话优先 1e-3）
            init_linear_small_normal(head.fc2, std=1e-3)

        # 3) WF/WO/WQ: 用 orthogonal 或 xavier（推荐 orthogonal）
        init_orthogonal(self.WF, gain=1.0)
        init_orthogonal(self.WO, gain=1.0)
        init_orthogonal(self.WQ, gain=1.0)

        # 4) BN（一般默认即可，这里显式更清晰）
        init_norm(self.normalize)
        
    def _split_segments(self, protein):
        segs = []
        start = 0
        for dim_i in self.num_concat:
            end = start + dim_i
            segs.append(protein[:, start:end])
            start = end
        return segs  # list of (B, dim_i)

        
    def forward(self, protein, config, epoch, training=False):
        """
        router_prior: (B, M) 可选。你原来的 Router 输出可以作为 prior，
                      最终融合权重用 normalize(router_prior * beta)。
        """
        x = self.normalize(protein)                         # (B, D)
        if config['training']['iteration'] >= 10 * epoch and training:
            test = True
        else:
            test = False


        # ---- Step A: adapters -> features φ_i ----
        segs = self._split_segments(x)
        feats = [self.adapters[i](segs[i]) for i in range(self.num_models)]    # list of (B, d_feat)

        # ---- Step B: Agile α（intra-domain weights）----
        # alpha[i]: (B, M)  表示对固定 feature i，选择哪个 head j
        # y_tilde[i]: (B, out_dim)  assembled output
        alpha_all = []
        y_tilde_all = []
        if test:
            for i in range(self.num_models):
            # cross-head outputs: 对同一 φ_i，跑所有 head_j
                logits_ij = []
                sem_ij = []
                y_ij, h_ij = self.heads[i](feats[i], return_sem=True)     # (B,out_dim), (B,head_hidden)

                y_tilde_all.append(y_ij)
        else :
            for i in range(self.num_models):
                # cross-head outputs: 对同一 φ_i，跑所有 head_j
                logits_ij = []
                sem_ij = []
                for j in range(self.num_models):
                    y_ij, h_ij = self.heads[j](feats[i], return_sem=True)     # (B,out_dim), (B,head_hidden)
                    logits_ij.append(y_ij)
                    sem_ij.append(h_ij)

                Y = torch.stack(logits_ij, dim=1)       # (B, M, out_dim)
                H = torch.stack(sem_ij, dim=1)          # (B, M, head_hidden)

                # similarity: cos( WF(φ_i), WO(H_ij) )
                f = F.normalize(self.WF(feats[i]), dim=-1)          # (B, d_emb)
                o = F.normalize(self.WO(H), dim=-1)                 # (B, M, d_emb)
                sim = (o * f.unsqueeze(1)).sum(dim=-1)              # (B, M)

                alpha_i = F.softmax(sim / self.alpha_temp, dim=-1)  # (B, M)
                y_tilde_i = (alpha_i.unsqueeze(-1) * Y).sum(dim=1)  # (B, out_dim)

                alpha_all.append(alpha_i)
                y_tilde_all.append(y_tilde_i)

        # (B, M, M): alpha[b, i, j]
        #alpha = torch.stack(alpha_all, dim=1)
        # (B, M, out_dim): y_tilde[b, i, :]
        y_tilde = torch.stack(y_tilde_all, dim=1)

        # ---- Step C: Agile β（inter-domain weights）----
        feat_cat = torch.cat(feats, dim=-1)                     # (B, M*d_feat)
        q = F.normalize(self.WQ(feat_cat), dim=-1)              # (B, d_emb)

        K = torch.stack([self.WF(f_i) for f_i in feats], dim=1) # (B, M, d_emb) 共享 WF
        K = F.normalize(K, dim=-1)
        sim_beta = (K * q.unsqueeze(1)).sum(dim=-1)             # (B, M)
        beta = F.softmax(sim_beta / self.beta_temp, dim=-1)     # (B, M)

        w = beta
        w_soft = w  # 保存 top-k 前的稠密分布
        if config['training']['iteration'] >= 10 * epoch and training:
            topk = None
        else:
            topk = self.topk
        # ---- Step D: 可选 top-k 稀疏化（对 w 做）----
        if topk is not None and topk < self.num_models:
            _, idx = w.topk(topk, dim=-1)                  # (B, k)
            mask = torch.zeros_like(w)
            mask.scatter_(1, idx, 1.0)

            w_hard = w * mask
            w_hard = w_hard / (w_hard.sum(dim=-1, keepdim=True) + 1e-8)
            w = w_hard.detach() + (w - w.detach())              # ST
        else:
            mask = torch.ones_like(w)

        # ---- Step E: final output ----
        y_hat = (w.unsqueeze(-1) * y_tilde).sum(dim=1)           # (B, out_dim)


        #所谓“对角线”其实是“每个专家只对自己的输出 head 负责”
        y_diag_list = []
        for i in range(self.num_models):
            y_ii = self.heads[i](feats[i], return_sem=False)  # (B, out_dim)
            y_diag_list.append(y_ii)
        y_diag = torch.stack(y_diag_list, dim=1)              # (B, M, out_dim)

        aux = {
            "alpha": 0,          # (B,M,M)
            "beta": beta,            # (B,M)
            "w_final": w,            # (B,M)
            "y_diag": y_diag,        # (B,M,out_dim)
            "y_tilde": y_tilde,      # (B,M,out_dim)  你已经算了
            "mask": mask,
            "w_soft": w_soft,
            "segs": segs,       # ← 新增: list of (B, D_i), BN后原始PLM embedding
            "feats": feats,      # ← 新增: list of (B, d_feat), adapter输出
        }
        return y_hat, mask, aux
    


class WideEmbeddingNetwork(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension, width_multiplier, dropout_rate):
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.width = int(hidden_dimension * width_multiplier)
        self.dropout_rate = dropout_rate

        self.normalize = nn.BatchNorm1d(self.input_dimension)
        self.input_layer = nn.Linear(self.input_dimension, self.width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.hidden_layer = nn.Linear(self.width, self.hidden_dimension)
        self.output_layer = nn.Linear(self.hidden_dimension, self.output_dimension)
        
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, protein):
        x = self.normalize(protein)
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x.squeeze()



# Model architecture based on https://www.biorxiv.org/content/10.1101/2024.01.29.577794v1 (https://github.com/navid-naderi/PLM_SWE/tree/main)
class PPIClassifier(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension

        self.fully_connected = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fully_connected.weight)
        
    def forward(self, proteinA, proteinB):
        
        combined_proteins = torch.cat((proteinA, proteinB), dim=0)
        projected_combined = self.relu(self.fully_connected(combined_proteins))
        num_pairs  = len(combined_proteins) // 2
        cosine_similarity = F.cosine_similarity(projected_combined[:num_pairs], projected_combined[num_pairs:])
        
        return cosine_similarity




