import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, confusion_matrix, precision_recall_curve, precision_score, recall_score, confusion_matrix, f1_score, matthews_corrcoef
import numpy as np
import scipy.stats
import torch.nn.functional as F
import torch.nn as nn
from im_loss import expert_orthogonality_loss, diversity_loss_from_probs
############ Utils for AAV, GB1, GFP, Location, Meltome, Stability Dataset Models ############
class BenchmarkDataset(Dataset):
    def __init__(self, dictionary, file_path, target_type=float):
        self.dictionary = dictionary
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                protein, target = line.strip().split("\t")
                if target_type == float:
                    target_value = float(target)
                    target_tensor = torch.tensor(target_value, dtype=torch.float32)
                else:
                    target_value = int(target)
                    target_tensor = torch.tensor(target_value, dtype=torch.long)
                self.data.append((protein, target_tensor))

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, index):
    #     protein, target = self.data[index]
    #     return self.dictionary[protein], target

    def __getitem__(self, index):
        protein, target = self.data[index]
        return protein, self.dictionary[protein], target #gfp的protein直接就是蛋白质序列 location/aav 可能是蛋白质编号或者序列
    
def get_data_loader(dataset_type, dictionary, action_file, batch_size, shuffle):

    dataset = BenchmarkDataset(dictionary, action_file, target_type=int if dataset_type == 'location' else float)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_step(model, config, data_loader, optimizer, criterion, device, seq_dict, training=True):
    model.train()
    total_loss = 0
    total_samples = 0


    # 【新增】初始化计数器：形状为 (Num_Models,)
    # 假设 model.num_models 可访问，或者硬编码，或者动态获取
    # 这里我们动态获取
    total_selection_counts = 0
    total_samples = 0
    total_selection_samples = 0

    for pids, protein, target in data_loader: #pids可能是代号/也可能是直接的序列
        optimizer.zero_grad()

        protein, target = protein.to(device), target.to(device)
        predictions, mask, aux = model(protein, training)
        

        # 更新计数器
        batch_counts = mask.sum(dim=0).detach().cpu()
        total_selection_counts += batch_counts
        
        total_selection_samples += mask.size(0) # 累加样本总数
        '''
        loss = l_task + l_explore*gamma + l_exploit
        '''
        
        # ===== 1) 主任务损失：先对齐形状，彻底消灭 broadcasting =====
        pred = predictions
        if pred.dim() == 2 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)          # (B,)

        tgt = target
        if tgt.dim() == 2 and tgt.size(-1) == 1:
            tgt = tgt.squeeze(-1)           # (B,)

        assert pred.shape == tgt.shape, (pred.shape, tgt.shape)
        l_task = F.mse_loss(pred, tgt)       # scalar
        # ===== 2) Masked Exploit Loss：只对被激活的专家计算损失 =====
        y_diag = aux["y_diag"]
        assert y_diag.dim() == 3 and y_diag.size(1) == 6 and y_diag.size(2) == 1, y_diag.shape
        y_diag = y_diag.squeeze(-1)          # (B, 6)

        tgt6 = tgt.unsqueeze(1).expand_as(y_diag)   # (B, 6)
        loss_diag_b6 = (y_diag - tgt6).pow(2)       # (B, 6), 每个专家的 MSE

        # 【关键】使用 mask.detach()，让门控梯度不通过 exploit loss
        # 这样门控不能"通过改变选择来降低 exploit"，而是专注让被选中的专家变强
        mask_detached = mask.detach().float()  # (B, 6)

        # Masked exploit loss: 只对激活的专家计算加权平均
        l_exploit = (mask_detached * loss_diag_b6).sum() / (mask_detached.sum() + 1e-8)

        l_aux_diag = 0.0  #全能预测器损失
        lambda_exploit = config['training']['l_exploit']
        lambda_aux = config['training']['l_aux']


        # ===== 3) IM loss =====
        lambda_intra = config["training"].get("l_im_intra", 0.0)
        lambda_gate  = config["training"].get("l_im_gate", 0.0)
        im_reduce    = config["training"].get("im_reduce", "mean")
        im_eps       = config["training"].get("im_eps", 1e-8)

        is_cls = isinstance(criterion, torch.nn.CrossEntropyLoss)

        # base loss 只拼一次
        loss = l_task + lambda_exploit * l_exploit + lambda_aux * l_aux_diag

        # 统一的统计字典（你可以返回或打印）
        stats = {}

        # ---- (A) 原文 intra IM：仅分类且 out_dim=C>1 时有意义
        if lambda_intra > 0 and is_cls:
            # (B, M, C)
            y_tilde = aux["y_tilde"]
            l_intra, l_ent, l_div = agile_intra_im_from_y_tilde(y_tilde, reduce=im_reduce, eps=im_eps)
            loss = loss + lambda_intra * l_intra
            stats.update({
                "im/intra":      l_intra.detach(),
                "im/intra_ent":  l_ent.detach(),
                "im/intra_div":  l_div.detach(),
            })

        # ---- (B) gate IM：回归推荐；分类也可选（看你是否希望约束专家使用）
        if lambda_gate > 0 and (not is_cls):
            p_gate = aux.get("w_soft", aux["w_final"])  # (B, M)
            l_gate, l_gate_ent, l_gate_div = im_loss_from_probs(p_gate, reduce=im_reduce, eps=im_eps)
            loss = loss + lambda_gate * l_gate
            stats.update({
                "im/gate":      l_gate.detach(),
                "im/gate_ent":  l_gate_ent.detach(),
                "im/gate_div":  l_gate_div.detach(),
            })

        loss.backward()
        optimizer.step()
        
        batch_size = len(protein)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
    
    return total_loss / total_samples, total_selection_counts, total_selection_samples


from transfer_loss import jmmd_hsic_transfer_loss
def train_step_dual_optimizer(
    model, config, data_loader,
    optimizer_gate, optimizer_expert,
    gate_params, expert_params,  # 传入参数列表用于梯度控制
    criterion, device, seq_dict, epoch, training=True
):
    """
    彻底解耦的训练步骤（单次前向）：
    - Gate:   学习 "选谁"    ← l_task + l_im
    - Expert: 学习 "拟合好"  ← l_task + l_exploit
    
    关键：使用 retain_graph + 手动梯度控制
    """
    model.train()
    total_loss = 0
    total_samples = 0
    total_selection_counts = 0
    total_selection_samples = 0

    lambda_im = config["training"].get("l_im_gate", 0.1)
    lambda_exploit = config["training"].get("l_exploit", 0.5)


    gate_ids = {id(p) for p in gate_params}
    expert_ids = {id(p) for p in expert_params}
    assert gate_ids.isdisjoint(expert_ids), "gate_params 与 expert_params 存在重叠参数"


    for pids, protein, target in data_loader:
        protein = protein.to(device)
        target = target.to(device)
        
        tgt = target.squeeze() if target.dim() > 1 else target
        is_cls = isinstance(criterion, torch.nn.CrossEntropyLoss)
        # ================================================================
        # Step 1: 单次前向传播
        # ================================================================
        pred, mask, aux = model(protein, config, epoch, training=True)
        
        # 更新统计
        batch_counts = mask.sum(dim=0).detach().cpu()
        total_selection_counts += batch_counts
        total_selection_samples += mask.size(0)

        # ================================================================
        # Step 2: 计算各个 Loss
        # ================================================================
        if is_cls:
            # 分类任务：pred 保持 (B, num_classes)，tgt 是 (B,) 的类别标签
            l_task = F.cross_entropy(pred, tgt)
        else:
            # 回归任务：squeeze 后计算 MSE
            pred_sq = pred.squeeze() if pred.dim() > 1 else pred
            l_task = F.mse_loss(pred_sq, tgt)
        
        

        # l_exploit: Exploit 损失（只应该更新 expert）
        y_diag = aux["y_diag"]  # (B, M, out_dim)
        mask_detached = mask.detach().float()  # (B, M)，关键：阻断到 gate 的梯度
        
        if is_cls:
            # 分类任务：对每个专家计算 cross entropy
            # y_diag: (B, M, num_classes), tgt: (B,)
            B, M, C = y_diag.shape
            # 将 y_diag reshape 成 (B*M, C)，tgt 扩展成 (B*M,)
            y_diag_flat = y_diag.view(B * M, C)
            tgt_flat = tgt.unsqueeze(1).expand(B, M).reshape(B * M)
            # 计算每个 (样本, 专家) 对的 cross entropy
            loss_per_expert_flat = F.cross_entropy(y_diag_flat, tgt_flat, reduction='none')  # (B*M,)
            loss_per_expert = loss_per_expert_flat.view(B, M)  # (B, M)
        else:
            # 回归任务：squeeze 后计算 MSE
            y_diag_sq = y_diag.squeeze(-1)  # (B, M)
            tgt_expand = tgt.unsqueeze(1).expand_as(y_diag_sq)
            loss_per_expert = (y_diag_sq - tgt_expand).pow(2)  # (B, M)

        p_gate = aux.get("w_soft", aux["w_final"])  # (B, M)
        # l_im: IM 损失（只应该更新 gate）
        # p_gate = aux.get("w_soft", aux["w_final"])
        # l_im, _, _ = im_loss_from_probs(p_gate, reduce="mean")
        # #l_im = 0.0

        #l_im, _, _ = im_loss_from_probs(p_gate, reduce="mean")
        l_im = diversity_loss_from_probs(p_gate)
        #l_im_balance = load_balance_loss(p_gate, mask)

        l_exploit = (mask_detached * loss_per_expert).sum() / (mask_detached.sum() + 1e-8)

        # Step :加入transfer loss：
        # 原: loss_expert = l_task + lambda_exploit * l_exploit
        # 新:
        lambda_transfer = config["training"].get("l_transfer", 0.0)
        transfer_cfg = config["training"].get("transfer_cfg", None)
        l_transfer = torch.tensor(0.0, device=device)
        if lambda_transfer > 0:
            l_transfer = jmmd_hsic_transfer_loss(
                segs       = aux["segs"],
                feats      = aux["feats"],
                projectors = model.pred_head.projectors,
                y          = tgt,
                beta       = aux.get("w_soft", aux["w_final"]),
                is_cls     = is_cls,
                cfg        = transfer_cfg,
            )
        if epoch % 10 ==0 :
            print(aux["w_soft"])

            
        # ================================================================
        # Step 3: 组合 Loss
        # ================================================================
        # Gate Loss: l_task + l_im
        # Expert Loss: l_task + l_exploit
        # 
        # 由于 l_task 是共享的，我们可以这样处理：
        # total_loss = l_task + λ_im * l_im + λ_exploit * l_exploit
        # 
        # 但这样 gate 会收到 l_exploit 的梯度（虽然 mask.detach 阻断了部分路径，
        # 但 y_diag 仍然依赖 alpha，alpha 依赖 gate 参数）
        #
        # 更好的方案：分两次 backward

        # ================================================================
        # Step 4: 双阶段反向传播
        # ================================================================
        
        loss_gate   = l_task + lambda_im * l_im
        loss_expert = l_task + lambda_exploit * l_exploit + 5 * l_transfer

        # 清零所有梯度
        optimizer_gate.zero_grad()
        optimizer_expert.zero_grad()

        # -----------------------------
        # 只对 gate_params 求导
        # -----------------------------
        grads_gate = torch.autograd.grad(
            loss_gate, gate_params,
            retain_graph=True,     # 下面还要对 expert 求导
            allow_unused=True
        )

        # -----------------------------
        # 只对 expert_params 求导
        # -----------------------------
        grads_expert = torch.autograd.grad(
            loss_expert, expert_params,
            retain_graph=False,
            allow_unused=True
        )

        # -----------------------------
        # 9) 写回 grad 并分别 step
        # -----------------------------
        assign_grads(gate_params, grads_gate)
        optimizer_gate.step()

        assign_grads(expert_params, grads_expert)
        optimizer_expert.step()
        # ================================================================
        # Step 5: 统计
        # ================================================================
        batch_size = len(protein)
        total_samples += batch_size
        total_loss += (loss_gate.item() + loss_expert.item()) * batch_size

    return total_loss / total_samples, total_selection_counts, total_selection_samples

def test_step(model, config, data_loader, criterion, device, seq_dict, epoch, return_preds=False):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for pids, protein, target in data_loader:
            protein, target = protein.to(device), target.to(device)

            predictions, _ = model(protein, config, epoch)

            # ---- 分类任务保持不动 ----
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(predictions, target)
                preds_for_metric = predictions.argmax(dim=1)

            # ---- 回归任务：必须 squeeze + mse_loss，消灭 broadcasting ----
            else:
                pred = predictions                  # (B, 1)
                if pred.dim() == 2 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)          # (B,)
                tgt = target                         # (B,)
                if tgt.dim() == 2 and tgt.size(-1) == 1:
                    tgt = tgt.squeeze(-1)           # (B,)
                assert pred.shape == tgt.shape, (pred.shape, tgt.shape)

                loss = F.mse_loss(pred, tgt)
                preds_for_metric = pred

            batch_size = len(protein)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            all_targets.append(target.detach())
            all_predictions.append(preds_for_metric.detach())

    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        metric = calculate_accuracy(all_targets, all_predictions)
    else:
        # 强制变成一维，避免 (N,1) 干扰 spearman
        metric = calculate_spearman(all_targets.reshape(-1), all_predictions.reshape(-1))

    if return_preds:
        return total_loss / total_samples, metric, all_predictions
    return total_loss / total_samples, metric


def calculate_spearman(y_true, y_pred):
    return scipy.stats.spearmanr(y_true, y_pred).correlation

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

############ Protein-protein interaction classification utils ############
class ProteinInteractionDataset(Dataset):
    def __init__(self, protein_dict, protein_interactions_file):
        self.protein_dict = protein_dict
        self.protein_interactions = []
        with open(protein_interactions_file, 'r') as f:
            for line in f:
                protein_a, protein_b, label = line.strip().split("\t")
                self.protein_interactions.append((protein_a, protein_b, int(label)))

    def __len__(self):
        return len(self.protein_interactions)

    def __getitem__(self, index):
        protein_a, protein_b, label = self.protein_interactions[index]
        return self.protein_dict[protein_a], self.protein_dict[protein_b], torch.tensor(label, dtype=torch.float)

def get_ppi_data_loader(dictionary, action_file, batch_size, shuffle):
    dataset = ProteinInteractionDataset(dictionary, action_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def ppi_train_step(model, data_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_samples = 0

    for proteinA, proteinB, labels in data_loader:
        optimizer.zero_grad()
        proteinA, proteinB, labels = proteinA.to(device), proteinB.to(device), labels.to(device)
        probabilities = model(proteinA, proteinB)
        loss = criterion(probabilities, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(proteinA)
        total_samples += batch_size

        total_loss += loss.item()*batch_size
    return total_loss / total_samples

def ppi_test_step(model, data_loader, criterion, device, return_preds=False):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for proteinA, proteinB, labels in data_loader:
            proteinA, proteinB, labels = proteinA.to(device), proteinB.to(device), labels.to(device)
            probabilities = model(proteinA, proteinB)
            loss = criterion(probabilities, labels)

            batch_size = len(proteinA)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            all_labels.append(labels)
            all_probabilities.append(probabilities)
        
    all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()
    all_probabilities = torch.cat(all_probabilities, dim=0).detach().cpu().numpy()

    aucroc, prc, accuracy, sensitivity, specificity, precision, f1, mcc = calculate_classification_metrics(all_labels, all_probabilities)
   
    if return_preds:
        return total_loss / total_samples, aucroc, prc, accuracy, sensitivity, specificity, precision, f1, mcc, all_probabilities
    
    else:
        return total_loss / total_samples, aucroc, prc, accuracy, sensitivity, specificity, precision, f1, mcc
    

def calculate_classification_metrics(label, probabilities):

    predictions = np.round(probabilities)

    aucroc = roc_auc_score(label, probabilities)
    precision, recall, _ = precision_recall_curve(label, probabilities)
    prc = auc(recall, precision)
    accuracy = accuracy_score(label, predictions)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(label, predictions).ravel()
    sensitivity = recall_score(label, predictions)
    specificity = true_negative / (true_negative + false_positive)
    precision = precision_score(label, predictions)
    f1 = f1_score(label, predictions)
    mcc = matthews_corrcoef(label, predictions)
    return aucroc, prc, accuracy, sensitivity, specificity, precision, f1, mcc



#Gumbel采样工具

import torch
import torch.nn.functional as F

def sample_gumbel(shape, device, eps: float = 1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_logits(logits: torch.Tensor, tau: float):
    g = sample_gumbel(logits.shape, device=logits.device)
    return (logits + g) / tau


#loss_exploration
def add_gaussian_noise(logits_p: torch.Tensor, logits_n: torch.Tensor):
    """
    noisy_logits = logits_p + eps * softplus(logits_n)
    eps ~ N(0,1)
    """
    eps = torch.randn_like(logits_p)
    scale = F.softplus(logits_n)                   # 非负
    return logits_p + eps * scale

def exploration_loss_entropy(y_soft: torch.Tensor):
    """
    y_soft: (B, M)
    Lexplore = -H(mean_batch(y_soft))
    """
    p_bar = y_soft.mean(dim=0)                      # (M,)
    p_bar = p_bar.clamp_min(1e-8)
    entropy = -(p_bar * p_bar.log()).sum()          # 标准熵
    lexplore = -entropy                             # 最小化 lexplore 等价最大化 entropy
    return lexplore

#loss_exploitation

@torch.no_grad()
def _unique_activated_experts(mask: torch.Tensor):
    # mask: (B, M) 0/1
    active = (mask.sum(dim=0) > 0).nonzero(as_tuple=False).squeeze(-1)      #(M,)
    return active                                                  #返回激活的专家索引

def exploitation_loss_activated(
    pred_head: nn.Module,
    protein,
    mask: torch.Tensor,
    target: torch.Tensor,
    num_concat: list,
    criterion,
):
    """
    Lexploit = sum_i E_{mask[:,i]=1} loss(pred_i, y)
    pred_i 通过 w_onehot(i) 调用 pred_head 得到
    """
    B, M = mask.shape
    device = mask.device

    active_experts = _unique_activated_experts(mask)
    if active_experts.numel() == 0:
        return torch.zeros((), device=device)

    total = 0.0
    denom = 0.0

    for i in active_experts.tolist():
        idx = (mask[:, i] > 0).nonzero(as_tuple=False).squeeze(-1)    # 激活该 expert 的样本
        if idx.numel() == 0:
            continue

        w = torch.zeros((B, M), device=device)
        w[idx, i] = 1.0                                              # 只对这些样本用 one-hot

        pred_i = pred_head(protein, w, num_concat)                    # (B, ...) 但只取 idx
        loss_i = criterion(pred_i[idx], target[idx])

        # 按样本数加权，避免某个 expert 样本多就主导
        total = total + loss_i * idx.numel()
        denom = denom + idx.numel()

    return total / (denom + 1e-8)


def st_gumbel_topk(score, k, tau=1.0, eps=1e-8):
    # score: (B, M) 非负即可；更稳定可用 log(score+eps) 当 logits
    logits = (score + eps).log()

    # 1) 加 Gumbel 噪声
    g = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    noisy = (logits + g) / tau

    # 2) 取 top-k（离散）
    _, topk_idx = noisy.topk(k, dim=-1)
    mask = torch.zeros_like(score).scatter_(1, topk_idx, 1.0)

    # 3) 连续权重（用 softmax/noisy softmax）作为反向代理
    soft = F.softmax(noisy, dim=-1)          # (B, M)
    hard = mask * soft
    hard = hard / (hard.sum(dim=-1, keepdim=True) + eps)

    # 4) ST：前向 hard，反向 soft（或 hard 对 soft）
    final = hard.detach() + (soft - soft.detach())
    return final, mask



def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p: (..., K), each row sums to 1
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)

#Switch Transformer 风格的 importance + load balance loss

def load_balance_loss(gate_probs, mask):
    """
    gate_probs: (B, M) softmax后的β权重
    mask: (B, M) top-k的0/1 mask
    """
    # f_i: 每个专家实际被路由到的样本比例
    f = mask.float().mean(dim=0)           # (M,)
    # P_i: 每个专家的平均gate概率  
    P = gate_probs.mean(dim=0)             # (M,)
    # 惩罚 f 和 P 的相关性（防止富者越富）
    return (f * P).sum() * len(f)

def im_loss_from_probs(p: torch.Tensor, reduce: str = "mean", eps: float = 1e-8):
    """
    p: (B, ..., K)  概率分布
    返回: (L_im, L_ent, L_div) 三个标量（按 reduce 聚合）
    """
    # L_ent: E_b[ H(p_b) ]，先对 K 求熵，再对 batch 求均值
    H_inst = _entropy_from_probs(p, eps=eps).mean(dim=0)   # (...,)

    # L_div: H( mean_b p_b )
    p_bar = p.mean(dim=0)                                 # (..., K)
    H_div = _entropy_from_probs(p_bar, eps=eps)            # (...,)

    L_im = H_inst - H_div                                  # (...,)

    if reduce == "sum":
        return L_im.sum(), H_inst.sum(), H_div.sum()
    else:  # "mean"
        return L_im.mean(), H_inst.mean(), H_div.mean()

def agile_intra_im_from_y_tilde(y_tilde_logits: torch.Tensor, reduce: str = "mean", eps: float = 1e-8):
    """
    严格对应 Agile Eq.(5)：对每个 i 的 Softmax(tilde y^i) 做 IM，再对 i 聚合
    y_tilde_logits: (B, M, C)
    """
    p = F.softmax(y_tilde_logits, dim=-1)  # (B, M, C)
    return im_loss_from_probs(p, reduce=reduce, eps=eps)

# train_utils.py

def split_params(model):
    """
    将模型参数分为 gate 参数和 expert 参数
    返回两个列表，用于双优化器
    """
    net = model.pred_head  # EmbeddingNetwork

    # ---- Gate 参数：控制选择策略 ----
    gate_params = []
    # α attention
    gate_params.extend(list(net.WF.parameters()))
    gate_params.extend(list(net.WO.parameters()))
    # β attention
    gate_params.extend(list(net.WQ.parameters()))

    # ---- Expert 参数：各专家的拟合能力 ----
    expert_params = []
    # 各专家的 adapter 和 head
    expert_params.extend(list(net.adapters.parameters()))
    expert_params.extend(list(net.heads.parameters()))
    # 共享的归一化层
    expert_params.extend(list(net.normalize.parameters()))
    # 如果有全连接层
    if hasattr(net, 'fully_connected'):
        expert_params.extend(list(net.fully_connected.parameters()))
    if hasattr(net, 'output_layer'):
        expert_params.extend(list(net.output_layer.parameters()))
    if hasattr(net, 'projectors'):
        expert_params.extend(list(net.projectors.parameters()))
    return gate_params, expert_params

def init_linear_kaiming_relu(m: nn.Linear):
    nn.init.kaiming_uniform_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def init_linear_xavier(m: nn.Linear, gain: float = 1.0):
    nn.init.xavier_uniform_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def init_linear_small_normal(m: nn.Linear, std: float = 1e-3):
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def init_norm(m):
    # LayerNorm / BatchNorm1d 都按这个
    if hasattr(m, "weight") and m.weight is not None:
        nn.init.ones_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.zeros_(m.bias)

def init_orthogonal(m: nn.Linear, gain: float = 1.0):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)

def assign_grads(params, grads):
    """把 autograd.grad 返回的 grads 写回 params[i].grad；None 表示该参数本步未参与图"""
    for p, g in zip(params, grads):
        if g is None:
            p.grad = None
        else:
            p.grad = g.detach()




#better expert specialization

def expert_orthogonality_loss(feats, gate_weights):
    """
    Args:
        feats:        list[Tensor(B, d_feat)]  adapter 输出, len = M
        gate_weights: Tensor(B, M)             gate 权重 (会被 detach)
 
    Returns:
        l_ortho: scalar tensor, 梯度只流向 feats (adapter 参数)
    """
    M = len(feats)
    #w = gate_weights.detach()  # (B, M), 阻断 gate 梯度
 
    prototypes = []
    for i in range(M):
        h_norm = F.normalize(feats[i], dim=-1)      # (B, d_feat)
        wi = gate_weights[:, i]                                  # (B,)
 
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
 