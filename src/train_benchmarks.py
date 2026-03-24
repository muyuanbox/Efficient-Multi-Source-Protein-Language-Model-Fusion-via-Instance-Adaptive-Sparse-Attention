import argparse
import torch
import torch.optim as optim
from model import EmbeddingNetwork, WideEmbeddingNetwork, SequenceProjector
from main_utils import (
    load_config, override_config, set_seed, get_device, get_run_name,
    setup_logging_directories, load_embeddings, concatenate_embeddings,
    count_parameters, save_model, get_tsv_dirs
)
from train_utils import (
    get_data_loader, train_step, test_step, split_params, train_step_dual_optimizer
)
import os, csv, uuid, time
import pandas as pd
import datetime

def train(args, config, model, embedding_dict, device, log_path, checkpoint_path,seq_dict):
    gate_params, expert_params = split_params(model)
    
    # 双优化器
    optimizer_gate = optim.Adam(gate_params, lr=config['training']['lr_gate'], eps=1e-7)
    optimizer_expert = optim.Adam(expert_params, lr=config['training']['lr_expert'], eps=1e-7)
    #optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], eps=1e-7)
    criterion = torch.nn.CrossEntropyLoss() if args.dataset == 'location' else torch.nn.MSELoss()

    train_loader = get_data_loader(args.dataset, embedding_dict, config['directories']['train'], config['training']['batch_size'], True)
    val_loader = get_data_loader(args.dataset, embedding_dict, config['directories']['validation'], config['training']['batch_size'], False)
    test_loader = get_data_loader(args.dataset, embedding_dict, config['directories']['test'], config['training']['batch_size'], False)

    best_val_loss = float('inf')

    with open(log_path, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file, delimiter='\t')

        for epoch in range(1, config['training']['iteration'] + 1):
            train_loss, train_selection_counts, train_selection_samples = train_step_dual_optimizer(
                model, config, train_loader,
                optimizer_gate, optimizer_expert,
                gate_params, expert_params,
                criterion, device, seq_dict, epoch, training=True
            )
            val_loss, val_metric = test_step(model, config, val_loader, criterion, device, seq_dict, epoch)
            # 打印进度 (可选，方便终端查看)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{config['training']['iteration']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}")
                # ==========================================
                # 【核心监控逻辑】
                # ==========================================
                # 1. 计算每个模型的被选中概率 (Selection Rate)
                # k * total_samples 是所有被选中的总次数（因为每个样本选 k 个）
                # 但这里我们只关心相对占比
                selection_probs = train_selection_counts / train_selection_samples
                
                # 2. 计算熵 (Entropy): H = -sum(p * log(p))
                # 熵越大 -> 分布越均匀 (Diversity好)
                # 熵越小 -> 分布越尖锐 (可能 Collapse)
                entropy = -torch.sum(selection_probs * torch.log(selection_probs + 1e-9)).item()
                
                # 3. 格式化输出 (打印 Top-k 的分布情况)
                # 转为 list 方便打印
                probs_list = [f"{p:.2f}" for p in selection_probs.tolist()]
                
                print(f"\n--- Epoch {epoch} Router Stats ---")
                print(f"Selection Dist : {probs_list}")
                    
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, checkpoint_path)

        if args.evaluate:
            model.load_state_dict(torch.load(checkpoint_path))
            _, val_metric = test_step(model, config, val_loader, criterion, device, seq_dict, 0)
            _, test_metric = test_step(model, config, test_loader, criterion, device, seq_dict, 0)

            param_count = count_parameters(model)
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            run_time = time.time() - args.start_time
            
            log_writer.writerow([args.dataset, args.embeddings, args.hidden_dimension, args.seed, val_metric, test_metric, param_count, max_mem, run_time, args.topk, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def inference(args, config, model, embedding_dict, device, seq_dict):
    test_loader = get_data_loader(args.dataset, embedding_dict, config['directories']['test'], config['training']['batch_size'], shuffle=False)
    criterion = torch.nn.CrossEntropyLoss() if args.dataset == 'location' else torch.nn.MSELoss()

    checkpoint_path = args.model_path
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, _, all_probs = test_step(model, config, test_loader, criterion, device, seq_dict, 0, return_preds=True)

    model_id = os.path.basename(os.path.dirname(args.model_path))
    output_path = os.path.join('results','predicted_probs', args.dataset, f"{model_id}.tsv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for prob in all_probs:
            f.write(f"{prob}\n")

def main():
    parser = argparse.ArgumentParser(description="Train or run inference on pLM tasks")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--dataset", choices=['aav', 'gb1', 'gfp', 'location', 'meltome', 'stability'], required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--hidden_dimension", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--model_path", type=str, help="Path to saved model for inference")
    parser.add_argument("--project_dim", type=int, default=32)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--explore_a", type=float, default=0.3)
    parser.add_argument("--batch_size",     type=int,   default=None)
    parser.add_argument("--iteration",      type=int,   default=None)
    parser.add_argument("--lr_gate",        type=float, default=None)
    parser.add_argument("--lr_expert",      type=float, default=None)
    parser.add_argument("--learning_rate",  type=float, default=None)
    parser.add_argument("--l_exploit",      type=float, default=None)
    parser.add_argument("--l_aux",          type=float, default=None)
    parser.add_argument("--l_im_intra",     type=float, default=None)
    parser.add_argument("--l_im_gate",      type=float, default=None)
    parser.add_argument("--l_transfer",     type=float, default=None)
    parser.add_argument("--im_reduce",      type=str,   default=None, choices=["mean", "sum"])

    args = parser.parse_args()
    args.start_time = time.time()
    set_seed(args.seed)
    config = load_config(args.dataset) #读取 configs/{dataset}.yaml 中的训练/数据路径（当前光标位置说明该行的作用），以便后续加载 batch size、学习率等超参
    config = override_config(config, args)   # <-- 命令行 > yaml
    device = get_device(args.device)

    run_name = get_run_name(args.dataset, args.embeddings, args.hidden_dimension, args.seed) #生成例如 aav_B_32_2 的唯一名称，用来区分不同嵌入/seed/隐藏维度组合
    checkpoint_dir = os.path.join("model_checkpoints", args.dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = setup_logging_directories(args.dataset, run_name, args.embeddings) #根据是单一嵌入还是组合嵌入切分

    try:
        dicts = load_embeddings(args.embeddings, args.dataset)
        embedding_dict, input_dim = concatenate_embeddings(dicts) #组合后的embedding
    except ValueError as e: #如果加载失败则抛出带路径的错误；
        print(f"Error loading embeddings: {e}")
        return



    num_models = len(dicts)
    num_concat = [len(list(v.values())[0]) for v in dicts]

    #调用蛋白质读取函数生成embedding->
    tsv_path = get_tsv_dirs(args.dataset)
    df = pd.read_csv(tsv_path, sep="\t", header=None)
    protein_ids = df[0].tolist()
    seqs = df[1].tolist()
    # 1) 组合成一个 id -> 序列 的字典，方便后面用/这里根据数据集需要考虑不同的方案
    seq_dict = dict(zip(protein_ids, seqs))

    # 2) 调用整个路由模块
    model = SequenceProjector(
        project_dim=args.project_dim,
        input_dimension=input_dim,
        output_dimension=10 if args.dataset == 'location' else 1,
        hidden_dimension=args.hidden_dimension,
        dropout_rate=args.dropout,
        num_models=num_models,
        num_concat=num_concat,
        topk=args.topk,
    ).to(device)

    model_dir = f"{run_name}_{uuid.uuid4().hex[:8]}"
    checkpoint_path = os.path.join(checkpoint_dir, model_dir, "best_model.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    if args.mode == "train":
        #
        train(args, config, model, embedding_dict, device, log_path, checkpoint_path, seq_dict)
    else:
        #
        inference(args, config, model, embedding_dict, device, seq_dict)

if __name__ == "__main__":
    main()