### 控制embedding到[-L,L]范围，为了控制loss的边界，但即使不控范围，而是通过规定有限的迭代次数也是可以得到优秀结果的。



import gc
import math
import random
import time

import math
import random
import time
from torch_sparse import index_select, SparseTensor
import networkx as nx
import numpy
import numpy as np
import torch
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib
from torch.utils.checkpoint import checkpoint
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import to_dense_adj, to_torch_coo_tensor, subgraph
import os


from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix, coo_matrix
from torch import nn, optim
from GCNEmbedding import *
# from GATembedding import *
from utils import *
# from torch.nn.functional import cosine_similarity
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score, jaccard_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
os.environ["OMP_NUM_THREADS"] = "3"
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
color_red = "\033[1;31m"  # 红色字体
color_green = "\033[1;32m"  # 绿色字体
color_reset = "\033[0m"  # 恢复默认颜色





###接收脚本参数
args1 = argparse.ArgumentParser()
args1.add_argument('--data', type=str, required=True, help="Param1")
args1.add_argument('--setting', type=int, required=True, help="Param2")
args1.add_argument('--L', type=int, required=True, help="Param2")
args1.add_argument('--seed', type=int, required=True, help="Param3")
args1 = args1.parse_args()
args.dataset = args1.data
setting = args1.setting
L = args1.L
args.seed = args1.seed





torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果。
torch.cuda.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


#
datasets = [ "ca-GrQc", "ca-HepTh" ,"SC",  "crime",
              "netsci", "wiki_mi", "euroroad","twitch_gamers"]
k_values = [2, 3,  3,  2, 2, 3, 4,2]
label_rate = [0.2,0.05,0.2,0.2,0.2,0.2,0.05,0.2]
con_rate =[0.01,0.01,0.02,0.02,0.05,0.02,0.01,0.02]


# # 创建字典
data_k_map = dict(zip(datasets,k_values))
data_labelrate_map = dict(zip(datasets,label_rate))
data_conrate_map = dict(zip(datasets,con_rate))

###
num_fea = data_k_map[args.dataset]
args.rate = data_labelrate_map[args.dataset]
args.constraint_rate = data_conrate_map[args.dataset]
args.constraint_rate_neg = data_conrate_map[args.dataset]




print("num_fea:",num_fea)
print("rate",args.rate)
print("constraint_rate",args.constraint_rate)



data = load_data(args.dataset,num_fea)
data.to(device)

####合并mask得到15%节点的mask
merged_mask = data.train_mask+data.test_mask
merged_mask = numpy.array(merged_mask)
total_mask = data.train_mask+data.test_mask
total_mask = torch.tensor(total_mask).to(device)


##记录列表（部分代码的耗时）
cal_loss_list = []
kl_time = []
op_time = []


sim_type="resim"



n = data.train_node_number
pos_k = int(n * args.constraint_rate)
neg_k = int(n * args.constraint_rate_neg)





setting_configs = {
    0: "0_仅使用kl散度优化",
    1: "1_使用相似性对比学习",
    2: "2_使用标签对比学习",
    3: "3_使用相似性加标签对比学习",
    4: "4_Jaccard相似性",
    5: "5_Cosine相似性",
    6: "6_SimRank相似性",
    7: "7_node2vec相似性",
    8: "8_RoleSim相似性",
    9: "9_特征_ReEig",
    10: "10_特征_degree",
    11: "11_特征_allcentrality",
    12: "12_特征_localcentrality",
    13: "13_特征_reEig+allcentrality",
    14: "14_特征_reEig+localcentrality",
    15: "15_reEigcosine做相似性",
    16: "16_reEigdistance的倒数做相似性"
}
setting_label = setting_configs.get(setting, f"{setting}_未知配置")
print(f"[Config] setting={setting_label}")





def clustering_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    accuracy = sum(w[row_ind[i], col_ind[i]] for i in range(len(row_ind))) / y_pred.size
    return accuracy


def KL_loss(graph_emb: torch.Tensor,
                 centers: torch.Tensor,
                 eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    graph_emb : [N, d]
    centers   : [K, d]
    return    : (kl_scalar, p_ij  [N, K])
    """
    # 1. 距离矩阵 -> 一次矩阵乘法 + 广播（等价 cdist，但更快）
    #    ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    kl_start = time.time()
    # 1. 距离矩阵 -> 一次矩阵乘法 + 广播（等价 cdist，但更快）
    #    ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    x2 = graph_emb.square().sum(1, keepdim=True)  # [N,1]
    y2 = centers.square().sum(1, keepdim=True).t()  # [1,K]
    xy = graph_emb @ centers.t()  # [N,K]
    dist_sq = x2 + y2 - 2 * xy  # 无显式开方
    dist = torch.sqrt(torch.relu(dist_sq) + eps)  # 开方一次

    # 2. 相似度矩阵
    sim = 1.0 / (1.0 + dist)  # [N,K]

    # 3. 行归一化得到 q_ij
    q_ij = sim / sim.sum(1, keepdim=True)

    # 4. 列和 f_j
    f_j = sim.sum(0)  # [K]

    # 5. p_ij（一次广播完成）
    tmp = q_ij.square() / f_j  # [N,K]
    p_ij = tmp / tmp.sum(1, keepdim=True)

    # 6. KL 散度（log-sum-exp 稳定）
    kl = torch.sum(p_ij * (p_ij + eps).log() - p_ij * (q_ij + eps).log())
    kl_time.append(time.time() - kl_start)
    return kl, p_ij  


def reconstruction_loss(A, embedding):
    A_hat = torch.sigmoid(torch.matmul(embedding, embedding.T))
    r_loss = torch.norm(A - A_hat)
    return r_loss


def train_semi_supervise_clustering_model(data, centers, label_matrix, sim_matrix, label_coef=1.0):
    model.train()
    out = model(data.x, data.adj_normalized)

    kl = KL_loss(out[total_mask], centers)[0]

     ##更新后裁剪
    out = torch.clamp(out, min=-L, max=L)
    
    # 分别计算各约束分量（与原始总 loss 等价：sum(values*norm_sq) = ml - cl）
    sim_ml = sim_cl = label_ml = label_cl = torch.tensor(0.0, device=device)
    if sim_matrix is not None:
        sim_ml, sim_cl = pair_constraint_loss_split(out[total_mask], sim_matrix)
    if label_matrix is not None:
        label_ml, label_cl = pair_constraint_loss_split(out[total_mask], label_matrix)
    
    loss = kl
    gama_cl = 1
    gama_con = 1
    loss_sml = torch.tensor(0.0, device=device)
    loss_lml = torch.tensor(0.0, device=device)
    loss_scl = torch.tensor(0.0, device=device)
    loss_lcl = torch.tensor(0.0, device=device)
    if sim_matrix is not None:
      loss_sml = sim_ml
      loss_scl = sim_cl
    if label_matrix is not None:
      loss_lml = label_ml
      loss_lcl = label_cl
    
    loss_con =( loss_sml + loss_lml + loss_lcl+loss_scl )
    if setting!=0:
      loss += loss_con
     
    # 只读计算各分量梯度范数（不影响梯度流）
    comps = {'kl': kl, 'sim_ml': sim_ml, 'sim_cl': sim_cl,
             'label_ml': label_ml, 'label_cl': label_cl}
    grad_info = {}
    for name, val in comps.items():
        grad_info[name] = compute_grad_norm(val, model, retain_graph=True)
    l_con = ( loss_sml + loss_lml + loss_lcl+loss_scl )
    grad_info['con'] = compute_grad_norm(l_con, model, retain_graph=True)
    grad_info['total'] = compute_grad_norm(loss, model, retain_graph=True)

    op_start = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    op_time.append(time.time() - op_start)
    
    detail = {
        'kl': kl.item(),
        'label_ml': loss_lml.item(), 'label_cl': 0-loss_lcl.item(),
        'sim_ml': loss_sml.item(), 'sim_cl': 0-loss_scl.item(),
        'l_con': l_con.item(),
        'total_contrastive_loss': loss_con.item(),
        'total_loss': loss.item(),
        'grad_kl': grad_info['kl'],
        'grad_label_ml': grad_info['label_ml'], 'grad_label_cl': grad_info['label_cl'],
        'grad_sim_ml': grad_info['sim_ml'], 'grad_sim_cl': grad_info['sim_cl'],
        'grad_con': grad_info['con'], 'grad_total': grad_info['total'],
    }
   
    return loss, out, detail
     
    
def pre_train_model(data):
    model.train()
    out = model(data.x, data.adj_normalized)

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    adj_matrix_sparse = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(edge_index.shape[1], device=device),
        size=(num_nodes, num_nodes)
    )

    adj_matrix_dense = adj_matrix_sparse.to_dense()
    pre_loss = reconstruction_loss(adj_matrix_dense, out)
    optimizer.zero_grad()
    pre_loss.backward()
    optimizer.step()

    return pre_loss
   

@torch.no_grad()
def test_model(data):
    model.eval()
    embeddings = model(data.x,data.adj_normalized)
    return embeddings


def min_max_normalize(tensor, min_value=0.0, max_value=1.0):
    """
    对输入张量进行Max-Min归一化，将其缩放到指定范围内。

    参数:
    tensor (torch.Tensor): 需要归一化的张量
    min_value (float): 归一化后的最小值
    max_value (float): 归一化后的最大值

    返回:
    torch.Tensor: 归一化后的张量
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    normalized_tensor = normalized_tensor * (max_value - min_value) + min_value
     
    return normalized_tensor
     
    
def pair_constraint_loss_split(embedding, constraint_matrix):
    """返回 (ml_loss, cl_loss), 分别对应 must-link (+1) 和 cannot-link (-1) 的 L2 距离和"""
    constraint_matrix = constraint_matrix.to_torch_sparse_coo_tensor()
    constraint_sparse = constraint_matrix.coalesce().to(device)
    indices = constraint_sparse.indices()
    values = constraint_sparse.values()

    row, col = indices[0], indices[1]
    diff = embedding[row] - embedding[col]
    norm_sq = torch.sum(diff ** 2, dim=1)
    # norm_sq = torch.sqrt(norm_sq + 1e-6)
    ml_mask= values > 0
    cl_mask = values < 0
    
    ml_loss = torch.sum(norm_sq[ml_mask]) if ml_mask.any() else torch.tensor(0.0, device=device)
    cl_loss = 0-torch.sum(norm_sq[cl_mask]) if cl_mask.any() else torch.tensor(0.0, device=device)
    # ml_loss = torch.mean(norm_sq[ml_mask]) if ml_mask.any() else torch.tensor(0.0, device=device)
    # cl_loss = torch.mean(norm_sq[cl_mask]) if cl_mask.any() else torch.tensor(0.0, device=device)
    
    return ml_loss, cl_loss

@torch.no_grad()
def compute_grad_norm(loss_val, model, retain_graph=True):
    """只读计算 loss 关于 model 参数的梯度 L2 范数，不修改梯度"""
    if not (isinstance(loss_val, torch.Tensor) and loss_val.requires_grad and loss_val.grad_fn is not None):
        return 0.0
    grads = torch.autograd.grad(loss_val, model.parameters(), retain_graph=retain_graph, allow_unused=True)
    valid = [g.view(-1) for g in grads if g is not None]
    if not valid:
        return 0.0
    return torch.norm(torch.cat(valid)).item()


def capture_param_stats(model):
    """保存训练前参数快照 {name: tensor}，只读不参与计算图"""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


@torch.no_grad()
def log_model_update(pre_snapshot, model, dataset, setting, epoch, lr, csv_path, write_header):
    """训练后对比快照，记录每层参数更新量 + 梯度 + 参数分布"""
    records = []
    for name, p in model.named_parameters():
        delta = p - pre_snapshot[name]
        grad = p.grad

        has_nan = bool(torch.isnan(p).any())
        has_inf = bool(torch.isinf(p).any())

        row = {
            'time': time.strftime("%Y_%m_%d_%H_%M_%S"),
            'dataset': dataset,
            'setting': setting,
            'epoch': epoch,
            'lr': lr,
            'layer': name,
            'update_norm': torch.norm(delta.view(-1)).item(),
            'grad_norm': torch.norm(grad.view(-1)).item() if grad is not None else 0.0,
            'param_mean': p.mean().item(),
            'param_std': p.std().item(),
            'param_min': p.min().item(),
            'param_max': p.max().item(),
            'grad_mean': grad.mean().item() if grad is not None else 0.0,
            'grad_max_abs': grad.abs().max().item() if grad is not None else 0.0,
            'has_nan': has_nan,
            'has_inf': has_inf,
        }
        records.append(row)

    pd.DataFrame(records).to_csv(csv_path, mode='a', index=False, header=write_header)


def Constraint_matrix():
    """
    完全向量化构造约束稀疏矩阵，无 Python 层 for-loop。

    参数
    ----
    regular_simi_df : pd.DataFrame 或 np.ndarray  (n,n)
    train_label     : 1-D array-like 长度 n
    total_node      : set / list / 1-D array  允许建立约束的节点编号
    constraint_rate : float   控制 top-k/bottom-k 比例
    device          : torch.device

    返回
    ----
    torch.sparse_coo_tensor  (n,n)
    """
    regular_simi = data.regular_simi
    train_label = data.train_label
    total_node = data.total_node
    
    #### 相似性选择
    if setting not in [4,5,6,7,8,15,16]:
        # if args.dataset!="wiki_mi": 
        if args.dataset=="twitch_gamers":
            # 规则等价相似性
            # 1. 保留数值列并填充缺失
            regular_similarity_matrix = regular_simi.select_dtypes(include='number').fillna(0)

            # 2. 转 numpy
            regular_similarity_matrix = regular_similarity_matrix.to_numpy(dtype=np.float32)  # 或 .values
        
            # 3. 转 tensor
            sim = torch.from_numpy( regular_similarity_matrix).to(device)
        else:
            regular_simi = regular_simi[total_mask]
            sim = regular_simi[:,total_mask]

    
    if setting==4:
        ## Jaccard相似性
        # 基于全图结构计算Jaccard相似度: J(i,j) = |N(i)∩N(j)| / |N(i)∪N(j)|
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        merged_idx = np.where(merged_mask)[0]
        n_merged = len(merged_idx)

        edge_np = edge_index.cpu().numpy()
        e0, e1 = edge_np[0], edge_np[1]

        # 全图邻接矩阵 [num_nodes, num_nodes]
        adj_full = coo_matrix((np.ones(len(e0)), (e0, e1)), shape=(num_nodes, num_nodes)).tocsr()
        # 取merged节点的行，邻域来自全图
        adj_m = adj_full[merged_idx]  # [n_merged, num_nodes]

        common = adj_m.dot(adj_m.T)   # [n_merged, n_merged], 共同邻居数
        deg = np.array(adj_m.sum(axis=1)).flatten().astype(np.float32)

        common_dense = common.toarray()
        union = deg[:, None] + deg[None, :] - common_dense
        with np.errstate(divide='ignore', invalid='ignore'):
            jaccard = np.where(union > 0, common_dense / union, 0.0)
        np.fill_diagonal(jaccard, 1.0)

        sim = torch.tensor(jaccard, dtype=torch.float32, device=device)

    if setting==5:
    # Cosine相似性
    # 基于邻接向量的余弦相似度: cos(A,B) = |N(A)∩N(B)| / sqrt(|N(A)|*|N(B)|)
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        merged_idx = np.where(merged_mask)[0]
        n_merged = len(merged_idx)
        
        edge_np = edge_index.cpu().numpy()
        e0, e1 = edge_np[0], edge_np[1]

        adj_full = coo_matrix((np.ones(len(e0)), (e0, e1)), shape=(num_nodes, num_nodes)).tocsr()
        adj_m = adj_full[merged_idx]

        common = adj_m.dot(adj_m.T)
        deg = np.array(adj_m.sum(axis=1)).flatten().astype(np.float32)

        common_dense = common.toarray()
        denom = np.sqrt(deg[:, None] * deg[None, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_sim = np.where(denom > 0, common_dense / denom, 0.0)
        np.fill_diagonal(cos_sim, 1.0)
        sim = torch.tensor(cos_sim, dtype=torch.float32, device=device)

    if setting==6:
    # SimRank
    # 基于全图结构的SimRank相似度, 最后取merged子集
    # S(a,b) = C/|N(a)|/|N(b)| * Sum_{x∈N(a)} Sum_{y∈N(b)} S(x,y)
        simi_dir = os.path.join(os.path.dirname(__file__), 'similarity', args.dataset)
        os.makedirs(simi_dir, exist_ok=True)
        simi_file = os.path.join(simi_dir, f"{args.dataset}_simrank.npy")
        if os.path.exists(simi_file):
            sim = torch.tensor(np.load(simi_file), dtype=torch.float32, device=device)
            print(f"[Similarity] 读取缓存 SimRank: {simi_file}")
        else:
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            merged_idx = np.where(merged_mask)[0]
            n_merged = len(merged_idx)

            edge_np = edge_index.cpu().numpy()
            e0, e1 = edge_np[0], edge_np[1]

            # 全图邻接矩阵 [num_nodes, num_nodes]
            adj_full = coo_matrix((np.ones(len(e0)), (e0, e1)), shape=(num_nodes, num_nodes)).tocsr()
            deg = np.array(adj_full.sum(axis=1)).flatten().astype(np.float32)
            deg_inv = np.where(deg > 1e-8, 1.0 / deg, 0.0)
            D_inv = coo_matrix((deg_inv, (range(num_nodes), range(num_nodes))), shape=(num_nodes, num_nodes)).tocsr()
            A_norm = D_inv @ adj_full

            C = 0.8
            max_iter = 50
            tol = 1e-4
            S = np.eye(num_nodes, dtype=np.float32)
            for i in range(max_iter):
                S_new = C * (A_norm @ S @ A_norm.T)
                np.fill_diagonal(S_new, 1.0)
                change = np.max(np.abs(S_new - S))
                S = S_new
                if change < tol:
                    print(f"SimRank converged at iteration {i+1}, change={change:.6e}")
                    break
            else:
                print(f"SimRank reached max_iter={max_iter}, change={change:.6e}")
            # 取merged节点子集 [n_merged, n_merged]
            sim = torch.tensor(S[merged_idx][:, merged_idx], dtype=torch.float32, device=device)
            np.save(simi_file, sim.cpu().numpy())
            print(f"[Similarity] 保存 SimRank: {simi_file}")

    if setting==7:
    ## node2vec
    # 用node2vec学习节点embedding，再计算余弦相似度矩阵
        simi_dir = os.path.join(os.path.dirname(__file__), 'similarity', args.dataset)
        os.makedirs(simi_dir, exist_ok=True)
        simi_file = os.path.join(simi_dir, f"{args.dataset}_node2vec.npy")
        if os.path.exists(simi_file):
            sim = torch.tensor(np.load(simi_file), dtype=torch.float32, device=device)
            print(f"[Similarity] 读取缓存 node2vec: {simi_file}")
        else:
            from torch_geometric.nn import Node2Vec as PyGNode2Vec
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            merged_idx = np.where(merged_mask)[0]
            n_merged = len(merged_idx)

            walk_length = 20
            walks_per_node = 10
            p = 1.0
            q = 1.0
            embedding_dim = 128
            context_size = 10
            num_neg_samples = 5
            epochs = 50
            lr = 0.01

            model = PyGNode2Vec(
                edge_index,
                embedding_dim=embedding_dim,
                walk_length=walk_length,
                context_size=context_size,
                walks_per_node=walks_per_node,
                num_negative_samples=num_neg_samples,
                p=p,
                q=q,
                num_nodes=num_nodes,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loader = model.loader(batch_size=128, shuffle=True, num_workers=0)

            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 10 == 0:
                    print(f"Node2Vec epoch {epoch}/{epochs}, loss={total_loss:.4f}")

            model.eval()
            with torch.no_grad():
                z = model()                                   # (num_nodes, embedding_dim)
                z_norm = z / z.norm(dim=1, keepdim=True).clamp(min=1e-12)
            sim = (z_norm @ z_norm.T)[merged_idx][:, merged_idx]
            np.save(simi_file, sim.cpu().numpy())
            print(f"[Similarity] 保存 node2vec: {simi_file}")

    if setting==8:
    #RoleSim
    # R(a,b) = (1-β) * Σ_{x∈N(a)} max_{y∈N(b)} R(x,y) / max(|N(a)|,|N(b)|) + β
        simi_dir = os.path.join(os.path.dirname(__file__), 'similarity', args.dataset)
        os.makedirs(simi_dir, exist_ok=True)
        simi_file = os.path.join(simi_dir, f"{args.dataset}_rolesim.npy")
        if os.path.exists(simi_file):
            sim = torch.tensor(np.load(simi_file), dtype=torch.float32, device=device)
            print(f"[Similarity] 读取缓存 RoleSim: {simi_file}")
        else:
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            merged_idx = np.where(merged_mask)[0]
            n_merged = len(merged_idx)

            edge_np = edge_index.cpu().numpy()
            e0, e1 = edge_np[0], edge_np[1]

            # 全图邻接矩阵 [num_nodes, num_nodes]
            adj_full = coo_matrix((np.ones(len(e0)), (e0, e1)), shape=(num_nodes, num_nodes)).tocsr()
            deg = np.array(adj_full.sum(axis=1)).flatten().astype(np.float32)
            adj_bool = adj_full.astype(bool)

            beta = 0.15
            max_iter = 50
            tol = 1e-4

            R = np.eye(num_nodes, dtype=np.float32)
            for iteration in range(max_iter):
                R_new = np.zeros_like(R)
                for j in range(num_nodes):
                    if deg[j] < 0.5:
                        R_new[:, j] = beta
                        continue
                    nj = adj_bool[:, j].nonzero()[0]               # N(j)
                    col_max = R[:, nj].max(axis=1)                 # (n,)  max_{y∈N(j)} R(x,y)
                    numer   = adj_bool @ col_max                   # (n,)  Σ_{x∈N(i)} ...
                    denom   = np.maximum(deg, deg[j])
                    R_new[:, j] = (1 - beta) * numer / denom + beta

                np.fill_diagonal(R_new, 1.0)
                change = np.max(np.abs(R_new - R))
                R = R_new
                if change < tol:
                    print(f"RoleSim converged at iteration {iteration+1}, change={change:.6e}")
                    break
            else:
                print(f"RoleSim reached max_iter={max_iter}, change={change:.6e}")
            # 取merged节点子集 [n_merged, n_merged]
            sim = torch.tensor(R[merged_idx][:, merged_idx], dtype=torch.float32, device=device)
            np.save(simi_file, sim.cpu().numpy())
            print(f"[Similarity] 保存 RoleSim: {simi_file}")

    if setting==15:
       #reeig余弦相似性
        x = data.x[merged_mask]
        cosine_similarity_matrix = cosine_similarity(x.cpu())
        cosine_similarity_matrix = torch.tensor(cosine_similarity_matrix)
        sim = cosine_similarity_matrix.to(device)
    if setting==16:
        ###距离转相似性 1/(1+d)
        x = data.x[merged_mask]
        distance_matrix = euclidean_distances(x.cpu())
        distance_matrix = torch.tensor(distance_matrix)
        sim = (1.0 / (1.0 + distance_matrix)).to(device)



    ####对生成的sim的统计检验
    # 对每行 sim 的非零值计数，统计非零数不足节点数5%的行所占比例并打印
    _row_nz = (sim != 0).sum(dim=1)                        # 每行非零个数 [n]
    _n_rows = sim.size(0)
    _thresh = max(1, int(_n_rows * 0.05))                  # 5% 节点数
    _sparse = int((_row_nz < _thresh).sum().item())        # 非零数不足5%的行数
    _mean_nz = _row_nz.float().mean().item()               # 每行非零个数平均
    _mean_ratio = (_row_nz.float() / _n_rows).mean().item()  # 每行非零比例平均
    print(f"[Similarity] 统计检验: 总行数={_n_rows}, 非零不足5%({_thresh}个)的行数={_sparse}, 比例={_sparse / _n_rows:.4f}, 非零个数平均={_mean_nz:.4f}, 非零比例平均={_mean_ratio:.4f}")
    



    ##后续处理
    sim = sim.to(device)
    n = sim.size(0)
    label = torch.as_tensor(train_label, device=device)
    total_node = torch.as_tensor(list(total_node), device=device)
    
    
    # 排除自环后再选，保证每节点恰好 pos_k / neg_k 个
    sim_pos = sim.clone(); sim_pos.fill_diagonal_(float('-inf'))
    _, topk_idx = sim_pos.topk(pos_k, dim=1, largest=True)    # (n, pos_k)
    sim_neg = sim.clone(); sim_neg.fill_diagonal_(float('inf'))
    _, bot_idx  = sim_neg.topk(neg_k, dim=1, largest=False)   # (n, neg_k)
    
    
    
    # 自环已排除，mask 全部有效（保留 != 检查作为安全兜底）
    arange_n = torch.arange(n, device=device)[:, None]
    top_mask = (topk_idx != arange_n)
    bot_mask = (bot_idx  != arange_n)

    ##打印输出实际选择的正负样本个数（每个节点平均）
    pos_per_node = top_mask.sum(dim=1).float().mean().item()
    neg_per_node = bot_mask.sum(dim=1).float().mean().item()
    print(f"[Constraint] pos_k={pos_k}, avg positive per node: {pos_per_node:.2f}  |  neg_k={neg_k}, avg negative per node: {neg_per_node:.2f}")
     
    # 4. 构造「行-列」坐标
    row_top = arange_n.expand(-1, pos_k)[top_mask]        # 1-D
    col_top = topk_idx[top_mask]
    row_bot = arange_n.expand(-1, neg_k)[bot_mask]
    col_bot = bot_idx[bot_mask]

    # 5. 各种 mask（全部 torch.bool）
    in_total   = torch.isin(torch.arange(n, device=device), total_node)
    node_mask  = in_total[:, None] & in_total[None, :]    # (n,n)
    label_eq   = (label[:, None] == label[None, :])       # (n,n)
    
    # 6. 稀疏 mask
    top_sparse    = torch.zeros(n, n, dtype=torch.bool, device=device)
    bottom_sparse = torch.zeros(n, n, dtype=torch.bool, device=device)
    top_sparse[row_top, col_top] = True
    bottom_sparse[row_bot, col_bot] = True

    # 7. 合并规则（统一去除自环）
    eye_mask = torch.eye(n, dtype=torch.bool, device=device)
    mask_pos = label_eq & node_mask & ~eye_mask
    mask_neg = (~label_eq) & node_mask & ~eye_mask
    mask_top = (~node_mask) & top_sparse & ~eye_mask
    mask_bot = (~node_mask) & bottom_sparse & ~eye_mask
    
    # 8. 转成 scipy COO（保证稀疏）
    def to_coo(torch_mask, val):
        rc = torch_mask.nonzero(as_tuple=False).cpu().numpy()  # (nnz,2)
        return coo_matrix((np.full(rc.shape[0], val, dtype=np.float32),
                          (rc[:, 0], rc[:, 1])),
                          shape=(n, n))
    
    coo_pos = to_coo(mask_pos,  1)
    coo_neg = to_coo(mask_neg, -1)
    coo_top = to_coo(mask_top,  1)
    coo_bot = to_coo(mask_bot, -1)
    
    if setting == 0:
        final_label = None
        final_sim = None
        print("use none")
    elif setting==1:
        final_label = None
        final_sim = (coo_top + coo_bot).tocoo()
        print("use sim")
    elif setting==2:
        final_label = (coo_pos + coo_neg).tocoo()
        final_sim = None
        print("use label")
    elif setting==3:
        final_label = (coo_pos + coo_neg).tocoo()
        final_sim = (coo_top + coo_bot).tocoo()
        print("use label + sim")
    else:
        final_label = (coo_pos + coo_neg).tocoo()
        final_sim = (coo_top + coo_bot).tocoo()
        print("use label + sim")
    
    ## 关于生成的约束矩阵相关信息统计
    # === 约束矩阵统计分析 ===
    from scipy.stats import mannwhitneyu
    import pandas as pd

    # must-link: mask_top (高相似跨类)
    # cannot-link:  mask_bot (低相似跨类)
    # ml_mask = mask_pos      # 保留 node_mask 影响
    # cl_mask = mask_neg 
    ml_mask = mask_top      # 保留 node_mask 影响
    cl_mask = mask_bot      # 保留 node_mask 影响
    
    # SIR差值 = |SIR[i] - SIR[j]|，从SIR文件读取
    sir_path = os.path.join(os.path.dirname(__file__), '..','..','..', '..', 'data', 'influ_score', f'{args.dataset}-graph_SIR.txt')
    sir_dict = {}
    with open(sir_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                sir_dict[int(parts[0])] = float(parts[1])
    merged_idx_local = np.where(merged_mask)[0]
    sir_vals = np.array([sir_dict.get(idx, 0.0) for idx in merged_idx_local], dtype=np.float32)
     
    ml_idx = ml_mask.nonzero(as_tuple=False)
    ml_r = ml_idx[:, 0].cpu().numpy()       
    ml_c = ml_idx[:, 1].cpu().numpy()       
    s_ml = np.abs(sir_vals[ml_r] - sir_vals[ml_c])    
     
    cl_idx = cl_mask.nonzero(as_tuple=False)
    cl_r = cl_idx[:, 0].cpu().numpy()
    cl_c = cl_idx[:, 1].cpu().numpy()
    s_cl = np.abs(sir_vals[cl_r] - sir_vals[cl_c])
     
    # 均值
    mean_ml = float(s_ml.mean()) if len(s_ml) > 0 else 0.0
    mean_cl = float(s_cl.mean()) if len(s_cl) > 0 else 0.0
    gap = mean_ml - mean_cl
    
    # Mann-Whitney U test
    if len(s_ml) > 0 and len(s_cl) > 0:
        u_stat, p_value = mannwhitneyu(s_ml, s_cl, alternative="less")
    else:
        u_stat, p_value = 0.0, 1.0
    
    # 用节点真实标签判断约束对的同异类
    ml_r_t, ml_c_t = ml_mask.nonzero(as_tuple=False).unbind(dim=1)
    ml_total = ml_r_t.size(0)
    ml_same = int((label[ml_r_t] == label[ml_c_t]).sum().item())
    ml_diff = ml_total - ml_same

    cl_r_t, cl_c_t = cl_mask.nonzero(as_tuple=False).unbind(dim=1)
    cl_total = cl_r_t.size(0)
    cl_same = int((label[cl_r_t] == label[cl_c_t]).sum().item())
    cl_diff = cl_total - cl_same

    # 按标签 0/1 计算正负样本准确率
    ml_lbl0 = (label[ml_r_t] == 0)
    ml_lbl1 = (label[ml_r_t] == 1)
    pos_acc_l0 = (label[ml_r_t[ml_lbl0]] == label[ml_c_t[ml_lbl0]]).sum().item() / ml_lbl0.sum().item() if ml_lbl0.sum() > 0 else 0.0
    pos_acc_l1 = (label[ml_r_t[ml_lbl1]] == label[ml_c_t[ml_lbl1]]).sum().item() / ml_lbl1.sum().item() if ml_lbl1.sum() > 0 else 0.0
    cl_lbl0 = (label[cl_r_t] == 0)
    cl_lbl1 = (label[cl_r_t] == 1)
    neg_acc_l0 = (label[cl_r_t[cl_lbl0]] != label[cl_c_t[cl_lbl0]]).sum().item() / cl_lbl0.sum().item() if cl_lbl0.sum() > 0 else 0.0
    neg_acc_l1 = (label[cl_r_t[cl_lbl1]] != label[cl_c_t[cl_lbl1]]).sum().item() / cl_lbl1.sum().item() if cl_lbl1.sum() > 0 else 0.0
    
    row = {
        "dataset": args.dataset,
        "constraint_rate": args.constraint_rate,
        "n_nodes": n,
        "n_must_link": ml_total,
        "n_cannot_link": cl_total,
        "mean_S_ML": round(mean_ml, 6),
        "mean_S_CL": round(mean_cl, 6),
        "gap": round(gap, 6),
        "MannWhitneyU_stat": round(float(u_stat), 4),
        "p_value": f"{p_value:.2e}",
        "ML_same_ratio": round(ml_same / ml_total, 4) if ml_total > 0 else 0.0,
        "ML_diff_ratio": round(ml_diff / ml_total, 4) if ml_total > 0 else 0.0,
        "CL_diff_ratio": round(cl_diff / cl_total, 4) if cl_total > 0 else 0.0,
        "CL_same_ratio": round(cl_same / cl_total, 4) if cl_total > 0 else 0.0,
        "pos_acc_label0": round(pos_acc_l0, 4),
        "pos_acc_label1": round(pos_acc_l1, 4),
        "neg_acc_label0": round(neg_acc_l0, 4),
        "neg_acc_label1": round(neg_acc_l1, 4),
        "time": time.strftime("%Y_%m_%d_%H_%M_%S"),
    }
    df_new = pd.DataFrame([row])

    save_dir = os.path.join(os.path.dirname(__file__), 'experiment_result', 'constrain_matrix_message', time.strftime("%Y_%m_%d"))
    os.makedirs(save_dir, exist_ok=True)
    time_suffix = time.strftime("%Y_%m_%d")
    save_path = os.path.join(save_dir, f"constraint_stats_{time_suffix}_s{setting}.csv")
    write_header = not os.path.exists(save_path)
    df_new.to_csv(save_path, mode='a', index=False, header=write_header)
    print(f"[Stats] dataset={args.dataset}, mean_ML={mean_ml:.4f}, mean_CL={mean_cl:.4f}, "
          f"gap={gap:.4f}, p-value={p_value:.2e}")
    print(f"[Stats] must-link: {ml_total} (mask_top); cannot-link: {cl_total} (mask_bot)")
      
 
    ### 记录每个数据集基于标签得到的约束个数和基于相似性得到的约束个数
    label_must = mask_pos.sum().item()
    label_cannot = mask_neg.sum().item()
    simi_must = mask_top.sum().item()
    simi_cannot = mask_bot.sum().item()
    if setting == 3:
        label_total = label_must + label_cannot
        simi_total = simi_must + simi_cannot
        label_coef_val = 2 * simi_total / label_total if label_total > 0 else 1.0
    else:
        label_coef_val = 1.0
    con_row = {
        "dataset": args.dataset,
        "label_must_link": label_must,
        "label_cannot_link": label_cannot,
        "label_total": label_must + label_cannot,
        "simi_must_link": simi_must,
        "simi_cannot_link": simi_cannot,
        "simi_total": simi_must + simi_cannot,
        "label_coef_val": label_coef_val,
        "label_rate": args.rate,
        "constraint_rate": args.constraint_rate,
        "total_node_count": len(data.total_node),
        "total_train_node_number": n,
        "time": time.strftime("%Y_%m_%d_%H_%M_%S"),
    }
    con_df = pd.DataFrame([con_row])
    con_save_dir = os.path.join(os.path.dirname(__file__),  'experiment_result', 'constrain_matrix_message', time.strftime("%Y_%m_%d"))
    os.makedirs(con_save_dir, exist_ok=True)
    con_save_path = os.path.join(con_save_dir, f"constraint_count_s{setting}_{time_suffix}.csv")
    con_header = not os.path.exists(con_save_path)
    con_df.to_csv(con_save_path, mode='a', index=False, header=con_header)
    print(f"[ConstraintCount] {args.dataset}: label=({label_must},{label_cannot}), simi=({simi_must},{simi_cannot})")

    print(f"[ConstraintCount] label_coef = {label_coef_val:.4f}")
      


    # 9. 转回 torch sparse（分别构建标签和相似性矩阵）
    if final_label is not None:
        indices_l = torch.from_numpy(np.vstack([final_label.row, final_label.col])).long().to(device)
        values_l = torch.from_numpy(final_label.data).to(device)
        label_sparse = torch.sparse_coo_tensor(indices_l, values_l, (n, n), device=device)
    else:
        label_sparse = None

    if final_sim is not None:
        indices_s = torch.from_numpy(np.vstack([final_sim.row, final_sim.col])).long().to(device)
        values_s = torch.from_numpy(final_sim.data).to(device)
        sim_sparse = torch.sparse_coo_tensor(indices_s, values_s, (n, n), device=device)
    else:
        sim_sparse = None

    return label_sparse, sim_sparse, label_coef_val

def pre_train_model_big(sampler):
    model.train()
    total_loss = 0

    count = 0
    for batch_size, n_id, adjs in sampler:
        print("次数：",count)
        count = count + 1
        x_full = data.x[n_id].to(device)  # [N_sub, F]
        edge_index_full = adjs[-1][0].to(device)

        # 1. 只保留前 batch_size 个节点的边
        node_mask = torch.zeros(x_full.size(0), dtype=torch.bool, device=device)
        node_mask[:batch_size] = True
        edge_index, _ = subgraph(node_mask, edge_index_full,
                                 relabel_nodes=True, num_nodes=x_full.size(0))

        # 2. 特征也截断
        x = x_full[:batch_size]  # [batch_size, F]
        adj = to_torch_coo_tensor(edge_index, size=(batch_size, batch_size))
        # 1. 加自环
        loop_index = torch.arange(batch_size, device=device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+V]

        # 3. 计算度向量
        deg = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 孤立点

        # 4. 计算归一化系数：每条边 (i,j) 对应 deg_inv_sqrt[i]*deg_inv_sqrt[j]
        row, col = edge_index
        norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E+V]

        # 5. 重新组装稀疏张量
        normalized_adj_sparse = torch.sparse_coo_tensor(
            indices=edge_index,
            values=norm_vals,
            size=(batch_size, batch_size),
            device=device)
         



        # 3. 前向
        z = model(x, normalized_adj_sparse)
        # 5. 计算损失并回传
        loss = reconstruction_loss(adj.to_dense(), z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size

    return total_loss


def split_mask(mask: torch.Tensor, n_parts: int):
    """
    将一维 bool 型 mask 均分成 n_parts 份，每份长度与原 mask 相同，
    但只在对应区间保留原 mask 为 True 的元素，其余置 False。

    参数
    ----
    mask   : torch.Tensor，一维，dtype=torch.bool
    n_parts: int，要分成的份数

    返回
    ----
    list[torch.Tensor]，长度为 n_parts，每个元素都是与原 mask 同形状的 bool 张量
    """
    if mask.ndim != 1:
        raise ValueError("mask 必须是一维张量")
    if n_parts <= 0:
        raise ValueError("n_parts 必须是正整数")

    total = mask.size(0)
    step  = (total + n_parts - 1) // n_parts   # 向上取整，保证最后一份不越界

    # 找到所有 True 的索引
    true_idx = torch.where(mask)[0]

    sub_masks = []
    for i in range(n_parts):
        start = i * step
        end   = min((i + 1) * step, total)
        # 落在 [start, end) 区间的 True 索引
        keep = true_idx[(true_idx >= start) & (true_idx < end)]
        # 构造子 mask，先全 False，再把选中的位置置 True
        sub = torch.zeros_like(mask)
        sub[keep] = True
        sub_masks.append(sub)

    return sub_masks
def train_semi_supervise_clustering_model_big(data, centers, label_matrix, sim_matrix, label_coef=1.0):

    model.train()
    count_loss = float('inf')
    # 合并 label_matrix 和 sim_matrix 为 constraint_matrix
    if label_matrix is not None and sim_matrix is not None:
        constraint_matrix = label_matrix + sim_matrix
    elif label_matrix is not None:
        constraint_matrix = label_matrix
    elif sim_matrix is not None:
        constraint_matrix = sim_matrix
    else:
        constraint_matrix = None
    sub_masks = split_mask(total_mask,4)
    loss_total = 0
    for i, mask in enumerate(sub_masks):
        print("i:",i)
        out = model(data.x, data.adj_normalized)
        out = out[mask]
        if setting!=0:
            c_mask = mask[total_mask]
            row_mask = c_mask  # 长度 = 原矩阵行数
            col_mask = c_mask  # 如果你要按同样规则筛列


            rows_to_keep = row_mask.nonzero(as_tuple=False).squeeze(1)  # (k,) LongTensor
            cols_to_keep = col_mask.nonzero(as_tuple=False).squeeze(1)  # (m,) LongTensor

            # print('矩阵行数:', constraint_matrix.size(0))
            # print('rows_to_keep 最大:', rows_to_keep.max().item())
            # print('rows_to_keep 最小:', rows_to_keep.min().item())

            constraint_matrix_choose = index_select(constraint_matrix, 0, rows_to_keep)  # 先筛行
            constraint_matrix_choose = index_select(constraint_matrix_choose, 1, cols_to_keep)
            
            ml_loss, cl_loss =  pair_constraint_loss_split(out, constraint_matrix_choose)
            kl = KL_loss(out, centers)[0]

            loss = kl + ml_loss + cl_loss
        else:
            kl = KL_loss(out, centers)[0]
            loss = kl 
        print("loss:",loss)
        loss_total+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    out = model(data.x, data.adj_normalized)
    detail = {
            'kl': 0,
            'label_ml': 0, 'label_cl': 0,
            'sim_ml': 0, 'sim_cl': 0,
            'l_con': 0,
            'total_contrastive_loss': 0,
            'total_loss': 0,
            'grad_kl': 0.0,
            'grad_label_ml': 0.0, 'grad_label_cl': 0.0,
            'grad_sim_ml': 0.0, 'grad_sim_cl': 0.0,
            'grad_con': 0.0, 'grad_total': 0.0,
    }

    return loss_total, out,detail

def pair_constraint_loss( choose_embedding, constraint_matrix):
    cal_start = time.time()
    constraint_matrix = constraint_matrix.to_torch_sparse_coo_tensor()
    constraint_sparse = constraint_matrix.coalesce().to(device)
    indices = constraint_sparse.indices()  # [2, nnz]
    values = constraint_sparse.values()
    
    # choose_embedding = embedding[total_mask]  # [N, F]
    N = choose_embedding.size(0)

    # 1. 只计算“稀疏张量里出现过的行、列”对应的范数
    row, col = indices[0], indices[1]  # [nnz]
    diff = choose_embedding[row] - choose_embedding[col]  # [nnz, F]
    norm_sq = torch.sum(diff ** 2, dim=1)  # [nnz]  L2^2

    # 2. 与非零值加权求和
    constraint_loss = torch.sum(values * norm_sq)

    
    cal_end = time.time()
    cal_loss_list.append(cal_end-cal_start)
    # return total_loss
    return constraint_loss

if args.dataset == "crime":
    args.pre_learning_rate=0.00008 
    args.learning_rate=0.00008

if args.dataset == "euroroad":
    args.pre_learning_rate=0.0008
    args.learning_rate=0.0008

if args.dataset == "ca-GrQc":
    args.learning_rate=0.000008
    args.pre_learning_rate=0.000008

if args.dataset == "netsci":
    args.learning_rate=0.0008
    args.pre_learning_rate=0.0008

if args.dataset=="wiki_mi":
    args.pre_learning_rate = 0.0015
    args.learning_rate = 0.0015

if __name__ == '__main__':

    """#########  Cluster control  ############"""
    cluster_number = 2
    centriods = torch.rand((cluster_number, 128), dtype=torch.float).to(device)
    con_start = time.time()
    constraint_matrix_label, constraint_matrix_sim, label_coef_val = Constraint_matrix()
    constraint_time = time.time() - con_start
    print(f"[Config] label_coef = {label_coef_val:.4f}")
    
    ### 转换稀疏格式
    def to_sparse_tensor(coo_matrix):
        return SparseTensor.from_edge_index(
            coo_matrix.indices(),
            coo_matrix.values(),
            sparse_sizes=coo_matrix.shape).to(device)
     
    if constraint_matrix_label is not None:
        constraint_matrix_label = constraint_matrix_label.coalesce()
        constraint_matrix_label = to_sparse_tensor(constraint_matrix_label)
    if constraint_matrix_sim is not None:
        constraint_matrix_sim = constraint_matrix_sim.coalesce()
        constraint_matrix_sim = to_sparse_tensor(constraint_matrix_sim)
    del data.regular_simi
    data.regular_simi = None
    gc.collect()
    torch.cuda.empty_cache()
   
     
    '''################  Feature reconstruction #############'''

    data.x = torch.cat([data.x, data.added_fea], dim=-1)
    
    
   
    
    
    
     
    data_label = data.label.cpu().numpy().astype(int)
    print("fea:", data.x)
    """###############   Initial model ##########"""
    if args.type == "Binary":
        # model = GATemb(in_channels=1,out_channels=2,hidden_channels=128).to(device)
        model = GCNemb(num_node_features=data.x.shape[1] , output_dim=2, hidden_dim=128, cluster_num=4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # pre_optimizer = optim.Adam(model.parameters(), lr=args.pre_learning_rate, weight_decay=args.weight_decay)
        # optimizer = optim.SGD(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)
    print(model)

    loss_set = []
    pretrain_losses = []

    ### 选出15%节点带了标签的节点的mask，针对后续额外测试
    length = data.num_nodes  # 指定长度
    unlabel_mask = torch.ones(length, dtype=torch.bool)
    label_data = data.total_node
    label_data.sort()
    unlabel_mask[label_data] = False
    """##############  Sample construction  ###############"""
    batch_size = 40000
    num_neighbors = [15, 10]
    
    sampler = NeighborSampler(data.edge_index,
                              sizes=num_neighbors,
                              batch_size=batch_size,
                              shuffle=True,
                              num_nodes=data.num_nodes,
                              num_workers=0)
    
    """#########  training control ###########"""
    
       
    
    pre_train_flag = 1
    train_flag = 1
    if args.dataset=="twitch_gamers":
        load_path = os.path.join(os.path.dirname(__file__),  'pretrain', args.dataset, f'pre_train_model_{args.pre_learning_rate}.pth')
        if os.path.exists(load_path):
            pre_train_flag = 0
            print(f"预训练模型已存在，跳过预训练，直接加载: {load_path}")
    
    
    kmeans_time = []
    count_loss = float('inf')
    pre_train = time.time()
    if pre_train_flag == 1:
        for i in range(100):
            if args.dataset=='twitch_gamers':
                pre_loss = pre_train_model_big(sampler)
            else:
                pre_loss = pre_train_model(data)
            print(f'pretrain loss: {pre_loss}')
            pretrain_losses.append(pre_loss.item() if hasattr(pre_loss, 'item') else pre_loss)
            if count_loss > pre_loss:
                save_path = os.path.join(os.path.dirname(__file__),  'pretrain', args.dataset)
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, f'pre_train_model_{args.pre_learning_rate}.pth'))
    pre_time = time.time()-pre_train
    print("pre_time:",pre_time)
    load_path = os.path.join(os.path.dirname(__file__),  'pretrain', args.dataset, f'pre_train_model_{args.pre_learning_rate}.pth')
    model.load_state_dict(torch.load(load_path))
    mode = 'kmeans'
    train_start = time.time()
    # loss detail CSV
    loss_detail_dir = os.path.join(os.path.dirname(__file__),  'experiment_result', 'loss_analysis')
    os.makedirs(loss_detail_dir, exist_ok=True)
    loss_detail_file = os.path.join(loss_detail_dir, f"loss_detail_{time.strftime('%Y_%m_%d')}.csv")
    loss_detail_header = not os.path.exists(loss_detail_file)
    # model update CSV（可注释整个块来关闭）
    track_update = False
    update_csv = os.path.join(loss_detail_dir, f"model_update_{time.strftime('%Y_%m_%d')}.csv")
    update_header = not os.path.exists(update_csv)
    loss_detail_list = []
    if train_flag == 1:
        for i in range(100):
            if mode == 'kmeans':
                # 训练前：保存参数快照
                if track_update:
                    pre_params = capture_param_stats(model)
                if args.dataset=="twitch_gamers":
                    loss, out, loss_detail = train_semi_supervise_clustering_model_big(data, centriods, 
                                                                                   constraint_matrix_label, constraint_matrix_sim, label_coef_val)
                else:
                    loss, out, loss_detail = train_semi_supervise_clustering_model(data, centriods, 
                                                                               constraint_matrix_label, constraint_matrix_sim, label_coef_val)
                loss_detail_list.append(loss_detail)
                # 训练后：记录参数更新
                if track_update:
                    log_model_update(pre_params, model, args.dataset, setting, i, args.learning_rate, update_csv, update_header)
                    update_header = False
                kmean_satrt = time.time()
                k_input = out.detach().cpu().numpy()
                k_input = k_input[merged_mask]
                kmeans = KMeans(n_clusters=cluster_number)
                kmeans.fit(k_input)
                centriods = torch.tensor(kmeans.cluster_centers_).to(device)
                cluster_results = kmeans.fit_predict(k_input)
                kmeans_time.append(time.time() - kmean_satrt)
                loss_set.append(loss.item())
                # 记录 loss detail
                row = {'time': time.strftime("%Y_%m_%d_%H_%M_%S"), 'dataset': args.dataset,
                       'setting': setting, 'epoch': i,
                       'kl': loss_detail['kl'],
                       'label_ml': loss_detail['label_ml'], 'label_cl': loss_detail['label_cl'],
                       'sim_ml': loss_detail['sim_ml'], 'sim_cl': loss_detail['sim_cl'],
                       'l_con': loss_detail['l_con'],
                       'grad_kl': loss_detail['grad_kl'],
                       'grad_label_ml': loss_detail['grad_label_ml'], 'grad_label_cl': loss_detail['grad_label_cl'],
                       'grad_sim_ml': loss_detail['grad_sim_ml'], 'grad_sim_cl': loss_detail['grad_sim_cl'],
                       'grad_con': loss_detail['grad_con'], 'grad_total': loss_detail['grad_total'],
                       'lr': args.learning_rate}
                pd.DataFrame([row]).to_csv(loss_detail_file, mode='a', index=False, header=loss_detail_header)
                loss_detail_header = False
            elif mode == 'no_kl':
                loss, out = train_no_kl_model(data, topk, bottomk)
                k_input = out.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=cluster_number)
                kmeans.fit(k_input)
                cluster_results = kmeans.fit_predict(k_input)
            else:
                loss, out = train_no_kl_model(data, topk, bottomk)
                s_input = out.detach().cpu().numpy()
                spec_cluster = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
                cluster_results = spec_cluster.fit_predict(s_input)


            train_nmi = normalized_mutual_info_score(data.label.cpu().numpy()[merged_mask],
                                                     cluster_results)
            train_acc = clustering_accuracy(data_label[merged_mask], cluster_results)
            print(f"epoch: {i}, total_loss: {loss}, ACC: {train_acc}, NMI: {train_nmi}")
        train_end = time.time()


        "#####################loss picture###############"
        loss_dir = os.path.join(os.path.dirname(__file__), 'loss_picture')
        os.makedirs(loss_dir, exist_ok=True)

        # 预训练损失曲线
        if len(pretrain_losses) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(pretrain_losses)+1), pretrain_losses,
                    marker='o', markersize=2, linewidth=1, color='#4CAF50')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'pretrain_{args.dataset}_s{setting}_lr{args.learning_rate}')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = f"pretrain_{args.dataset}_s{setting}_{args.pre_learning_rate}.png"
            plt.savefig(os.path.join(loss_dir, fname), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'[LossPic] saved {fname}')

        

        # 各部分 loss 单独曲线（从 loss_detail_list 绘制）
        if len(loss_detail_list) > 0:
            pic_time_str = time.strftime('%Y%m%d_%H%M%S')
            pic_date_dir = time.strftime('%Y_%m_%d')
            loss_pic_dir = os.path.join(os.path.dirname(__file__),  'experiment_result', 'loss_analysis', pic_date_dir, 'picture', args.dataset, f'setting_{setting}', pic_time_str)
            os.makedirs(loss_pic_dir, exist_ok=True)

            epochs = range(len(loss_detail_list))
            loss_items = [
                ('kl', '1) kl loss'),
                ('label_ml', '2) label must-link loss'),
                ('label_cl', '3) label cannot-link loss'),
                ('sim_ml', '4) sim must-link loss'),
                ('sim_cl', '5) sim cannot-link loss'),
                ('total_contrastive_loss', '6) total_contrastive_loss'),
                ('total_loss', '7) total_loss'),
            ]

            n = len(loss_items)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
            axes = axes.flatten()

            for idx, (key, title) in enumerate(loss_items):
                ax = axes[idx]
                vals = [d[key] for d in loss_detail_list]
                ax.plot(epochs, vals, linewidth=1.2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{title} ({args.dataset} s{setting})')
                ax.grid(True, alpha=0.3)

            for j in range(idx + 1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            fname = f"individual_loss_{args.dataset}_s{setting}.png"
            plt.savefig(os.path.join(loss_pic_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()
            print(f'[LossPic] saved {fname}')

            # 各分量组合曲线
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            loss_colors = {'kl': '#1f77b4', 'label_ml': '#ff7f0e', 'label_cl': '#2ca02c',
                           'sim_ml': '#d62728', 'sim_cl': '#9467bd',
                           'total_contrastive_loss': '#8c564b', 'total_loss': '#000000'}

            ax = axes[0, 0]
            for key in ['kl', 'label_ml', 'sim_ml']:
                vals = [d[key] for d in loss_detail_list]
                ax.plot(epochs, vals, label=key, color=loss_colors[key], linewidth=1)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('KL & Must-link Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            for key in ['label_cl', 'sim_cl']:
                vals = [d[key] for d in loss_detail_list]
                ax.plot(epochs, vals, label=key, color=loss_colors[key], linewidth=1)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Cannot-link Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 0]
            ax.plot(epochs, [d['total_contrastive_loss'] for d in loss_detail_list],
                    label='total_contrastive_loss', color='#8c564b', linewidth=1)
            ax.plot(epochs, [d['total_loss'] for d in loss_detail_list],
                    label='total_loss', color='#000000', linewidth=1, linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Total Contrastive Loss & Total Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            grad_keys = [('grad_kl', '#1f77b4'), ('grad_label_ml', '#ff7f0e'),
                         ('grad_label_cl', '#2ca02c'), ('grad_sim_ml', '#d62728'),
                         ('grad_sim_cl', '#9467bd'), ('grad_total', '#000000')]
            for key, color in grad_keys:
                vals = [d[key] for d in loss_detail_list]
                ax.plot(epochs, vals, label=key, color=color, linewidth=0.8, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradients Overview')
            ax.legend(ncol=2, fontsize=7)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = f"all_loss_curves_{args.dataset}_s{setting}.png"
            plt.savefig(os.path.join(loss_pic_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()
            print(f'[LossPic] saved {fname}')

            # 写入配置文件 txt
            config_path = os.path.join(loss_pic_dir, 'config.txt')
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(f'dataset: {args.dataset}\n')
                f.write(f'setting: {setting}\n')
                f.write(f'learning_rate: {args.learning_rate}\n')
                f.write(f'pre_learning_rate: {args.pre_learning_rate}\n')
                f.write(f'label_rate: {args.rate}\n')
                f.write(f'constraint_rate: {args.constraint_rate}\n')
                f.write(f'weight_decay: {args.weight_decay}\n')
                f.write(f'seed: {args.seed}\n')
                f.write(f'model_type: {args.type}\n')
                f.write(f'hidden_dim: 128\n')
                f.write(f'label_coef: {label_coef_val}\n')
                f.write(f'sim_type: {sim_type}\n')
                f.write(f'time: {pic_time_str}\n')
            print(f'[Config] saved {config_path}\n')

        test_start = time.time()
        emd = test_model(data)
        emd = emd.cpu().numpy()

        """############# Downstream task ##############"""
     

        merged_data = emd[merged_mask]
        merged_label = data.label[merged_mask]
        print("data_number:", len(merged_label))

        kmeans = KMeans(n_clusters=cluster_number)
        predict_labels = kmeans.fit_predict(merged_data)
        # predict_labels_all = kmeans.fit_predict(emd)
        cen = kmeans.cluster_centers_
        print(mode)
        
         
            ### 对测试结果进行2维聚类图分析
        from sklearn.decomposition import PCA

        # 如果嵌入超过2维，用PCA降到2维
        plot_data = emd
        if emd.shape[1] > 2:
            pca = PCA(n_components=2)
            plot_data = pca.fit_transform(emd)

        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(plot_data[:, 0], plot_data[:, 1], c=data.label.cpu().numpy(),
                    cmap='viridis', s=10, alpha=0.7)
        ax.set_title(f'{args.dataset} - Predicted Clusters (seed={args.seed})')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        plt.tight_layout()

        # 保存PDF
        plot_dir = os.path.join(os.path.dirname(__file__), 'experiment_result', 'chlster',
                                time.strftime("%Y_%m_%d"), f'setting_{setting}', args.dataset)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{args.dataset}_{args.seed}.pdf')
        fig.savefig(plot_path, format='pdf')
        print(f'[Cluster Plot] saved -> {plot_path}')
        plt.close(fig)
                    
        
        ##### 正式测试结果
        print(args.dataset)
        acc = clustering_accuracy(merged_label.cpu().numpy().astype(int), predict_labels)
        nmi = normalized_mutual_info_score(merged_label.cpu().numpy(), predict_labels)
        ari = adjusted_rand_score(merged_label.cpu().numpy(), predict_labels)
        print(f"ACC: {acc}")
        print(f"NMI: {nmi}")
        print(f'ARI: {ari}')
        test_end = time.time()
        train_time = train_end - train_start
        test_time = test_end-test_start
        cal_loss_time = np.array(cal_loss_list)
        op_time = np.array(op_time)
        kl_time = np.array(kl_time)
        kmeans_time = np.array(kmeans_time)
        print("train_time:",train_time)
        # plot_clusters_4(emd, predict_labels.astype(int), merged_label.cpu().numpy().astype(int), cen, merged_mask, dataset_name=args.dataset)
        columns = ['metrics', 'ACC', 'NMI', "ARI", "train_time", "test_time", "calcontra_time", "epoch", "label_rate", "args.constraint_rate",
                   "pos_k", "neg_k", "lr", "pre_learning_rate", "set", "op_time", "kl_time", "kmeans_time","pre_train_time", "constraint_time", "setting_config",
                   "args.seed", "L"]
        records = pd.DataFrame(columns=columns)
        records = records._append(
            {'metrics': args.dataset, 'ACC': acc, 'NMI': nmi, 'ARI': ari, 'train_time': train_time,
             'test_time': test_time,
             'calcontra_time': cal_loss_time.mean(), 'epoch': 100,
             'label_rate': args.rate,'args.constraint_rate': args.constraint_rate, 'pos_k': pos_k, 'neg_k': neg_k,
             'lr': args.learning_rate, 'pre_learning_rate': args.pre_learning_rate, 'set': sim_type, 'op_time': op_time.mean(),
             'kl_time': kl_time.mean(), 'kmeans_time': kmeans_time.mean(),'pre_train_time':pre_time,
             'constraint_time': constraint_time,
             'setting_config': f"{setting}：{setting_configs.get(setting, '未知配置')}",
             'label_coef_val': label_coef_val,
             'args.seed': args.seed,
             'L': L,
             'time': time.strftime("%Y_%m_%d_%H_%M_%S")}, ignore_index=True)
        file_name = f"pcgcn_{time.strftime('%Y_%m_%d')}.csv"
        file_path = os.path.join(os.path.dirname(__file__), 'experiment_result', 'result', file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        write_header = not os.path.exists(file_path)
        records.to_csv(file_path, mode='a', index=False, header=write_header)
       




