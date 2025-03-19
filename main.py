import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import scipy.sparse as ssp
import torch
from ogb.linkproppred import Evaluator
from scipy.sparse import SparseEfficiencyWarning
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from pyg_dataloader import DataLoader
from models import GIN, SEAL, SGNN
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.utils import k_hop_subgraph as pyg_k_hop_subgraph
from complete import CN, AA, RA, PA, shortest_path, katz_close, katz_apro
from pyg_dataset import QueryDataset, SEALDataset, SEALDynamicDataset, get_foundation_dataset
from utils import (MAX_Z, LIST_VALIDATION, Logger, get_pos_neg_edges,
                   load_data, get_git_revision_short_hash,
                   save_model, load_model, update_args, set_random_seeds,
                   str2bool, evaluate, evaluate_mrr)

import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.data import Data

# 导入外部库中的编码器
from ast_encoder import ASTNodeEncoder, ASTEdgeEncoder
from dummy_edge_encoder import DummyEdgeEncoder
from equivstable_laplace_pos_encoder import EquivStableLapPENodeEncoder
from graphormer_encoder import GraphormerEncoder
from kernel_pos_encoder import RWSENodeEncoder, HKdiagSENodeEncoder, ElstaticSENodeEncoder
from laplace_pos_encoder import LapPENodeEncoder
from linear_edge_encoder import LinearEdgeEncoder
from linear_node_encoder import LinearNodeEncoder
from ppa_encoder import PPANodeEncoder, PPAEdgeEncoder
from signnet_pos_encoder import SignNetNodeEncoder
from type_dict_encoder import TypeDictNodeEncoder, TypeDictEdgeEncoder
from voc_superpixels_encoder import VOCNodeEncoder, VOCEdgeEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter('ignore', SparseEfficiencyWarning)


# 新增函数：对图编码器生成的内容进行 K-means 聚类并生成子图
def cluster_and_generate_subgraphs(encoded_graphs, num_clusters):
    """
    使用 K-means 聚类对图编码器生成的内容进行粗化生成子图。
    Args:
        encoded_graphs: 图编码器生成的内容列表
        num_clusters: 聚类的数量
    Returns:
        subgraphs: 聚类后的子图列表
    """
    # 将所有编码后的图特征堆叠成一个矩阵
    all_features = torch.cat([graph.x for graph in encoded_graphs], dim=0)
    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(all_features.cpu().numpy())
    # 根据聚类结果生成子图
    subgraphs = []
    for cluster_id in range(num_clusters):
        cluster_indices = torch.tensor(np.where(cluster_labels == cluster_id)[0])
        subgraph_nodes = torch.cat([graph.x[cluster_indices] for graph in encoded_graphs], dim=0)
        subgraph_edges = torch.cat([graph.edge_index[:, cluster_indices] for graph in encoded_graphs], dim=1)
        subgraph = Data(x=subgraph_nodes, edge_index=subgraph_edges)
        subgraphs.append(subgraph)
    return subgraphs


# 新增函数：通过线性层对齐子图的维度
def align_subgraph_dimensions(subgraphs, target_dim):
    """
    使用线性层对齐子图的维度。
    Args:
        subgraphs: 子图列表
        target_dim: 目标维度
    Returns:
        aligned_subgraphs: 对齐维度后的子图列表
    """
    aligned_subgraphs = []
    for subgraph in subgraphs:
        linear_layer = torch.nn.Linear(subgraph.x.shape[1], target_dim)
        aligned_subgraph = Data(x=linear_layer(subgraph.x), edge_index=subgraph.edge_index)
        aligned_subgraphs.append(aligned_subgraph)
    return aligned_subgraphs


def train(epoch):
    model.train()

    total_loss = 0
    pbar = tqdm(train_loader, ncols=70, desc='Training')
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        edge_weight = data.edge_weight if args.use_edge_weight else None
        logits = model(None, data, edge_weight)
        labels = data.y[data.query_graph].to(torch.float)
        loss = criterion(logits.view(-1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)

def get_loaders(args, loggers, run, train_dataset=None, train_loader=None, one_dataset_name=None):
    if one_dataset_name is None:  # not running multi dataset inference
        one_dataset_name = args.dataset
    elif one_dataset_name == "ValidationDataset":
        val_dataset_list = []
        test_dataset_list = []
        num_samples = 200
        for one_dataset_name in LIST_VALIDATION.split(','):
            data, split_edge = load_data(one_dataset_name, args.dataset_dir, False, False, 0)
            # select the first m edges from training set as in-context link set
            if split_edge["train"]["edge"].shape[0] >= args.k:
                split_edge["support"] = {}
                split_edge["support"]["edge"] = split_edge["train"]["edge"][:args.k]
                split_edge["support"]["edge_neg"] = split_edge["train"]["edge_neg"][:args.k]
                split_edge["valid"]["edge"] = split_edge["valid"]["edge"][:num_samples]
                split_edge["valid"]["edge_neg"] = split_edge["valid"]["edge_neg"][:num_samples]
                split_edge["test"] = {
                    "edge": torch.tensor([[0, 1]], dtype=torch.long),
                    "edge_neg": torch.tensor([[0, 1]], dtype=torch.long),  # just a placeholder
                }
            else:
                raise ValueError(
                    f"the number of training edges ({split_edge['train']['edge'].shape[0]}) is smaller than m_ways ({args.k})")

            path = args.dataset_dir / one_dataset_name
            use_coalesce = True if one_dataset_name == 'ogbl-collab' else False
            dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
            val_dataset = eval(dataset_class)(
                path,
                data,
                split_edge,
                num_hops=args.num_hops,
                num_samples=args.val_samples,
                split='valid',
                use_coalesce=use_coalesce,
                node_label=args.node_label,
                ratio_per_hop=args.ratio_per_hop,
                max_nodes_per_hop=args.max_nodes_per_hop,
                directed=directed,
            )
            dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
            test_dataset = eval(dataset_class)(
                path,
                data,
                split_edge,
                num_hops=args.num_hops,
                num_samples=args.test_samples,
                split='test',
                use_coalesce=use_coalesce,
                node_label=args.node_label,
                ratio_per_hop=args.ratio_per_hop,
                max_nodes_per_hop=args.max_nodes_per_hop,
                directed=directed,
            )
            support_dataset = SEALDataset(
                path,
                data,
                split_edge,
                num_hops=args.num_hops,
                num_samples=None,
                split='support',
                use_coalesce=use_coalesce,
                node_label=args.node_label,
                ratio_per_hop=args.ratio_per_hop,
                max_nodes_per_hop=args.max_nodes_per_hop,
                directed=directed,
            )
            val_dataset = QueryDataset(path, args.k, val_dataset, support_dataset)
            test_dataset = QueryDataset(path, args.k, test_dataset, support_dataset)
            val_dataset_list.append(val_dataset)
            test_dataset_list.append(test_dataset)
        val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)
        test_dataset = torch.utils.data.ConcatDataset(test_dataset_list)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 num_workers=0)
        return train_dataset, train_loader, val_loader, test_loader

    # normal load data
    data, split_edge = load_data(one_dataset_name, args.dataset_dir, args.use_valedges_as_input, False, run)
    # select the first m edges from training set as support set
    if 'edge' in split_edge['train']:
        if split_edge["train"]["edge"].shape[0] >= args.k:
            split_edge["support"] = {}
            split_edge["support"]["edge"] = split_edge["train"]["edge"][:args.k]
            split_edge["support"]["edge_neg"] = split_edge["train"]["edge_neg"][:args.k]
        else:
            raise ValueError(
                f"the number of training edges ({split_edge['train']['edge'].shape[0]}) is smaller than m_ways ({args.k})")
    elif 'source_node' in split_edge['train']:
        split_edge["support"] = {}
        split_edge["support"]["edge"] = torch.stack(
            [split_edge["train"]["source_node"], split_edge["train"]["target_node"]], dim=1)[:args.k]
        split_edge["support"]["edge_neg"] = split_edge["train"]["edge_neg"][:args.k]
    if args.use_heuristic:
        # loggers = {
        #     'Hits@20': Logger(2, args),
        #     'Hits@50': Logger(2, args),
        #     'Hits@100': Logger(2, args),
        #     'aucroc': Logger(2, args),
        #     'aucpr': Logger(2, args),
        # }
        # Test link prediction heuristics.
        num_nodes = data.num_nodes
        if 'edge_weight' in data:
            edge_weight = data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

        A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])),
                           shape=(num_nodes, num_nodes))

        pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge,
                                                       data.edge_index,
                                                       data.num_nodes)
        pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge,
                                                         data.edge_index,
                                                         data.num_nodes)
        pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge)
        neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge)
        pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge)
        neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge)

        results = evaluate(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, one_dataset_name, evaluator)

        for key, result in results.items():
            loggers[key].add_result(run, result)
            # loggers[key].add_result(0, result)
            # loggers[key].add_result(1, result)
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run=run, f=f)
        return 0

    path = args.dataset_dir / one_dataset_name
    use_coalesce = True if one_dataset_name == 'ogbl-collab' else False
    dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
    val_dataset = eval(dataset_class)(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        num_samples=args.val_samples,
        split='valid',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        directed=directed,
    )
    dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
    test_dataset = eval(dataset_class)(
        path,
        data,
        split_edge,
        num_hops=args.num_hops,
        num_samples=args.test_samples,
        split='test',
        use_coalesce=use_coalesce,
        node_label=args.node_label,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        directed=directed,
    )
    if args.foundation_mode:
        support_dataset = SEALDataset(
            path,
            data,
            split_edge,
            num_hops=args.num_hops,
            num_samples=None,
            split='support',
            use_coalesce=use_coalesce,
            node_label=args.node_label,
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
            directed=directed,
        )
        val_dataset = QueryDataset(path, args.k, val_dataset, support_dataset)
        test_dataset = QueryDataset(path, args.k, test_dataset, support_dataset)
    elif args.pretrain_datasets is None:
        dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
        train_dataset = eval(dataset_class)(
            path,
            data,
            split_edge,
            num_hops=args.num_hops,
            num_samples=args.train_samples,
            split='train',
            use_coalesce=use_coalesce,
            node_label=args.node_label,
            ratio_per_hop=args.ratio_per_hop,
            max_nodes_per_hop=args.max_nodes_per_hop,
            directed=directed,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=0)
    return train_dataset, train_loader, val_loader, test_loader
global results
def test_print(cache_support=True):
    results = test(cache_support=cache_support)

    if not isinstance(val_loader.dataset, QueryDataset):
        print("Enforce cache_support=False for multi dataset inference")
        cache_support = False
        results = test(cache_support=cache_support)

        # 处理 KeyError
        if "Hits@20" not in results:
            print("Warning: 'Hits@20' not found in test results. Using default value 0.0")
            results["Hits@20"] = 0.0  # 设定默认值以避免崩溃

        print(f"Test results: Hits@20 = {results['Hits@20']}")
    results = test(cache_support=cache_support)
    for key, result in results.items():
        loggers[key].add_result(max(run, 0), result)

    for key, result in results.items():
        valid_res, test_res = result
        to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                    f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                    f'Test: {100 * test_res:.2f}%')
        print(key)
        print(to_print)
        with open(log_file, 'a') as f:
            print(key, file=f)
            print(to_print, file=f)


@torch.no_grad()
def test(cache_support=False):
    model.eval()

    y_pred, y_true = [], []
    support_cache = None
    if cache_support:
        # only when all the query graphs use the same set of support graphs
        data = val_loader.collate_fn(val_loader.dataset.get(0))
        data = data.to(device)
        edge_weight = data.edge_weight if args.use_edge_weight else None
        support_cache = model.get_support(None, data, edge_weight)
    for data in tqdm(val_loader, ncols=70, desc='Valid'):
        data = data.to(device)
        edge_weight = data.edge_weight if args.use_edge_weight else None

        # 替换上下文采样生成的子图
        encoded_graphs = [data]  # 假设 data 是图编码器生成的内容
        num_clusters = 5  # 假设聚类数量为 5
        subgraphs = cluster_and_generate_subgraphs(encoded_graphs, num_clusters)
        target_dim = 128  # 假设目标维度为 128
        aligned_subgraphs = align_subgraph_dimensions(subgraphs, target_dim)
        data = aligned_subgraphs[0]  # 使用第一个对齐后的子图替换原始数据

        logits = model(None, data, edge_weight, support_cache)
        labels = data.y[data.query_graph].to(torch.float)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(labels.view(-1).cpu())
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true == 1]
    neg_val_pred = val_pred[val_true == 0]

    y_pred, y_true = [], []
    if cache_support:
        # only when all the query graphs use the same set of support graphs
        data = test_loader.collate_fn(test_loader.dataset.get(0))
        data = data.to(device)
        edge_weight = data.edge_weight if args.use_edge_weight else None
        support_cache = model.get_support(None, data, edge_weight)
    for data in tqdm(test_loader, ncols=70, desc='Test'):
        data = data.to(device)
        edge_weight = data.edge_weight if args.use_edge_weight else None

        # 替换上下文采样生成的子图
        encoded_graphs = [data]  # 假设 data 是图编码器生成的内容
        num_clusters = 5  # 假设聚类数量为 5
        subgraphs = cluster_and_generate_subgraphs(encoded_graphs, num_clusters)
        target_dim = 128  # 假设目标维度为 128
        aligned_subgraphs = align_subgraph_dimensions(subgraphs, target_dim)
        data = aligned_subgraphs[0]  # 使用第一个对齐后的子图替换原始数据

        logits = model(None, data, edge_weight, support_cache)
        labels = data.y[data.query_graph].to(torch.float)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(labels.view(-1).cpu())
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true == 1]
    neg_test_pred = test_pred[test_true == 0]

    if args.eval_metric == 'hits':
        results = evaluate(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, one_dataset_name, evaluator)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, one_dataset_name, evaluator)
    if "Hits@20" not in results:
        print("Warning: 'Hits@20' is missing in test() results. Setting default value.")
        results["Hits@20"] = 0.0  # 添加默认值，防止 KeyError

    return results


# Data settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbl-collab', help="the dataset to evaluate on")
parser.add_argument('--pretrain_datasets', type=str, help="the datasets to pretrain on")
parser.add_argument('--inference_datasets', type=str, help="the datasets to inference on")
parser.add_argument('--dataset_dir', type=str, default='./data/')
# GNN settings
parser.add_argument('--model', type=str, default='SAGE')
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--pooling', type=str, default='mean', choices=["center", "sum", "mean", "max"],
                    help='the subgraph pooling method')
parser.add_argument('--jk', type=str2bool, default='True', help='whether to use JumpingKnowledge')
# Subgraph extraction settings
parser.add_argument('--num_hops', type=int, default=2)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='drnl+',
                    help="which specific labeling trick to use")
# Training settings
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=5, help='number of patience steps for early stopping')
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--train_samples', type=int, help="the number of training samples, per graph")
parser.add_argument('--val_samples', type=int)
parser.add_argument('--test_samples', type=int)
parser.add_argument('--dynamic_train', type=str2bool, default='False',
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_val', type=str2bool, default='True', )
parser.add_argument('--dynamic_test', type=str2bool, default='True', )
parser.add_argument('--num_workers', type=int, default=8,
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--log_dir', type=str, default='results')
# Testing settings
parser.add_argument('--eval_metric', type=str, default='hits')
parser.add_argument('--use_valedges_as_input', type=str2bool, default='False', )
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--use_heuristic', type=str, default=None,
                    help="test a link prediction heuristic (CN or AA)")

# Foundation settings
parser.add_argument('--foundation_mode', type=str2bool, default="True",
                    help="whether to enable the foundation mode so that the inference is made on support graphs")
parser.add_argument('--use_graph_embedding', type=str2bool, default="True",
                    help="whether to use graph embedding when predicting the edge score. If not, it will only combine the meta edge embedding, which reflects the links labels")
parser.add_argument('--save_model', type=str, default=None,
                    help="whether to save the model under the dir")
parser.add_argument('--load_model', type=str, default=None,
                    help="whether to load the model with the path")
parser.add_argument('--k', type=int, default=40, help="the number of positive and negative In-context links")
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--add_self_loops', type=str2bool, default='True',
                    help='whether to add self loops during attention')
parser.add_argument('--finetune', type=str, default='all',
                    help="can be 'none', 'all', 'emb', 'encoder', 'agg'(only for SEAL), 'classifier' and seperated by ','")

set_random_seeds(123)
args = parser.parse_args()
if args.use_heuristic:
    args.model = args.use_heuristic
args.use_edge_weight = False
args.use_feature = False  # not using feature for all datasets for now
print(args)
args.dataset_dir = Path(args.dataset_dir)
args.pretrain_datasets = args.pretrain_datasets.split(",") if args.pretrain_datasets is not None else None

data_appendix = '_h{}_{}_rph{}'.format(
    args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
if args.max_nodes_per_hop is not None:
    data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
if args.use_valedges_as_input:
    data_appendix += '_uvai'

if args.load_model or args.save_model:
    args.sortpool_k = 30

pid = os.getpid()
time_str = int(time.time())
appendix = f"jobID_{os.getenv('JOB_ID', 'None')}_PID_{pid}_{time_str}"
if args.inference_datasets is None:
    log_file = Path(args.log_dir) / f"{args.dataset}_{appendix}.log"
else:
    log_file = Path(args.log_dir) / f"incontext_{appendix}.log"
    args.inference_datasets = args.inference_datasets.split(",")
if not Path(args.log_dir).exists():
    Path(args.log_dir).mkdir(parents=True)

# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
hostname = os.getenv('HOSTNAME', 'None')
print('Command line input: ' + cmd_input + ' is saved.')
# Save git revision.
git_hash = get_git_revision_short_hash()
with open(log_file, 'a') as f:
    print(args, file=f)
    f.write('\n' + cmd_input)
    print(f"HOSTNAME: {hostname}", file=f)
    print('Git revision: ' + git_hash + '\n', file=f)

if args.load_model:
    args = update_args(args, args.load_model)
    args_str = "###### Override args ######\n" + args.__repr__()
    print(args_str)
    with open(log_file, 'a') as f:
        print(args_str, file=f)

directed = False
if args.eval_metric == 'hits':
    evaluator = Evaluator(name='ogbl-ddi')
elif args.eval_metric == 'mrr':
    evaluator = Evaluator(name='ogbl-citation2')
criterion = BCEWithLogitsLoss()
if args.inference_datasets is None:
    loggers = {
        f'Hits@20{args.dataset}': Logger(args.runs, args),
        f'Hits@50{args.dataset}': Logger(args.runs, args),
        f'Hits@100{args.dataset}': Logger(args.runs, args),
        f'aucroc{args.dataset}': Logger(args.runs, args),
        f'aucpr{args.dataset}': Logger(args.runs, args),
    }
    metric = "Hits@50" + args.dataset
    one_dataset_name = ''
else:
    loggers = {}
    for one_dataset_name in args.inference_datasets:  # + ["ValidationDataset"]:
        if args.eval_metric == 'mrr':
            loggers[f'MRR{one_dataset_name}'] = Logger(args.runs, args)
        else:
            loggers.update({
                f'Hits@20{one_dataset_name}': Logger(args.runs, args),
                f'Hits@50{one_dataset_name}': Logger(args.runs, args),
                f'Hits@100{one_dataset_name}': Logger(args.runs, args),
                f'aucroc{one_dataset_name}': Logger(args.runs, args),
                f'aucpr{one_dataset_name}': Logger(args.runs, args),
            })
    one_dataset_name = "ValidationDataset"
    loggers.update({
        f'Hits@20{one_dataset_name}': Logger(1, args),
        f'Hits@50{one_dataset_name}': Logger(1, args),
        f'Hits@100{one_dataset_name}': Logger(1, args),
        f'aucroc{one_dataset_name}': Logger(1, args),
        f'aucpr{one_dataset_name}': Logger(1, args),
    })
    metric = "Hits@50ValidationDataset"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = None
train_loader = None
start_run = 0
if args.pretrain_datasets and not args.load_model:
    pretrain_data_list = []
    start_run = -1
    for pretrain_dataset in args.pretrain_datasets:
        path = args.dataset_dir / pretrain_dataset / data_appendix
        use_coalesce = True if pretrain_dataset == 'ogbl-collab' else False
        pretrain_data, pretrain_split_edge = load_data(pretrain_dataset, args.dataset_dir, args.use_valedges_as_input,
                                                       no_split=True)
        train_dataset_dataset = get_foundation_dataset(root=path,
                                                       data=pretrain_data,
                                                       split_edge=pretrain_split_edge,
                                                       num_hops=args.num_hops,
                                                       num_samples=args.train_samples,
                                                       split='all',
                                                       use_coalesce=use_coalesce,
                                                       node_label=args.node_label,
                                                       ratio_per_hop=args.ratio_per_hop,
                                                       max_nodes_per_hop=args.max_nodes_per_hop,
                                                       directed=directed, )
        tmp = train_dataset_dataset.multi_get([1, 2, 3])
        pretrain_dataset = QueryDataset(path, args.k, train_dataset_dataset, train_dataset_dataset)
        pretrain_dataset.get(0)
        pretrain_data_list.append(pretrain_dataset)
    train_dataset = torch.utils.data.ConcatDataset(pretrain_data_list)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

emb = None
train_flg = True
save_model_path = "C:/Users/Jay/Desktop/git/model.pt"

for run in range(start_run, args.runs):
    if args.inference_datasets is None:  # only test on args.dataset
        if args.use_heuristic:
            get_loaders(args, loggers, run, train_dataset, train_loader)
            continue
        else:
            train_dataset, train_loader, val_loader, test_loader = get_loaders(args, loggers, run, train_dataset, train_loader)
    else:
        train_dataset, train_loader, val_loader, test_loader = get_loaders(args, loggers, run, train_dataset, train_loader)

    if args.model not in ['GCN', 'SAGE', 'SEAL', 'GIN']:
        raise ValueError(f"Unsupported model type: {args.model}. Supported models are 'GCN', 'SAGE', 'SEAL', and 'GIN'.")

    model = None  # Initialize model to None
    if args.foundation_mode:
        assert args.model in ['GCN', 'SAGE'], "only GCN and SAGE are supported for foundation mode"
        model = SGNN(eval(f"{args.model}Conv"), args.hidden_channels, args.num_layers, MAX_Z, None,
                     args.use_feature, node_embedding=emb, pooling=args.pooling,
                     add_self_loops=args.add_self_loops, foundation_mode=True, heads=args.heads,
                     use_graph_embedding=args.use_graph_embedding).to(device)

        if args.load_model:  # 如果有预训练模型路径，则加载模型
            load_model(model, args.load_model)
            train_flg = False
        elif run >= 0:  # 如果是第一次运行之后，加载保存的最佳模型
            if save_model_path is None:
                raise ValueError("save_model_path is not defined. Please ensure the model is saved correctly during pretraining.")
            load_model(model, save_model_path)
            train_flg = False
    else:
        if args.pretrain_datasets and run >= 0:  # after the first run (pretraining), we load the best model
            if save_model_path is None:
                raise ValueError("save_model_path is not defined. Please ensure the model is saved correctly during pretraining.")
            load_model(model, save_model_path)
            train_flg = False
        elif args.model in ['GCN', 'SAGE']:
            model = SGNN(eval(f"{args.model}Conv"), args.hidden_channels, args.num_layers, MAX_Z, None,
                         args.use_feature, node_embedding=emb, pooling=args.pooling,
                         add_self_loops=args.add_self_loops, foundation_mode=False).to(device)
        elif args.model == 'SEAL':
            model = SEAL(args.hidden_channels, args.num_layers, MAX_Z, args.sortpool_k,
                         train_dataset, args.dynamic_train, use_feature=args.use_feature,
                         node_embedding=emb).to(device)
        elif args.model == 'GIN':
            model = GIN(args.hidden_channels, args.num_layers, MAX_Z, train_dataset,
                        args.use_feature, node_embedding=emb).to(device)
        if args.load_model:
            load_model(model, args.load_model)
            train_flg = False

    if train_flg:
        parameters_to_train = []
        # freeze all parameters first:
        for param in model.parameters():
            param.requires_grad = False
        # only require grad for the parameters in the finetune parts:
        if args.finetune == 'none':
            train_flg = False
        else:
            train_flg = True
            for each_part in args.finetune.split(","):
                p = model.get_parameters(each_part)
                for each in p:
                    each.requires_grad = True
                parameters_to_train += p
        if len(parameters_to_train) > 0:
            optimizer = torch.optim.Adam(params=parameters_to_train, lr=args.lr)
        total_params = sum(p.numel() for param in model.parameters() for p in param)
        print(f'Total number of parameters is {total_params}')

    start_epoch = 1
    cnt_wait = 0
    best_val = 0.0
    loss = -9999
    one_dataset_name = ''

    # Training starts
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if not train_flg:
            if args.inference_datasets is None:
                test_print(cache_support=True)
            else:
                for one_dataset_name in args.inference_datasets:
                    train_dataset, train_loader, val_loader, test_loader = get_loaders(args, loggers, run, train_dataset, train_loader, one_dataset_name)
                    test_print(cache_support=True)
            break
        loss = train(epoch)

        if epoch % args.eval_steps == 0:
            if args.inference_datasets is None:
                results = test_print(cache_support=True)
            else:
                one_dataset_name = "ValidationDataset"
                train_dataset, train_loader, val_loader, test_loader = get_loaders(args, loggers, run, train_dataset, train_loader, one_dataset_name)
                results = test_print()

            if results[metric][0] >= best_val:
                best_val = results[metric][0]
                cnt_wait = 0
                if args.save_model:
                    state_dict = model.state_dict()
                    save_model_path = save_model(state_dict, args.save_model, appendix, cmd_input,
                                                 git_hash, hostname, args)
                    model = None  # Reset model to None to avoid multiple saves
            else:
                cnt_wait += 1
            # Early stopping
            if cnt_wait >= args.patience:
                break

    if "Hits@20" in results:
        if results["Hits@20"] >= best_val:
            best_val = results["Hits@20"]

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(run)
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(run, f=f)

for key in loggers.keys():
    print(key)
    loggers[key].print_statistics()
    with open(log_file, 'a') as f:
        print(key, file=f)
        loggers[key].print_statistics(f=f)

    if train_flg:
        parameters_to_train = []
        # freeze all parameters first:
        for param in model.parameters():
            param.requires_grad = False
        # only require grad for the parameters in the finetune parts:
        if args.finetune == 'none':
            train_flg = False
        else:
            train_flg = True
            for each_part in args.finetune.split(","):
                p = model.get_parameters(each_part)
                for each in p:
                    each.requires_grad = True
                parameters_to_train += p
        if len(parameters_to_train) > 0:
            optimizer = torch.optim.Adam(params=parameters_to_train, lr=args.lr)
        total_params = sum(p.numel() for param in model.parameters() for p in param)
        print(f'Total number of parameters is {total_params}')

    start_epoch = 1
    cnt_wait = 0
    best_val = 0.0
    loss = -9999
    one_dataset_name = ''

    # Training starts
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if not train_flg:
            if args.inference_datasets is None:
                test_print(cache_support=True)
            else:
                for one_dataset_name in args.inference_datasets:
                    train_dataset, train_loader, val_loader, test_loader = get_loaders(args, loggers, run, train_dataset, train_loader, one_dataset_name)
                    test_print(cache_support=True)
            break
        loss = train(epoch)

        if epoch % args.eval_steps == 0:
            if args.inference_datasets is None:
                test_print(cache_support=True)
            else:
                one_dataset_name = "ValidationDataset"
                train_dataset, train_loader, val_loader, test_loader = get_loaders(args, loggers, run, train_dataset, train_loader, one_dataset_name)
                test_print()

            if results[metric][0] >= best_val:
                best_val = results[metric][0]
                cnt_wait = 0
                if args.save_model:
                    state_dict = model.state_dict()
                    save_model_path = save_model(state_dict, args.save_model, appendix, cmd_input, git_hash, hostname, args)
            else:
                cnt_wait += 1
            # Early stopping
            if cnt_wait >= args.patience:
                break

    if "Hits@20" in results:
        if results["Hits@20"] >= best_val:
            best_val = results["Hits@20"]

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(run)
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(run, f=f)

for key in loggers.keys():
    print(key)
    loggers[key].print_statistics()
    with open(log_file, 'a') as f:
        print(key, file=f)
        loggers[key].print_statistics(f=f)