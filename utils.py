import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as ssp
import torch
import torch_geometric.transforms as T
import torch_sparse
import yaml, json
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, PolBlogs
from torch_geometric.data.collate import collate
from snap_dataset import SNAPDataset
from torch_geometric.utils import (add_self_loops, degree,
                                   from_scipy_sparse_matrix, is_undirected,
                                   negative_sampling, spmm,
                                   to_undirected, train_test_split_edges)
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
from tqdm import tqdm
from torch_geometric.utils.num_nodes import maybe_num_nodes

from complete import AA, CN, GCN, RA, SAGE

MAX_Z=1000 # set a large max_z so that every z has embeddings to look up
LIST_VALIDATION="Celegans,USAir,PB,NS,Cora"
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def edge_split(data, root, name='cora', split_val_percent=5, split_test_percent=10, split_edge=None, run=0):
    # run=0
    if split_edge is None:
        val_ratio=split_val_percent/100
        test_ratio=split_test_percent/100
        data_folder = root / name
        data_folder.mkdir(parents=True, exist_ok=True)    
        file_path = data_folder / f"split{run}_{split_val_percent}_{split_test_percent}.pt"
        # load data
        if file_path.exists():
            print(f"load split edge from {file_path}")
            split_edge = torch.load(file_path)
        else:
            split_edge = randomsplit(data, val_ratio=val_ratio, test_ratio=test_ratio)
            torch.save(split_edge, file_path)
            print(f"save split edges to {file_path}")
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    print("-"*20)
    if 'edge' in split_edge['train']:
        print(f"train: {split_edge['train']['edge'].shape[0]}")
        print(f"{split_edge['train']['edge'][:10,:]}")
        print(f"valid: {split_edge['valid']['edge'].shape[0]}")
        print(f"test: {split_edge['test']['edge'].shape[0]}")
    elif 'source_node' in split_edge['train']:
        pass
    print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
    return data, split_edge

def load_unsplitted_data(dataset, dataset_dir):
    # read .mat format files
    data_dir = dataset_dir / '{}.mat'.format(dataset)
    print('Load data from: '+ str(data_dir))
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index,_ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index,num_nodes = torch.max(edge_index).item()+1)
    if is_undirected(data.edge_index) == False: #in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    return data

def load_data(dataset_name,dataset_dir,use_valedges_as_input, no_split=False, run=0, split_val_percent=10, split_test_percent=20):
    directed=False
    split_edge = None
    print("Loading dataset {}".format(dataset_name))
    if dataset_name.startswith('ogbl') and run == 0:
        dataset = PygLinkPropPredDataset(name=dataset_name, root=Path().home()/"files")
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        if dataset_name == 'ogbl-collab':
            data, split_edge = filter_by_year(data, split_edge, 2010)
        if dataset_name.startswith('ogbl-vessel'):
            # normalize node features
            data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
            data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
            data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)
        if use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            if not directed:
                val_edge_index = to_undirected(val_edge_index)
            data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
            val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
            data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)
        dataset = [data]
        # create edge_neg in train
        new_edge_index, _ = add_self_loops(data.edge_index)
        if 'edge' in split_edge['train']:
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=data.num_nodes,
                num_neg_samples=split_edge['train']['edge'].size(0))
        elif 'source_node' in split_edge['train']:
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=data.num_nodes,
                num_neg_samples=20000)
        split_edge['train']['edge_neg'] = neg_edge.t()
    elif dataset_name.lower() in ["cora",\
                                "citeseer",\
                                "pubmed",]:
        dataset = Planetoid(dataset_dir, dataset_name.capitalize())
    elif dataset_name in ["BlogCatalog",
                            "Celegans",
                            "Ecoli",
                            "NS",
                            "PB",
                            "Power",
                            "Router",
                            "USAir",
                            "Yeast"]:
        data = load_unsplitted_data(dataset_name, dataset_dir)
        dataset = [data]
    elif dataset_name.lower() in ["computers","photo"]:
        dataset = Amazon(dataset_dir, dataset_name.capitalize())
    elif dataset_name.lower() in ["cs","physics"]:
        dataset = Coauthor(dataset_dir, dataset_name.capitalize())
    elif dataset_name.lower() == "polblogs":
        dataset = PolBlogs(dataset_dir/ "PolBlogs")
    elif dataset_name.startswith('snap'):
        dataset_name_snap = dataset_name[dataset_name.index('-')+1:]
        dataset = SNAPDataset(dataset_dir, dataset_name_snap)
    elif dataset_name.startswith('syn-'):
        from pyg_dataset import SyntheticDataset
        dataset = SyntheticDataset(dataset_dir, dataset_name[dataset_name.index('-')+1:])
    else:
        raise ValueError("dataset not found")
    data, _, _ = collate(
        dataset[0].__class__,
        data_list=list(dataset),
        increment=True,
        add_batch=False,
    )
    if no_split:
        row,col = data.edge_index
        train_edges = data.edge_index[:,row<col]
        split_edge = {"train":{"edge":train_edges.t()}}
        data.num_nodes = maybe_num_nodes(data.edge_index, data.num_nodes)
    else:
        data, split_edge = edge_split(data, dataset_dir, dataset_name, \
                                    split_val_percent=split_val_percent, split_test_percent=split_test_percent,\
                                    run=run, split_edge=split_edge)
        data.num_nodes = maybe_num_nodes(data.edge_index, data.num_nodes)
    return data, split_edge

# adpoted from https://github.com/facebookresearch/SEAL_OGB

def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features=None, 
                   y=1, directed=False, A_csc=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]


    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)

def drnl_plus_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL) plus.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    
    dist2both_fill = torch.nan_to_num(dist2src,posinf=0) + torch.nan_to_num(dist2dst,posinf=0)
    z[torch.isnan(z)] = MAX_Z - dist2both_fill[torch.isnan(z)] # last z to denote those 0s to distance to one of the end nodes

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More 
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl'):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])

    # if node_features is None:
    node_features = torch.zeros([1])
    # else:
    #     # run GCN twice
    #     K=2
    #     adj_t = gcn_norm(SparseTensor.from_edge_index(edge_index,
    #                            sparse_sizes=(num_nodes, num_nodes)), 
    #                     edge_weight, num_nodes, False, True)
    #     for k in range(K):
    #         node_features=spmm(adj_t, node_features, reduce='add')
    #     # normalize
    #     node_features = torch.nn.functional.normalize(node_features, dim=1)
    #     # inner product between 0 and 1
    #     node_features = (node_features[0]*node_features[1]).sum().unsqueeze(0)

    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'drnl+':  # DRNL+
        z = drnl_plus_node_labeling(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists)==0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z>100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(None, edge_index, edge_weight=edge_weight, y=y, z=z, node_features=node_features,
                node_id=node_ids, num_nodes=num_nodes, query_graph = torch.ones(1, dtype=torch.bool), query_node = torch.ones( num_nodes, dtype=torch.long))
    return data

 
def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl', 
                                ratio_per_hop=1.0, max_nodes_per_hop=None, 
                                directed=False, A_csc=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop, 
                             max_nodes_per_hop, node_features=x, y=y, 
                             directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(*tmp, node_label)
        data_list.append(data)

    return data_list


# random split dataset
def randomsplit(data, val_ratio: float=0.10, test_ratio: float=0.2):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei

    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()

    # create edge_neg in train
    edge_index = to_undirected(split_edge['train']['edge'].t())
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(
        new_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=split_edge['train']['edge'].size(0))
    split_edge['train']['edge_neg'] = neg_edge.t()
    return split_edge


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, num_samples=None):
    if 'edge' in split_edge['train'] or split == 'support':
        pos_edge = split_edge[split]['edge'].t()

        if 'edge_neg' in split_edge[split] and split != 'train': # for 'train', we always use negative sampling
            # use presampled  negative training edges for ogbl-vessel
            neg_edge = split_edge[split]['edge_neg'].t()

        else:
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))

        num_pos = pos_edge.size(1)
        num_neg = neg_edge.size(1)
        if num_samples is None:
            num_samples_pos = num_pos
            num_samples_neg = num_neg
        else:
            num_samples_pos = min(num_samples, num_pos)
            num_samples_neg = min(num_samples, num_neg)
            # assert num_samples_pos <= num_pos, f"num_samples {num_samples} > num_pos {num_pos}"
        # subsample for pos_edge
        perm = np.random.permutation(num_pos)
        perm = perm[:num_samples_pos]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        perm = np.random.permutation(num_neg)
        perm = perm[:num_samples_neg]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:num_samples]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])
    print("-"*20)
    print(f"neg_edge:\n{neg_edge.t()[:10]}")
    return pos_edge, neg_edge


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_argmax(self, result, last_best=True):
        if last_best:
            # get last max value index by reversing result tensor
            argmax = result.size(0) - result[:, 0].flip(dims=[0]).argmax().item() - 1
        else:
            argmax = result[:, 0].argmax().item()
        return argmax

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            if run < 0  or (run+1) >= len(self.results):
                result = torch.tensor([(0,0)])
                argmax = 0
                print(f'No Result', file=f)
                return
            else:
                result = 100 * torch.tensor(self.results[run])
                argmax = self.get_argmax(result)
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'Highest Eval Point: {argmax + 1}', file=f)
            print(f'   Final Test: {result[argmax, 1]:.2f}', file=f)
        else:
            best_results = []
            if len(self.results[0]) == 0:
                print(f'No Result', file=f)
                return
            for r in self.results:
                r = torch.tensor(r) * 100
                valid = r[:, 0].max().item()
                argmax = self.get_argmax(r)
                test = r[argmax, 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)

import subprocess

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


# adopted from "https://github.com/melifluos/subgraph-sketching/tree/main"
def filter_by_year(data, split_edge, year):
    """
    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge

from sklearn import metrics
def evaluate(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, one_dataset_name, evaluator):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}{one_dataset_name}'] = (valid_hits, test_hits)
    results.update(evaluate_auc_pr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, one_dataset_name))
    return results

def evaluate_auc_pr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, one_dataset_name):
    result = {}
    val_label = torch.concat([torch.tensor([1]).repeat(pos_val_pred.shape[0]),
                              torch.tensor([0]).repeat(neg_val_pred.shape[0])])
    val_pred = torch.concat([pos_val_pred, neg_val_pred])

    test_label = torch.concat([torch.tensor([1]).repeat(pos_test_pred.shape[0]),
                              torch.tensor([0]).repeat(neg_test_pred.shape[0])])
    test_pred = torch.concat([pos_test_pred, neg_test_pred])

    fpr, tpr, thresholds = metrics.roc_curve(val_label, val_pred, pos_label=1) # (y,pred)
    val_aucroc = metrics.auc(fpr, tpr)
    
    precision, recall, thresholds = metrics.precision_recall_curve(val_label, val_pred, pos_label=1) # y, pred
    # Use AUC function to calculate the area under the curve of precision recall curve
    val_aucpr = metrics.auc(recall, precision)

    fpr, tpr, thresholds = metrics.roc_curve(test_label, test_pred, pos_label=1) # (y,pred)
    test_aucroc = metrics.auc(fpr, tpr)
    
    precision, recall, thresholds = metrics.precision_recall_curve(test_label, test_pred, pos_label=1) # y, pred
    # Use AUC function to calculate the area under the curve of precision recall curve
    test_aucpr = metrics.auc(recall, precision)
    
    result[f"aucroc{one_dataset_name}"] = (val_aucroc, test_aucroc)
    result[f"aucpr{one_dataset_name}"] = (val_aucpr, test_aucpr)
    return result

def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, one_dataset_name, evaluator):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results[f'MRR{one_dataset_name}'] = (valid_mrr, test_mrr)
    return results

MODEL_ARGS = ["model", "hidden_channels", "num_layers", "use_feature", "pooling", "add_self_loops", "heads", "use_graph_embedding"]

def save_model(state_dict, dir, appendix, cmd, git_hash, hostname, args):
    checkpoints = Path(dir)
    model_folder = checkpoints / appendix
    model_folder.mkdir(parents=True, exist_ok=True)
    
    # dump model args
    args_dict = {}
    for arg in MODEL_ARGS:
        args_dict[arg] = args.__dict__[arg]
    dump = {"weights": state_dict, "args": args_dict}
    torch.save(dump, model_folder / "model.pt")
    print(f"save model to {model_folder / 'model.pt'}")

    # save command and git hash
    j = {}
    j["CMD"] = cmd
    j["git_hash"] = git_hash
    j["hostname"] = hostname
    with open(model_folder/f"config.json",'w') as f:
       json.dump(j, f, indent = 6)
    return model_folder / 'model.pt'

def load_model(model, load_model_path):
    dump = torch.load(load_model_path, map_location=torch.device('cpu'))
    ## for backward compatibility
    if "weights"  in dump:
        model.load_state_dict(dump['weights'])
    else:
        model.load_state_dict(dump)
    print(f"load model from {load_model_path}")

def update_args(args, load_model_path):
    dump = torch.load(load_model_path)
    ## for backward compatibility
    if "args" in dump:
        args.__dict__.update(dump['args'])
    return args