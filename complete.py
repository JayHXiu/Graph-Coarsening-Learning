import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN as GCN_model, GraphSAGE as SAGE_model
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import Evaluator
from tqdm import tqdm

def CN(A, edge_index, batch_size=1000000, **kwargs):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index


def AA(A, edge_index, batch_size=1000000, **kwargs):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    #edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

def RA(A, edge_index, batch_size=1000000, **kwargs):
    # The Resource Allocation heuristic score.
    multiplier = 1 / A.sum(axis=0)
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def PA(A, edge_index, batch_size=1000000, **kwargs):
    # Preferential Attachment heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = A[src].sum(axis=1).A1 * A[dst].sum(axis=1).A1
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


## adopted from https://github.com/Juanhui28/HeaRT/blob/master/benchmarking/get_heuristic.py
from tqdm import tqdm
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from torch_sparse import spspmm
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
import networkx as nx
import math

def shortest_path(A, edge_index, remove=False):
    
    scores = []
    G = nx.from_scipy_sparse_array(A)
    add_flag1 = 0
    add_flag2 = 0
    count = 0
    count1 = count2 = 0
    print('remove: ', remove)
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        # if (s,t) in train_pos_list: train_pos_list.remove((s,t))
        # if (t,s) in train_pos_list: train_pos_list.remove((t,s))


        # G = nx.Graph(train_pos_list)
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        if nx.has_path(G, source=s, target=t):

            sp = nx.shortest_path_length(G, source=s, target=t)
            # if sp == 0:
            #     print(1)
        else:
            sp = 999
        

        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
    

        scores.append(1/(sp))
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores), edge_index

def katz_apro(A, edge_index, beta=0.005, path_len=3, remove=False):
    scores = []
    G = nx.from_scipy_sparse_array(A)
    path_len = int(path_len)
    count = 0
    add_flag1 = 0
    add_flag2 = 0
    count1 = count2 = 0
    betas = np.zeros(path_len)
    print('remove: ', remove)
    for i in range(len(betas)):
        betas[i] = np.power(beta, i+1)
    
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        if s == t:
            count += 1
            scores.append(0)
            continue
        
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
                
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1


        paths = np.zeros(path_len)
        for path in nx.all_simple_paths(G, source=s, target=t, cutoff=path_len):
            paths[len(path)-2] += 1  
        
        kz = np.sum(betas * paths)

        scores.append(kz)
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
        
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores), edge_index


def katz_close(A, edge_index, beta=0.005):

    scores = []
    G = nx.from_scipy_sparse_array(A)

    adj = nx.adjacency_matrix(G, nodelist=range(len(G.nodes)))
    aux = adj.T.multiply(-beta).todense()
    np.fill_diagonal(aux, 1+aux.diagonal())
    sim = np.linalg.inv(aux)
    np.fill_diagonal(sim, sim.diagonal()-1)

    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        scores.append(sim[s,t])

    
    return torch.FloatTensor(scores), edge_index

def initialize(data, method):
    if data.x is None:
        if method == 'one-hot':
            data.x = F.one_hot(torch.arange(data.num_nodes),num_classes=data.num_nodes).float()
            input_size = data.num_nodes
        elif method == 'trainable':
            node_emb_dim = 512
            emb = torch.nn.Embedding(data.num_nodes, node_emb_dim)
            data.emb = emb
            input_size = node_emb_dim
        else:
            raise NotImplementedError
    else:
        input_size = data.x.shape[1]
    return data, input_size

def create_input(data):
    if hasattr(data, 'emb') and data.emb is not None:
        x = data.emb.weight
    else:
        x = data.x
    return x

def GCN(A, all_edges, split_edge, data):
    model_class = GCN_model
    return GNN(data, split_edge, model_class, all_edges)

def SAGE(A, all_edges, split_edge, data):
    model_class = SAGE_model
    return GNN(data, split_edge, model_class, all_edges)


def GNN(data, split_edge, model_class, all_edges):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_layers=2
    hidden_channels=256
    dropout=0.5
    batch_size=64*1024
    lr=0.005
    epochs=20000
    patience=200
    metric="Hits@50"
    evaluator = Evaluator(name='ogbl-ddi')
    data, input_size = initialize(data, 'trainable')
    data = data.to(device)

    model = model_class(in_channels=input_size,
                        hidden_channels=hidden_channels, 
                        out_channels=hidden_channels, 
                        num_layers=num_layers, 
                        dropout=dropout).to(device)
    model.reset_parameters()

    parameters = list(model.parameters())
    if hasattr(data, "emb"):
        parameters += list(data.emb.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    cnt_wait = 0
    best_val = 0.0

    for epoch in range(epochs):
        loss = train(model, data, split_edge,
                        optimizer, batch_size)

        results = test(model, data, split_edge,
                        evaluator, batch_size)

        if results[metric][0] >= best_val:
            best_val = results[metric][0]
            cnt_wait = 0
        else:
            cnt_wait +=1


        if cnt_wait >= patience:
            break
    scores = inference(model, data, all_edges, batch_size)
    data.cpu()
    return scores, None


def train(model, data, split_edge, optimizer, batch_size):
    model.train()

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    pos_train_edge = split_edge['train']['edge'].to(create_input(data).device).t()
    # if dataset != "collab" and dataset != "ppa":
    neg_train_edge = negative_sampling(data.edge_index, num_nodes=data.num_nodes,
                            num_neg_samples=pos_train_edge.size(1), method='dense')
    
    optimizer.zero_grad()
    total_loss = total_examples = 0
    for perm in (pbar := tqdm(DataLoader(range(pos_train_edge.size(1)), batch_size,
                           shuffle=True)) ):
        h = model(create_input(data), data.edge_index)
        pos_edge = pos_train_edge[:,perm]
        neg_edge = neg_train_edge[:,perm]

        train_edges = torch.cat((pos_edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(pos_edge.size(1)), torch.zeros(neg_edge.size(1))), dim=0).to(train_edges.device)

        out = (h[train_edges[0]]*h[train_edges[1]]).mean(1)
        link_loss = criterion(out, train_label)
        loss = link_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(create_input(data), 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_examples += train_label.size(0)
        total_loss += loss.detach() * train_label.size(0)
    
    return total_loss.item() / total_examples


@torch.no_grad()
def test(model, data, split_edge, evaluator, batch_size):
    model.eval()
    # if extractor is None: # run once if no extractor
    h = model(create_input(data), data.edge_index)
    
    pos_train_edge = split_edge['train']['edge'].to(data.edge_index.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.edge_index.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.edge_index.device)
    pos_test_edge = split_edge['test']['edge'].to(data.edge_index.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.edge_index.device)

    concat_edge = torch.cat([pos_valid_edge, neg_valid_edge, pos_test_edge, neg_test_edge], dim=0).t()
    split = [pos_valid_edge.size(0), neg_valid_edge.size(0), pos_test_edge.size(0), neg_test_edge.size(0)]

    preds = []
    for perm in DataLoader(range(concat_edge.size(1)), batch_size):
        edge = concat_edge[:,perm]
        out = (h[edge[0]]* h[edge[1]]).mean(1)
        preds += [out.squeeze().cpu()]
    pred = torch.cat(preds, dim=0)
    pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred = pred.split(split, dim=0)
    results = {}
    for K in [10, 20, 30, 50]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (roc_auc_score(valid_result.cpu().numpy(),valid_pred.cpu().numpy()),roc_auc_score(test_result.cpu().numpy(),test_pred.cpu().numpy()))

    return results


@torch.no_grad()
def inference(model, data, all_edges, batch_size):
    model.eval()
    # if extractor is None: # run once if no extractor
    h = model(create_input(data), data.edge_index)
    
    all_edges = all_edges.to(data.edge_index.device)

    preds = []
    for perm in DataLoader(range(all_edges.size(1)), batch_size):
        edge = all_edges[:,perm]
        out = (h[edge[0]]* h[edge[1]]).mean(1)
        preds += [out.squeeze().cpu()]
    pred = torch.cat(preds, dim=0)
    return pred