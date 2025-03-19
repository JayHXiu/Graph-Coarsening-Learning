# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import (Conv1d, Embedding, Linear, MaxPool1d, ModuleList, ReLU,
                      Sequential)
from torch_geometric.nn import GATv2Conv, GCNConv, MessagePassing
from torch_geometric.nn import GINConv as GINConv_
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn import SAGEConv as SAGEConv_
from torch_geometric.nn import (global_add_pool, global_max_pool,
                                global_mean_pool, global_sort_pool)
from torch_geometric.utils import (degree, is_undirected, sort_edge_index,
                                   to_undirected,softmax)
from torch_sparse import transpose


class SGNN(torch.nn.Module):
    def __init__(self, GNN, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, pooling='sum',
                 jk=False, add_self_loops=False, foundation_mode=True, heads=4,
                 use_graph_embedding=False):

        super(SGNN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.pooling_name = pooling
        self.foundation_mode = foundation_mode

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(GNN(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.jk = jk
        if self.jk:
            graph_dim = num_layers * hidden_channels
        else:
            graph_dim = hidden_channels
        self.lin1 = Linear(graph_dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

        if self.foundation_mode:
            self.gat = SupportAttention(graph_dim, hidden_channels, add_self_loops, 
                                        heads=heads, use_graph_embedding=use_graph_embedding)

    def reset_parameters(self):
        self.z_embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.foundation_mode:
            self.gat.reset_parameters()

    def get_emb(self, z, edge_index, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x[node_id].to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]
        for conv in self.convs[:-1]:
            x = conv(xs[-1], edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]
        x = self.convs[-1](xs[-1], edge_index, edge_weight)
        xs += [x]
        if self.jk:
            x = torch.cat(xs[1:], dim=1)
        else:
            x = xs[-1]
        return x

    def pooling(self, x, batch):
        if self.pooling_name == "center":  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
        else:
            if self.pooling_name == "sum":  # sum pooling
                pooling_func = global_add_pool
            elif self.pooling_name == "mean":
                pooling_func = global_mean_pool
            elif self.pooling_name == "max":
                pooling_func = global_max_pool
            else:
                raise NotImplementedError
            x = pooling_func(x, batch)
        return x
    
    def forward(self, x, data, edge_weight=None, support_cache=None):
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        node_id = data.node_id# ;torch.cuda.synchronize()
        if support_cache is None:
            all_x = self.get_emb(z, edge_index, x, edge_weight, node_id)# ;torch.cuda.synchronize()
            graph_x = self.pooling(all_x, batch)# ;torch.cuda.synchronize()
        else:
            # mask the edges from support graphs, 
            # since the embedding of support graphs are already in support_cache
            # also subset the graph to nodes in query graph
            query_node_idx_subset = []# ;torch.cuda.synchronize()
            batchs = []# ;torch.cuda.synchronize()
            i=0# ;torch.cuda.synchronize()
            bs = data.query_graph.argwhere().view(-1)# ;torch.cuda.synchronize()
            edges_indices = torch.zeros_like(edge_index[0], dtype=torch.bool, device=support_cache.device)# ;torch.cuda.synchronize()
            slice_dict_edge_index = data._slice_dict["edge_index"]# ;torch.cuda.synchronize()
            slice_dict_z = data._slice_dict["z"]# ;torch.cuda.synchronize()
            for b in bs:
                edges_indices[slice_dict_edge_index[b]:slice_dict_edge_index[b+1]] = 1# ;torch.cuda.synchronize()
                batchs.extend([i]*(slice_dict_z[b+1]-slice_dict_z[b]))# ;torch.cuda.synchronize()
                query_node_idx_subset.extend(list(range(slice_dict_z[b],slice_dict_z[b+1])))# ;torch.cuda.synchronize()
                i+=1
            edge_index = edge_index[:,edges_indices]# ;torch.cuda.synchronize()
            if edge_weight is not None:
                edge_weight = edge_weight[edges_indices]# ;torch.cuda.synchronize()
            batch = torch.LongTensor(batchs).to(support_cache.device)# ;torch.cuda.synchronize()
            query_node_idx_subset = torch.LongTensor(query_node_idx_subset).to(support_cache.device)# ;torch.cuda.synchronize()
            # relabel edge_index
            mapping = edge_index[0].new_full((z.size(0), ), -1)
            mapping[query_node_idx_subset] = torch.arange(query_node_idx_subset.size(0), device=mapping.device)
            edge_index = mapping[edge_index]
            z = z[query_node_idx_subset]
            node_id = node_id[query_node_idx_subset]
            query_node_x = self.get_emb(z, edge_index, x, edge_weight, node_id)# ;torch.cuda.synchronize()
            query_graph_x = self.pooling(query_node_x, batch)# ;torch.cuda.synchronize()
            # support_cache: m_ways x dim -->  (m_ways * batch_size) x dim


        if self.foundation_mode:
            # Support graph to interact with query graph
            if support_cache is None:
                graph_x, alpha = self.gat(data, graph_x)
            else:
                graph_x, alpha = self.gat.forward_cache_support(query_graph_x, support_cache, data)
        else:
            graph_x = graph_x[data.query_graph]
        # MLP.
        results = self.classifier(graph_x)
        return results

    def classifier(self, graph_x):
        graph_x = F.relu(self.lin1(graph_x))
        graph_x = F.dropout(graph_x, p=self.dropout, training=self.training)
        graph_x = self.lin2(graph_x)
        return graph_x


    def get_support(self, x, data, edge_weight=None):
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        node_id = data.node_id
        all_x = self.get_emb(z, edge_index, x, edge_weight, node_id)
        support_graph_x = self.pooling(all_x, batch)[~data.query_graph]

        return support_graph_x

    def get_parameters(self, parts:str = 'all'):
        """
            parts: 'all', 'emb', 'encoder', 'classifier'.
        """
        if parts == 'all':
            return list(self.parameters())
        elif parts == 'emb':
            return list(self.z_embedding.parameters())
        elif parts == 'encoder':
            return list(self.convs.parameters())
        elif parts == 'classifier':
            return list(self.lin1.parameters()) + list(self.lin2.parameters())
        else:
            raise NotImplementedError


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class SEAL(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None, 
                 dynamic_train=False, GNN=GCNConv, use_feature=False, 
                 node_embedding=None):
        super(SEAL, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:  # Transform percentile to number.
            if train_dataset is None:
                k = 30
            else:
                if dynamic_train:
                    sampled_train = train_dataset[:1000]
                else:
                    sampled_train = train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)
        print(f"Top k nodes: {self.k}")

        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for i in range(0, num_layers-1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def get_emb(self, z, edge_index, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x[node_id].to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        all_x = torch.cat(xs[1:], dim=-1)
        return all_x
    
    def pooling(self, all_x, batch):
        # Global pooling.
        x = global_sort_pool(all_x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        return x

    def forward(self, x, data, edge_weight=None, epoch=None, **kwargs):
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        node_id = data.node_id
        all_x = self.get_emb(z, edge_index, x, edge_weight, node_id)
        graph_x = self.pooling(all_x, batch)

        # MLP.
        graph_x = F.relu(self.lin1(graph_x))
        graph_x = F.dropout(graph_x, p=0.5, training=self.training)
        graph_x = self.lin2(graph_x)

        results = graph_x
        return results

    def get_parameters(self, parts:str = 'all'):
        """
            parts: 'all', 'emb', 'encoder', 'classifier'.
        """
        if parts == 'all':
            return list(self.parameters())
        elif parts == 'emb':
            return list(self.z_embedding.parameters())
        elif parts == 'encoder':
            return list(self.convs.parameters())
        elif parts == 'agg':
            return list(self.conv1.parameters()) + list(self.conv2.parameters())
        elif parts == 'classifier':
            return list(self.lin1.parameters()) + list(self.lin2.parameters())
        else:
            raise NotImplementedError


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, train_dataset,
                 use_feature=False, node_embedding=None, dropout=0.5, 
                 jk=True, train_eps=False):
        super(GIN, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)
        self.jk = jk

        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += train_dataset.num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.conv1 = GINConv(
            Sequential(
                Linear(initial_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ),
            train_eps=train_eps)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ),
                    train_eps=train_eps))

        self.dropout = dropout
        if self.jk:
            edge_dim = num_layers * hidden_channels
            self.lin1 = Linear(edge_dim, hidden_channels)
        else:
            edge_dim = hidden_channels
            self.lin1 = Linear(edge_dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x, data, edge_weight=None, epoch=None, **kwargs):
        z = data.z
        edge_index = data.edge_index
        batch = data.batch
        node_id = data.node_id
        all_x = self.get_emb(z, edge_index, x, edge_weight, node_id)
        graph_x = self.pooling(all_x, batch)
        
        # MLP.
        graph_x = F.relu(self.lin1(graph_x))
        graph_x = F.dropout(graph_x, p=self.dropout, training=self.training)
        graph_x = self.lin2(graph_x)

        results = graph_x
        return results

    def get_emb(self, z, edge_index, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x[node_id].to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        
        x = self.conv1(x, edge_index, edge_atten=edge_weight)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_atten=edge_weight)
            xs += [x]
        if self.jk:
            all_x = torch.cat(xs, dim=1)
        else:
            all_x = xs[-1]
        return all_x
    
    def pooling(self, all_x, batch):
        return global_mean_pool(all_x, batch)

    def get_parameters(self, parts:str = 'all'):
        """
            parts: 'all', 'emb', 'encoder', 'classifier'.
        """
        if parts == 'all':
            return list(self.parameters())
        elif parts == 'emb':
            return list(self.z_embedding.parameters())
        elif parts == 'encoder':
            return list(self.conv1.parameters()) + list(self.convs.parameters())
        elif parts == 'classifier':
            return list(self.lin1.parameters()) + list(self.lin2.parameters())
        else:
            raise NotImplementedError



class SupportAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels, add_self_loops=False, heads=4, use_graph_embedding=False):
        super(SupportAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T_embedding = Embedding(3, in_channels) # 0 --> neg, 1 --> pos, 2 --> query
        self.add_self_loops = add_self_loops
        self.gat = AttentionLayer(in_channels=in_channels, out_channels=out_channels, 
                                  heads=heads, concat=False, add_self_loops=False,
                                  use_graph_embedding=use_graph_embedding)

    def reset_parameters(self):
        self.gat.reset_parameters()

    def forward(self, data, graph_x):
        query_graph_x = graph_x[data.query_graph]
        support_graph_xs = graph_x[~data.query_graph]
        support_graph_ys = data.y[~data.query_graph]
        support_graph_batch = data.query_graph_idx[~data.query_graph]

        # query_graph_x += self.T_embedding.weight[2]
        # support_graph_xs += self.T_embedding(support_graph_ys)

        data_graph_x = torch.concat([query_graph_x, support_graph_xs], dim=0)
        row = torch.arange(start=query_graph_x.size(0), end=data_graph_x.size(0)).to(query_graph_x.device)
        col = support_graph_batch
        data_graph_edge_index = torch.stack([row, col], dim=0) # no need to to_undirected()

        # T_embedding
        query_T_embedding = self.T_embedding.weight[2].view(1,-1).expand(query_graph_x.size(0),-1)
        support_T_embedding = self.T_embedding(support_graph_ys)
        T_embedding = torch.cat([query_T_embedding, support_T_embedding], dim=0)

        if self.add_self_loops:
            query_graph_ids = torch.arange(start=0, end=data.query_graph.sum()).to(query_graph_x.device)
            self_loops_query_graph = torch.stack([query_graph_ids, query_graph_ids], dim=0)
            data_graph_edge_index = torch.cat([data_graph_edge_index, self_loops_query_graph], dim=1)
        x, (_,alpha) = self.gat(data_graph_x, data_graph_edge_index, T_embedding, return_attention_weights=True)
        query_x = x[:query_graph_x.size(0)]
        return query_x, alpha
    
    def forward_cache_support(self, query_graph_x, support_cache, data):
        support_graph_ys = data.y[~data.query_graph]
        support_graph_batch = data.query_graph_idx[~data.query_graph]
        counts = torch.bincount(support_graph_batch)
        assert torch.unique(counts).size(0) == 1, "Support graph must be the same across different batch"
        assert (support_graph_ys.view(-1,counts[0]).float().mean(0)==support_graph_ys[:counts[0]]).all(), "the labels must be the same"

        data_graph_x = torch.concat([query_graph_x, support_cache], dim=0)
        row1 = torch.arange(start=query_graph_x.size(0), 
                            end=data_graph_x.size(0)).to(query_graph_x.device)
        row = row1.repeat(support_graph_batch.max()+1) # repeat batch_size times
        col = support_graph_batch
        data_graph_edge_index = torch.stack([row, col], dim=0)

        # T_embedding
        query_T_embedding = self.T_embedding.weight[2].view(1,-1).expand(query_graph_x.size(0),-1)
        support_T_embedding = self.T_embedding(support_graph_ys[:counts[0]])
        T_embedding = torch.cat([query_T_embedding, support_T_embedding], dim=0)

        if self.add_self_loops:
            query_graph_ids = torch.arange(start=0, end=data.query_graph.sum()).to(query_graph_x.device)
            self_loops_query_graph = torch.stack([query_graph_ids, query_graph_ids], dim=0)
            data_graph_edge_index = torch.cat([data_graph_edge_index, self_loops_query_graph], dim=1)
        x, (_,alpha) = self.gat(data_graph_x, data_graph_edge_index, T_embedding, return_attention_weights=True)
        query_x = x[:query_graph_x.size(0)]
        return query_x, alpha





class MLP(nn.Sequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


from typing import Any, Callable, Optional, Tuple, Union

from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)

# Adopted from PyG library (https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/dropout.py)
from torch import Tensor
from torch_geometric.typing import (Adj, OptPairTensor, OptTensor, PairTensor,
                                    Size)

class AttentionLayer(GATv2Conv):
    def __init__(self, **kwargs):
        self.use_graph_embedding = kwargs.pop('use_graph_embedding', False)
        super(AttentionLayer, self).__init__(**kwargs)
    
    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], edge_index_j) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        num_edges, num_heads, hidden_channels = x_j.shape
        edge_attr = edge_attr.view(size_i,1,hidden_channels).expand(size_i,num_heads,hidden_channels)
        edge_attr =  edge_attr.index_select(self.node_dim, edge_index_j)
        if self.use_graph_embedding:
            out = x_j + edge_attr
        else:
            out = edge_attr
        return out * alpha.unsqueeze(-1)


class SAGEConv(SAGEConv_):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_atten: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        # when source_to_target, msg flows from x[edge_index[0]] to x[edge_index[1]]. scatter(src=x[edge_index[0]], index=edge_index[1])
        out = self.propagate(edge_index, x=x, size=size, edge_atten=edge_atten)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten.view(-1,1)
        else:
            return x_j
        

class GINConv(GINConv_):

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_atten: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_atten=edge_atten)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten.view(-1,1)
        else:
            return x_j

    def message_and_aggregate(self, adj_t,
                              x) -> Tensor:
        raise NotImplementedError