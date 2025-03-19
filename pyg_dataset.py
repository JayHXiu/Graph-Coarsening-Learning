import os
import torch
import random
import shutil
import os.path as osp
from pathlib import Path
from torch import Tensor
from torch_geometric.data import Data, Dataset, InMemoryDataset, OnDiskDataset
from torch_geometric.utils import coalesce, to_undirected, from_networkx

from torch_sparse import SparseTensor

from database import SQLiteDatabase
import scipy.sparse as ssp
from tqdm import tqdm
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Tuple, Union)

from utils import (construct_pyg_graph, extract_enclosing_subgraphs,
                   get_pos_neg_edges, k_hop_subgraph)
from graph_generation import generate_graph, GraphType



class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, num_samples=None, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.num_samples = num_samples
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        if False: # self.split == "all"
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = self.cache_data, self.cache_slices
    @property
    def processed_file_names(self):
        name = 'SEAL_{}'.format(self.split)
        name += '.pt'
        return [name]

    def process(self):
        if self.split == "all":
            split = "train"
        else:
            split = self.split
        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.num_samples)#;exit(0)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None
        
        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, 1, self.num_hops, self.node_label, 
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, 0, self.num_hops, self.node_label, 
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        if False: # self.split == "all"
            torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
            del pos_list, neg_list
        else: # no cache
            self.cache_data, self.cache_slices = self.collate(pos_list + neg_list)


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, num_samples=None, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False, **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.num_samples = num_samples
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDynamicDataset, self).__init__(root)

        if split == "all":
            split = "train"
        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.num_samples)
        print(f"split: {split}, pos_edge: {pos_edge.size()}, neg_edge: {neg_edge.size()}")
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight, 
                self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), 
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None
        
    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def shuffle(self):
        # only works for train set
        pos_edge, neg_edge = get_pos_neg_edges('train', self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes, 
                                               self.num_samples)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)
        return self

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop, 
                             self.max_nodes_per_hop, node_features=self.data.x, 
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label)

        return data

def get_foundation_dataset(**kwargs):
    ### parse schema
    dynamic_dataset = SEALDynamicDataset(**kwargs)
    ref_data = dynamic_dataset.get(0)
    schema: Dict[str, Any] = {} # {'name': str}
    for key, value in ref_data.to_dict().items():
        if isinstance(value, (int, float, str)):
            schema[key] = value.__class__
        elif isinstance(value, Tensor) and value.dim() == 0:
            schema[key] = dict(dtype=value.dtype, size=(-1, ))
        elif isinstance(value, Tensor):
            size = list(value.size())
            size[ref_data.__cat_dim__(key, value)] = -1
            schema[key] = dict(dtype=value.dtype, size=tuple(size))
        else:
            schema[key] = object

    on_disk_data = SEALOnDiskDataset(root=osp.join(kwargs['root'], 'on_disk'), schema=schema, kwargs_for_dataset=kwargs)
    return on_disk_data

class SEALOnDiskDataset(OnDiskDataset):
    BACKENDS = {
        'sqlite': SQLiteDatabase,
    }
    def __init__(
        self,
        root: str,
        schema: dict,
        kwargs_for_dataset: dict = {},
    ):
        self.kwargs_for_dataset = kwargs_for_dataset
        super().__init__(
            root=root,
            transform=None,
            backend='sqlite',
            schema=schema,
        )
        # connect to the /tmp database
        self.db.close()
        src = Path(self.root) / 'processed' / self.processed_file_names
        src_split = str(src).split('/')
        self.in_memory_dataset_name = src_split[-5]
        config_name = src_split[-4]
        dst = Path(os.getenv('TMPDIR',f"/tmp/kdong2/{os.getenv('JOB_ID','None')}"))/self.in_memory_dataset_name/config_name/'processed'/self.processed_file_names
        if not dst.parent.exists():
            dst.parent.mkdir(parents=True)
        print(f"copying {src} to {dst}")
        shutil.copyfile(src, dst)
        print(f"copy done")
        # call once for connection
        self.db.path = dst
        self.db.connect()

    @property
    def processed_file_names(self) -> str:
        num_samples = self.kwargs_for_dataset.get('num_samples', '')
        return f'{self.backend}_{num_samples}.db'

    def process(self):
        in_memory_dataset = SEALDataset(**self.kwargs_for_dataset)
        dataset_name = str(in_memory_dataset.root).split('/')[-2]
        self.in_memory_dataset_name = dataset_name
        _iter = [
            in_memory_dataset.get(i)
            for i in in_memory_dataset.indices()
        ]
        if True:  # pragma: no cover
            _iter = tqdm(_iter, desc='Converting to OnDiskDataset')

        data_list: List[Data] = []
        for i, data in enumerate(_iter):
            # data.name = dataset_name
            data_list.append(data)
            if i + 1 == len(in_memory_dataset) or (i + 1) % 1000 == 0:
                self.extend(data_list)
                data_list = []

    def serialize(self, data: Data) -> Dict[str, Any]:
        return data.to_dict()

    def deserialize(self, data: Dict[str, Any]) -> Data:
        return Data.from_dict(data)

    # this one seems to work. but it is slow
    # def connect(self):
    #     import sqlite3
    #     self._connection = sqlite3.connect(self.path,isolation_level=None,check_same_thread=False,)
    #     self._connection.execute('pragma journal_mode=wal')
    #     self._cursor = self._connection.cursor()
    
    # def multi_get(
    #     self,
    #     indices: Union[Iterable[int], Tensor, slice, range],
    #     batch_size: Optional[int] = None,
    # ):
    #     r"""Gets a list of data objects from the specified indices."""
    #     if len(indices) == 1:
    #         data_list = [self.db.get(indices[0])]
    #     else:
    #         data_list = self.db.multi_get(indices, batch_size)

    #     return [self.deserialize(data) for data in data_list]

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return (f'OnDisk{self.in_memory_dataset_name}('
                f'{arg_repr})')

# deprecated: we can use torch.utils.data.ConcatDataset
# class MergeDataset(Dataset):
#     def __init__(self, root, dataset_list):
#         self.dataset_list = dataset_list
#         super(MergeDataset, self).__init__(root)
    
#     def __len__(self):
#         return sum([len(dataset) for dataset in self.dataset_list])
    
#     def len(self):
#         return self.__len__()

#     def get(self, idx):
#         for dataset in self.dataset_list:
#             if idx < len(dataset):
#                 return dataset[idx]
#             else:
#                 idx -= len(dataset)
#         raise IndexError

class QueryDataset(Dataset):
    def __init__(self, root, m, query_dataset, support_dataset: SEALDataset):
        """
            support_dataset: By default, the first half is positive and the second half is negative.
        """
        assert len(support_dataset) >= 2*m, \
            "support dataset is smaller than m"
        self.m = m
        self.query_dataset = query_dataset
        self.support_dataset = support_dataset
        super(QueryDataset, self).__init__(root)

    def __len__(self):
        return len(self.query_dataset)
    
    def len(self):
        return self.__len__()

    def get(self, idx: int) -> Data:
        # two ways to get the query and support graph together:
        # 1. each idx get a batch of graphs including both query and support graphs
        # Pros:
        #   - the structure of each Data passed to Dataloader is clear
        # Cons:
        #   - when num_workers>1, each worker needs to get a batch of graphs. not sure if this is optimal for multi-processing

        # 2. each idx only get a query graph or a support graph, then resemble it in the dataloader
        # Pros:
        #   - each worker only needs to get a single graph
        # Cons:
        #   - very strong coupling between the dataloader and the dataset
        #   - hard to support shuffled data access

        # Take the #1 for cleaner structure
        if (id(self.query_dataset) == id(self.support_dataset)) and isinstance(self.support_dataset, SEALOnDiskDataset):
            query_is_support_flg = True
        else:
            query_is_support_flg = False
        support_data_idxes = self.slice_select(idx, len(self.support_dataset)//2).tolist()
        if isinstance(self.support_dataset, SEALOnDiskDataset):
            if query_is_support_flg:
                # it is possible idx is in support_data_idxes, SQLite will not return the same row twice
                if idx in support_data_idxes:
                    query_idx_in_idxes = support_data_idxes.index(idx)
                else:
                    support_data_idxes += [idx]
                    query_idx_in_idxes = None
            # SQL will fetch data according to support_data_idxes but with sorted order
            subset, inv = torch.tensor(support_data_idxes).unique(return_inverse=True)
            support_data = self.support_dataset.multi_get(support_data_idxes)
            support_data = [support_data[i] for i in inv]
            if query_is_support_flg:
                if query_idx_in_idxes is None:
                    query_data = support_data[-1]
                    support_data = support_data[:-1]
                else:
                    query_data = support_data[query_idx_in_idxes].clone()
        else:
            support_data = [self.support_dataset[i].clone() for i in support_data_idxes]

        support_and_query_data = []
        for each_data in support_data:
            data_copy = each_data
            data_copy.query_graph = False
            data_copy.query_node = torch.zeros(data_copy.num_nodes, dtype=torch.long)
            support_and_query_data.append(data_copy)

        if not query_is_support_flg:
            # get query graph
            query_data = self.query_dataset[idx].clone()
        query_data.query_graph = True
        query_data.query_node = torch.ones(query_data.num_nodes, dtype=torch.long)
        support_and_query_data.append(query_data)

        # combine query graph and support graph into one data instance
        # assert len(support_and_query_data) == self.m*2+1

        # optional assert
        # hit_query = 0
        # hit_support_pos = 0
        # hit_support_neg = 0
        # # name = query_data.name
        # for each in support_and_query_data:
        #     # assert name == each.name
        #     if each.query_graph:
        #         hit_query += 1
        #     else:
        #         if each.y == 1:
        #             hit_support_pos += 1
        #         else:
        #             hit_support_neg += 1
        # assert hit_query == 1
        # assert hit_support_pos == self.m
        # assert hit_support_neg == self.m
        return support_and_query_data

    def slice_select(self, idx, max_len):
        # Dataset is randomly shuffled when `get_pos_neg_edges`
        all_idx = torch.arange(max_len)
        start_idx = (idx*self.m) % max_len
        if start_idx + self.m > max_len:
            select = torch.cat([all_idx[start_idx:], all_idx[:start_idx+self.m-max_len]])
        else:
            select = all_idx[start_idx:start_idx+self.m]
        
        # concat pos and neg edges
        select = torch.cat([select, select+max_len])
        return select

def random_select(l, m):
    perm = torch.randperm(len(l))
    select = [l[i] for i in perm[:m]]
    return select


class SyntheticDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        N: int=10000,
    ):
        self.dataset_name = dataset_name
        self.N = N
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'{self.dataset_name}_{self.N}.pt'

    def process(self):
        graph_type_str = f"GraphType.{self.dataset_name}"
        nx_data = generate_graph(self.N, eval(graph_type_str), seed=0)
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])


