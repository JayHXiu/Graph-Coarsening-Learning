"""Dataset loading, structural precomputation, and train/val/test splits."""

import logging
import os

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj, to_networkx, to_scipy_sparse_matrix, to_undirected

from graph_precompute import (
    get_electrostatic_function_encoding,
    get_heat_kernels_diag,
    get_lap_decomp_stats,
    get_rw_landing_probs,
)


def load_dataset(args):
    """Load MoleculeNet data, precompute encodings, and return PyG loaders."""
    print(f"加载数据集: {args.dataset}")
    logging.info(f"加载数据集: {args.dataset}")

    original_dataset = MoleculeNet(
        root=os.path.join(args.data_dir, args.dataset),
        name=args.dataset,
    )

    num_features = original_dataset.num_features
    if args.dataset in ['Tox21', 'ToxCast', 'SIDER', 'ClinTox', 'MUV']:
        num_tasks = original_dataset.num_tasks
        task_type = 'classification'
        out_channels = num_tasks
    else:
        num_tasks = 1
        task_type = 'classification'
        out_channels = 2

    num_edge_features = 1
    sample = original_dataset[0]
    if getattr(sample, 'edge_attr', None) is not None:
        num_edge_features = sample.edge_attr.shape[1]

    print("为数据集中每个图计算位置编码...")
    logging.info("为数据集中每个图计算位置编码...")
    processed_data_list = []

    for data_idx in range(len(original_dataset)):
        data = original_dataset.get(data_idx).clone()

        if data_idx % 100 == 0:
            print(f"  处理图 {data_idx + 1}/{len(original_dataset)}")
            logging.info(f"  处理图 {data_idx + 1}/{len(original_dataset)}")

        N_graph = data.num_nodes if data.num_nodes is not None else data.x.shape[0]
        graph_orig_edge_index = data.edge_index
        if graph_orig_edge_index is None or graph_orig_edge_index.numel() == 0:
            device = data.x.device if hasattr(data.x, 'device') else 'cpu'
            graph_orig_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            graph_undir_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            graph_undir_edge_index = to_undirected(data.edge_index, num_nodes=N_graph)

        if 'RWSE' in args.pe_types:
            kernel_param = args.RWSE_times_func
            if len(kernel_param) == 0:
                raise ValueError("List of kernel times required for RWSE")
            data.pestat_RWSE = get_rw_landing_probs(
                ksteps=kernel_param,
                edge_index=graph_orig_edge_index,
                num_nodes=N_graph,
            )

        if 'HKdiagSE' in args.pe_types or 'HKfullPE' in args.pe_types:
            if graph_undir_edge_index.numel() > 0:
                L_heat = to_scipy_sparse_matrix(
                    *get_laplacian(graph_undir_edge_index, normalization=None, num_nodes=N_graph)
                )
                evals_heat_np, evects_heat_np = np.linalg.eigh(L_heat.toarray())
            else:
                evals_heat_np = np.zeros(N_graph)
                evects_heat_np = np.eye(N_graph)

            evals_heat = torch.from_numpy(evals_heat_np).float()
            evects_heat = torch.from_numpy(evects_heat_np).float()

            if 'HKfullPE' in args.pe_types:
                raise NotImplementedError("HKfullPE per graph needs specific implementation")
            if 'HKdiagSE' in args.pe_types:
                kernel_param_hk = args.HKSE_times_func
                if len(kernel_param_hk) == 0:
                    raise ValueError("Diffusion times are required for heat kernel")
                data.pestat_HKdiagSE = get_heat_kernels_diag(
                    evects_heat, evals_heat, kernel_times=kernel_param_hk, space_dim=0
                )

        if 'ElstaticSE' in args.pe_types:
            if graph_undir_edge_index.numel() > 0:
                elstatic = get_electrostatic_function_encoding(graph_undir_edge_index, N_graph)
            else:
                elstatic = torch.zeros((N_graph, args.dim_pe_ETSE), dtype=torch.float)
            data.pestat_ElstaticSE = elstatic

        if 'LapPE' in args.pe_types:
            if graph_undir_edge_index.numel() > 0:
                L_lap = to_scipy_sparse_matrix(
                    *get_laplacian(
                        graph_undir_edge_index,
                        normalization=args.laplacian_norm,
                        num_nodes=N_graph,
                    )
                )
                evals_lap_np, evects_lap_np = np.linalg.eigh(L_lap.toarray())
            else:
                evals_lap_np = np.zeros(N_graph)
                evects_lap_np = np.eye(N_graph)

            data.EigVals, data.EigVecs = get_lap_decomp_stats(
                evals=evals_lap_np,
                evects=evects_lap_np,
                max_freqs=args.max_freqs,
                eigvec_norm=args.eigvec_norm,
            )

        if 'EquivStableLapPE' in args.pe_types:
            if graph_undir_edge_index.numel() > 0:
                L_es = to_scipy_sparse_matrix(
                    *get_laplacian(
                        graph_undir_edge_index,
                        normalization=args.laplacian_norm_ES,
                        num_nodes=N_graph,
                    )
                )
                evals_es_np, evects_es_np = np.linalg.eigh(L_es.toarray())
            else:
                evals_es_np = np.zeros(N_graph)
                evects_es_np = np.eye(N_graph)

            data.EigVals_ES, data.EigVecs_ES = get_lap_decomp_stats(
                evals=evals_es_np,
                evects=evects_es_np,
                max_freqs=args.max_freqs_ES,
                eigvec_norm=args.eigvec_norm_ES,
            )

        if 'SignNet' in args.pe_types:
            if graph_undir_edge_index.numel() > 0:
                L_sn = to_scipy_sparse_matrix(
                    *get_laplacian(
                        graph_undir_edge_index,
                        normalization=args.laplacian_norm_SN,
                        num_nodes=N_graph,
                    )
                )
                evals_sn_np, evects_sn_np = np.linalg.eigh(L_sn.toarray())
            else:
                evals_sn_np = np.zeros(N_graph)
                evects_sn_np = np.eye(N_graph)

            data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
                evals=evals_sn_np,
                evects=evects_sn_np,
                max_freqs=args.max_freqs_SN,
                eigvec_norm=args.eigvec_norm_SN,
            )

        if args.edge_types or args.global_types:
            nx_graph = to_networkx(data, to_undirected=True)

        if 'ShortestPathEdge' in args.edge_types:
            path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph))
            edge_spds = []
            for i, j in data.edge_index.t().tolist():
                try:
                    edge_spds.append(path_lengths[i][j])
                except KeyError:
                    edge_spds.append(float('inf'))

            edge_spds_tensor = torch.tensor(edge_spds, dtype=torch.long)
            max_len = 0
            if torch.any(torch.isfinite(edge_spds_tensor.float())):
                max_len = torch.max(
                    edge_spds_tensor[torch.isfinite(edge_spds_tensor.float())].float()
                )
            edge_spds_tensor[edge_spds_tensor == float('inf')] = int(max_len) + 1
            data.precomputed_shortest_paths = edge_spds_tensor

        if 'HeatKernelEdge' in args.edge_types and hasattr(data, 'pestat_HKdiagSE'):
            node_hk = data.pestat_HKdiagSE
            data.heat_kernels = (node_hk[data.edge_index[0]] + node_hk[data.edge_index[1]]) / 2

        if 'RandomWalkEdge' in args.edge_types and hasattr(data, 'pestat_RWSE'):
            node_rw = data.pestat_RWSE
            data.rw_edge_features = (node_rw[data.edge_index[0]] + node_rw[data.edge_index[1]]) / 2

        if 'SpectralGraph' in args.global_types:
            max_freqs = args.config['posenc_SpectralGraph']['max_freqs']
            eigenval_type = args.config['posenc_SpectralGraph'].get('eigenval_type', 'lap')

            if eigenval_type == 'lap':
                if graph_undir_edge_index.numel() > 0:
                    L = to_scipy_sparse_matrix(
                        *get_laplacian(graph_undir_edge_index, normalization=None, num_nodes=N_graph)
                    )
                    evals = np.linalg.eigvalsh(L.toarray())
                else:
                    evals = np.zeros(N_graph)
            elif eigenval_type == 'adj':
                if graph_undir_edge_index.numel() > 0:
                    adj = to_dense_adj(graph_undir_edge_index, max_num_nodes=N_graph)[0].numpy()
                    evals = np.linalg.eigvalsh(adj)
                else:
                    evals = np.zeros(N_graph)
            else:
                raise ValueError(f"Unknown eigenval type: {eigenval_type}")

            evals = np.sort(evals)
            graph_evals = torch.zeros(max_freqs, dtype=torch.float)
            num_evals_to_copy = min(len(evals), max_freqs)
            graph_evals[:num_evals_to_copy] = torch.from_numpy(evals[:num_evals_to_copy])
            setattr(data, f"graph_eigenvals_{eigenval_type}", graph_evals)

        if 'DegreeDistribution' in args.global_types:
            degrees = [d for _, d in nx_graph.degree()]
            max_degree = args.config['posenc_DegreeDistribution']['max_degree']
            degree_hist = torch.zeros(max_degree + 1)
            for d in degrees:
                if d <= max_degree:
                    degree_hist[d] += 1
            if degree_hist.sum() > 0:
                degree_hist /= degree_hist.sum()
            data.precomputed_degree_dist = degree_hist

        if 'ClusteringCoefficient' in args.global_types:
            clustering_coeffs_list = list(nx.clustering(nx_graph).values())
            clustering_coeffs = torch.tensor(clustering_coeffs_list, dtype=torch.float)
            num_bins = args.config['posenc_ClusteringCoefficient']['num_histogram_bins']
            cluster_hist = torch.histc(clustering_coeffs, bins=num_bins, min=0, max=1)
            if cluster_hist.sum() > 0:
                cluster_hist /= cluster_hist.sum()
            mean_coeff = clustering_coeffs.mean() if clustering_coeffs_list else torch.tensor(0.0)
            std_coeff = clustering_coeffs.std() if clustering_coeffs_list else torch.tensor(0.0)
            data.precomputed_cluster_coeff = torch.cat([
                cluster_hist,
                mean_coeff.unsqueeze(0),
                std_coeff.unsqueeze(0),
            ])

        processed_data_list.append(data)

    print("位置编码计算完成。")
    logging.info("位置编码计算完成。")

    labels = [d.y.item() for d in processed_data_list]
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n数据集类别分布:")
    logging.info("\n数据集类别分布:")
    for label, count in zip(unique_labels, counts):
        msg = f"类别 {label}: {count} 样本 ({count / len(processed_data_list) * 100:.2f}%)"
        print(msg)
        logging.info(msg)

    print(f"边特征维度: {num_edge_features}")
    logging.info(f"边特征维度: {num_edge_features}")

    indices = np.arange(len(processed_data_list))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=args.seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        stratify=[labels[i] for i in train_val_idx],
        random_state=args.seed,
    )

    train_dataset = [processed_data_list[i] for i in train_idx]
    valid_dataset = [processed_data_list[i] for i in val_idx]
    test_dataset = [processed_data_list[i] for i in test_idx]

    print(f"数据集大小 - 训练集: {len(train_dataset)}, 验证集: {len(valid_dataset)}, 测试集: {len(test_dataset)}")
    logging.info(
        f"数据集大小 - 训练集: {len(train_dataset)}, 验证集: {len(valid_dataset)}, 测试集: {len(test_dataset)}"
    )

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = PyGDataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size)

    return train_loader, valid_loader, test_loader, num_features, out_channels, task_type, num_tasks, num_edge_features
