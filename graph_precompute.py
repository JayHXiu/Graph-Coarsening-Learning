"""Graph structural feature precomputation for positional encodings."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """Normalize Laplacian eigenvectors."""
    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)
    elif normalization == "L2":
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)
    elif normalization == "abs-max":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
    elif normalization == "wavelength":
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom * 2 / np.pi
    elif normalization == "wavelength-asin":
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = eigval_denom
    elif normalization == "wavelength-soft":
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1
        denom = denom * eigval_denom
    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    return EigVecs / denom


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Laplacian eigen-decomposition statistics for PE."""
    N = len(evals)
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None, num_nodes=None, space_dim=0):
    """Random-walk landing probabilities."""
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source = edge_index[0]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)

    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
    return torch.cat(rws, dim=0).transpose(0, 1)


def get_heat_kernels_diag(evects, evals, kernel_times=None, space_dim=0):
    """Heat kernel diagonal."""
    if kernel_times is None:
        kernel_times = []
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]
        evals = evals.unsqueeze(-1)
        evects = evects.transpose(0, 1)
        eigvec_mul = evects ** 2
        for t in kernel_times:
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul, dim=0, keepdim=False)
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)
    return heat_kernels_diag


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Electrostatic Green's function node encoding."""
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()

    return torch.stack([
        electrostatic.min(dim=0)[0],
        electrostatic.max(dim=0)[0],
        electrostatic.mean(dim=0),
        electrostatic.std(dim=0),
        electrostatic.min(dim=1)[0],
        electrostatic.max(dim=1)[0],
        electrostatic.mean(dim=1),
        electrostatic.std(dim=1),
        (DinvA * electrostatic).sum(dim=0),
        (DinvA * electrostatic).sum(dim=1),
    ], dim=1)
