import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_scatter import scatter
import torch.nn.functional as F


class LapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)

    e.g.
        posenc_LapPE:
        enable: True
        eigen:
            laplacian_norm: none
            eigvec_norm: L2
            max_freqs: 1
        model: DeepSet
        dim_pe: 4
        layers: 2
        raw_norm_type: none
    """
    def __init__(self, cfg, expand_x=False):
        super().__init__()
        # dim_in = cfg.share.dim_in  # Expected original input node features dim
        pecfg = cfg['posenc_LapPE']
        dim_pe = pecfg["dim_pe"]  # Size of Laplace PE embedding
        model_type = pecfg["model"]  # Encoder NN model type for PEs
        if model_type not in ['Transformer', 'DeepSet']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type
        n_layers = pecfg["layers"]  # Num. layers in PE encoder model
        n_heads = pecfg["n_heads"]  # Num. attention heads in Trf PE encoder
        max_freqs = pecfg["eigen"]["max_freqs"]  # Num. eigenvectors (frequencies)
        norm_type = pecfg["raw_norm_type"]  # Raw PE normalization layer type
        self.pass_as_var = pecfg["pass_as_var"]  # Pass PE also as a separate variable

        self.linear_A = nn.Linear(2, dim_pe)
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'Transformer':
            # Transformer model for LapPE
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe,
                                                       nhead=n_heads,
                                                       batch_first=True)
            self.pe_encoder = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=n_layers)
        else:
            # DeepSet model for LapPE
            layers = []
            if n_layers == 1:
                layers.append(activation())
            else:
                self.linear_A = nn.Linear(2, 2 * dim_pe)
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)  # 将层序列化为 nn.Sequential 模型。

        # self.post_mlp = None
        # if post_n_layers > 0:
        #     # MLP to apply post pooling 
        #     layers = []
        #     if post_n_layers == 1:
        #         layers.append(nn.Linear(dim_pe, dim_pe))
        #         layers.append(activation())
        #     else:
        #         layers.append(nn.Linear(dim_pe, 2 * dim_pe))
        #         layers.append(activation())
        #         for _ in range(post_n_layers - 2):
        #             layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
        #             layers.append(activation())
        #         layers.append(nn.Linear(2 * dim_pe, dim_pe))
        #         layers.append(activation())
        #     self.post_mlp = nn.Sequential(*layers)


    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        EigVals = batch.EigVals
        EigVecs = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)  # 对特征向量应用符号翻转，增强模型鲁棒性。同一个特征值可能对应着不同的特征向量.

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2 ， 创建掩码，标记 pos_enc 中的 NaN 值。

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2，将 pos_enc 中的 NaN 值替换为 0。
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(src=pos_enc,
                                      src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)   # 将掩码位置的值设为 0，使用 clone() 避免 inplace 操作问题。

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        # if self.post_mlp is not None:
        #     pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:  # 是否将 PE 作为单独变量传递（例如用于跳跃连接）， 不加到x上.
            batch.pe_LapPE = pos_enc
            # batch.pe_LapPE = torch.cat((h, pos_enc), 1)
        return batch




class EquivStableLapPENodeEncoder(torch.nn.Module):
    """Equivariant and Stable Laplace Positional Embedding node encoder.

    This encoder simply transforms the k-dim node LapPE to d-dim to be
    later used at the local GNN module as edge weights.
    Based on the approach proposed in paper https://openreview.net/pdf?id=e95i1IHcWj
    
    Args:
        dim_emb: Size of final node embedding
    """

    def __init__(self, cfg):
        super().__init__()

        pecfg = cfg['posenc_EquivStableLapPE']
        max_freqs = pecfg['eigen']['max_freqs']  # Num. eigenvectors (frequencies)
        norm_type = pecfg['raw_norm_type'] # Raw PE normalization layer type

        dim_pe_new = pecfg['dim_pe_new']

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        self.linear_encoder_eigenvec = nn.Linear(max_freqs, dim_pe_new)

    def forward(self, batch):
        # if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
        #     raise ValueError("Precomputed eigen values and vectors are "
        #                      f"required for {self.__class__.__name__}; set "
        #                      f"config 'posenc_EquivStableLapPE.enable' to True")
        # pos_enc = batch.EigVecs
        pos_enc = batch.EigVecs_ES

        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors)
        pos_enc[empty_mask] = 0.  # (Num nodes) x (Num Eigenvectors)

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_encoder_eigenvec(pos_enc)

        # Keep PE separate in a variable
        batch.pe_EquivStableLapPE = pos_enc

        return batch


class KernelPENodeEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding node encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, cfg, expand_x=False):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        # dim_in = cfg.share.dim_in  # Expected original input node features dim

        # pecfg = getattr(cfg, f"posenc_{self.kernel_type}")  # 从配置 cfg 中获取与 kernel_type 对应的位置编码配置对象
        name = f"posenc_{self.kernel_type}"
        pecfg = cfg[name]
        dim_pe = pecfg["dim_pe"]  # Size of the kernel-based PE embedding
        num_rw_steps = len(pecfg["kernel"]["times"])
        model_type = pecfg["model"] # Encoder NN model type for PEs
        n_layers = pecfg["layers"]  # Num. layers in PE encoder model
        norm_type = pecfg["raw_norm_type"] # Raw PE normalization layer type
        self.pass_as_var = pecfg["pass_as_var"]  # Pass PE also as a separate variable

        # if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
        #     raise ValueError(f"PE dim size {dim_pe} is too large for "
        #                      f"desired embedding size of {dim_emb}.")

        # if expand_x and dim_emb - dim_pe > 0:
        #     self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        # self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        # print(batch.pestat_ElstaticSE)
        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        pos_enc = torch.nan_to_num(pos_enc, nan=0.0)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        # if self.expand_x:
        #     h = self.linear_x(batch.x)
        # else:
        #     h = batch.x
        # Concatenate final PEs to input embedding
        # batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch


class RWSENodeEncoder(KernelPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'RWSE'


class HKdiagSENodeEncoder(KernelPENodeEncoder):
    """Heat kernel (diagonal) Structural Encoding node encoder.
    """
    kernel_type = 'HKdiagSE'


class ElstaticSENodeEncoder(KernelPENodeEncoder):
    """Electrostatic interactions Structural Encoding node encoder.
    """
    kernel_type = 'ElstaticSE'





class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 use_bn=False, use_ln=False, dropout=0.5, activation='relu',
                 residual=False):
        super().__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()

        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
                 use_bn=True, dropout=0.5, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels,
                             2, use_bn=use_bn, dropout=dropout,
                             activation=activation)
            self.layers.append(GINConv(update_net))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i - 1](x)
                    elif x.ndim == 3:
                        x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)
                    else:
                        raise ValueError('invalid x dim')
            x = layer(x, edge_index)
        return x


class GINDeepSigns(nn.Module):
    """ Sign invariant neural network with MLP aggregation.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 k, dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        x = x.transpose(0, 1) # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)
        x = x.transpose(0, 1).reshape(N, -1)  # K x N x Out -> N x (K * Out)
        x = self.rho(x)  # N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class MaskedGINDeepSigns(nn.Module):
    """ Sign invariant neural network with sum pooling and DeepSet.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        self.rho = MLP(out_channels, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def batched_n_nodes(self, batch_index):
        batch_size = batch_index.max().item() + 1
        one = batch_index.new_ones(batch_index.size(0))
        n_nodes = scatter(one, batch_index, dim=0, dim_size=batch_size,
                          reduce='add')  # Number of nodes in each graph.
        n_nodes = n_nodes.unsqueeze(1)
        return torch.cat([size * n_nodes.new_ones(size) for size in n_nodes])

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        K = x.shape[1]  # Max. number of eigen vectors / frequencies.
        x = x.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)  # K x N x Out
        x = x.transpose(0, 1)  # K x N x Out -> N x K x Out

        batched_num_nodes = self.batched_n_nodes(batch_index)
        mask = torch.cat([torch.arange(K).unsqueeze(0) for _ in range(N)])
        mask = (mask.to(batch_index.device) < batched_num_nodes.unsqueeze(1)).bool()
        # print(f"     - mask: {mask.shape} {mask}")
        # print(f"     - num_nodes: {num_nodes}")
        # print(f"     - batched_num_nodes: {batched_num_nodes.shape} {batched_num_nodes}")
        x[~mask] = 0
        x = x.sum(dim=1)  # (sum over K) -> N x Out
        x = self.rho(x)  # N x Out -> N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class SignNetNodeEncoder(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \phi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)

    E.g.
        posenc_SignNet:
        enable: True
        eigen:
            laplacian_norm: none
            eigvec_norm: L2
            max_freqs: 37  # Max graph size in molpcba is 332, but 97.1% are <= 37
        model: DeepSet
        dim_pe: 32  # Note: In original SignNet codebase dim_pe is always equal to max_freq
        layers: 8  # Num. layers in \phi model
        post_layers: 3  # Num. layers in \rho model; The original uses the same as in \phi
        phi_hidden_dim: 64
        phi_out_dim: 64
    """

    def __init__(self, cfg, expand_x=False):
        super().__init__()
        # dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = cfg["posenc_SignNet"]
        dim_pe = pecfg["dim_pe"]  # Size of PE embedding
        model_type = pecfg["model"]  # Encoder NN model type for SignNet
        if model_type not in ['MLP', 'DeepSet']:
            raise ValueError(f"Unexpected SignNet model {model_type}")
        self.model_type = model_type
        sign_inv_layers = pecfg["layers"]  # Num. layers in \phi GNN part
        rho_layers = pecfg["post_layers"]  # Num. layers in \rho MLP/DeepSet
        if rho_layers < 1:
            raise ValueError(f"Num layers in rho model has to be positive.")
        max_freqs = pecfg["eigen"]["max_freqs"]  # Num. eigenvectors (frequencies)
        self.pass_as_var = pecfg["pass_as_var"]  # Pass PE also as a separate variable

        # if dim_emb - dim_pe < 0:
        #     raise ValueError(f"SignNet PE size {dim_pe} is too large for "
        #                      f"desired embedding size of {dim_emb}.")

        # if expand_x:
        #     self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        # Sign invariant neural network.
        if self.model_type == 'MLP':
            self.sign_inv_net = GINDeepSigns(
                in_channels=1,
                hidden_channels=pecfg["phi_hidden_dim"],
                out_channels=pecfg["phi_out_dim"],
                num_layers=sign_inv_layers,
                k=max_freqs,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        elif self.model_type == 'DeepSet':
            self.sign_inv_net = MaskedGINDeepSigns(
                in_channels=1,
                hidden_channels=pecfg["phi_hidden_dim"],
                out_channels=pecfg["phi_out_dim"],
                num_layers=sign_inv_layers,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        else:
            raise ValueError(f"Unexpected model {self.model_type}")

    def forward(self, batch):
        if not (hasattr(batch, 'eigvals_sn') and hasattr(batch, 'eigvecs_sn')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_SignNet.enable' to True")
        # eigvals = batch.eigvals_sn
        eigvecs = batch.eigvecs_sn

        # pos_enc = torch.cat((eigvecs.unsqueeze(2), eigvals), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc = eigvecs.unsqueeze(-1)  # (Num nodes) x (Num Eigenvectors) x 1

        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 1

        # SignNet
        pos_enc = self.sign_inv_net(pos_enc, batch.edge_index, batch.batch)  # (Num nodes) x (pos_enc_dim)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        # batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_SignNet = pos_enc
        return batch

# pestat_RWSE=[2708, 20], pestat_HKdiagSE=[2708, 16], 
# pestat_ElstaticSE=[2708, 10], in_degrees=[2708], 
# out_degrees=[2708], batch=[2708], 
# ptr=[2], split='train', 
# pe_LapPE=[2708, 20], pe_SignNet=[2708, 20])

