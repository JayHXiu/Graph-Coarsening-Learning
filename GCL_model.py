import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import degree, add_self_loops
from torch_geometric.nn import GCNConv, GINEConv
from torch_geometric.nn import global_mean_pool
import math
from node_encoder import LapPENodeEncoder, EquivStableLapPENodeEncoder,\
    RWSENodeEncoder, HKdiagSENodeEncoder, ElstaticSENodeEncoder, SignNetNodeEncoder
from edge_encoder import ShortestPathEdgeEncoder, HeatKernelEdgeEncoder, RandomWalkEdgeEncoder, MultiScaleEdgeEncoder
from global_encoder import SpectralGraphEncoder, DegreeDistributionEncoder, ClusteringCoefficientEncoder, MultiFeatureGraphEncoder
from torch_geometric.nn.dense import dense_diff_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch

class DifferentiableCoarseningNetwork(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_embed_dim, assign_hidden_dim, num_coarse_nodes, use_edge_features_in_embedding=True):
        """
        可微图粗化网络.

        参数:
            node_in_dim (int): 输入节点特征的维度.
            edge_in_dim (int): 输入边特征的维度. 如果为0或 use_edge_features_in_embedding=False, 则在嵌入GNN中不显式使用边特征.
            node_embed_dim (int): 嵌入GNN输出的节点嵌入维度.
            assign_hidden_dim (int): 分配GNN的隐藏层维度.
            num_coarse_nodes (int): 粗化后图中节点的数量 (簇的数量 K).
            use_edge_features_in_embedding (bool): 是否在嵌入GNN中使用边特征 (如果 edge_in_dim > 0).
        """
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.num_coarse_nodes = num_coarse_nodes
        self.use_edge_features_in_embedding = use_edge_features_in_embedding and self.edge_in_dim > 0

        # 1. 嵌入GNN (Embedding GNN)
        #    学习节点表示，用于后续的簇分配.
        #    我们使用两层GNN来获取更丰富的节点表示.
        current_dim = self.node_in_dim

        if self.use_edge_features_in_embedding:
            # 重新思考GINEConv的用法：
            # `nn`: A neural network $h_{\mathbf{\Theta}}$ that maps node features
            # `x_j` of shape `[-1, in_channels]` to shape `[-1, out_channels]`, *e.g.*, defined by `torch.nn.Sequential`.
            # `edge_dim`: Edge feature dimensionality.
            # GINEConv: (1+eps) * x_i + sum_{j \in N(i)} MLP(x_j + MLP_edge(e_ij))
            # 我们将使用一个MLP来处理节点特征，并指定edge_dim。
            self.embedding_conv1_node_mlp = nn.Sequential(nn.Linear(current_dim, node_embed_dim // 2), nn.ReLU())
            self.embedding_conv1 = GINEConv(self.embedding_conv1_node_mlp, edge_dim=self.edge_in_dim)
            current_dim = node_embed_dim // 2

            self.embedding_conv2_node_mlp = nn.Sequential(nn.Linear(current_dim, node_embed_dim), nn.ReLU())
            self.embedding_conv2 = GINEConv(self.embedding_conv2_node_mlp, edge_dim=self.edge_in_dim)
            current_dim = node_embed_dim

        else: # 不使用边特征，或边特征维度为0
            self.embedding_conv1 = GCNConv(current_dim, node_embed_dim // 2)
            current_dim = node_embed_dim // 2
            self.embedding_conv2 = GCNConv(current_dim, node_embed_dim)
            current_dim = node_embed_dim

        # 2. 分配GNN (Assignment GNN)
        #    接收节点嵌入，并预测每个节点到 K 个簇的分配.
        #    通常使用 GCNConv 序列.
        self.assignment_gnn = nn.ModuleList()
        self.assignment_gnn.append(GCNConv(current_dim, assign_hidden_dim)) # 输入来自嵌入GNN
        self.assignment_gnn.append(GCNConv(assign_hidden_dim, self.num_coarse_nodes)) # 输出 K 个簇的原始分数 (logits)


    def forward(self, x_orig, edge_index_orig, edge_attr_orig=None, batch_orig=None):
        """
        前向传播.

        参数:
            x_orig (torch.Tensor): 原始节点特征, shape [N_total_orig, node_in_dim]. N_total_orig 是批次中所有图的原始节点总数.
            edge_index_orig (torch.Tensor): 原始图的边索引 (COO格式), shape [2, E_total_orig]. E_total_orig 是批次中所有图的原始边总数.
            edge_attr_orig (torch.Tensor, optional): 原始边的特征, shape [E_total_orig, edge_in_dim]. 默认为 None.
            batch_orig (torch.Tensor, optional): 将节点映射到其各自图的批处理向量, shape [N_total_orig]. 如果为 None, 则假设只有一个图.

        返回:
            dict: 包含粗化图信息的字典:
                "node_features_coarse" (torch.Tensor): 粗化后节点的特征, shape [N_coarse_total, node_in_dim].
                "edge_index_coarse" (torch.Tensor): 粗化后图的边索引, shape [2, E_coarse_total].
                "edge_features_coarse" (torch.Tensor): 粗化后边的特征 (标量权重), shape [E_coarse_total].
                "batch_coarse" (torch.Tensor): 粗化后节点到其图的批处理向量.
                "link_prediction_loss" (torch.Tensor): DiffPool 的链接预测损失.
                "entropy_loss" (torch.Tensor): DiffPool 的熵损失 (用于正则化分配).
                "assignment_matrix_softmax" (torch.Tensor): 计算得到的软分配矩阵 S (稠密格式), shape [batch_size, N_max_orig, K].
        """
        if batch_orig is None: # 如果没有提供批次信息，则假设是单个图
            batch_orig = torch.zeros(x_orig.size(0), dtype=torch.long, device=x_orig.device)

        # --- 步骤 1: 获取用于分配的节点嵌入 ---
        node_embeddings = x_orig
        if self.use_edge_features_in_embedding and edge_attr_orig is not None:
            node_embeddings = F.relu(self.embedding_conv1(node_embeddings, edge_index_orig, edge_attr=edge_attr_orig))
            node_embeddings = self.embedding_conv2(node_embeddings, edge_index_orig, edge_attr=edge_attr_orig) # 通常最后一层不加ReLU，除非后面还有处理
        else:
            node_embeddings = F.relu(self.embedding_conv1(node_embeddings, edge_index_orig))
            node_embeddings = self.embedding_conv2(node_embeddings, edge_index_orig)
        # node_embeddings shape: [N_total_orig, node_embed_dim]

        # --- 步骤 2: 预测分配矩阵 S 的 logits ---
        # 分配GNN也使用原始图结构来传播信息
        assignment_logits = F.relu(self.assignment_gnn[0](node_embeddings, edge_index_orig))
        assignment_logits = self.assignment_gnn[1](assignment_logits, edge_index_orig)
        # assignment_logits shape: [N_total_orig, num_coarse_nodes]

        # 对分配 logits 应用 softmax 得到软分配矩阵 S (对于每个图中的每个节点)
        # dense_diff_pool 需要批处理的稠密分配矩阵
        assignment_matrix_s_softmax = torch.softmax(assignment_logits, dim=-1)

        # --- 步骤 3: 为 dense_diff_pool 准备输入 ---
        # dense_diff_pool 函数期望的输入格式:
        #   x: [batch_size, N_max_orig, F_node]
        #   adj: [batch_size, N_max_orig, N_max_orig]
        #   s: [batch_size, N_max_orig, K]
        #   mask: [batch_size, N_max_orig] (可选, 用于指示哪些节点是真实节点 vs padding)

        # 将稀疏数据转换为批处理的稠密数据
        # x_dense: [batch_size, N_max_orig, node_in_dim] (使用原始节点特征进行池化)
        # mask: [batch_size, N_max_orig]
        x_dense, mask = to_dense_batch(x_orig, batch_orig, fill_value=0)
        # adj_dense: [batch_size, N_max_orig, N_max_orig]
        # 注意：如果 edge_attr_orig 是多维的，to_dense_adj 通常用于生成加权邻接矩阵，
        # 权重通常是标量。如果 edge_attr_orig 是多维的，你需要决定如何将其转换为标量权重。
        # 例如，取平均值、范数，或者通过一个小型MLP投影。
        # 为简单起见，如果 edge_attr_orig 是多维的，我们将不传递它给 to_dense_adj，
        # 这意味着 adj_dense 将是一个无权重的邻接矩阵（或者说权重为1）。
        # 如果 edge_attr_orig 本身就是标量权重 [E_total_orig]，则可以直接使用。
        scalar_edge_weights_for_adj = None
        if edge_attr_orig is not None:
            if self.edge_in_dim == 1 and edge_attr_orig.ndim == 1: # 已经是标量权重
                scalar_edge_weights_for_adj = edge_attr_orig
            elif self.edge_in_dim == 1 and edge_attr_orig.ndim == 2 and edge_attr_orig.size(1) == 1: # [E, 1]
                scalar_edge_weights_for_adj = edge_attr_orig.squeeze(-1)
            # else: 多维边特征，此处不转换为标量权重，adj将是0/1矩阵

        adj_dense = to_dense_adj(edge_index_orig, batch=batch_orig, edge_attr=scalar_edge_weights_for_adj)

        # s_dense: [batch_size, N_max_orig, num_coarse_nodes]
        s_dense, _ = to_dense_batch(assignment_matrix_s_softmax, batch_orig, fill_value=0)


        # --- 步骤 4: 执行可微池化操作 ---
        # dense_diff_pool 的 x 输入是用于计算新的粗化节点特征的特征矩阵。
        # 通常，我们使用原始节点特征 x_orig，而不是学习到的 node_embeddings。
        # 这是因为 DiffPool 的目标是池化原始特征。
        # 如果希望池化学习到的嵌入，可以将 node_embeddings 转换为稠密格式并作为 x 传入。
        # 这里我们遵循标准做法，池化 x_orig。
        x_coarse_dense, adj_coarse_dense, link_prediction_loss, entropy_loss = \
            dense_diff_pool(x_dense, adj_dense, s_dense, mask=mask)
        # x_coarse_dense: shape [batch_size, num_coarse_nodes, node_in_dim]
        # adj_coarse_dense: shape [batch_size, num_coarse_nodes, num_coarse_nodes]

        # --- 步骤 5: 导出粗化后的图结构 (边索引) 和边特征 ---
        # 从批处理的稠密格式转换回稀疏格式 (PyG Batch 对象)
        # 首先，我们需要为粗化后的节点创建一个新的批处理向量 batch_coarse
        batch_size = x_coarse_dense.size(0)
        batch_coarse = torch.arange(batch_size, device=x_orig.device).repeat_interleave(self.num_coarse_nodes)

        # 将 x_coarse_dense 展平回 [N_coarse_total, node_in_dim]
        node_features_coarse = x_coarse_dense.reshape(-1, self.node_in_dim) # N_coarse_total = batch_size * num_coarse_nodes

        # 将 adj_coarse_dense 转换为稀疏的 edge_index_coarse 和 edge_features_coarse
        # dense_to_sparse 通常处理单个邻接矩阵。我们需要迭代每个批次中的图。
        edge_indices_list = []
        edge_attrs_list = []
        current_node_offset = 0
        for i in range(batch_size):
            adj_i = adj_coarse_dense[i] # shape [num_coarse_nodes, num_coarse_nodes]
            edge_index_i, edge_attr_i = dense_to_sparse(adj_i)
            if edge_index_i.numel() > 0: # 如果存在边
                edge_indices_list.append(edge_index_i + current_node_offset)
                edge_attrs_list.append(edge_attr_i)
            current_node_offset += self.num_coarse_nodes

        if len(edge_indices_list) > 0:
            edge_index_coarse = torch.cat(edge_indices_list, dim=1)
            edge_features_coarse = torch.cat(edge_attrs_list, dim=0) # 这些是标量权重
        else: # 如果粗化后没有边 (例如，如果 num_coarse_nodes 很小或图很稀疏)
            edge_index_coarse = torch.empty((2, 0), dtype=torch.long, device=x_orig.device)
            edge_features_coarse = torch.empty((0,), dtype=x_orig.dtype, device=x_orig.device)


        # 注意: `edge_features_coarse` 是从 `adj_coarse_dense` 中新生成的标量权重。
        # 这些标量权重可以被视为 "粗化后的边特征"。
        # 如果需要通过池化原始多维边特征得到多维粗化边特征，
        # 则需要一个更复杂的、自定义的边池化机制，这超出了 `dense_diff_pool` 的标准功能。

        return {
            "node_features_coarse": node_features_coarse,
            "edge_index_coarse": edge_index_coarse,
            "edge_features_coarse": edge_features_coarse, # 这是标量权重
            "batch_coarse": batch_coarse,
            "link_prediction_loss": link_prediction_loss,
            "entropy_loss": entropy_loss,
            "assignment_matrix_softmax": s_dense # 返回批处理的稠密分配矩阵
        }


class MultiViewAttention(nn.Module):
    """多视图注意力模块
    完全按照设计实现，原图embedding作为Query，粗化图embedding作为Key
    """
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.5, temperature=1.0):
        super(MultiViewAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size必须能被num_heads整除"
        
        # Query投影层 - 用于原图特征
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        
        # Key投影层 - 用于粗化图特征
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        
        # Value投影层 - 用于粗化图特征
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, keys, values=None):
        """
        多头注意力前向传播
        
        Args:
            query: Query特征 [batch_size, hidden_size]
            keys: Key特征 [batch_size, n_views, hidden_size]
            values: Value特征，如果为None则使用keys [batch_size, n_views, hidden_size]
        """
        if values is None:
            values = keys
            
        batch_size = query.size(0)
        n_views = keys.size(1)
        
        # 应用Layer Normalization
        query = self.norm1(query)
        keys = self.norm2(keys)
        values = self.norm2(values)
        
        # 线性投影
        q = self.query_proj(query)  # [batch_size, hidden_size]
        k = self.key_proj(keys)     # [batch_size, n_views, hidden_size]
        v = self.value_proj(values) # [batch_size, n_views, hidden_size]
        
        # 重塑为多头形式
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_views, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_views, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores / self.temperature
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权聚合value
        out = torch.matmul(attn_weights, v)  # [batch_size, num_heads, 1, head_dim]
        
        # 重塑回原始维度
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_size)
        out = out.squeeze(1)  # [batch_size, hidden_size]
        
        # 输出投影
        out = self.output_proj(out)
        
        return out, attn_weights

class GraphCoarseningModel(nn.Module):
    """
    图粗化学习模型(GCL)
    """

    def __init__(self,args, cfg):
        super(GraphCoarseningModel, self).__init__()
        self.cfg = cfg
        self.in_channels = cfg['in_channels']
        self.hidden_channels = cfg['hidden_channels']
        self.out_channels = cfg['out_channels']
        self.encoder_type = args.pe_types
        self.n_layers = cfg['n_layers']
        self.task_type = cfg['task_type']
        self.edge_encoder_type = getattr(args, 'edge_types', []) or cfg.get('edge_types', [])
        self.global_encoder_type = getattr(args, 'global_types', []) or cfg.get('global_types', [])
        
        # 选择激活函数
        if cfg['activation'] == 'relu':
            self.activation = nn.ReLU()
        elif cfg['activation'] == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif cfg['activation'] == 'elu':
            self.activation = nn.ELU()
        elif cfg['activation'] == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()  # 默认使用ReLU
        
        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(cfg['raw_node_feature'], cfg['in_channels']),
            nn.LayerNorm(cfg['in_channels']) if cfg['layer_norm'] else nn.Identity(),
            self.activation,
            nn.Dropout(cfg['dropout']) 
        )
        
        ### 节点编码器初始化
        if 'EquivStableLapPE' in self.encoder_type:
            self.node_encoder_EquivStableLapPE = EquivStableLapPENodeEncoder(cfg)
        if 'RWSE' in self.encoder_type:
            self.node_encoder_RWSE = RWSENodeEncoder(cfg)
        if 'HKdiagSE' in self.encoder_type:
            self.node_encoder_HKdiagSE = HKdiagSENodeEncoder(cfg)
        if 'ElstaticSE' in self.encoder_type:
            self.node_encoder_ElstaticSE = ElstaticSENodeEncoder(cfg)
        if 'LapPE' in self.encoder_type:
            self.node_encoder_LapPE = LapPENodeEncoder(cfg)
        if 'SignNet' in self.encoder_type:
            self.node_encoder_SignNet = SignNetNodeEncoder(cfg)

        ### 边级编码器初始化
        if 'ShortestPathEdge' in self.edge_encoder_type:
            self.edge_encoder_ShortestPath = ShortestPathEdgeEncoder(cfg)
        if 'HeatKernelEdge' in self.edge_encoder_type:
            self.edge_encoder_HeatKernel = HeatKernelEdgeEncoder(cfg)
        if 'RandomWalkEdge' in self.edge_encoder_type:
            self.edge_encoder_RandomWalk = RandomWalkEdgeEncoder(cfg)
        if 'MultiScaleEdge' in self.edge_encoder_type:
            self.edge_encoder_MultiScale = MultiScaleEdgeEncoder(cfg)

        ### 图级编码器初始化
        if 'SpectralGraph' in self.global_encoder_type:
            self.global_encoder_Spectral = SpectralGraphEncoder(cfg)
        if 'DegreeDistribution' in self.global_encoder_type:
            self.global_encoder_DegreeDist = DegreeDistributionEncoder(cfg)
        if 'ClusteringCoefficient' in self.global_encoder_type:
            self.global_encoder_ClusterCoef = ClusteringCoefficientEncoder(cfg)
        if 'MultiFeatureGraph' in self.global_encoder_type:
            self.global_encoder_MultiFeature = MultiFeatureGraphEncoder(cfg)

        # GNN层
        model_list = [GCNConv(cfg['in_channels'], cfg['hidden_channels'])]
        for _ in range(cfg['n_layers']-1):
            model_list.append(GCNConv(cfg['hidden_channels'], cfg['hidden_channels']))
        self.original_gnn = nn.ModuleList(model_list)

        # 图粗化模型
        self.coarsening_model = DifferentiableCoarseningNetwork(
            node_in_dim=cfg['in_channels'],
            edge_in_dim=cfg['edge_feature_dim'],
            node_embed_dim=cfg['hidden_channels'],
            assign_hidden_dim=cfg['hidden_channels']*2,
            num_coarse_nodes=5,
            use_edge_features_in_embedding=True
        )

        # 为图级编码器添加投影层，以匹配hidden_channels
        self.global_proj_layers = nn.ModuleDict()
        if 'SpectralGraph' in self.global_encoder_type:
            self.global_proj_layers['SpectralGraph'] = nn.Linear(cfg['posenc_SpectralGraph']['dim_pe'], cfg['hidden_channels'])
        if 'DegreeDistribution' in self.global_encoder_type:
            self.global_proj_layers['DegreeDistribution'] = nn.Linear(cfg['posenc_DegreeDistribution']['dim_pe'], cfg['hidden_channels'])
        if 'ClusteringCoefficient' in self.global_encoder_type:
            self.global_proj_layers['ClusteringCoefficient'] = nn.Linear(cfg['posenc_ClusteringCoefficient']['dim_pe'], cfg['hidden_channels'])
        if 'MultiFeatureGraph' in self.global_encoder_type:
            self.global_proj_layers['MultiFeatureGraph'] = nn.Linear(cfg['posenc_MultiFeatureGraph']['dim_pe'], cfg['hidden_channels'])
 
        # 多视图注意力融合
        self.multiview_attention = MultiViewAttention(
            hidden_size=cfg['hidden_channels'],
            num_heads=cfg['n_heads'],
            dropout=cfg['dropout'],
            temperature=cfg['temperature']
        )
        
        # 输出层
        self.linear = nn.Linear(cfg['hidden_channels'], cfg['out_channels'])
        
        # 初始化模型权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.numel() > 0:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None and module.bias.numel() > 0:
                    nn.init.zeros_(module.bias)
    
    def compute_loss(self, predictions, targets, edge_index, temperature=0.1):
        """
        计算包含对比学习损失的完整损失函数
        
        Args:
            predictions: 模型预测输出
            targets: 目标标签
            edge_index: 边索引
            temperature: 对比学习温度参数
            
        Returns:
            losses: 包含各部分损失的字典
        """
        losses = {}
        
        # 1. 任务特定损失 (分类/回归)
        if self.task_type == 'classification':
            if targets.dim() > 1 and targets.size(1) > 1:
                # 多标签分类任务
                task_loss = F.binary_cross_entropy_with_logits(predictions, targets)
            else:
                # 单标签分类任务
                task_loss = F.cross_entropy(predictions, targets.long().view(-1))
        else:
            # 回归任务
            task_loss = F.mse_loss(predictions, targets)
            
        losses['task_loss'] = task_loss

        # 2. 对比学习损失 (基于不同视图之间的互信息最大化)
        # 获取所有粗化图的嵌入表示
        n_views = len(self.coarsened_representations)
        if n_views > 1:  # 仅当有多个视图时计算对比学习损失
            contrastive_loss = 0.0
            batch_size = self.coarsened_representations[0].size(0)
            
            # 遍历所有视图对
            for i in range(n_views):
                for j in range(i+1, n_views):
                    # 获取两个视图的表示
                    z_i = F.normalize(self.coarsened_representations[i], p=2, dim=1)
                    z_j = F.normalize(self.coarsened_representations[j], p=2, dim=1)
                    
                    # 计算相似度矩阵
                    sim_matrix = torch.matmul(z_i, z_j.t()) / temperature
                    
                    # InfoNCE损失
                    labels = torch.arange(batch_size, device=z_i.device)
                    loss_i = F.cross_entropy(sim_matrix, labels)
                    loss_j = F.cross_entropy(sim_matrix.t(), labels)
                    
                    # 平均损失
                    contrastive_loss += (loss_i + loss_j) / 2
                    
            # 按视图对数量归一化
            n_pairs = n_views * (n_views - 1) // 2
            contrastive_loss = contrastive_loss / n_pairs
            losses['contrastive_loss'] = contrastive_loss
        else:
            losses['contrastive_loss'] = torch.tensor(0.0, device=predictions.device)
            
        # 3. 结构保持正则化损失
        # 这部分用于确保粗化图与原图的结构一致性
        structure_loss = 0.0
        for coarsened_graph in self.coarsened_graphs:
            # 链接预测损失 (来自DiffPool)
            structure_loss += coarsened_graph['link_prediction_loss']
            
            # 熵正则化损失 (鼓励每个节点明确分配给一个簇)
            structure_loss += coarsened_graph['entropy_loss']
        
        # 平均损失
        if len(self.coarsened_graphs) > 0:
            structure_loss = structure_loss / len(self.coarsened_graphs)
            losses['structure_loss'] = structure_loss
        else:
            losses['structure_loss'] = torch.tensor(0.0, device=predictions.device)
        
        # 总损失：任务损失 + α*对比学习损失 + β*结构保持损失
        alpha = self.cfg.get('alpha', 0.1)  # 对比学习权重系数，如图片所示
        beta = self.cfg.get('beta', 0.05)   # 结构保持正则化系数
        
        total_loss = task_loss + alpha * losses['contrastive_loss'] + beta * losses['structure_loss']
        losses['total_loss'] = total_loss
        
        return losses
        
    def forward(self, batch):
        """
        前向传播函数
        
        Args:
            batch: PyG批次数据对象
            
        Returns:
            output: 模型预测输出
            attention_weights: 多视图注意力权重
            attention_dict: 按编码类型组织的注意力权重字典
        """
        # 确保输入为float类型
        x = batch.x.float()
        edge_index = batch.edge_index
        if batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()

        # 特征投影
        x = self.feature_proj(x)

        # 原图通过GCN层
        query_features = x
        for i, gnn in enumerate(self.original_gnn):
            query_features = F.relu(gnn(query_features, edge_index))
            query_features = F.dropout(query_features, p=0.5, training=self.training)

        # 获取图级表示作为查询
        graphs_features = global_mean_pool(query_features, batch.batch)

        # 应用节点编码器
        if 'LapPE' in self.encoder_type:
            batch = self.node_encoder_LapPE(batch)
        if 'SignNet' in self.encoder_type:
            batch = self.node_encoder_SignNet(batch)
        if 'RWSE' in self.encoder_type:
            batch = self.node_encoder_RWSE(batch)
        if 'HKdiagSE' in self.encoder_type:
            batch = self.node_encoder_HKdiagSE(batch)
        if 'ElstaticSE' in self.encoder_type:
            batch = self.node_encoder_ElstaticSE(batch)
        if 'EquivStableLapPE' in self.encoder_type:
            batch = self.node_encoder_EquivStableLapPE(batch)
            
        # 应用边级编码器
        if 'ShortestPathEdge' in self.edge_encoder_type:
            batch = self.edge_encoder_ShortestPath(batch)
        if 'HeatKernelEdge' in self.edge_encoder_type:
            batch = self.edge_encoder_HeatKernel(batch)
        if 'RandomWalkEdge' in self.edge_encoder_type:
            batch = self.edge_encoder_RandomWalk(batch)
        if 'MultiScaleEdge' in self.edge_encoder_type:
            batch = self.edge_encoder_MultiScale(batch)
                
        # 应用图级编码器
        if 'SpectralGraph' in self.global_encoder_type:
            batch = self.global_encoder_Spectral(batch)
        if 'DegreeDistribution' in self.global_encoder_type:
            batch = self.global_encoder_DegreeDist(batch)
        if 'ClusteringCoefficient' in self.global_encoder_type:
            batch = self.global_encoder_ClusterCoef(batch)
        if 'MultiFeatureGraph' in self.global_encoder_type:
            batch = self.global_encoder_MultiFeature(batch)

        # 1. 首先粗化原始特征
        raw_coarsened = self.coarsening_model(x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # 2. 收集所有可用编码，并进行粗化
        self.coarsened_graphs = [raw_coarsened]
        all_encoding_names = ['raw']
        
        # 节点编码粗化
        if hasattr(batch, 'pe_LapPE'):
            LapPE_coarsened = self.coarsening_model(batch.pe_LapPE, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(LapPE_coarsened)
            all_encoding_names.append('LapPE')
        if hasattr(batch, 'pe_SignNet'):
            SignNet_coarsened = self.coarsening_model(batch.pe_SignNet, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(SignNet_coarsened)
            all_encoding_names.append('SignNet')
        if hasattr(batch, 'pe_RWSE'):
            RWSE_coarsened = self.coarsening_model(batch.pe_RWSE, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(RWSE_coarsened)
            all_encoding_names.append('RWSE')
        if hasattr(batch, 'pe_HKdiagSE'):
            HKdiagSE_coarsened = self.coarsening_model(batch.pe_HKdiagSE, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(HKdiagSE_coarsened)
            all_encoding_names.append('HKdiagSE')
        if hasattr(batch, 'pe_ElstaticSE'):
            ElstaticSE_coarsened = self.coarsening_model(batch.pe_ElstaticSE, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(ElstaticSE_coarsened)
            all_encoding_names.append('ElstaticSE')
        if hasattr(batch, 'pe_EquivStableLapPE'):
            EquivStableLapPE_coarsened = self.coarsening_model(batch.pe_EquivStableLapPE, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(EquivStableLapPE_coarsened)
            all_encoding_names.append('EquivStableLapPE')

        # 3. 边级编码转换与粗化
        if hasattr(batch, 'pe_ShortestPath'):
            row, col = batch.edge_index
            edge_feature = batch.pe_ShortestPath
            node_edge_feature = scatter_add(edge_feature, col, dim=0, dim_size=batch.x.size(0))
            degrees = degree(col, batch.x.size(0), dtype=node_edge_feature.dtype).clamp(min=1)
            node_edge_feature = node_edge_feature / degrees.view(-1, 1)
            ShortestPath_coarsened = self.coarsening_model(node_edge_feature, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(ShortestPath_coarsened)
            all_encoding_names.append('ShortestPath')
        if hasattr(batch, 'pe_HeatKernel'):
            row, col = batch.edge_index
            edge_feature = batch.pe_HeatKernel
            node_edge_feature = scatter_add(edge_feature, col, dim=0, dim_size=batch.x.size(0))
            degrees = degree(col, batch.x.size(0), dtype=node_edge_feature.dtype).clamp(min=1)
            node_edge_feature = node_edge_feature / degrees.view(-1, 1)
            HeatKernel_coarsened = self.coarsening_model(node_edge_feature, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(HeatKernel_coarsened)
            all_encoding_names.append('HeatKernel')
        if hasattr(batch, 'pe_RandomWalk'):
            row, col = batch.edge_index
            edge_feature = batch.pe_RandomWalk
            node_edge_feature = scatter_add(edge_feature, col, dim=0, dim_size=batch.x.size(0))
            degrees = degree(col, batch.x.size(0), dtype=node_edge_feature.dtype).clamp(min=1)
            node_edge_feature = node_edge_feature / degrees.view(-1, 1)
            RandomWalk_coarsened = self.coarsening_model(node_edge_feature, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(RandomWalk_coarsened)
            all_encoding_names.append('RandomWalk')
        if hasattr(batch, 'pe_MultiScaleEdge'):
            row, col = batch.edge_index
            edge_feature = batch.pe_MultiScaleEdge
            node_edge_feature = scatter_add(edge_feature, col, dim=0, dim_size=batch.x.size(0))
            degrees = degree(col, batch.x.size(0), dtype=node_edge_feature.dtype).clamp(min=1)
            node_edge_feature = node_edge_feature / degrees.view(-1, 1)
            MultiScaleEdge_coarsened = self.coarsening_model(node_edge_feature, batch.edge_index, batch.edge_attr, batch.batch)
            self.coarsened_graphs.append(MultiScaleEdge_coarsened)
            all_encoding_names.append('MultiScaleEdge')

        # 4. 收集图级编码 - "图级全局结构编码视角"
        graph_encodings_map = {}
        if hasattr(batch, 'pe_SpectralGraph'):
            graph_encodings_map['SpectralGraph'] = batch.pe_SpectralGraph
        if hasattr(batch, 'pe_DegreeDistribution'):
            graph_encodings_map['DegreeDistribution'] = batch.pe_DegreeDistribution
        if hasattr(batch, 'pe_ClusteringCoefficient'):
            graph_encodings_map['ClusteringCoefficient'] = batch.pe_ClusteringCoefficient
        if hasattr(batch, 'pe_MultiFeatureGraph'):
            graph_encodings_map['MultiFeatureGraph'] = batch.pe_MultiFeatureGraph
        
        # 5. 对粗化图进行学习
        coarsened_graphs_embedding = []
        self.coarsened_representations = []
        
        for i, coarsened_graph in enumerate(self.coarsened_graphs):
            coarsened_x = coarsened_graph['node_features_coarse']
            coarsened_edge_index = coarsened_graph['edge_index_coarse']
            coarsened_edge_attr = coarsened_graph['edge_features_coarse']
            
            # 将粗化后的图通过相同的GNN层
            coarsened_features = coarsened_x
            for gnn_layer_idx, gnn in enumerate(self.original_gnn):
                coarsened_features = F.relu(gnn(coarsened_features, coarsened_edge_index, coarsened_edge_attr))
                coarsened_features = F.dropout(coarsened_features, p=0.5, training=self.training)
            
            # 获取粗化图的图级表示
            coarsened_graph_embedding = global_mean_pool(coarsened_features, coarsened_graph['batch_coarse'])
            coarsened_graphs_embedding.append(coarsened_graph_embedding)
            self.coarsened_representations.append(coarsened_graph_embedding)
            
        # 6. 添加经过投影的图级编码到嵌入列表
        for name, encoding in graph_encodings_map.items():
            projected_encoding = self.global_proj_layers[name](encoding)
            coarsened_graphs_embedding.append(projected_encoding)
            self.coarsened_representations.append(projected_encoding)
            all_encoding_names.append(name)

        # 7. 应用多视图注意力机制 - 与图片模型完全匹配
        query = graphs_features
        
        if len(coarsened_graphs_embedding) > 0:
            keys = torch.stack(coarsened_graphs_embedding, dim=1)
            
            fused_features, attention_weights = self.multiview_attention(query, keys)
        else:
            fused_features = graphs_features
            attention_weights = None
        
        # 8. 通过最终线性层生成输出
        output = self.linear(fused_features)
        
        # 9. 构造注意力权重字典
        attention_dict = {}
        if attention_weights is not None:
            if attention_weights.dim() == 4:
                attention_weights = attention_weights.mean(dim=1).squeeze(1)
            elif attention_weights.dim() == 3:
                attention_weights = attention_weights.mean(dim=1)
                
            avg_attention = attention_weights.mean(dim=0)
            
            attention_dict = {name: float(weight.item()) for name, weight in zip(all_encoding_names, avg_attention)}
        
        return output, attention_weights, attention_dict
    
    