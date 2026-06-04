import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class SpectralGraphEncoder(torch.nn.Module):
    """谱图编码器。
    
    利用图的邻接矩阵或拉普拉斯矩阵的特征值来编码图的全局属性。
    
    Args:
        cfg: 配置对象，包含谱编码器的参数
    """
    
    def __init__(self, cfg):
        super().__init__()
        pecfg = cfg['posenc_SpectralGraph']
        self.dim_pe = pecfg["dim_pe"]  # 图位置编码的维度
        max_freqs = pecfg["max_freqs"]  # 最大频率数（特征值数量）
        model_type = pecfg["model"]  # 编码器模型类型
        n_layers = pecfg["layers"]  # 编码器模型的层数
        norm_type = pecfg["raw_norm_type"]  # 原始PE归一化类型
        self.pass_as_var = pecfg["pass_as_var"]  # 是否将PE作为单独变量传递
        self.eigenval_type = pecfg["eigenval_type"]  # 'adj' 或 'lap'
        
        # 归一化层
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None
            
        # 选择激活函数
        activation = nn.ReLU
        
        # 构建编码模型
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(max_freqs, self.dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(max_freqs, 2 * self.dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
                layers.append(activation())
            self.encoder = nn.Sequential(*layers)
        elif model_type == 'deep_set':
            # Deep Sets 架构
            layers = []
            layers.append(nn.Linear(1, 2 * self.dim_pe))
            layers.append(activation())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                layers.append(activation())
            layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
            layers.append(activation())
            self.phi = nn.Sequential(*layers)
            
            post_layers = []
            post_layers.append(nn.Linear(self.dim_pe, 2 * self.dim_pe))
            post_layers.append(activation())
            post_layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
            post_layers.append(activation())
            self.rho = nn.Sequential(*post_layers)
        else:
            raise ValueError(f"Unsupported graph encoder model type: {model_type}")
        
        self.model_type = model_type
            
    def forward(self, batch):
        """前向传播函数。
        
        Args:
            batch: PyG批次对象，必须包含预计算的图特征值
            
        Returns:
            batch: 更新后的批次对象
        """
        attr_name = f"graph_eigenvals_{self.eigenval_type}"
        if not hasattr(batch, attr_name):
            raise ValueError(f"Precomputed graph eigenvalues of type {self.eigenval_type} are "
                             f"required for {self.__class__.__name__}")
        
        # 获取特征值
        # eigenvals形状应为[B, max_freqs]，其中B为批次中图的数量
        eigenvals = getattr(batch, attr_name)
        
        # 处理缺失值
        mask = torch.isnan(eigenvals)
        eigenvals = eigenvals.clone()
        eigenvals[mask] = 0
        
        # 应用归一化（如果配置）
        if self.raw_norm:
            if eigenvals.dim() == 1:
                eigenvals = eigenvals.view(batch.num_graphs, -1)
            eigenvals = self.raw_norm(eigenvals)
        
        # 编码特征值
        if self.model_type == 'mlp':
            graph_pe = self.encoder(eigenvals)
        elif self.model_type == 'deep_set':
            # Deep Sets 处理: 先映射每个值，再聚合
            x = eigenvals.unsqueeze(-1)  # [B, max_freqs, 1]
            x = self.phi(x)  # [B, max_freqs, dim_pe]
            # 对特征值维度求和
            x = torch.sum(x, dim=1)  # [B, dim_pe]
            graph_pe = self.rho(x)  # [B, dim_pe]
        
        # 将全局图编码保存为批次的属性
        if self.pass_as_var:
            batch.pe_SpectralGraph = graph_pe
            
        return batch


class DegreeDistributionEncoder(torch.nn.Module):
    """度分布编码器。
    
    利用图的度分布统计来编码图的全局拓扑属性。
    
    Args:
        cfg: 配置对象，包含度分布编码器的参数
    """
    
    def __init__(self, cfg):
        super().__init__()
        pecfg = cfg['posenc_DegreeDistribution']
        self.dim_pe = pecfg["dim_pe"]  # 图位置编码的维度
        self.max_degree = pecfg["max_degree"]  # 考虑的最大节点度数
        model_type = pecfg["model"]  # 编码器模型类型
        n_layers = pecfg["layers"]  # 编码器模型的层数
        norm_type = pecfg["raw_norm_type"]  # 原始PE归一化类型
        self.pass_as_var = pecfg["pass_as_var"]  # 是否将PE作为单独变量传递
        self.degree_histogram_bins = pecfg.get("degree_histogram_bins", self.max_degree + 1)
        
        # 归一化层
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(self.degree_histogram_bins)
        else:
            self.raw_norm = None
            
        # 选择激活函数
        activation = nn.ReLU
        
        # 构建编码模型
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(self.degree_histogram_bins, self.dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(self.degree_histogram_bins, 2 * self.dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
                layers.append(activation())
            self.encoder = nn.Sequential(*layers)
        elif model_type == 'cnn':
            # 使用一维卷积处理直方图
            layers = []
            layers.append(nn.Conv1d(1, 16, kernel_size=3, padding=1))
            layers.append(activation())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Conv1d(16, 32, kernel_size=3, padding=1))
            layers.append(activation())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Flatten())
            
            # 计算扁平化后的维度
            flat_dim = 32 * (self.degree_histogram_bins // 4)
            layers.append(nn.Linear(flat_dim, self.dim_pe))
            layers.append(activation())
            
            self.encoder = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported graph encoder model type: {model_type}")
            
        self.model_type = model_type
            
    def forward(self, batch):
        """前向传播函数。
        
        Args:
            batch: PyG批次对象，包含图的节点度信息
            
        Returns:
            batch: 更新后的批次对象
        """
        if not hasattr(batch, 'precomputed_degree_dist'):
             raise ValueError("Precomputed degree distribution is required.")
        
        # 假设 precomputed_degree_dist 已经是 batch a [B, num_bins] 张量
        # DataLoader 应该能正确处理图级别的属性
        degree_features = batch.precomputed_degree_dist
        
        if self.raw_norm:
            if degree_features.dim() == 1:
                degree_features = degree_features.view(batch.num_graphs, -1)
            degree_features = self.raw_norm(degree_features)
            
        graph_pe = self.encoder(degree_features)
        
        if self.pass_as_var:
            batch.pe_DegreeDistribution = graph_pe
            
        return batch


class ClusteringCoefficientEncoder(torch.nn.Module):
    """聚类系数编码器。
    
    利用图的聚类系数分布来编码图的全局结构特性。
    
    Args:
        cfg: 配置对象，包含聚类系数编码器的参数
    """
    
    def __init__(self, cfg):
        super().__init__()
        pecfg = cfg['posenc_ClusteringCoefficient']
        self.dim_pe = pecfg["dim_pe"]  # 图位置编码的维度
        model_type = pecfg["model"]  # 编码器模型类型
        n_layers = pecfg["layers"]  # 编码器模型的层数
        norm_type = pecfg["raw_norm_type"]  # 原始PE归一化类型
        self.pass_as_var = pecfg["pass_as_var"]  # 是否将PE作为单独变量传递
        self.num_bins = pecfg.get("num_histogram_bins", 10)
        
        # 归一化层
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(self.num_bins + 2)  # +2 用于全局平均和方差
        else:
            self.raw_norm = None
            
        # 选择激活函数
        activation = nn.ReLU
        
        # 构建编码模型
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(self.num_bins + 2, self.dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(self.num_bins + 2, 2 * self.dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
                layers.append(activation())
            self.encoder = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported graph encoder model type: {model_type}")
            
    def forward(self, batch):
        """前向传播函数。
        
        Args:
            batch: PyG批次对象，必须包含预计算的聚类系数数据
            
        Returns:
            batch: 更新后的批次对象
        """
        if not hasattr(batch, 'precomputed_cluster_coeff'):
            raise ValueError("Precomputed clustering coefficient is required.")
            
        cluster_features = batch.precomputed_cluster_coeff
        
        if self.raw_norm:
            if cluster_features.dim() == 1:
                cluster_features = cluster_features.view(batch.num_graphs, -1)
            cluster_features = self.raw_norm(cluster_features)
            
        graph_pe = self.encoder(cluster_features)
        
        if self.pass_as_var:
            batch.pe_ClusteringCoefficient = graph_pe
            
        return batch


class MultiFeatureGraphEncoder(torch.nn.Module):
    """多特征图编码器。
    
    组合多种图特征编码，为图提供综合的全局表示。
    
    Args:
        cfg: 配置对象，包含多特征图编码器的参数
    """
    
    def __init__(self, cfg):
        super().__init__()
        pecfg = cfg['posenc_MultiFeatureGraph']
        self.dim_pe = pecfg["dim_pe"]  # 图位置编码的维度
        encoders_list = pecfg["encoders"]  # 要使用的编码器列表
        fusion_method = pecfg["fusion"]  # 融合方法
        n_layers = pecfg["layers"]  # 融合MLP的层数
        self.pass_as_var = pecfg["pass_as_var"]  # 是否将PE作为单独变量传递
        
        # 初始化所有指定的编码器
        self.encoders = nn.ModuleDict()
        self.encoder_dims = {}
        for enc_name in encoders_list:
            if enc_name == 'SpectralGraph':
                self.encoders[enc_name] = SpectralGraphEncoder(cfg)
                self.encoder_dims[enc_name] = cfg['posenc_SpectralGraph']["dim_pe"]
            elif enc_name == 'DegreeDistribution':
                self.encoders[enc_name] = DegreeDistributionEncoder(cfg)
                self.encoder_dims[enc_name] = cfg['posenc_DegreeDistribution']["dim_pe"]
            elif enc_name == 'ClusteringCoefficient':
                self.encoders[enc_name] = ClusteringCoefficientEncoder(cfg)
                self.encoder_dims[enc_name] = cfg['posenc_ClusteringCoefficient']["dim_pe"]
            else:
                raise ValueError(f"Unknown graph encoder: {enc_name}")
                
        # 选择激活函数
        activation = nn.ReLU
        
        # 合并编码的方法
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            total_dim = sum(self.encoder_dims.values())
            # 投影到目标维度
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(total_dim, self.dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(total_dim, 2 * self.dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
                layers.append(activation())
            self.fusion_mlp = nn.Sequential(*layers)
        elif fusion_method == 'attention':
            # 多头注意力融合
            num_heads = pecfg.get("num_heads", 4)
            total_dim = sum(self.encoder_dims.values())
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=total_dim // len(self.encoders),
                num_heads=num_heads,
                batch_first=True
            )
            self.linear_proj = nn.Linear(total_dim // len(self.encoders), self.dim_pe)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
            
    def forward(self, batch):
        """前向传播函数。
        
        Args:
            batch: PyG批次对象，必须包含各编码器所需的预计算特征
            
        Returns:
            batch: 更新后的批次对象
        """
        # 运行所有编码器
        encodings = {}
        for name, encoder in self.encoders.items():
            batch = encoder(batch)
            attr_name = f"pe_{name}"
            if hasattr(batch, attr_name):
                encodings[name] = getattr(batch, attr_name)
                
        if not encodings:
            raise ValueError("No valid encodings were generated by any encoder")
        
        # 应用融合
        if self.fusion_method == 'concat':
            # 拼接所有编码并通过MLP
            graph_pe = torch.cat([encodings[name] for name in self.encoders.keys()], dim=-1)
            graph_pe = self.fusion_mlp(graph_pe)
        elif self.fusion_method == 'attention':
            # 将编码重塑为序列，应用多头注意力
            encodings_list = [encodings[name] for name in self.encoders.keys()]
            seq_len = len(encodings_list)
            batch_size = encodings_list[0].size(0)
            feature_dim = encodings_list[0].size(1)
            
            # 形状: [batch_size, seq_len, feature_dim]
            stacked = torch.stack(encodings_list, dim=1)
            
            # 自注意力: query, key, value都是相同的
            attn_output, _ = self.multihead_attn(stacked, stacked, stacked)
            
            # 聚合: 在seq_len维度上平均或求和
            graph_pe = torch.mean(attn_output, dim=1)
            
            # 投影到输出维度
            graph_pe = self.linear_proj(graph_pe)
        
        # 将融合后的图编码保存为批次的属性
        if self.pass_as_var:
            batch.pe_MultiFeatureGraph = graph_pe
            
        return batch 