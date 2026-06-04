import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_scatter import scatter
import torch.nn.functional as F


class ShortestPathEdgeEncoder(torch.nn.Module):
    """最短路径距离边编码器。
    
    将最短路径距离编码成边特征。这个编码器将预先计算的最短路径距离
    转换为边的嵌入表示。
    
    Args:
        cfg: 配置对象，包含边编码器的参数
    """
    
    def __init__(self, cfg):
        super().__init__()
        pecfg = cfg['posenc_ShortestPathEdge']
        self.dim_pe = pecfg["dim_pe"]  
        model_type = pecfg["model"]  
        n_layers = pecfg["layers"]  
        norm_type = pecfg["raw_norm_type"]  
        max_path_length = pecfg["max_path_length"]  
        self.pass_as_var = pecfg["pass_as_var"]  
        
        # 增加max_path_length的存储
        self.max_path_length = max_path_length

        # 归一化层
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_path_length)
        else:
            self.raw_norm = None
            
        # 选择激活函数
        activation = nn.ReLU
        
        # 构建边距离编码模型
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(max_path_length, self.dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(max_path_length, 2 * self.dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
                layers.append(activation())
            self.edge_encoder = nn.Sequential(*layers)
        elif model_type == 'attention':
            # 基于注意力的编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.dim_pe, 
                nhead=pecfg["n_heads"],
                batch_first=True
            )
            self.edge_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers
            )
            self.linear_proj = nn.Linear(max_path_length, self.dim_pe)
        else:
            raise ValueError(f"Unsupported edge encoder model type: {model_type}")
            
    def forward(self, batch):
        """前向传播函数。
        
        Args:
            batch: PyG批次对象，必须包含预计算的最短路径距离矩阵
            
        Returns:
            batch: 更新后的批次对象
        """
        # 检查预计算的SPD是否存在
        if not hasattr(batch, 'precomputed_shortest_paths'):
            raise ValueError("Precomputed shortest path matrix is "
                             f"required for {self.__class__.__name__}")
        
        # 提取边对应的最短路径长度
        edge_attr = batch.precomputed_shortest_paths

        # 将路径长度限制在max_path_length以内
        edge_attr = edge_attr.clamp(0, self.max_path_length -1)

        # 进行独热编码
        edge_attr_one_hot = F.one_hot(edge_attr, num_classes=self.max_path_length).float()
        
        # 应用归一化（如果配置）
        if self.raw_norm:
            edge_attr_one_hot = self.raw_norm(edge_attr_one_hot)
        
        # 编码边距离特征
        if isinstance(self.edge_encoder, nn.Sequential):
            edge_pe = self.edge_encoder(edge_attr_one_hot)
        else:
            # 对于Transformer编码器
            edge_pe = self.linear_proj(edge_attr_one_hot)
            # 创建填充掩码
            pad_mask = (edge_attr == self.max_path_length) # 使用超出范围的值作为mask
            edge_pe = self.edge_encoder(
                src=edge_pe,
                src_key_padding_mask=pad_mask
            )
            # 对掩码位置的结果清零
            edge_pe = edge_pe.masked_fill(pad_mask.unsqueeze(-1), 0)
        
        # 将边位置编码保存为批次的属性
        if self.pass_as_var:
            batch.pe_ShortestPath = edge_pe
            
        return batch


class HeatKernelEdgeEncoder(torch.nn.Module):
    """热核距离边编码器。
    
    基于热核方法的边编码，利用热传播模拟计算边的结构信息。
    
    Args:
        cfg: 配置对象，包含热核编码器的参数
    """
    
    def __init__(self, cfg):
        super().__init__()
        pecfg = cfg['posenc_HeatKernelEdge']
        self.dim_pe = pecfg["dim_pe"]  
        num_kernel_times = len(pecfg["kernel"]["times"])  
        model_type = pecfg["model"]  
        n_layers = pecfg["layers"]  
        norm_type = pecfg["raw_norm_type"]  
        self.pass_as_var = pecfg["pass_as_var"]  
        
        # 归一化层
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_kernel_times)
        else:
            self.raw_norm = None
            
        # 选择激活函数
        activation = nn.ReLU
        
        # 构建热核编码模型
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_kernel_times, self.dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_kernel_times, 2 * self.dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
                layers.append(activation())
            self.edge_encoder = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported edge encoder model type: {model_type}")
            
    def forward(self, batch):
        """前向传播函数。
        
        Args:
            batch: PyG批次对象，必须包含预计算的热核距离
            
        Returns:
            batch: 更新后的批次对象
        """
        if not hasattr(batch, 'heat_kernels'):
            raise ValueError("Precomputed heat kernel matrix is "
                             f"required for {self.__class__.__name__}")
        
        # 获取热核距离
        # heat_kernels形状应为[E, num_kernel_times]，其中E为边的数量
        heat_kernels = batch.heat_kernels
        
        # 处理缺失值
        mask = torch.isnan(heat_kernels)
        heat_kernels = heat_kernels.clone()
        heat_kernels[mask] = 0
        
        # 应用归一化（如果配置）
        if self.raw_norm:
            heat_kernels = self.raw_norm(heat_kernels)
        
        # 编码热核特征
        edge_pe = self.edge_encoder(heat_kernels)
        
        # 将热核编码保存为批次的属性
        if self.pass_as_var:
            batch.pe_HeatKernel = edge_pe
            
        return batch


class RandomWalkEdgeEncoder(torch.nn.Module):
    """随机游走边编码器。
    
    基于随机游走统计信息的边编码，捕获边的结构上下文。
    
    Args:
        cfg: 配置对象，包含随机游走编码器的参数
    """
    
    def __init__(self, cfg):
        super().__init__()
        pecfg = cfg['posenc_RandomWalkEdge']
        self.dim_pe = pecfg["dim_pe"] 
        num_rw_steps = len(pecfg["kernel"]["steps"])  
        model_type = pecfg["model"]  
        n_layers = pecfg["layers"]
        norm_type = pecfg["raw_norm_type"]  
        self.pass_as_var = pecfg["pass_as_var"]  
        
        # 归一化层
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None
            
        # 选择激活函数
        activation = nn.ReLU
        
        # 构建随机游走编码模型
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, self.dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * self.dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
                layers.append(activation())
            self.edge_encoder = nn.Sequential(*layers)
        elif model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.dim_pe,
                nhead=pecfg["n_heads"],
                batch_first=True
            )
            self.edge_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers
            )
            self.linear_proj = nn.Linear(num_rw_steps, self.dim_pe)
        else:
            raise ValueError(f"Unsupported edge encoder model type: {model_type}")
            
    def forward(self, batch):
        """前向传播函数。
        
        Args:
            batch: PyG批次对象，必须包含预计算的随机游走统计信息
            
        Returns:
            batch: 更新后的批次对象
        """
        if not hasattr(batch, 'rw_edge_features'):
            raise ValueError("Precomputed random walk edge features are "
                             f"required for {self.__class__.__name__}")
        
        # 获取随机游走特征
        # rw_edge_features形状应为[E, num_rw_steps]，其中E为边的数量
        rw_edge_features = batch.rw_edge_features
        
        # 处理缺失值
        mask = torch.isnan(rw_edge_features)
        rw_edge_features = rw_edge_features.clone()
        rw_edge_features[mask] = 0
        
        # 应用归一化（如果配置）
        if self.raw_norm:
            rw_edge_features = self.raw_norm(rw_edge_features)
        
        # 编码随机游走特征
        if isinstance(self.edge_encoder, nn.Sequential):
            edge_pe = self.edge_encoder(rw_edge_features)
        else:
            # 对于Transformer编码器
            edge_pe = self.linear_proj(rw_edge_features)
            edge_pe = self.edge_encoder(
                src=edge_pe,
                src_key_padding_mask=mask
            )
            # 对掩码位置的结果清零
            edge_pe = edge_pe.masked_fill(mask.unsqueeze(-1), 0)
        
        # 将随机游走编码保存为批次的属性
        if self.pass_as_var:
            batch.pe_RandomWalk = edge_pe
            
        return batch


class MultiScaleEdgeEncoder(torch.nn.Module):
    """多尺度边编码器。
    
    同时整合多种边结构编码器并进行融合，提供全面的边表示。
    
    Args:
        cfg: 配置对象，包含多尺度编码器的参数
    """
    
    def __init__(self, cfg):
        super().__init__()
        pecfg = cfg['posenc_MultiScaleEdge']
        self.dim_pe_out = pecfg["dim_pe_out"]  # 输出位置编码的维度
        encoders_list = pecfg["encoders"]  # 要使用的编码器列表
        fusion_type = pecfg["fusion"]  # 融合方法
        self.pass_as_var = pecfg["pass_as_var"]  # 是否将PE作为单独变量传递
        
        # 初始化所有指定的编码器
        self.encoders = nn.ModuleDict()
        self.encoder_dims = {}
        for enc_name in encoders_list:
            if enc_name == 'ShortestPathEdge':
                self.encoders[enc_name] = ShortestPathEdgeEncoder(cfg)
                self.encoder_dims[enc_name] = cfg['posenc_ShortestPathEdge']["dim_pe"]
            elif enc_name == 'HeatKernelEdge':
                self.encoders[enc_name] = HeatKernelEdgeEncoder(cfg)
                self.encoder_dims[enc_name] = cfg['posenc_HeatKernelEdge']["dim_pe"]
            elif enc_name == 'RandomWalkEdge':
                self.encoders[enc_name] = RandomWalkEdgeEncoder(cfg)
                self.encoder_dims[enc_name] = cfg['posenc_RandomWalkEdge']["dim_pe"]
            else:
                raise ValueError(f"Unknown edge encoder: {enc_name}")
                
        # 编码融合层
        total_dim = sum(self.encoder_dims.values())
        if fusion_type == 'concat':
            # 如果是简单拼接，确保总维度匹配输出维度
            if total_dim != self.dim_pe_out:
                self.fusion_proj = nn.Linear(total_dim, self.dim_pe_out)
            else:
                self.fusion_proj = nn.Identity()
        elif fusion_type == 'attention':
            # 注意力融合
            self.fusion_weights = nn.ParameterDict({
                name: nn.Parameter(torch.ones(1)) for name in self.encoders.keys()
            })
            if total_dim != self.dim_pe_out:
                self.fusion_proj = nn.Linear(total_dim, self.dim_pe_out)
            else:
                self.fusion_proj = nn.Identity()
        elif fusion_type == 'mlp':
            # MLP融合
            self.fusion_proj = nn.Sequential(
                nn.Linear(total_dim, 2 * self.dim_pe_out),
                nn.ReLU(),
                nn.Linear(2 * self.dim_pe_out, self.dim_pe_out)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        self.fusion_type = fusion_type
            
    def forward(self, batch):
        """前向传播函数。
        
        Args:
            batch: PyG批次对象，必须包含所有编码器所需的预计算特征
            
        Returns:
            batch: 更新后的批次对象
        """
        # 运行所有编码器
        encodings = {}
        for name, encoder in self.encoders.items():
            batch = encoder(batch)
            attr_name = f"pe_{name.replace('Edge', '')}"
            if hasattr(batch, attr_name):
                encodings[name] = getattr(batch, attr_name)
            
        # 应用融合
        if self.fusion_type == 'concat':
            # 按顺序拼接所有编码
            edge_pe = torch.cat([encodings[name] for name in self.encoders.keys()], dim=-1)
            edge_pe = self.fusion_proj(edge_pe)
        elif self.fusion_type == 'attention':
            # 加权融合
            weighted_encodings = [
                self.fusion_weights[name] * encodings[name] 
                for name in self.encoders.keys()
            ]
            edge_pe = torch.cat(weighted_encodings, dim=-1)
            edge_pe = self.fusion_proj(edge_pe)
        elif self.fusion_type == 'mlp':
            # MLP融合
            edge_pe = torch.cat([encodings[name] for name in self.encoders.keys()], dim=-1)
            edge_pe = self.fusion_proj(edge_pe)
            
        # 将融合后的边位置编码保存为批次的属性
        if self.pass_as_var:
            batch.pe_MultiScaleEdge = edge_pe
            
        return batch 