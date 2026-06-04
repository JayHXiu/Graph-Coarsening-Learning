# Graph Coarsening Learning (GCL)

**图粗化学习**项目：研究如何在端到端训练中学习图的粗化表示，并在多粒度视图之间进行有效融合。实现包含可微粗化（DiffPool）、多视图注意力、对比学习与结构保持正则等组件。

本仓库**以图粗化学习方法为主**，使用 [MoleculeNet](https://moleculenet.org/) 中的分子图数据作为**实验测试基准**——分子图是常见的图结构数据形式，便于验证方法在真实图学习任务上的效果；项目本身并非围绕 MoleculeNet 构建的数据集平台或分子性质预测专用框架。

## 核心思想

图粗化将原图映射为节点更少、结构更紧凑的粗化图，从而在保留关键拓扑信息的同时降低计算复杂度。本项目在此基础上进一步提出：

1. **多视角粗化**：对原始特征及多种结构编码（节点/边/图级）分别进行可微粗化，形成多个粗化视图；
2. **多视图融合**：以原图级表示为 Query，以各粗化视图（及图级全局特征）为 Key/Value，通过多头注意力自适应加权；
3. **联合优化**：在下游监督任务之外，用对比学习对齐不同粗化视图，用 DiffPool 的链接预测与熵正则约束粗化质量。

## 方法概览

```
输入图 G
    │
    ├─► 特征投影 + 原图 GNN ──────────────────► 图级 Query
    │
    ├─► 结构编码（节点 / 边 / 图级 PE）
    │         │
    │         └─► 各编码通道独立 DiffPool 粗化 ─► 粗化图 GNN ─► 多视图 Key
    │
    └─► MultiViewAttention 融合 ─► 分类头 ─► 预测
```

总损失：

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathcal{L}_{\text{contrastive}} + \beta \cdot \mathcal{L}_{\text{structure}}$$

| 项 | 作用 |
|----|------|
| $\mathcal{L}_{\text{task}}$ | 下游图分类任务（交叉熵 / BCE） |
| $\mathcal{L}_{\text{contrastive}}$ | 多粗化视图间的 InfoNCE，增强表示一致性 |
| $\mathcal{L}_{\text{structure}}$ | DiffPool 链接预测 + 分配熵正则，保持结构可辨 |

## 主要模块

| 模块 | 说明 |
|------|------|
| `DifferentiableCoarseningNetwork` | 嵌入 GNN + 分配 GNN + `dense_diff_pool` 可微粗化 |
| `MultiViewAttention` | 原图 Query × 粗化视图 Key 的多头注意力 |
| `node_encoder` / `edge_encoder` / `global_encoder` | 多层级结构编码，为各粗化通道提供输入 |
| `graph_precompute.py` | 拉普拉斯、随机游走、热核等结构统计量预计算 |

## 项目结构

```
Graph_Coarsening_Learning/
├── main.py                 # 训练入口
├── GCL_model.py            # 图粗化学习主模型
├── data.py                 # 数据加载与划分
├── graph_precompute.py     # 图结构特征预计算
├── train.py                # 训练 / 验证 / 指标
├── node_encoder.py         # 节点级结构编码
├── edge_encoder.py         # 边级结构编码
├── global_encoder.py       # 图级结构编码
├── config.yaml             # 模型与编码器配置
├── scripts/evaluate.py     # 加载权重在测试集上评估
├── data/                   # 实验数据缓存（MoleculeNet 自动下载）
├── models/                 # 保存的 checkpoint
└── logs/                   # 训练日志
```

## 环境要求

- Python 3.9+
- PyTorch ≥ 1.9
- PyTorch Geometric ≥ 2.0，以及 `torch-scatter`、`torch-sparse` 等配套包

```bash
pip install -r requirements.txt
```

PyG 扩展包需与当前 PyTorch/CUDA 版本匹配，参见 [PyG 安装文档](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

## 快速开始

### 训练（在指定 MoleculeNet 基准上测试）

```bash
python main.py --dataset BACE --config config.yaml --epochs 200 --batch_size 32 --gpu 0
```

通过 `--dataset` 切换不同的 MoleculeNet 子集，用于对比实验；数据首次运行时会下载到 `data/{DATASET}/`。

| 参数 | 说明 | 默认 |
|------|------|------|
| `--dataset` | MoleculeNet 子集名称（实验用） | `BACE` |
| `--config` | 模型与粗化相关配置 | `config.yaml` |
| `--epochs` | 训练轮数 | `200` |
| `--batch_size` | 批大小 | `32` |
| `--alpha` | 对比学习权重 | `0.1` |
| `--beta` | 结构正则权重 | `0.05` |
| `--patience` | 早停耐心值 | `20` |
| `--gpu` | GPU 编号（CPU 可用 `-1`） | `0` |

最佳权重：`models/{DATASET}_best.pt`；日志：`logs/`。

### 评估

```bash
python scripts/evaluate.py --dataset BACE --checkpoint models/BACE_best.pt --config config.yaml
```

评估流程与训练一致，会重新进行结构特征预计算。

## 实验测试数据（MoleculeNet）

以下子集仅作为**图分类实验的测试基准**，用于检验图粗化学习方法的表现，可按需切换 `--dataset`：

| 子集 | 说明 |
|------|------|
| BACE, BBBP, HIV | 单任务二分类 |
| Tox21, ToxCast, SIDER, ClinTox, MUV | 多标签分类 |

- 数据来源：`torch_geometric.datasets.MoleculeNet`
- 划分：训练 60% / 验证 20% / 测试 20%（分层采样）

若需在其他图数据上验证本方法，可参照 `data.py` 中的加载与预计算流程进行扩展；当前实现针对 MoleculeNet 图格式做了适配。

## 配置说明

`config.yaml` 中与**图粗化学习**相关的主要项：

- `pe_types` / `edge_types` / `global_types`：参与粗化的结构编码通道
- `hidden_channels`, `n_layers`, `dropout`, `n_heads`：GNN 与注意力规模
- `alpha`, `beta`：对比学习与结构正则系数
- `posenc_*`：各结构编码器的维度与超参

命令行参数（如 `--hidden_channels`、`--n_layers`）可覆盖 YAML 中对应字段。

## 评估指标

在测试基准上报告：AUROC、AUPRC、Accuracy、Precision、Recall、F1。验证集以 AUROC（或 Accuracy）选取最佳 checkpoint。

## 使用建议

1. **预计算耗时**：每个图需拉普拉斯分解、最短路径等；调试时可于 `config.yaml` 减少 `pe_types` / `edge_types`。
2. **显存**：DiffPool 采用批内稠密表示，大图或较大 `batch_size` 时注意显存占用。
3. **可复现**：固定 `--seed` 以保证划分与初始化一致。

## 引用

使用 MoleculeNet 作为测试基准时请引用其原始论文。若采用本仓库中的图粗化与多视图融合设计，请同时引用 DiffPool、相关位置编码（LapPE、SignNet 等）等方法的原始文献。
