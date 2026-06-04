import os
import argparse
import logging
import datetime
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import torch.nn as nn
import torch.optim as optim

from GCL_model import GraphCoarseningModel
from data import load_dataset
from train import train_epoch, validate, evaluate

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='图粗化学习用于分子性质预测')

    # 配置文件参数
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')

    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='BACE', help='数据集名称')

    # 模型相关参数
    parser.add_argument('--hidden_channels', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--n_layers', type=int, default=3, help='GNN层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比例')
    parser.add_argument('--n_views', type=int, default=3, help='多视图数量')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--coarsening_ratio', type=float, default=0.5, help='图粗化比例')
    parser.add_argument('--coarsening_levels', type=int, default=2, help='图粗化层数')
    parser.add_argument('--temperature', type=float, default=0.1, help='注意力温度参数')
    parser.add_argument('--batch_norm', type=bool, default=True, help='是否使用批归一化')
    parser.add_argument('--alpha', type=float, default=0.1, help='对比学习权重系数')
    parser.add_argument('--beta', type=float, default=0.05, help='结构正则化系数')

    # 训练相关参数
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 其他参数
    parser.add_argument('--save_model', type=bool, default=True, help='是否保存模型')
    parser.add_argument('--model_dir', type=str, default='./models', help='模型保存路径')
    parser.add_argument('--runs', type=int, default=1, help='运行次数')


    parser.add_argument('--is_undirected', type=int, default=True)

    ### LapPE
    parser.add_argument('--max_freqs', type=int, default=1)
    parser.add_argument('--eigvec_norm', type=str, default='L2')
    parser.add_argument('--laplacian_norm', default=None)
    parser.add_argument('--dim_pe', type=int, default=20)   
    parser.add_argument('--pass_as_var', type=bool, default=True)    
    parser.add_argument('--raw_norm', default=None)   
    ### EquivStableLapPE
    parser.add_argument('--max_freqs_ES', type=int, default=10)
    parser.add_argument('--eigvec_norm_ES', type=str, default='L2')
    parser.add_argument('--laplacian_norm_ES', type=str, default=None)
    parser.add_argument('--raw_norm_ES', default=None)   
    parser.add_argument('--dim_pe_new_ES', type=int, default=20)       
    ### RWSE
    parser.add_argument('--RWSE_times_func', type=list, default=[1,2,3,4,5])
    parser.add_argument('--dim_pe_RWSE', type=int, default=20)   
    parser.add_argument('--raw_norm_RWSE', default='BatchNorm')   
    parser.add_argument('--pass_as_var_RWSE', type=bool, default=True)   
    ### HKdiagSE
    parser.add_argument('--laplacian_norm_HKSE', type=str, default=None)
    parser.add_argument('--HKSE_times_func', type=list, default=[1,2,3,4,5])
    parser.add_argument('--dim_pe_HKSE', type=int, default=20)
    parser.add_argument('--raw_norm_HKSE', default='BatchNorm')
    parser.add_argument('--pass_as_var_HKSE', type=bool, default=True)
    ### ElstaticSE
    parser.add_argument('--dim_pe_ETSE', type=int, default=20)
    parser.add_argument('--raw_norm_ETSE', default='BatchNorm')
    parser.add_argument('--pass_as_var_ETSE', type=bool, default=True)
    ### SignNet
    parser.add_argument('--max_freqs_SN', type=int, default=20)
    parser.add_argument('--eigvec_norm_SN', type=str, default='L2')
    parser.add_argument('--laplacian_norm_SN', type=str, default=None)
    parser.add_argument('--Layer_SN', type=int, default=8)   
    parser.add_argument('--post_layer_SN', type=int, default=3)   
    parser.add_argument('--dim_pe_SN', type=int, default=20)   
    parser.add_argument('--phi_hidden_dim', type=int, default=64)   
    parser.add_argument('--phi_out_dim', type=int, default=64)   
    parser.add_argument('--pass_as_var_SN', type=bool, default=True)    
    parser.add_argument('--pe_types', type=list, default=['LapPE', 'EquivStableLapPE']) #,'SignNet','RWSE','HKdiagSE','ElstaticSE'])

    return parser.parse_args()



def run_training(args, config, train_loader, valid_loader, test_loader, num_features, out_channels, task_type, num_tasks):
    """运行训练和评估过程"""

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"使用设备: {device}")
    logging.info(f"使用设备: {device}") # 新增

    # 存储所有运行的结果
    all_epochs_results = []

    print("\n开始训练...")
    logging.info("\n开始训练...") # 新增

    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    logging.info(f"设置随机种子: {seed}") # 新增

    # 初始化模型
    model = GraphCoarseningModel(args, config).to(device)
    logging.info("模型初始化完成") # 新增
    logging.info(f"模型结构: {model}") # 新增

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logging.info(f"优化器: Adam, 学习率: {args.lr}, 权重衰减: {args.weight_decay}") # 新增

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.patience
    )

    # 损失函数
    if task_type == 'classification':
        if num_tasks > 1:  # 多标签分类
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:  # 单标签分类
            criterion = nn.CrossEntropyLoss()
    else:  # 回归
        criterion = nn.MSELoss()

    # 记录最佳模型
    best_val_metric = 0
    best_metric_name = None
    best_epoch = 0
    best_test_metrics = None

    # 早停计数器
    patience_counter = 0

    # 保存每个epoch的测试结果
    epoch_results = []

    ############################## node encoding and edge encoding in there.



    # 训练模型
    for epoch in range(args.epochs):
        print(f"\n轮次: {epoch + 1}/{args.epochs}")
        logging.info(f"\n轮次: {epoch + 1}/{args.epochs}") # 新增

        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logging.info(f"Epoch {epoch + 1} - 训练损失: {train_loss:.4f}") # 新增

        # 对训练集进行评估
        train_eval_loss, train_y_true, train_y_pred, train_y_scores = validate(model, train_loader, criterion, device)
        train_metrics = evaluate(train_y_true, train_y_pred, train_y_scores, args.dataset)
        logging.info(f"Epoch {epoch + 1} - 训练集评估损失: {train_eval_loss:.4f}") # 新增

        # 验证
        val_loss, val_y_true, val_y_pred, val_y_scores = validate(model, valid_loader, criterion, device)
        val_metrics = evaluate(val_y_true, val_y_pred, val_y_scores, args.dataset)
        logging.info(f"Epoch {epoch + 1} - 验证集损失: {val_loss:.4f}") # 新增

        # 测试
        test_loss, test_y_true, test_y_pred, test_y_scores = validate(model, test_loader, criterion, device)
        test_metrics = evaluate(test_y_true, test_y_pred, test_y_scores, args.dataset)
        logging.info(f"Epoch {epoch + 1} - 测试集损失: {test_loss:.4f}") # 新增

        # 更新学习率
        if 'AUROC' in val_metrics:
            scheduler.step(val_metrics['AUROC'])
        else:
            scheduler.step(val_metrics['Accuracy'])  # 使用准确率作为备选指标

        # 记录当前epoch的测试结果
        epoch_results.append({
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        })

        # 构建详细指标的字符串输出
        train_metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items() if not np.isnan(v)])
        val_metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if not np.isnan(v)])
        test_metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items() if not np.isnan(v)])

        # 打印结果
        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        print(f"训练集指标: {train_metrics_str}")
        print(f"验证集指标: {val_metrics_str}")
        print(f"测试集指标: {test_metrics_str}")
        logging.info(f"Epoch {epoch + 1} - 训练集指标: {train_metrics_str}") # 新增
        logging.info(f"Epoch {epoch + 1} - 验证集指标: {val_metrics_str}") # 新增
        logging.info(f"Epoch {epoch + 1} - 测试集指标: {test_metrics_str}") # 新增

        # 保存最佳模型
        if 'AUROC' in val_metrics:
            current_metric = val_metrics['AUROC']
            metric_name = 'AUROC'
        else:
            current_metric = val_metrics['Accuracy']
            metric_name = 'Accuracy'

        if current_metric > best_val_metric:
            # 记录新的最佳值
            previous_best = best_val_metric
            best_val_metric = current_metric
            best_metric_name = metric_name
            best_epoch = epoch
            best_test_metrics = test_metrics
            patience_counter = 0 # 重置耐心计数器
            
            # 打印发现新的最佳模型信息
            print(f"\n发现新的最佳模型!")
            print(f"  轮次: {epoch + 1}")
            print(f"  验证集{metric_name}: {best_val_metric:.4f} (提升: {best_val_metric - previous_best:.4f})")
            print(f"  测试集指标:")
            logging.info(f"\n发现新的最佳模型!") # 新增
            logging.info(f"  轮次: {epoch + 1}") # 新增
            logging.info(f"  验证集{metric_name}: {best_val_metric:.4f} (提升: {best_val_metric - previous_best:.4f})") # 新增
            logging.info(f"  测试集指标:") # 新增
            for metric, value in test_metrics.items():
                print(f"    {metric}: {value:.4f}")
                logging.info(f"    {metric}: {value:.4f}") # 新增
            
            if args.save_model:
                # 确保目录存在
                os.makedirs(args.model_dir, exist_ok=True)
                model_path = os.path.join(args.model_dir, f"{args.dataset}_best.pt")
                torch.save(model.state_dict(), model_path)
                print(f"最佳模型已保存至: {model_path}")
                logging.info(f"最佳模型已保存至: {model_path}") # 新增
        else:
            patience_counter += 1
            print(f"验证集指标没有提升，耐心计数: {patience_counter}/{args.patience}")
            logging.info(f"验证集指标没有提升，耐心计数: {patience_counter}/{args.patience}")

        # 检查是否可以早停
        if patience_counter >= args.patience:
            print(f"\n连续 {args.patience} 个轮次验证集指标没有提升，触发早停。")
            logging.info(f"连续 {args.patience} 个轮次验证集指标没有提升，触发早停。")
            break

    # 加载最佳模型
    if args.save_model:
        model_path = os.path.join(args.model_dir, f"{args.dataset}_best.pt")

        # 添加文件存在性检查
        if os.path.exists(model_path):
            print(f"加载最佳模型: {model_path}")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"警告: 未找到模型文件 {model_path}，无法加载最佳模型")

    # 测试最佳模型
    test_loss, test_y_true, test_y_pred, test_y_scores = validate(model, test_loader, criterion, device)
    final_test_metrics = evaluate(test_y_true, test_y_pred, test_y_scores, args.dataset)

    print(f"\n训练结果总结:")
    print(f"  最佳轮次: {best_epoch + 1}")
    if best_metric_name == 'AUROC':
        print(f"  最佳验证集 AUROC: {best_val_metric:.4f}")
    else:
        print(f"  最佳验证集准确率: {best_val_metric:.4f}")
    print(f"  最终测试集指标:")
    for metric, value in final_test_metrics.items():
        if not np.isnan(value):
            print(f"    {metric}: {value:.4f}")

    # 保存训练的结果
    all_epochs_results = epoch_results

    return all_epochs_results


def calculate_stability(all_epochs_results):
    """计算最后5个epoch的稳定性范围"""

    # 获取最后5个epoch的指标
    last_epochs = min(5, len(all_epochs_results))
    train_metrics_last_epochs = [run['train_metrics'] for run in all_epochs_results[-last_epochs:]]
    val_metrics_last_epochs = [run['val_metrics'] for run in all_epochs_results[-last_epochs:]]
    test_metrics_last_epochs = [run['test_metrics'] for run in all_epochs_results[-last_epochs:]]

    # 提取所有可能的指标名称
    metric_names = set()
    for metrics_list in [train_metrics_last_epochs, val_metrics_last_epochs, test_metrics_last_epochs]:
        for metrics in metrics_list:
            metric_names.update(metrics.keys())

    # 计算每个数据集的每个指标的平均值、最小值和最大值
    stability_results = {
        'train': {},
        'val': {},
        'test': {}
    }

    # 处理训练集指标
    for metric in metric_names:
        values = []
        for run_metrics in train_metrics_last_epochs:
            if metric in run_metrics and not np.isnan(run_metrics[metric]):
                values.append(run_metrics[metric])

        if values:
            stability_results['train'][metric] = {
                'mean': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values)
            }

    # 处理验证集指标
    for metric in metric_names:
        values = []
        for run_metrics in val_metrics_last_epochs:
            if metric in run_metrics and not np.isnan(run_metrics[metric]):
                values.append(run_metrics[metric])

        if values:
            stability_results['val'][metric] = {
                'mean': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values)
            }

    # 处理测试集指标
    for metric in metric_names:
        values = []
        for run_metrics in test_metrics_last_epochs:
            if metric in run_metrics and not np.isnan(run_metrics[metric]):
                values.append(run_metrics[metric])

        if values:
            stability_results['test'][metric] = {
                'mean': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values)
            }

    return stability_results


def main():
    # 获取命令行参数
    args = get_args()

    # 创建日志文件和目录
    log_dir = Path('./logs') # 新增
    log_dir.mkdir(parents=True, exist_ok=True) # 新增
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # 新增
    log_file_name = f"{args.dataset}_{current_time}.log" # 新增
    log_file_path = log_dir / log_file_name # 新增
    # 配置日志记录
    logging.basicConfig( # 新增
        level=logging.INFO, # 新增
        format='%(asctime)s - %(levelname)s - %(message)s', # 新增
        handlers=[ # 新增
            logging.FileHandler(log_file_path), # 新增
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台 # 新增
        ] # 新增
    ) # 新增
    logging.info(f"日志文件保存在: {log_file_path}") # 新增

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    config = load_config(args.config)
    config['alpha'] = getattr(args, 'alpha', config.get('alpha', 0.1))
    config['beta'] = getattr(args, 'beta', config.get('beta', 0.05))
    for key in ('hidden_channels', 'n_layers', 'dropout', 'n_heads', 'temperature'):
        if hasattr(args, key):
            config[key] = getattr(args, key)
    logging.info(f"配置文件加载自: {args.config}")
    logging.info(f"配置参数: {config}") # 新增
    logging.info(f"命令行参数: {vars(args)}") # 新增
    # 修改运行次数为1（确保一致性）
    args.runs = 1

    # 创建保存模型的目录
    if args.save_model:
        os.makedirs(args.model_dir, exist_ok=True)

    # 加载数据集
    args.pe_types = config['pe_types'] 
    args.edge_types = config.get('edge_types', [])
    args.global_types = config.get('global_types', [])
    args.config = config # 将整个config字典附加到args上，方便后续使用
    train_loader, valid_loader, test_loader, num_features, out_channels, task_type, num_tasks, num_edge_features = load_dataset(args)
    config['out_channels'] = out_channels
    config['raw_node_feature'] = num_features
    config['edge_feature_dim'] = num_edge_features
    config['task_type'] = task_type
    logging.info(f"数据集加载完成: {args.dataset}") # 新增
    logging.info(f"节点特征维度: {num_features}, 输出通道数: {out_channels}, 任务类型: {task_type}, 任务数量: {num_tasks}, 边特征维度: {num_edge_features}") # 新增

    # 初始化所有编码器配置
    # print("初始化编码器配置...")
    # init_all_encoder_configs(num_features)

    # 运行训练
    all_epochs_results = run_training(args, config, train_loader, valid_loader, test_loader, num_features, out_channels,
                                      task_type, num_tasks)

    # 计算稳定性
    stability_results = calculate_stability(all_epochs_results)

    # 打印稳定性结果
    print("\n模型稳定性结果 (最后5个epoch):")
    print("训练集指标:")
    for metric, stats in stability_results['train'].items():
        print(f"  {metric}:")
        print(f"    平均值: {stats['mean']:.4f}")
        print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    标准差: {stats['std']:.4f}")

    print("\n验证集指标:")
    for metric, stats in stability_results['val'].items():
        print(f"  {metric}:")
        print(f"    平均值: {stats['mean']:.4f}")
        print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    标准差: {stats['std']:.4f}")

    print("\n测试集指标:")
    for metric, stats in stability_results['test'].items():
        print(f"  {metric}:")
        print(f"    平均值: {stats['mean']:.4f}")
        print(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    标准差: {stats['std']:.4f}")



if __name__ == '__main__':
    main()