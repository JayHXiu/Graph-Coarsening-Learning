#!/usr/bin/env python
"""Evaluate a saved checkpoint on the test split."""

import argparse
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from GCL_model import GraphCoarseningModel
from data import load_dataset
from train import evaluate, validate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GCL model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, default="BACE")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt state_dict")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_training_args(cli, config):
    """Namespace compatible with load_dataset and GraphCoarseningModel."""
    import argparse as ap

    args = ap.Namespace(**vars(cli))
    args.pe_types = config.get("pe_types", [])
    args.edge_types = config.get("edge_types", [])
    args.global_types = config.get("global_types", [])
    args.config = config
    args.laplacian_norm = None
    args.max_freqs = config.get("posenc_LapPE", {}).get("eigen", {}).get("max_freqs", 1)
    args.eigvec_norm = "L2"
    args.laplacian_norm_ES = None
    args.max_freqs_ES = config.get("posenc_EquivStableLapPE", {}).get("eigen", {}).get("max_freqs", 10)
    args.eigvec_norm_ES = "L2"
    args.RWSE_times_func = config.get("posenc_RWSE", {}).get("kernel", {}).get("times", [1, 2, 3, 4, 5])
    args.HKSE_times_func = config.get("posenc_HKdiagSE", {}).get("kernel", {}).get("times", [1, 2, 3, 4, 5])
    args.dim_pe_ETSE = config.get("posenc_ElstaticSE", {}).get("dim_pe", 20)
    args.laplacian_norm_SN = None
    args.max_freqs_SN = config.get("posenc_SignNet", {}).get("eigen", {}).get("max_freqs", 20)
    args.eigvec_norm_SN = "L2"
    return args


def main():
    cli = parse_args()
    if not os.path.exists(cli.config):
        raise FileNotFoundError(cli.config)
    if not os.path.exists(cli.checkpoint):
        raise FileNotFoundError(cli.checkpoint)

    with open(cli.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    args = build_training_args(cli, config)
    device = torch.device(
        f"cuda:{cli.gpu}" if torch.cuda.is_available() and cli.gpu >= 0 else "cpu"
    )

    _, _, test_loader, num_features, out_channels, task_type, _, num_edge_features = load_dataset(args)
    config["out_channels"] = out_channels
    config["raw_node_feature"] = num_features
    config["edge_feature_dim"] = num_edge_features
    config["task_type"] = task_type

    model = GraphCoarseningModel(args, config).to(device)
    try:
        state = torch.load(cli.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(cli.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    loss, y_true, y_pred, y_scores = validate(model, test_loader, criterion, device)
    metrics = evaluate(y_true, y_pred, y_scores, cli.dataset)

    print(f"Checkpoint: {cli.checkpoint}")
    print(f"Test loss (task): {loss:.4f}")
    for name, value in metrics.items():
        if value == value:
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
