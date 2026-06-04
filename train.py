"""Training, validation, and evaluation utilities."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output, _, _ = model(batch)

        y = batch.y
        if y.dim() == 0:
            y = y.unsqueeze(0)
        if y.dim() > 1 and y.size(1) == 1:
            y = y.squeeze(1)

        losses = model.compute_loss(output, y, edge_index=batch.edge_index, temperature=0.1)
        loss = losses['total_loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """Validate model and return loss plus predictions."""
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_scores = [], [], []
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output, _, _ = model(batch)

            y = batch.y
            if y.dim() == 0:
                y = y.unsqueeze(0)
            if y.dim() > 1 and y.size(1) == 1:
                y = y.squeeze(1)
            if y.numel() == 0:
                continue

            losses = model.compute_loss(output, y, edge_index=batch.edge_index, temperature=0.1)
            total_loss += losses['task_loss'].item() * batch.num_graphs
            total_samples += batch.num_graphs

            y_true.append(y.cpu().numpy())
            y_scores.append(torch.softmax(output, dim=1).cpu().numpy())
            y_pred.append(torch.argmax(output, dim=1).cpu().numpy())

    if not y_true or total_samples == 0:
        return 0.0, np.array([]), np.array([]), np.array([])

    return (
        total_loss / total_samples,
        np.concatenate(y_true, axis=0),
        np.concatenate(y_pred, axis=0),
        np.concatenate(y_scores, axis=0),
    )


def evaluate(y_true, y_pred, y_scores, dataset_name):
    """Compute classification metrics (AUROC, AUPRC, Accuracy, F1, etc.)."""
    _ = dataset_name  # reserved for dataset-specific metrics
    metrics = {}

    if y_true.ndim > 1:
        y_true = y_true.squeeze()
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()

    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        try:
            if y_scores.shape[1] == 2:
                metrics['AUROC'] = roc_auc_score(y_true, y_scores[:, 1])
                precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])
                metrics['AUPRC'] = auc(recall, precision)
            else:
                metrics['AUROC'] = roc_auc_score(y_true, y_scores, multi_class='ovr')
        except Exception as exc:
            print(f"计算 AUROC/AUPRC 时出错: {exc}")
            metrics['AUROC'] = float('nan')
            metrics['AUPRC'] = float('nan')

    try:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['F1'] = f1_score(y_true, y_pred, average='weighted')
    except Exception as exc:
        print(f"计算分类指标时出错: {exc}")
        for key in ('Accuracy', 'Precision', 'Recall', 'F1'):
            metrics[key] = float('nan')

    return metrics
