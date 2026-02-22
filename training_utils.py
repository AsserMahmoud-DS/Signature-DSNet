"""
Training utilities for DetailSemNet fine-tuning.
Includes checkpoint management, metrics computation, and training loops.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def save_checkpoint(save_path, epoch, model, optimizer, scheduler, best_loss, history, args=None):
    """
    Save training checkpoint for resuming later.
    
    Args:
        save_path: Path to save checkpoint file
        epoch: Current epoch number
        model: Model state dict
        optimizer: Optimizer state dict
        scheduler: Scheduler state dict
        best_loss: Best validation loss so far
        history: Training history dict (losses, metrics)
        args: Optional dict of training arguments
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'history': history,
    }
    
    if args is not None:
        checkpoint['args'] = args
    
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Resume training from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer (optional)
        scheduler: Scheduler (optional)
        device: Device to load to
        
    Returns:
        epoch, best_loss, history, args
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    history = checkpoint.get('history', {})
    args = checkpoint.get('args', {})
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"📂 Checkpoint loaded: {checkpoint_path}")
    print(f"   Resuming from epoch {epoch + 1}, best loss: {best_loss:.6f}")
    
    return epoch, best_loss, history, args


def compute_metrics(predictions, labels, step=None):
    """
    Compute signature verification metrics from predictions.
    
    Computes: accuracy, FAR, FRR, EER, and ROC curve.
    
    Args:
        predictions: Numpy array of predicted distances [N,]
        labels: Numpy array of labels (1=genuine, 0=forged) [N,]
        step: Threshold step size (default 5e-5)
        
    Returns:
        metrics dict with keys:
        - best_acc: Best accuracy across all thresholds
        - best_frr: FRR at best accuracy threshold
        - best_far: FAR at best accuracy threshold
        - eer: Equal error rate (FAR≈FRR)
        - eer_frr: FRR at EER threshold
        - eer_far: FAR at EER threshold
        - auc_roc: Area under ROC curve
        - tpr_arr, far_arr, frr_arr, d_arr: Arrays for plotting
    """
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
    
    if step is None:
        step = 0.00005
    
    max_acc = 0.0
    min_frr = 1.0
    min_far = 1.0
    min_dif = 1.0
    d_optimal = 0.0
    d_optimal_eer = 0.0
    
    tpr_arr, fpr_arr, far_arr, frr_arr, d_arr = [], [], [], [], []
    
    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d     # pred = 1 (genuine)
        idx2 = predictions.ravel() > d      # pred = 0 (forged)
        
        tp = float(np.sum(labels[idx1] == 1))
        tn = float(np.sum(labels[idx2] == 0))
        
        tpr = tp / nsame
        tnr = tn / ndiff
        
        frr = float(np.sum(labels[idx2] == 1)) / nsame
        far = float(np.sum(labels[idx1] == 0)) / ndiff
        
        tpr_arr.append(tpr)
        fpr_arr.append(1 - tnr)  # FPR = 1 - TNR for ROC
        far_arr.append(far)
        frr_arr.append(frr)
        d_arr.append(d)
        
        acc = (tp + tn) / (nsame + ndiff)
        
        # Track best accuracy
        if acc > max_acc:
            max_acc = acc
            d_optimal = d
            min_frr = frr
            min_far = far
        
        # Track EER (where FAR ≈ FRR)
        if abs(far - frr) < min_dif:
            min_dif = abs(far - frr)
            d_optimal_eer = d
            eer_frr = frr
            eer_far = far
    
    # Compute AUC of ROC curve
    # Sort by FPR and compute area under curve
    fpr_sorted = sorted(set(fpr_arr))
    tpr_interp = []
    for fpr_val in fpr_sorted:
        idx = fpr_arr.index(fpr_val)
        tpr_interp.append(tpr_arr[idx])
    
    auc_roc = auc(fpr_sorted, tpr_interp) if len(fpr_sorted) > 1 else 0.0
    
    eer = (eer_frr + eer_far) / 2.0
    
    metrics = {
        "best_acc": max_acc,
        "best_frr": min_frr,
        "best_far": min_far,
        "eer": eer,
        "eer_frr": eer_frr,
        "eer_far": eer_far,
        "auc_roc": auc_roc,
        "tpr_arr": tpr_arr,
        "far_arr": far_arr,
        "frr_arr": frr_arr,
        "fpr_arr": fpr_arr,
        "d_arr": d_arr,
        "d_optimal": d_optimal,
        "d_optimal_eer": d_optimal_eer
    }
    
    print(f"📊 Metrics: ACC={max_acc:.4f}, EER={(eer_frr+eer_far)/2.0:.4f} @d={d_optimal_eer:.4f}, AUC={auc_roc:.4f}")
    
    return metrics


def train_one_epoch(model, train_loader, optimizer, loss_fn, device='cuda', shift_aug=False):
    """
    Run one training epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        shift_aug: Optional shift augmentation transform
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch_data in enumerate(pbar):
        if len(batch_data) == 3:
            images, labels, writer_ids = batch_data
        else:
            images, labels = batch_data
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, loss = model(images, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, val_loader, loss_fn, device='cuda'):
    """
    Run validation epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch_data in pbar:
            if len(batch_data) == 3:
                images, labels, writer_ids = batch_data
            else:
                images, labels = batch_data
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, loss = model(images, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions for metrics
            all_predictions.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Compute metrics
    all_predictions = np.concatenate(all_predictions, axis=0).ravel()
    all_labels = np.concatenate(all_labels, axis=0).ravel()
    
    metrics = compute_metrics(all_predictions, all_labels)
    
    return avg_loss, metrics


def evaluate_at_threshold(predictions, labels, threshold, step=None):
    """
    Evaluate model predictions at a FIXED threshold (no threshold search).
    Used for cross-dataset validation to apply threshold learned from training set.
    Computes metrics at fixed threshold including EER and AUC.
    
    Args:
        predictions: Array of distance scores (lower = more similar)
        labels: Array of labels (1 = genuine pair, 0 = forged pair)
        threshold: Fixed distance threshold to use (typically d_optimal from training set)
        step: Ignored (kept for API compatibility)
        
    Returns:
        Dict with metrics at fixed threshold including EER and AUC
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    
    predictions = np.asarray(predictions).ravel()
    labels = np.asarray(labels).ravel()
    
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
    
    # Metrics at fixed threshold
    idx1 = predictions <= threshold      # pred = 1 (genuine)
    idx2 = predictions > threshold       # pred = 0 (forged)
    
    tp = float(np.sum(labels[idx1] == 1))
    tn = float(np.sum(labels[idx2] == 0))
    
    frr = float(np.sum(labels[idx2] == 1)) / nsame if nsame > 0 else 0.0
    far = float(np.sum(labels[idx1] == 0)) / ndiff if ndiff > 0 else 0.0
    acc = (tp + tn) / (nsame + ndiff) if (nsame + ndiff) > 0 else 0.0
    
    # Compute AUC
    try:
        auc_roc = roc_auc_score(labels, -predictions)
    except:
        auc_roc = float('nan')
    
    # Compute EER and ROC arrays for full curve
    fpr, tpr, thresholds = roc_curve(labels, -predictions)
    frr_arr = 1 - tpr
    far_arr = fpr
    
    # Find EER (where FAR = FRR)
    eer = float('nan')
    eer_far = float('nan')
    eer_frr = float('nan')
    for i in range(len(far_arr)):
        if abs(far_arr[i] - frr_arr[i]) < abs(eer - eer_far) if not np.isnan(eer) else True:
            eer = (far_arr[i] + frr_arr[i]) / 2.0
            eer_far = far_arr[i]
            eer_frr = frr_arr[i]
    
    d_arr = -predictions[np.argsort(predictions)]
    
    metrics = {
        "best_acc": acc,
        "best_frr": frr,
        "best_far": far,
        "eer": eer,
        "eer_frr": eer_frr,
        "eer_far": eer_far,
        "threshold": float(threshold),
        "auc_roc": auc_roc,
        "tpr_arr": tpr.tolist(),
        "far_arr": far_arr.tolist(),
        "frr_arr": frr_arr.tolist(),
        "fpr_arr": fpr.tolist(),
        "d_arr": d_arr.tolist(),
        "d_optimal": float(threshold),
        "d_optimal_eer": float(threshold)
    }
    
    print(f"📊 Fixed Threshold Metrics @d={threshold:.6f}: ACC={acc:.4f}, FAR={far:.4f}, FRR={frr:.4f}, AUC={auc_roc:.4f}, EER={eer:.4f}")
    
    return metrics


def plot_training_history(history, output_path=None):
    """
    Plot training history curves.
    
    Args:
        history: Dict with 'train_loss' and 'val_loss' keys
        output_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history.get('train_loss', []), label='Train Loss', marker='o')
    axes[0].plot(history.get('val_loss', []), label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid()
    
    # Metrics curve
    axes[1].plot(history.get('val_acc', []), label='Accuracy', marker='o')
    axes[1].plot(history.get('val_eer', []), label='EER', marker='s')
    axes[1].plot(history.get('val_auc', []), label='AUC', marker='^')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100)
        print(f"📊 Plot saved: {output_path}")
    
    plt.show()


def plot_roc_curve(metrics, output_path=None):
    """
    Plot ROC curve from metrics.
    
    Args:
        metrics: Metrics dict with fpr_arr and tpr_arr
        output_path: Path to save plot (optional)
    """
    fpr = metrics.get('fpr_arr', [])
    tpr = metrics.get('tpr_arr', [])
    auc_val = metrics.get('auc_roc', 0.0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc_val:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100)
        print(f"📊 ROC curve saved: {output_path}")
    
    plt.show()


def save_metrics_json(metrics, output_path):
    """
    Save metrics to JSON file (removes non-serializable arrays).
    
    Args:
        metrics: Metrics dict
        output_path: Path to save JSON
    """
    serializable_metrics = {
        'best_acc': float(metrics.get('best_acc', 0.0)),
        'best_far': float(metrics.get('best_far', 0.0)),
        'best_frr': float(metrics.get('best_frr', 0.0)),
        'eer': float(metrics.get('eer', 0.0)),
        'eer_far': float(metrics.get('eer_far', 0.0)),
        'eer_frr': float(metrics.get('eer_frr', 0.0)),
        'auc_roc': float(metrics.get('auc_roc', 0.0)),
        'd_optimal': float(metrics.get('d_optimal', 0.0)),
        'd_optimal_eer': float(metrics.get('d_optimal_eer', 0.0)),
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"📊 Metrics saved: {output_path}")
