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


def _init_margin_stats_acc():
    return {
        'pos_sum': 0.0,
        'pos_count': 0,
        'pos_max': float('-inf'),
        'neg_sum': 0.0,
        'neg_count': 0,
        'neg_min': float('inf'),
        'pos_violate_margin1_count': 0,
        'neg_violate_margin_count': 0,
    }


def _accumulate_margin_stats(acc, distances, labels, margin=None, margin_1=None):
    dist = distances.detach().view(-1)
    lab = labels.detach().view(-1)

    pos_mask = lab >= 0.5
    neg_mask = ~pos_mask

    pos_dist = dist[pos_mask]
    neg_dist = dist[neg_mask]

    if pos_dist.numel() > 0:
        acc['pos_sum'] += float(pos_dist.sum().item())
        acc['pos_count'] += int(pos_dist.numel())
        acc['pos_max'] = max(acc['pos_max'], float(pos_dist.max().item()))
        if margin_1 is not None:
            acc['pos_violate_margin1_count'] += int((pos_dist > margin_1).sum().item())

    if neg_dist.numel() > 0:
        acc['neg_sum'] += float(neg_dist.sum().item())
        acc['neg_count'] += int(neg_dist.numel())
        acc['neg_min'] = min(acc['neg_min'], float(neg_dist.min().item()))
        if margin is not None:
            acc['neg_violate_margin_count'] += int((neg_dist < margin).sum().item())


def _finalize_margin_stats(acc):
    pos_count = acc['pos_count']
    neg_count = acc['neg_count']

    pos_mean = (acc['pos_sum'] / pos_count) if pos_count > 0 else 0.0
    neg_mean = (acc['neg_sum'] / neg_count) if neg_count > 0 else 0.0

    pos_max = acc['pos_max'] if pos_count > 0 else 0.0
    neg_min = acc['neg_min'] if neg_count > 0 else 0.0

    pos_margin1_violation_rate = (
        acc['pos_violate_margin1_count'] / pos_count if pos_count > 0 else 0.0
    )
    neg_margin_violation_rate = (
        acc['neg_violate_margin_count'] / neg_count if neg_count > 0 else 0.0
    )

    return {
        'pos_mean': float(pos_mean),
        'neg_mean': float(neg_mean),
        'pos_max': float(pos_max),
        'neg_min': float(neg_min),
        'pos_count': int(pos_count),
        'neg_count': int(neg_count),
        'mean_gap_neg_minus_pos': float(neg_mean - pos_mean),
        'pos_margin1_violation_rate': float(pos_margin1_violation_rate),
        'neg_margin_violation_rate': float(neg_margin_violation_rate),
    }


def save_checkpoint(save_path, epoch, model, optimizer, scheduler, best_loss, history, args=None, best_eer=None):
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
        best_eer: Best validation EER so far (optional, for EER-based model selection)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'best_eer': best_eer,
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
    best_eer = checkpoint.get('best_eer', float('inf'))  # float('inf') for old checkpoints without EER
    history = checkpoint.get('history', {})
    args = checkpoint.get('args', {})
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"📂 Checkpoint loaded: {checkpoint_path}")
    print(f"   Resuming from epoch {epoch + 1}, best loss: {best_loss:.6f}, best EER: {best_eer:.6f}")
    
    return epoch, best_loss, best_eer, history, args


def compute_metrics(predictions, labels, step=None):
    """
    Compute signature verification metrics from predictions.
    
    Computes: accuracy, FAR, FRR, EER, and ROC curve.
    
    Notes:
        - `predictions` are distances where LOWER = more likely genuine.
        - Threshold semantics follow the repo: predict genuine if distance <= d.
        - Internally uses sklearn ROC utilities on scores = -distance.

    Args:
        predictions: Numpy array of predicted distances [N,]
        labels: Numpy array of labels (1=genuine, 0=forged) [N,]
        step: Ignored (kept for API compatibility)
        
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
    # Model outputs are distances where LOWER = more likely genuine.
    # For sklearn ROC utilities we use scores where HIGHER = more likely genuine.
    from sklearn.metrics import roc_auc_score, roc_curve

    predictions = np.asarray(predictions).ravel().astype(np.float64)
    labels = np.asarray(labels).ravel().astype(np.int64)

    nsame = int(np.sum(labels == 1))
    ndiff = int(np.sum(labels == 0))
    if (nsame + ndiff) == 0:
        raise ValueError("Empty labels passed to compute_metrics")

    scores = -predictions

    # ROC / AUC (robust)
    try:
        auc_roc = float(roc_auc_score(labels, scores))
    except Exception:
        auc_roc = float('nan')

    # ROC curve arrays
    # thresholds are in score-space; convert to distance-space via d = -threshold_score.
    fpr_arr, tpr_arr, thresholds_score = roc_curve(labels, scores)
    far_arr = fpr_arr
    frr_arr = 1.0 - tpr_arr
    d_arr = (-thresholds_score)

    # EER: point where FAR ~= FRR
    eer_idx = int(np.argmin(np.abs(far_arr - frr_arr)))
    eer_far = float(far_arr[eer_idx])
    eer_frr = float(frr_arr[eer_idx])
    eer = float((eer_far + eer_frr) / 2.0)
    d_optimal_eer = float(d_arr[eer_idx])

    # Best-accuracy threshold: sweep candidate distance thresholds.
    # Keep this consistent with repo semantics: pred genuine if distance <= d.
    max_acc = 0.0
    min_far = 1.0
    min_frr = 1.0
    d_optimal = float('nan')

    # Candidate thresholds: unique predicted distances + endpoints.
    # This is much cheaper than dense stepping and is exact for best-acc.
    unique_d = np.unique(predictions)
    # If extremely large, subsample to keep it fast.
    if unique_d.size > 50000:
        unique_d = np.quantile(unique_d, np.linspace(0, 1, 50000))
    for d in unique_d:
        idx1 = predictions <= d
        idx2 = ~idx1
        tp = float(np.sum(labels[idx1] == 1))
        tn = float(np.sum(labels[idx2] == 0))
        acc = (tp + tn) / float(nsame + ndiff)
        if acc > max_acc:
            max_acc = acc
            d_optimal = float(d)
            min_frr = float(np.sum(labels[idx2] == 1)) / float(nsame) if nsame > 0 else 0.0
            min_far = float(np.sum(labels[idx1] == 0)) / float(ndiff) if ndiff > 0 else 0.0
    
    metrics = {
        "best_acc": float(max_acc),
        "best_frr": float(min_frr),
        "best_far": float(min_far),
        "eer": float(eer),
        "eer_frr": float(eer_frr),
        "eer_far": float(eer_far),
        "auc_roc": float(auc_roc) if not np.isnan(auc_roc) else float('nan'),
        "tpr_arr": tpr_arr.tolist(),
        "far_arr": far_arr.tolist(),
        "frr_arr": frr_arr.tolist(),
        "fpr_arr": fpr_arr.tolist(),
        "d_arr": d_arr.tolist(),
        "d_optimal": float(d_optimal) if d_optimal == d_optimal else float('nan'),
        "d_optimal_eer": float(d_optimal_eer),
    }

    print(
        "📊 Metrics: "
        f"ACC={metrics['best_acc']:.4f} @d={metrics['d_optimal']:.6f} | "
        f"EER={metrics['eer']:.4f} @d={metrics['d_optimal_eer']:.6f} | "
        f"AUC={metrics['auc_roc']:.4f}"
    )
    
    return metrics


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device='cuda',
    shift_aug=False,
    log_margin_stats=False,
    return_margin_stats=False,
):
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
    
    margin_stats_acc = _init_margin_stats_acc() if (log_margin_stats or return_margin_stats) else None
    margin = getattr(loss_fn, 'margin', None)
    margin_1 = getattr(loss_fn, 'margin_1', None)

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch_data in enumerate(pbar):
        if len(batch_data) == 3:
            images, labels, writer_ids = batch_data
        else:
            images, labels = batch_data
        
        images = images.to(device)
        labels = labels.to(device)
        # Dataloaders in this repo often return labels shaped [B,1,1].
        # The model+loss expect a binary label per pair; normalize to [B,1].
        labels = labels.view(labels.shape[0], -1)[:, :1].float()
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, loss = model(images, labels)

        if margin_stats_acc is not None:
            _accumulate_margin_stats(margin_stats_acc, outputs, labels, margin=margin, margin_1=margin_1)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if margin_stats_acc is not None:
        margin_stats = _finalize_margin_stats(margin_stats_acc)
        if log_margin_stats:
            print(
                "📏 Margin stats: "
                f"pos_mean={margin_stats['pos_mean']:.4f}, "
                f"neg_mean={margin_stats['neg_mean']:.4f}, "
                f"pos_max={margin_stats['pos_max']:.4f}, "
                f"neg_min={margin_stats['neg_min']:.4f}, "
                f"gap(neg-pos)={margin_stats['mean_gap_neg_minus_pos']:.4f}"
            )
            if margin_1 is not None:
                print(
                    f"   pos>d1({margin_1:.4f}) rate={margin_stats['pos_margin1_violation_rate']:.4f}"
                )
            if margin is not None:
                print(
                    f"   neg<d2({margin:.4f}) rate={margin_stats['neg_margin_violation_rate']:.4f}"
                )
        if return_margin_stats:
            return avg_loss, margin_stats

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
            labels = labels.view(labels.shape[0], -1)[:, :1].float()
            
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
    
    # Compute EER and ROC arrays for full curve (thresholds in score-space)
    fpr, tpr, thresholds_score = roc_curve(labels, -predictions)
    frr_arr = 1.0 - tpr
    far_arr = fpr

    eer_idx = int(np.argmin(np.abs(far_arr - frr_arr)))
    eer_far = float(far_arr[eer_idx])
    eer_frr = float(frr_arr[eer_idx])
    eer = float((eer_far + eer_frr) / 2.0)

    d_arr = (-thresholds_score)
    d_optimal_eer = float(d_arr[eer_idx])
    
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
        "d_optimal_eer": d_optimal_eer
    }
    
    print(f"📊 Fixed Threshold Metrics @d={threshold:.6f}: ACC={acc:.4f}, FAR={far:.4f}, FRR={frr:.4f}, AUC={auc_roc:.4f}, EER={eer:.4f}")
    
    return metrics


def _extract_images_labels(batch_data):
    """Normalize dataloader batch payload to (images, labels)."""
    if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
        return batch_data[0], batch_data[1]
    raise ValueError("Expected batch_data to be tuple/list with at least (images, labels)")


def _resolve_mixed_batch_counts(primary_batch_size, replay_batch_size, mix_ratio=(1, 1), total_batch_size=None):
    if not isinstance(mix_ratio, (tuple, list)) or len(mix_ratio) != 2:
        raise ValueError(f"mix_ratio must be a length-2 tuple/list, got: {mix_ratio}")

    ratio_primary = int(mix_ratio[0])
    ratio_replay = int(mix_ratio[1])
    if ratio_primary <= 0 or ratio_replay <= 0:
        raise ValueError(f"mix_ratio entries must be > 0, got: {mix_ratio}")

    if total_batch_size is None:
        total_batch_size = int(primary_batch_size + replay_batch_size)
    total_batch_size = int(total_batch_size)
    if total_batch_size < 2:
        raise ValueError(f"total_batch_size must be >= 2, got: {total_batch_size}")

    ratio_sum = float(ratio_primary + ratio_replay)
    primary_count = int(round(total_batch_size * (ratio_primary / ratio_sum)))
    primary_count = max(1, min(primary_count, total_batch_size - 1))
    replay_count = int(total_batch_size - primary_count)

    if primary_count > int(primary_batch_size) or replay_count > int(replay_batch_size):
        raise ValueError(
            "Mixed-batch split exceeds available loader batch size. "
            f"Need primary/replay={primary_count}/{replay_count}, "
            f"but got primary/replay={primary_batch_size}/{replay_batch_size}. "
            "Increase the corresponding loader batch_size or reduce total_batch_size."
        )

    return primary_count, replay_count


def _build_mixed_batch(
    primary_images,
    primary_labels,
    replay_images,
    replay_labels,
    primary_count,
    replay_count,
    *,
    shuffle_within_batch=True,
    shuffle_generator=None,
):
    mixed_images = torch.cat(
        (primary_images[:primary_count], replay_images[:replay_count]),
        dim=0,
    )
    mixed_labels = torch.cat(
        (primary_labels[:primary_count], replay_labels[:replay_count]),
        dim=0,
    )

    if shuffle_within_batch and mixed_images.shape[0] > 1:
        perm = torch.randperm(mixed_images.shape[0], generator=shuffle_generator)
        if perm.device != mixed_images.device:
            perm = perm.to(mixed_images.device)
        mixed_images = mixed_images[perm]
        mixed_labels = mixed_labels[perm]

    return mixed_images, mixed_labels


def train_one_epoch_mixed(
    model,
    primary_loader,
    replay_loader,
    optimizer,
    loss_fn,
    device='cuda',
    *,
    primary_steps_per_replay=4,
    max_steps=None,
    log_margin_stats=False,
    return_margin_stats=False,
):
    """Train one epoch while mixing two dataloaders (for replay / anti-forgetting).

    Semantics:
    - Run `primary_steps_per_replay` batches from primary, then 1 batch from replay.
    - If `max_steps` is set, stop after that many optimizer steps.
    """

    model.train()
    total_loss = 0.0
    num_steps = 0
    primary_consumed = 0

    primary_iter = iter(primary_loader)
    replay_iter = iter(replay_loader)
    margin_stats_acc = _init_margin_stats_acc() if (log_margin_stats or return_margin_stats) else None
    margin = getattr(loss_fn, 'margin', None)
    margin_1 = getattr(loss_fn, 'margin_1', None)

    if max_steps is None:
        replay_steps_expected = int(np.ceil(len(primary_loader) / float(primary_steps_per_replay)))
        pbar_total = len(primary_loader) + replay_steps_expected
    else:
        pbar_total = max_steps
    pbar = tqdm(total=pbar_total, desc="Training (mixed)")
    while True:
        if max_steps is not None and num_steps >= max_steps:
            break

        # Primary batches
        for _ in range(primary_steps_per_replay):
            if max_steps is not None and num_steps >= max_steps:
                break
            try:
                batch_data = next(primary_iter)
            except StopIteration:
                return (total_loss / num_steps) if num_steps > 0 else 0.0

            if len(batch_data) == 3:
                images, labels, _ = batch_data
            else:
                images, labels = batch_data

            images = images.to(device)
            labels = labels.to(device)
            labels = labels.view(labels.shape[0], -1)[:, :1].float()

            optimizer.zero_grad()
            outputs, loss = model(images, labels)
            if margin_stats_acc is not None:
                _accumulate_margin_stats(margin_stats_acc, outputs, labels, margin=margin, margin_1=margin_1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            num_steps += 1
            primary_consumed += 1
            pbar.update(1)
            pbar.set_postfix({'loss': float(loss.item())})

        # If we've exhausted the primary loader, end the epoch without forcing an extra replay step.
        if max_steps is None and primary_consumed >= len(primary_loader):
            break

        # Replay batch
        if max_steps is not None and num_steps >= max_steps:
            break
        try:
            batch_data = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_loader)
            batch_data = next(replay_iter)

        if len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            images, labels = batch_data

        images = images.to(device)
        labels = labels.to(device)
        labels = labels.view(labels.shape[0], -1)[:, :1].float()

        optimizer.zero_grad()
        outputs, loss = model(images, labels)
        if margin_stats_acc is not None:
            _accumulate_margin_stats(margin_stats_acc, outputs, labels, margin=margin, margin_1=margin_1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_steps += 1
        pbar.update(1)
        pbar.set_postfix({'loss': float(loss.item())})

        if max_steps is None and primary_consumed >= len(primary_loader):
            break

    avg_loss = (total_loss / num_steps) if num_steps > 0 else 0.0

    if margin_stats_acc is not None:
        margin_stats = _finalize_margin_stats(margin_stats_acc)
        if log_margin_stats:
            print(
                "📏 Margin stats (mixed): "
                f"pos_mean={margin_stats['pos_mean']:.4f}, "
                f"neg_mean={margin_stats['neg_mean']:.4f}, "
                f"pos_max={margin_stats['pos_max']:.4f}, "
                f"neg_min={margin_stats['neg_min']:.4f}, "
                f"gap(neg-pos)={margin_stats['mean_gap_neg_minus_pos']:.4f}"
            )
            if margin_1 is not None:
                print(
                    f"   pos>d1({margin_1:.4f}) rate={margin_stats['pos_margin1_violation_rate']:.4f}"
                )
            if margin is not None:
                print(
                    f"   neg<d2({margin:.4f}) rate={margin_stats['neg_margin_violation_rate']:.4f}"
                )
        if return_margin_stats:
            return avg_loss, margin_stats

    return avg_loss


def train_one_epoch_mixed_batch(
    model,
    primary_loader,
    replay_loader,
    optimizer,
    loss_fn,
    device='cuda',
    *,
    mix_ratio=(1, 1),
    total_batch_size=None,
    max_steps=None,
    shuffle_within_batch=True,
    shuffle_seed=42,
    log_margin_stats=False,
    return_margin_stats=False,
):
    """Train one epoch using a single mixed batch from primary+replay on each step.

    Semantics:
    - Build each optimizer step from both loaders (sample-level mixing), then optional shuffle.
    - Default mix_ratio is 1:1 (primary:replay) inside each mixed batch.
    - If max_steps is None, epoch length is anchored to len(primary_loader).
    - replay_loader is cycled when exhausted.
    - Mixed-batch shuffle is seeded via `shuffle_seed` for reproducibility.

    Notes:
    - For best efficiency, configure loader batch sizes to satisfy the required split.
      Example: total_batch_size=32 and mix_ratio=(1,1) -> primary batch_size>=16, replay batch_size>=16.
    """

    if len(primary_loader) == 0:
        raise ValueError("primary_loader is empty")
    if len(replay_loader) == 0:
        raise ValueError("replay_loader is empty")

    model.train()
    total_loss = 0.0
    num_steps = 0

    primary_iter = iter(primary_loader)
    replay_iter = iter(replay_loader)
    margin_stats_acc = _init_margin_stats_acc() if (log_margin_stats or return_margin_stats) else None
    margin = getattr(loss_fn, 'margin', None)
    margin_1 = getattr(loss_fn, 'margin_1', None)

    pbar_total = int(max_steps) if max_steps is not None else int(len(primary_loader))
    pbar = tqdm(total=pbar_total, desc="Training (mixed-batch)")

    shuffle_generator = None
    if shuffle_within_batch and shuffle_seed is not None:
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(int(shuffle_seed))

    while True:
        if max_steps is not None and num_steps >= int(max_steps):
            break
        if max_steps is None and num_steps >= len(primary_loader):
            break

        try:
            primary_batch = next(primary_iter)
        except StopIteration:
            break

        try:
            replay_batch = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_loader)
            replay_batch = next(replay_iter)

        primary_images, primary_labels = _extract_images_labels(primary_batch)
        replay_images, replay_labels = _extract_images_labels(replay_batch)

        primary_count, replay_count = _resolve_mixed_batch_counts(
            primary_batch_size=int(primary_images.shape[0]),
            replay_batch_size=int(replay_images.shape[0]),
            mix_ratio=mix_ratio,
            total_batch_size=total_batch_size,
        )

        images, labels = _build_mixed_batch(
            primary_images,
            primary_labels,
            replay_images,
            replay_labels,
            primary_count,
            replay_count,
            shuffle_within_batch=bool(shuffle_within_batch),
            shuffle_generator=shuffle_generator,
        )

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels = labels.view(labels.shape[0], -1)[:, :1].float()

        optimizer.zero_grad()
        outputs, loss = model(images, labels)
        if margin_stats_acc is not None:
            _accumulate_margin_stats(margin_stats_acc, outputs, labels, margin=margin, margin_1=margin_1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_steps += 1
        pbar.update(1)
        pbar.set_postfix({'loss': float(loss.item())})

    avg_loss = (total_loss / num_steps) if num_steps > 0 else 0.0

    if margin_stats_acc is not None:
        margin_stats = _finalize_margin_stats(margin_stats_acc)
        if log_margin_stats:
            print(
                "📏 Margin stats (mixed-batch): "
                f"pos_mean={margin_stats['pos_mean']:.4f}, "
                f"neg_mean={margin_stats['neg_mean']:.4f}, "
                f"pos_max={margin_stats['pos_max']:.4f}, "
                f"neg_min={margin_stats['neg_min']:.4f}, "
                f"gap(neg-pos)={margin_stats['mean_gap_neg_minus_pos']:.4f}"
            )
            if margin_1 is not None:
                print(
                    f"   pos>d1({margin_1:.4f}) rate={margin_stats['pos_margin1_violation_rate']:.4f}"
                )
            if margin is not None:
                print(
                    f"   neg<d2({margin:.4f}) rate={margin_stats['neg_margin_violation_rate']:.4f}"
                )
        if return_margin_stats:
            return avg_loss, margin_stats

    return avg_loss


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
