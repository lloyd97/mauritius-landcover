"""
Evaluation Metrics for Land Cover Classification
=================================================

Provides comprehensive metrics for semantic segmentation:
    - Pixel accuracy
    - Mean IoU (Intersection over Union)
    - Per-class IoU
    - F1 Score
    - Confusion matrix
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class SegmentationMetrics:
    """
    Compute segmentation metrics.
    
    Tracks predictions over batches and computes aggregate metrics.
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize metrics tracker.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in metrics
            class_names: Optional class names for reporting
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Predicted class indices (B, H, W)
            targets: Ground truth class indices (B, H, W)
        """
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Filter out ignore index
        mask = targets != self.ignore_index
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        for p, t in zip(predictions, targets):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from confusion matrix.
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Pixel accuracy
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        metrics['accuracy'] = correct / max(total, 1)
        
        # Per-class IoU
        iou_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            iou = tp / max(tp + fp + fn, 1)
            iou_per_class.append(iou)
            metrics[f'iou_{self.class_names[i]}'] = iou
        
        # Mean IoU
        metrics['miou'] = np.mean(iou_per_class)
        
        # Per-class Dice/F1
        f1_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            f1_per_class.append(f1)
            metrics[f'f1_{self.class_names[i]}'] = f1
        
        # Mean F1
        metrics['mean_f1'] = np.mean(f1_per_class)
        
        # Weighted IoU (by class frequency)
        class_freq = self.confusion_matrix.sum(axis=1)
        total_freq = class_freq.sum()
        if total_freq > 0:
            weights = class_freq / total_freq
            metrics['weighted_iou'] = np.sum(np.array(iou_per_class) * weights)
        else:
            metrics['weighted_iou'] = 0
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.confusion_matrix.copy()
    
    def print_report(self):
        """Print detailed metrics report."""
        metrics = self.compute()
        
        print("\n" + "=" * 60)
        print("SEGMENTATION METRICS REPORT")
        print("=" * 60)
        
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean IoU: {metrics['miou']:.4f}")
        print(f"Mean F1: {metrics['mean_f1']:.4f}")
        print(f"Weighted IoU: {metrics['weighted_iou']:.4f}")
        
        print("\nPer-class metrics:")
        print("-" * 50)
        print(f"{'Class':<15} {'IoU':<10} {'F1':<10} {'Support'}")
        print("-" * 50)
        
        for i, name in enumerate(self.class_names):
            support = self.confusion_matrix[i, :].sum()
            print(f"{name:<15} {metrics[f'iou_{name}']:<10.4f} "
                  f"{metrics[f'f1_{name}']:<10.4f} {support}")
        
        print("=" * 60)


def compute_class_weights(
    dataset,
    num_classes: int,
    method: str = 'inverse_freq'
) -> torch.Tensor:
    """
    Compute class weights from dataset.
    
    Args:
        dataset: PyTorch dataset
        num_classes: Number of classes
        method: Weighting method ('inverse_freq', 'median_freq', 'log_freq')
        
    Returns:
        Tensor of class weights
    """
    class_counts = torch.zeros(num_classes)
    
    for sample in dataset:
        mask = sample.get('mask')
        if mask is not None:
            for c in range(num_classes):
                class_counts[c] += (mask == c).sum()
    
    total = class_counts.sum()
    
    if method == 'inverse_freq':
        weights = total / (num_classes * class_counts + 1)
    elif method == 'median_freq':
        median = torch.median(class_counts[class_counts > 0])
        weights = median / (class_counts + 1)
    elif method == 'log_freq':
        weights = 1 / torch.log(1.1 + class_counts / total)
    else:
        weights = torch.ones(num_classes)
    
    # Normalize
    weights = weights / weights.sum()
    
    return weights


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100
) -> Tuple[float, List[float]]:
    """
    Compute IoU score.
    
    Args:
        pred: Predictions (B, H, W)
        target: Ground truth (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore
        
    Returns:
        Tuple of (mean_iou, per_class_iou)
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        
        # Ignore mask
        if ignore_index >= 0:
            valid = target != ignore_index
            pred_c = pred_c & valid
            target_c = target_c & valid
        
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(float('nan'))
    
    # Filter NaN values for mean
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return mean_iou, ious


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> float:
    """
    Compute Dice score (F1).
    
    Args:
        pred: Predictions (B, C, H, W) - probabilities
        target: Ground truth (B, H, W) - class indices
        smooth: Smoothing factor
        
    Returns:
        Dice score
    """
    num_classes = pred.shape[1]
    
    # Convert predictions to class indices
    pred_classes = pred.argmax(dim=1)
    
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    
    return np.mean(dice_scores)


class ChangeDetectionMetrics:
    """
    Metrics for change detection evaluation.
    
    Computes:
        - Overall accuracy
        - Change/no-change accuracy
        - Kappa coefficient
        - False alarm rate
        - Miss rate
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.tp = 0  # True positive (correctly detected change)
        self.tn = 0  # True negative (correctly detected no change)
        self.fp = 0  # False positive (false alarm)
        self.fn = 0  # False negative (missed change)
    
    def update(
        self,
        pred_change: torch.Tensor,
        true_change: torch.Tensor
    ):
        """
        Update metrics.
        
        Args:
            pred_change: Predicted change mask (binary)
            true_change: Ground truth change mask (binary)
        """
        pred = pred_change.cpu().numpy().flatten().astype(bool)
        true = true_change.cpu().numpy().flatten().astype(bool)
        
        self.tp += np.sum(pred & true)
        self.tn += np.sum(~pred & ~true)
        self.fp += np.sum(pred & ~true)
        self.fn += np.sum(~pred & true)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        total = self.tp + self.tn + self.fp + self.fn
        
        metrics = {
            'accuracy': (self.tp + self.tn) / max(total, 1),
            'precision': self.tp / max(self.tp + self.fp, 1),
            'recall': self.tp / max(self.tp + self.fn, 1),
            'false_alarm_rate': self.fp / max(self.fp + self.tn, 1),
            'miss_rate': self.fn / max(self.fn + self.tp, 1),
        }
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = (2 * metrics['precision'] * metrics['recall'] / 
                           (metrics['precision'] + metrics['recall']))
        else:
            metrics['f1'] = 0
        
        # Kappa coefficient
        pe = ((self.tp + self.fp) * (self.tp + self.fn) + 
              (self.fn + self.tn) * (self.fp + self.tn)) / max(total ** 2, 1)
        po = metrics['accuracy']
        metrics['kappa'] = (po - pe) / max(1 - pe, 1e-6)
        
        return metrics


if __name__ == '__main__':
    # Test metrics
    print("Testing segmentation metrics...")
    
    # Create sample data
    num_classes = 7
    batch_size = 2
    height, width = 256, 256
    
    # Random predictions and targets
    pred = torch.randint(0, num_classes, (batch_size, height, width))
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test metrics
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        class_names=['bg', 'roads', 'water', 'forest', 'plantation', 'buildings', 'bare']
    )
    
    metrics.update(pred, target)
    metrics.print_report()
    
    # Test IoU
    miou, class_ious = iou_score(pred, target, num_classes)
    print(f"\nDirect IoU calculation: {miou:.4f}")
    
    # Test change detection metrics
    print("\nTesting change detection metrics...")
    cd_metrics = ChangeDetectionMetrics()
    
    pred_change = torch.randint(0, 2, (100, 100))
    true_change = torch.randint(0, 2, (100, 100))
    
    cd_metrics.update(pred_change, true_change)
    cd_results = cd_metrics.compute()
    
    print("Change detection results:")
    for k, v in cd_results.items():
        print(f"  {k}: {v:.4f}")
