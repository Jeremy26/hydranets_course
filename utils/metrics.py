"""
Evaluation Metrics for Multi-Task Learning

- Mean IoU for semantic segmentation
- RMSE for depth estimation
"""

import torch
import numpy as np


def compute_miou(predictions, targets, num_classes, ignore_index=255):
    """Compute Mean Intersection over Union for segmentation.

    Args:
        predictions: (B, num_classes, H, W) logits
        targets: (B, H, W) integer class labels
        num_classes: number of classes
        ignore_index: label index to ignore

    Returns:
        miou: float
    """
    preds = predictions.argmax(dim=1)  # (B, H, W)
    ious = []

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        valid = (targets != ignore_index)

        pred_mask = pred_mask & valid
        target_mask = target_mask & valid

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            ious.append((intersection / union).item())

    return np.mean(ious) if ious else 0.0


def compute_rmse(prediction, target):
    """Compute Root Mean Squared Error for depth estimation.

    Args:
        prediction: (B, 1, H, W) predicted depth
        target: (B, 1, H, W) ground truth depth

    Returns:
        rmse: float
    """
    mask = target > 0  # Only evaluate where depth is valid
    if mask.sum() == 0:
        return 0.0

    diff = prediction[mask] - target[mask]
    return torch.sqrt((diff ** 2).mean()).item()
