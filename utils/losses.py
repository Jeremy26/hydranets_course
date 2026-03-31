"""
Loss Functions for Multi-Task Learning

- InverseHuberLoss (berHu): robust loss for depth estimation
- MultiTaskLoss: weighted combination of task losses
"""

import torch
import torch.nn as nn


class InverseHuberLoss(nn.Module):
    """Inverse Huber (berHu) loss for depth estimation.

    Below threshold c: L1 loss (handles small errors well)
    Above threshold c: L2 loss (handles large errors well)

    c = 0.2 * max(|target - pred|) computed per batch.
    """

    def __init__(self):
        super(InverseHuberLoss, self).__init__()

    def forward(self, prediction, target):
        mask = target > 0  # Only compute loss where depth is valid
        prediction = prediction[mask]
        target = target[mask]

        if prediction.numel() == 0:
            return torch.tensor(0.0, device=prediction.device)

        diff = torch.abs(target - prediction)
        c = 0.2 * torch.max(diff).item()

        # berHu: L1 below c, L2 above c
        l1_mask = diff <= c
        l2_mask = diff > c

        loss = torch.zeros_like(diff)
        loss[l1_mask] = diff[l1_mask]
        if c > 0:
            loss[l2_mask] = (diff[l2_mask] ** 2 + c ** 2) / (2 * c)

        return loss.mean()


class MultiTaskLoss(nn.Module):
    """Combines multiple task losses with learnable or fixed weights.

    Args:
        task_names: list of task names
        learnable: if True, learns the loss weights (uncertainty weighting)
        initial_weights: dict of initial weights per task
    """

    def __init__(self, task_names, learnable=False, initial_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.task_names = task_names
        self.learnable = learnable

        if initial_weights is None:
            initial_weights = {name: 1.0 for name in task_names}

        if learnable:
            # Learnable log-variance (Kendall et al., 2018)
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(0.0))
                for name in task_names
            })
        else:
            self.weights = initial_weights

    def forward(self, losses):
        """
        Args:
            losses: dict of {task_name: loss_value}
        Returns:
            total_loss, loss_dict (for logging)
        """
        total = torch.tensor(0.0, device=list(losses.values())[0].device)
        loss_dict = {}

        for name in self.task_names:
            if name not in losses:
                continue

            if self.learnable:
                precision = torch.exp(-self.log_vars[name])
                weighted = precision * losses[name] + self.log_vars[name]
            else:
                weighted = self.weights[name] * losses[name]

            total = total + weighted
            loss_dict[name] = losses[name].item()

        loss_dict['total'] = total.item()
        return total, loss_dict
