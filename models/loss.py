"""Combined BCE + Focal + L2 regularization loss for binary depression classification."""

import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, lambda_reg=1e-4, alpha=1, gamma=2, focal_weight=1.0, l2_weight=1.0, smooth_eps=0.1):
        super(CombinedLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.l2_weight = l2_weight
        self.smooth_eps = smooth_eps
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, inputs, targets, model):
        smooth_targets = targets * (1 - self.smooth_eps) + 0.5 * self.smooth_eps
        bce_loss = self.bce_loss(inputs, smooth_targets)

        l2_reg = self.l2_weight * self.lambda_reg * sum(param.norm(2) for param in model.parameters())

        probs = torch.sigmoid(inputs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = torch.where(targets == 1, 1 - probs, probs)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        focal_loss = self.focal_weight * bce_loss * focal_weight
        focal_loss = focal_loss.mean()

        return bce_loss + focal_loss + l2_reg
