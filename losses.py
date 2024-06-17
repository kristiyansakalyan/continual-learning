"""
Below code is taken from here: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
And some changes are made for our semantic segmentation problem.
"""

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views,  num_classes, H, W ].
            labels: ground truth of shape [bsz, n_views, H, W ].
            mask: contrastive mask of shape [bsz * n_views * H * W, bsz * n_views * H * W], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
        bsz, n_views, num_classes, H, W = features.shape

        # Reshape features to [bsz * H * W, num_classes]
        features = features.permute(0, 1, 3, 4, 2).reshape(
            bsz * n_views * H * W, num_classes
        )

        # Reshape labels to [bsz * n_views * H * W]
        if labels is not None:
            labels = labels.view(bsz * n_views * H * W)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)

        else:
            mask = mask.float().to(device)

        contrast_count = n_views
        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        # mask_pos_pairs = mask.sum(1)
        # mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        mask_pos_pairs = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
