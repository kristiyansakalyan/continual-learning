"""
The PixelContrastLoss class is taken from: https://github.com/tfzhou/ContrastiveSeg
The prototype part is taken from: https://github.com/Simael/DSP/tree/main 
"""

from abc import ABC
from typing import Dict, Literal, NamedTuple, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelContrastLoss(nn.Module, ABC):
    def __init__(
        self,
        full_batch_sampling=False,
        weights=None,
        num_classes=None,
        num_prototypes_per_class=None,
        in_channels=None,
    ):
        super(PixelContrastLoss, self).__init__()

        self.temperature = 0.1  # This might also be 0.07 (github), 0.1 in the paper
        self.base_temperature = 0.07  # from github
        self.full_batch_sampling = full_batch_sampling
        self.ignore_label = -1  # from github

        self.max_samples = 1024  # from github
        self.max_views = 100  # 50 in the paper, 100 in github
        self.weights = (
            weights  # dictionary of class weights to be used in the contrastive loss
        )
        self.prototypes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (
            num_classes is not None
            and num_prototypes_per_class is not None
            and in_channels is not None
        ):
            # Setup random prototypes
            self.prototypes = torch.randn(
                (num_classes, num_prototypes_per_class, in_channels)
            )
            # Calculate norm
            norm = self.prototypes.pow(2).sum(2, keepdim=True).pow(1.0 / 2.0)
            # Initialize prototypes as learnable parameters
            self.prototypes = torch.nn.Parameter(
                self.prototypes.div(norm), requires_grad=True
            )

    def _hard_anchor_sampling(self, X, y_hat, y):
        """
        Sample positive and negative pixels for each class. Positive pixels will be easy samples (prediciton==gt)
        and the negative pixels will be hard samples (prediction!=gt)


        Parameters
        ----------
        X : torch.Tensor
            All the pixels in the given batch. Has shape [batch_size, H*W, feat_dim]
        y_hat : torch.Tensor
            Predicted semantic labels of size [batch_size, H*W]
        y : torch.Tensor
            Predicted semantic labels of size [batch_size, H*W]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Sampled pixels of size [total_classes, n_view, feat_dim] and their groundtruth labels
            of size [total_classes] are returned.
            (total_classes: total number of unique clases in B images,
            n_view: total number of positive and negative pixels for the given class in the current B image)
        """

        # X_B=X[:X.size(0)//2,...] # cataract images
        # X_A=X[X.size(0)//2:,...] # cadis images

        if (
            self.full_batch_sampling
        ):  # For each image, pixels from all the other images in the given batch will be sampled too (used in A training)
            batch_size, feat_dim = X.shape[0], X.shape[-1]
        else:  # For each image in the first half of the batch, pixels from the second half of the batch will be sampled too (used in B training)
            batch_size, feat_dim = X.shape[0] // 2, X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [
                x
                for x in this_classes
                if (this_y == x).nonzero().shape[0] > self.max_views
            ]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(
            self.device
        )  # sampled positive and negative pixels will be stored
        y_ = torch.zeros(total_classes, dtype=torch.float).to(
            self.device
        )  # GT labels of X_

        X_ptr = 0
        # Find positive and negative pixels for all the unique classes in B images
        for ii in range(batch_size):
            this_y_hat_B = y_hat[ii]  # prediction
            this_y_B = y[ii]  # groundtruth
            this_classes_B = classes[ii]  # unique classes

            for cls_id in this_classes_B:
                hard_indices_B = (
                    (this_y_hat_B == cls_id) & (this_y_B != cls_id)
                ).nonzero()  # negative samples
                easy_indices_B = (
                    (this_y_hat_B == cls_id) & (this_y_B == cls_id)
                ).nonzero()  # positive samples
                hard_pixels_A = []  # all negative A pixels will be stored
                easy_pixels_A = []  # all positive A pixels will be stored

                if self.full_batch_sampling:
                    # Traverse the rest of the batch, any other image except the current one
                    for i in range(batch_size):
                        if i == ii:
                            continue
                        this_y_A = y[i]  # groundtruth

                        if cls_id in torch.unique(this_y_A):
                            this_y_hat_A = y_hat[i]  # prediction

                            easy_indices_A = (
                                (this_y_hat_A == cls_id) & (this_y_A == cls_id)
                            ).nonzero()
                            if len(easy_indices_A) > 0:
                                easy_pixels_A.append(X[i, easy_indices_A, :].squeeze(1))

                        hard_indices_A = (
                            (this_y_hat_A == cls_id) & (this_y_A != cls_id)
                        ).nonzero()
                        if len(hard_indices_A) > 0:
                            hard_pixels_A.append(X[i, hard_indices_A, :].squeeze(1))

                else:
                    # Traverse replayed "A" images, the second half of the batch
                    for i in range(batch_size, batch_size * 2):
                        this_y_A = y[i]  # groundtruth

                        if cls_id in torch.unique(this_y_A):
                            this_y_hat_A = y_hat[i]  # prediction

                            easy_indices_A = (
                                (this_y_hat_A == cls_id) & (this_y_A == cls_id)
                            ).nonzero()
                            if len(easy_indices_A) > 0:
                                easy_pixels_A.append(X[i, easy_indices_A, :].squeeze(1))

                        hard_indices_A = (
                            (this_y_hat_A == cls_id) & (this_y_A != cls_id)
                        ).nonzero()
                        if len(hard_indices_A) > 0:
                            hard_pixels_A.append(X[i, hard_indices_A, :].squeeze(1))

                num_hard = (
                    len(hard_pixels_A) + hard_indices_B.shape[0]
                )  # total number of negative pixels
                num_easy = (
                    len(easy_pixels_A) + easy_indices_B.shape[0]
                )  # total number of positive pixels

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print(
                        "this shoud be never touched! {} {} {}".format(
                            num_hard, num_easy, n_view
                        )
                    )
                    raise Exception

                negative_B_keep = num_hard_keep // 2
                positive_B_keep = num_easy_keep // 2
                negative_A_keep = num_hard_keep - negative_B_keep
                positive_A_keep = num_easy_keep - positive_B_keep

                hard_pixels_A_ = None
                easy_pixels_A_ = None

                if len(hard_pixels_A) > 0:
                    hard_pixels_A_ = torch.cat(hard_pixels_A, dim=0).to(self.device)
                    if hard_pixels_A_.shape[0] < negative_A_keep:
                        negative_A_keep = hard_pixels_A_.shape[0]
                        negative_B_keep = num_hard_keep - negative_A_keep
                    elif hard_indices_B.shape[0] < negative_B_keep:
                        negative_B_keep = hard_indices_B.shape[0]
                        negative_A_keep = num_hard_keep - negative_B_keep
                else:
                    negative_B_keep = num_hard_keep
                    negative_A_keep = 0

                if len(easy_pixels_A) > 0:
                    easy_pixels_A_ = torch.cat(easy_pixels_A, dim=0).to(self.device)
                    if easy_pixels_A_.shape[0] < positive_A_keep:
                        positive_A_keep = easy_pixels_A_.shape[0]
                        positive_B_keep = num_easy_keep - positive_A_keep
                    elif easy_indices_B.shape[0] < positive_B_keep:
                        positive_B_keep = easy_indices_B.shape[0]
                        positive_A_keep = num_easy_keep - positive_B_keep
                else:
                    positive_B_keep = num_easy_keep
                    positive_A_keep = 0

                indices_B = get_random_elements(
                    negative_B_keep,
                    positive_B_keep,
                    hard_elements=hard_indices_B,
                    easy_elements=easy_indices_B,
                ).to(self.device)
                pixels_B = X[ii, indices_B, :].squeeze(1)

                if positive_A_keep + negative_A_keep > 0:
                    pixels_A = get_random_elements(
                        negative_A_keep,
                        positive_A_keep,
                        hard_elements=hard_pixels_A_,
                        easy_elements=easy_pixels_A_,
                    ).to(self.device)

                    pixels = torch.cat((pixels_B, pixels_A), dim=0).to(self.device)
                else:
                    pixels = pixels_B
                X_[X_ptr, :, :] = pixels
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_, prototype=None):
        """
        Contrastive loss. Taken directly from https://github.com/tfzhou/ContrastiveSeg


        Parameters
        ----------
        feats_ : torch.Tensor
            Pixels to be used in the contrastive loss. Has shape [total_classes, n_view, feat_dim]
        labels_ : torch.Tensor
            Groundtruth class labels of size [total_classes]

        (total_classes: total number of unique clases in B images,
        n_view: total number of positive and negative pixels for the given class in the current B image)


        Returns
        -------
        Float
            Returns contrastive loss
        """

        anchor_num, n_view = feats_.shape[0], feats_.shape[1]
        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().to(self.device)

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0).to(self.device)

        if prototype:
            anchor_feature = prototype
            # TODO !!!
            # anchor_count =?
        else:
            anchor_feature = contrast_feature

            anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
            self.temperature,
        ).to(self.device)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count).to(self.device)
        neg_mask = 1 - mask

        logits_mask = (
            torch.ones_like(mask)
            .scatter_(
                1,
                torch.arange(anchor_num * anchor_count).view(-1, 1).to(self.device),
                0,
            )
            .to(self.device)
        )
        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        if self.weights is not None:
            weight_tensor = torch.ones(labels_.size()).to(self.device)

            for idx, lbl in enumerate(labels_):
                weight_tensor[idx] = self.weights[int(lbl)]

            weight_tensor = weight_tensor.repeat(contrast_count, 1).reshape(-1)

            mean_log_prob_pos *= weight_tensor

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        """labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')"""

        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], "{} {}".format(
            labels.shape, feats.shape
        )

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)

        feats_normalized = F.normalize(feats, p=2, dim=1)
        feats = feats_normalized.permute(0, 2, 3, 1)

        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss


def get_random_elements(
    num_hard_keep, num_easy_keep, hard_elements=None, easy_elements=None
):
    """
    Randomly sample hard (negative) and easy (positive) elements for the given image


    Parameters
    ----------
    hard_elements : torch.Tensor
        Either hard pixels or hard pixel indices to randomly sample from.
    easy_elements : torch.Tensor
        Either easy pixels or easy pixel indices to randomly sample from.

    Returns
    -------
    torch.Tensor
        Returns the sampled and concatenated hard and easy elements
    """
    if hard_elements is not None and easy_elements is not None:
        num_hard = hard_elements.shape[0]
        num_easy = easy_elements.shape[0]
        perm = torch.randperm(num_hard)
        hard_elements = hard_elements[perm[:num_hard_keep]]
        perm = torch.randperm(num_easy)
        easy_elements = easy_elements[perm[:num_easy_keep]]

        elements = torch.cat((hard_elements, easy_elements), dim=0)

    elif hard_elements is not None:
        num_hard = hard_elements.shape[0]
        perm = torch.randperm(num_hard)
        elements = hard_elements[perm[:num_hard_keep]]

    elif easy_elements is not None:
        num_easy = easy_elements.shape[0]
        perm = torch.randperm(num_easy)
        elements = easy_elements[perm[:num_easy_keep]]

    return elements
