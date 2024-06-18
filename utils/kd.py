import torch
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
)

from utils.common import get_pred_logits


def _local_pod(
    x: torch.Tensor,
    spp_scales: list[int] = [1, 2, 4],
    normalize: bool = False,
    normalize_per_scale: bool = False,
) -> torch.Tensor:
    """Local Pooled Output Distillation.

    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.
    Parameters
    ----------
    x : torch.Tensor
        (b, n, w, h) Learned features.
    spp_scales : list[int], optional
        The scales to look at, by default [1, 2, 4]
    normalize : bool, optional
        Whether to normalize globally, by default False
    normalize_per_scale : bool, optional
        Whether to normalize each scale, by default False

    Returns
    -------
    torch.Tensor
        _description_
    """
    b = x.shape[0]
    w = x.shape[-1]
    emb = []

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale

        nb_regions = scale**2
        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k : (i + 1) * k, j * k : (j + 1) * k]

                horizontal_pool = tensor.mean(dim=2).view(b, -1)
                vertical_pool = tensor.mean(dim=1).view(b, -1)

                # in case of nan
                if horizontal_pool.numel() == 0 or vertical_pool.numel() == 0:
                    break

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                    horizontal_pool = horizontal_pool / factor
                    vertical_pool = vertical_pool / factor

                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def pod(
    list_attentions_a: torch.Tensor,
    list_attentions_b: torch.Tensor,
    spp_scales: list[int] = [1, 2, 4],
    collapse_channels="local",
    normalize_per_scale: bool = False,
    normalize: bool = True,
    memory_flags: bool = None,
    only_old: bool = False,
) -> torch.Tensor:
    """Pooled Output Distillation.

    Parameters
    ----------
    list_attentions_a : torch.Tensor
        A list of attention maps, each of shape (b, n, w, h).
    list_attentions_b : torch.Tensor
        A list of attention maps, each of shape (b, n, w, h).
    spp_scales : list[int], optional
        The scales to look at., by default [1, 2, 4]
    collapse_channels : str, optional
        How to pool the channels., by default "local"
    normalize_per_scale : bool, optional
        Normalize each scale or not., by default False
    normalize : bool, optional
        Whether to normalize or not., by default True
    memory_flags : bool, optional
        Integer flags denoting exemplars., by default None
    only_old : bool, optional
        Only apply loss to exemplars., by default False

    Returns
    -------
    torch.Tensor
        A float scalar loss.

    Raises
    ------
    ValueError
        If the collapse channel method is not supported.
    """

    assert len(list_attentions_a) == len(list_attentions_b)

    loss = torch.tensor(0.0).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if collapse_channels == "width":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=2).view(a.shape[0], -1)
            b_h = b.sum(dim=2).view(b.shape[0], -1)
            a_w = a.sum(dim=1).view(a.shape[0], -1)
            b_w = b.sum(dim=1).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "local":
            a = _local_pod(
                a, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
            )
            b = _local_pod(
                b, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
            )
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)


def query_kd_loss(
    student_queries: torch.Tensor,
    teacher_queries: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Class-aware query knowledge distillation

    Parameters
    ----------
    student_queries : torch.Tensor
        The queries from the student model.
    teacher_queries : torch.Tensor
        The queries from the teacher model.
    temperature : float, optional
        Temperature, by default 1.0

    Returns
    -------
    torch.Tensor
        Loss
    """
    loss = 0.0
    for sq, tq in zip(student_queries, teacher_queries):
        sq = sq / temperature
        tq = tq / temperature
        loss += torch.norm(sq - tq, p="fro") ** 2
    return (1.0 / len(student_queries)) * loss


def distillation_loss(
    student_class_logits: torch.Tensor,
    teacher_class_logits: torch.Tensor,
    temperature: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """Knowledge distillation loss using KL divergence.

    Parameters
    ----------
    student_logits : torch.Tensor
        The logits from the student model.
    teacher_logits : torch.Tensor
        The logits from the teacher model.
    temperature : float, optional
        Temperature for scaling logits, by default 1.0.

    Returns
    -------
    torch.Tensor
        The computed distillation loss.
    """
    # Scale the logits by temperature
    student_class_logits = student_class_logits / temperature
    teacher_class_logits = teacher_class_logits / temperature

    # Compute log probabilities for the student logits
    student_log_probs = F.log_softmax(student_class_logits, dim=dim)

    # Compute probabilities for the teacher logits
    teacher_probs = F.softmax(teacher_class_logits, dim=dim)

    # Compute KL divergence
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # Scale the KL divergence by the square of the temperature
    loss = kl_div * (temperature**2)

    # Clamp the loss to ensure it is non-negative
    loss = torch.clamp(loss, min=0.0)

    return loss


def dice_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Dice loss with BCE component.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted logits.
    targets : torch.Tensor
        Ground truth labels.

    Returns
    -------
    torch.Tensor
        Dice loss.
    """
    # Apply sigmoid to predictions to get probabilities
    preds = torch.sigmoid(preds)
    targets = torch.sigmoid(targets)

    # Clamp predictions to avoid log(0) errors
    preds = torch.clamp(preds, min=1e-3, max=1 - 1e-3)

    # Compute binary cross-entropy loss
    bce = F.binary_cross_entropy(preds, targets, reduction="none")

    # Compute Dice loss component
    dice = (targets - preds) ** 2 * bce

    return dice.mean()


def mask_kd_loss(
    student_mask_logits: torch.Tensor,
    teacher_mask_logits: torch.Tensor,
    lambda_ce: float = 1.0,
    lambda_dice: float = 1.0,
) -> torch.Tensor:
    """Predicted mask knowledge distillation loss.

    Parameters
    ----------
    student_mask_logits : torch.Tensor
        The predicted mask logits from the student model.
    teacher_mask_logits : torch.Tensor
        The predicted mask logits from the teacher model.
    lambda_ce : float, optional
        Weight for the Cross-Entropy loss, by default 1.0.
    lambda_dice : float, optional
        Weight for the Dice loss, by default 1.0.

    Returns
    -------
    torch.Tensor
        The combined loss value.
    """
    ce_loss = distillation_loss(student_mask_logits, teacher_mask_logits)
    dice = dice_loss(student_mask_logits, teacher_mask_logits)

    # Combined loss
    return lambda_ce * ce_loss + lambda_dice * dice


DEFAULT_LAMBDAS = {"q": 1, "c": 5, "m": 300, "pod": 100}


def compute_kd_loss(
    student_outputs: Mask2FormerForUniversalSegmentationOutput,
    teacher_outputs: Mask2FormerForUniversalSegmentationOutput,
    lambdas: dict[str, int] = DEFAULT_LAMBDAS,
    lambda_ce: float = 2.0,
    lambda_dice: float = 5.0,
    pred_temperature: float = 1.0,
    pred_dim: int = -1,
    query_temperature: float = 1.0,
    spp_scales: list[int] = [1, 2, 4],
    collapse_channels="local",
    normalize_per_scale: bool = False,
    normalize: bool = True,
    memory_flags: bool = None,
    only_old: bool = False,
    verbose: bool = False,
    compute_query_loss: bool = True,
    compute_class_loss: bool = True,
    compute_mask_loss: bool = True,
    compute_pod_loss: bool = True,
) -> torch.Tensor:
    """
    Compute the knowledge distillation (KD) loss for the student model outputs
    based on the teacher model outputs.

    Parameters
    ----------
    student_outputs : Mask2FormerForUniversalSegmentationOutput
        The outputs from the student model.
    teacher_outputs : Mask2FormerForUniversalSegmentationOutput
        The outputs from the teacher model.
    lambdas : dict[str, int], optional
        Weights for different loss components, by default DEFAULT_LAMBDAS.
    lambda_ce : float, optional
        Weight for the cross-entropy part of the mask KD loss, by default 2.0.
    lambda_dice : float, optional
        Weight for the dice part of the mask KD loss, by default 5.0.
    pred_temperature : float, optional
        Temperature for the prediction KD loss, by default 1.0.
    pred_dim : int, optional
        Dimension for class prediction KD loss, by default -1.
    query_temperature : float, optional
        Temperature for the query KD loss, by default 1.0.
    spp_scales : list[int], optional
        Scales for the pooled output distillation (POD) loss, by default [1, 2, 4].
    collapse_channels : str, optional
        Method for collapsing channels in POD loss, by default "local".
    normalize_per_scale : bool, optional
        Whether to normalize each scale in POD loss, by default False.
    normalize : bool, optional
        Whether to normalize in POD loss, by default True.
    memory_flags : bool, optional
        Flags denoting exemplars for POD loss, by default None.
    only_old : bool, optional
        Whether to apply loss only to exemplars, by default False.
    verbose : bool, optional
        Whether to print the losses.
    compute_query_loss : bool, optional
        Whether to compute the query KD loss, by default True.
    compute_class_loss : bool, optional
        Whether to compute the class KD loss, by default True.
    compute_mask_loss : bool, optional
        Whether to compute the mask KD loss, by default True.
    compute_pod_loss : bool, optional
        Whether to compute the POD loss, by default True.

    Returns
    -------
    torch.Tensor
        The computed KD loss.
    """
    total_loss = 0.0

    if compute_query_loss:
        student_transf_queries = student_outputs.transformer_decoder_hidden_states
        teacher_transf_queries = teacher_outputs.transformer_decoder_hidden_states

        loss_queries = sum(
            query_kd_loss(s_q, t_q, temperature=query_temperature)
            for s_q, t_q in zip(student_transf_queries, teacher_transf_queries)
        )
        loss_queries *= lambdas["q"]
        total_loss += loss_queries
    else:
        loss_queries = torch.tensor(0.0)

    if compute_class_loss:
        loss_class = distillation_loss(
            student_outputs.class_queries_logits,
            teacher_outputs.class_queries_logits,
            temperature=pred_temperature,
            dim=pred_dim,
        )
        loss_class *= lambdas["c"]
        total_loss += loss_class
    else:
        loss_class = torch.tensor(0.0)

    if compute_mask_loss:
        student_pred_masks = get_pred_logits(
            student_outputs.class_queries_logits,
            student_outputs.masks_queries_logits,
            resize=False,
        )
        teacher_pred_masks = get_pred_logits(
            teacher_outputs.class_queries_logits,
            teacher_outputs.masks_queries_logits,
            resize=False,
        )

        loss_mask = mask_kd_loss(
            # student_outputs.masks_queries_logits,
            # teacher_outputs.masks_queries_logits,
            student_pred_masks,
            teacher_pred_masks,
            lambda_ce=lambda_ce,
            lambda_dice=lambda_dice,
        )
        loss_mask *= lambdas["m"]
        total_loss += loss_mask
    else:
        loss_mask = torch.tensor(0.0)

    if compute_pod_loss:
        loss_pod_backbone = pod(
            student_outputs.encoder_last_hidden_state,
            teacher_outputs.encoder_last_hidden_state,
            spp_scales=spp_scales,
            collapse_channels=collapse_channels,
            normalize_per_scale=normalize_per_scale,
            normalize=normalize,
            memory_flags=memory_flags,
            only_old=only_old,
        )

        loss_pod_pixel_decoder = pod(
            student_outputs.pixel_decoder_last_hidden_state,
            teacher_outputs.pixel_decoder_last_hidden_state,
            spp_scales=spp_scales,
            collapse_channels=collapse_channels,
            normalize_per_scale=normalize_per_scale,
            normalize=normalize,
            memory_flags=memory_flags,
            only_old=only_old,
        )

        loss_pod = loss_pod_backbone + loss_pod_pixel_decoder
        loss_pod *= lambdas["pod"]
        total_loss += loss_pod
    else:
        loss_pod = torch.tensor(0.0)

    if verbose:
        print(
            f"L_q = {loss_queries.item():.4f}; L_c = {loss_class.item():.4f}; L_m = {loss_mask.item():.4f}; L_pod = {loss_pod.item():.4f}"
        )

    return total_loss
