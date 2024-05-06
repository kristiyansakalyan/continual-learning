import random
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import torch
from transformers import AutoImageProcessor

MASK_IGNORE_VALUE = 255


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Mask2FormerBatch(NamedTuple):
    pixel_values: torch.FloatTensor
    pixel_mask: torch.LongTensor
    mask_labels: list[torch.Tensor]
    class_labels: list[torch.LongTensor]


def _preprocess_mask2former(
    image_processor: AutoImageProcessor, batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> Mask2FormerBatch:
    """Preprocesses the data so that it is compatible with the
    expected Mask2Former input.

    Parameters
    ----------
    image_processor : AutoImageProcessor
        Mask2Former image processor
    batch : list[tuple[torch.Tensor, torch.Tensor]]
        A list of images and masks.

    Returns
    -------
    Mask2FormerBatch
        The expected input for the Mask2Former.
    """
    images = [img for img, _ in batch]
    segmentation_maps = [mask for _, mask in batch]

    return image_processor(
        images=images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
        do_rescale=False,
    )


# Preporcessor
mask2former_auto_image_processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-ade-semantic",
    ignore_index=MASK_IGNORE_VALUE,
    reduce_labels=False,
)

# Function to be used with dataloader collate_fn
preprocess_mask2former = partial(
    _preprocess_mask2former, mask2former_auto_image_processor
)


def m2f_extract_pred_maps_and_masks(
    batch: Mask2FormerBatch, outputs: Any, processor: AutoImageProcessor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract the predicted maps and masks for Mask2Former.

    Parameters
    ----------
    batch : Mask2FormerBatch
        The batch passed to the model.
    outputs : Any
        The output of the model.
    processor : AutoImageProcessor
        The image processor used for the model.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        Prediction maps and masks in HxW format.
    """
    # Extract predictions
    target_sizes = [
        (
            mask.shape[1],
            mask.shape[2],
        )
        for mask in batch["mask_labels"]
    ]
    pred_maps = processor.post_process_semantic_segmentation(
        outputs, target_sizes=target_sizes
    )

    # Get rid of warning in evaluate
    for i in range(len(pred_maps)):
        pred_maps[i] = pred_maps[i].to(torch.int32)

    # Convert CxHxW mask to HxW mask...
    masks = []

    # Fill the output mask with the category values
    for category_list, curr_mask in zip(batch["class_labels"], batch["mask_labels"]):
        output_mask = torch.full(
            (curr_mask.shape[1], curr_mask.shape[2]), 255, dtype=torch.int32
        )

        for i, category in enumerate(category_list):
            output_mask[curr_mask[i] == 1] = int(category)  # wtf?

        masks.append(output_mask)

    return pred_maps, masks
