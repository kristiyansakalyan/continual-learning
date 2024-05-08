import random
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import torch
from transformers import AutoImageProcessor

BG_VALUE = 0


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


def preprocess_mask2former_swin(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> Mask2FormerBatch:
    pixel_values = torch.stack([img for img, _ in batch])
    pixel_masks = torch.stack([torch.ones_like(img) for img, _ in batch])

    # Filter and collect class labels, ignoring the MASK_IGNORE_VALUE
    # class_labels = [torch.unique(mask) for _, mask in batch]
    class_labels = []
    for _, mask in batch:
        unique_labels = torch.unique(mask)
        # Filter out the ignore value
        unique_labels[unique_labels == BG_VALUE] = 0
        class_labels.append(unique_labels)

    mask_labels = []
    # Convert each mask from HxW to CxHxW
    for (_, mask), classes_list in zip(batch, class_labels):
        mask = mask.squeeze()
        mask_new = torch.zeros((len(classes_list), mask.shape[0], mask.shape[1]))

        for idx, curr in enumerate(classes_list):
            # Let's have the 0 class dedicated for background
            if curr == BG_VALUE:
                mask_new[0, mask == curr] = 1
                continue

            mask_new[idx, mask == curr] = 1

        mask_labels.append(mask_new)

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_masks,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }


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
    ignore_index=BG_VALUE,
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

    # We can use that: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mask2former/image_processing_mask2former.py
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
            (curr_mask.shape[1], curr_mask.shape[2]), BG_VALUE, dtype=torch.int32
        )

        for i, category in enumerate(category_list):
            output_mask[curr_mask[i] == 1] = int(category)  # wtf?

        masks.append(output_mask)

    return pred_maps, masks


def m2f_dataset_collate(examples):
    # Get the pixel values, pixel mask, mask labels, and class labels
    pixel_values = torch.stack(
        [example["pixel_values"].squeeze(0) for example in examples]
    )
    pixel_mask = torch.stack([example["pixel_mask"].squeeze(0) for example in examples])
    mask_labels = [example["mask_labels"][0] for example in examples]
    class_labels = [example["class_labels"][0] for example in examples]
    # Return a dictionary of all the collated features
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }
