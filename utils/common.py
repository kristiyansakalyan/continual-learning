import random
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import torch
from transformers import AutoImageProcessor

# BG_VALUE_255 = 255  # 0
BG_VALUE_0 = 0
TARGET_SIZE = (270, 480)
CADIS_PIXEL_MEAN = [0.57365126, 0.34606295, 0.19539679]
CADIS_PIXEL_STD = [0.15933991, 0.15584118, 0.10485045]
CAT1K_PIXEL_MEAN = [0.31471649, 0.28790466, 0.21170973]
CAT1K_PIXEL_STD = [0.30111405, 0.25568646, 0.22290612]
FULL_MERGE_PIXEL_MEAN = None
FULL_MERGE_PIXEL_STD = None
WEIGHTS_CADIS_TRAIN = [
    0.02725455,
    0.09069091,
    0.09068182,
    0.09065455,
    0.09082727,
    0.09083636,
    0.09075455,
    0.09073636,
    0.09062727,
    0.09062727,
    0.07541818,
    0.08089091,
]


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

    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set as {seed}")


def pixel_mean_std(dataset):
    sum_channels = np.zeros(3)
    sum_squares_channels = np.zeros(3)
    count_pixels = 0
    for im, _ in dataset:
        img = im.permute(1, 2, 0).numpy()
        sum_channels += np.sum(img, axis=(0, 1))
        sum_squares_channels += np.sum(img**2, axis=(0, 1))
        count_pixels += img.shape[0] * img.shape[1]
    pixel_mean = sum_channels / count_pixels
    pixel_std = np.sqrt((sum_squares_channels / count_pixels) - pixel_mean**2)

    return pixel_mean, pixel_std


def get_dataset_weights(dataset, num_classes):
    total_pixels = 0
    class_pixels = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(dataset)):
        _, mask = dataset[i]
        mask = mask.numpy()
        for cls_ in range(num_classes):
            class_pixels[cls_] += np.sum(mask == cls_)

    total_pixels = class_pixels.sum()
    class_distribution = class_pixels / total_pixels
    l = np.zeros(num_classes)
    for i in range(len(class_distribution)):
        l[i] += round(1 - class_distribution[i], 4)
    weights = l / l.sum()
    return weights


class Mask2FormerBatch(NamedTuple):
    pixel_values: torch.FloatTensor
    pixel_mask: torch.LongTensor
    mask_labels: list[torch.Tensor]
    class_labels: list[torch.LongTensor]


# def preprocess_mask2former_swin(
#     batch: list[tuple[torch.Tensor, torch.Tensor]]
# ) -> Mask2FormerBatch:
#     pixel_values = torch.stack([img for img, _ in batch])
#     pixel_masks = torch.stack([torch.ones_like(img) for img, _ in batch])

#     # Filter and collect class labels, ignoring the MASK_IGNORE_VALUE
#     # class_labels = [torch.unique(mask) for _, mask in batch]
#     class_labels = []
#     for _, mask in batch:
#         unique_labels = torch.unique(mask)
#         # Filter out the ignore value
#         unique_labels[unique_labels == BG_VALUE_255] = 0
#         class_labels.append(unique_labels)

#     mask_labels = []
#     # Convert each mask from HxW to CxHxW
#     for (_, mask), classes_list in zip(batch, class_labels):
#         mask = mask.squeeze()
#         mask_new = torch.zeros((len(classes_list), mask.shape[0], mask.shape[1]))

#         for idx, curr in enumerate(classes_list):
#             # Let's have the 0 class dedicated for background
#             if curr == BG_VALUE_255:
#                 mask_new[0, mask == curr] = 1
#                 continue

#             mask_new[idx, mask == curr] = 1

#         mask_labels.append(mask_new)

#     return {
#         "pixel_values": pixel_values,
#         "pixel_mask": pixel_masks,
#         "mask_labels": mask_labels,
#         "class_labels": class_labels,
#     }


# def _preprocess_mask2former(
#     image_processor: AutoImageProcessor, batch: list[tuple[torch.Tensor, torch.Tensor]]
# ) -> Mask2FormerBatch:
#     """Preprocesses the data so that it is compatible with the
#     expected Mask2Former input.

#     Parameters
#     ----------
#     image_processor : AutoImageProcessor
#         Mask2Former image processor
#     batch : list[tuple[torch.Tensor, torch.Tensor]]
#         A list of images and masks.

#     Returns
#     -------
#     Mask2FormerBatch
#         The expected input for the Mask2Former.
#     """
#     images = [img for img, _ in batch]
#     segmentation_maps = [mask for _, mask in batch]

#     return image_processor(
#         images=images,
#         segmentation_maps=segmentation_maps,
#         return_tensors="pt",
#         do_rescale=False,
#     )


# # Preporcessor
# mask2former_auto_image_processor = AutoImageProcessor.from_pretrained(
#     "facebook/mask2former-swin-large-ade-semantic",
#     ignore_index=BG_VALUE_255,
#     reduce_labels=False,
# )

# # Function to be used with dataloader collate_fn
# preprocess_mask2former = partial(
#     _preprocess_mask2former, mask2former_auto_image_processor
# )


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

    # With null classes
    # for i in range(len(pred_maps)):
    #     pred_maps[i][pred_maps[i] == 11] = 255

    # Get rid of warning in evaluate
    for i in range(len(pred_maps)):
        pred_maps[i] = pred_maps[i].to(torch.int32)

    # Convert CxHxW mask to HxW mask...
    masks = []

    # Fill the output mask with the category values
    for category_list, curr_mask in zip(batch["class_labels"], batch["mask_labels"]):
        output_mask = torch.full(
            (curr_mask.shape[1], curr_mask.shape[2]), BG_VALUE_0, dtype=torch.int32
        )

        for i, category in enumerate(category_list):
            output_mask[curr_mask[i] == 1] = int(category)  # wtf?

        masks.append(output_mask)

    return pred_maps, masks


def get_perpixel_features(
    pixel_decoder_hidden_states, pixel_decoder_last_hidden_state, avg=True
):
    """Upsample all pixel decoder features to the target (image) size and then
        take the average of them, return the average matrix

    Parameters
    ----------
    pixel_decoder_hidden_states : Tuple[torch.FloatTensor]
        Intermediate pixel decoder features of size [batch_size, feat_dim, height, width].
        Each feature has different height and width but the feat_dim=256
    pixel_decoder_last_hidden_state : torch.FloatTensor
        Last pixel decoder hidden state of size [batch_size, feat_dim, height, width]
    avg : Bool
        Default True. The per-pixel average of 4 pixel decoder matrices is returned. If it is set False,
        then a tensor of size [batch_size, feat_dim*4, height, width] is returned.

    Returns
    -------
    torch.FloatTensor
        Either the average of all the feature matrices from the pixel decoder is
        returned ([batch_size, feat_dim, height, width]) or a tensor
        which is the stack of 4 feature matrices is returned ([batch_size, feat_dim*4, height, width]).
    """
    upsampled_last_feature = torch.nn.functional.interpolate(
        pixel_decoder_last_hidden_state,
        size=TARGET_SIZE,
        mode="bilinear",
        align_corners=False,
    )
    if avg:
        features = upsampled_last_feature
        for feat in pixel_decoder_hidden_states:
            features = torch.add(
                features,
                torch.nn.functional.interpolate(
                    feat, size=TARGET_SIZE, mode="bilinear", align_corners=False
                ),
            )

        features /= len(pixel_decoder_hidden_states) + 1
    else:
        features = [upsampled_last_feature]
        for feat in pixel_decoder_hidden_states:
            features.append(
                torch.nn.functional.interpolate(
                    feat, size=TARGET_SIZE, mode="bilinear", align_corners=False
                )
            )
        features = torch.cat(features, dim=1).to(upsampled_last_feature.device)
    return features


def get_pred_logits(class_queries_logits, masks_queries_logits, resize=True):
    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = (
        masks_queries_logits.sigmoid()
    )  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation_pred = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    batch_size = class_queries_logits.shape[0]

    resized_logits = []
    if resize:
        for idx in range(batch_size):
            resized_logits.append(
                torch.nn.functional.interpolate(
                    segmentation_pred[idx].unsqueeze(dim=0),
                    size=TARGET_SIZE,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        resized_preds = torch.cat(resized_logits, dim=0)
        return resized_preds
    else:
        return segmentation_pred


def get_target_labels(
    class_labels, mask_labels, batch_size, num_classes, size_0=None, size_1=None
):
    if size_0 is not None:
        target_size_0 = size_0
    else:
        target_size_0 = TARGET_SIZE[0]

    if size_1 is not None:
        target_size_1 = size_1
    else:
        target_size_1 = TARGET_SIZE[1]

    target = torch.zeros((batch_size, num_classes, target_size_0, target_size_1))
    for b in range(batch_size):
        for i, label in enumerate(class_labels[b]):
            target[b, label, :, :] = mask_labels[b][i]
    return target


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


# def post_process_semantic_segmentation(
#     outputs, target_sizes: list[tuple[int, int]] | None = None
# ) -> "torch.Tensor":
#     """
#     Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
#     PyTorch.

#     Args:
#         outputs ([`Mask2FormerForUniversalSegmentation`]):
#             Raw outputs of the model.
#         target_sizes (`List[Tuple[int, int]]`, *optional*):
#             List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
#             final size (height, width) of each prediction. If left to None, predictions will not be resized.
#     Returns:
#         `List[torch.Tensor]`:
#             A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
#             corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
#             `torch.Tensor` correspond to a semantic class id.
#     """
#     class_queries_logits = (
#         outputs.class_queries_logits
#     )  # [batch_size, num_queries, num_classes+1]
#     masks_queries_logits = (
#         outputs.masks_queries_logits
#     )  # [batch_size, num_queries, height, width]

#     # # Scale back to preprocessed image size - (384, 384) for all models
#     masks_queries_logits = torch.nn.functional.interpolate(
#         masks_queries_logits, size=(270, 480), mode="bilinear", align_corners=False
#     )

#     # Remove the null class `[..., :-1]`
#     masks_classes = class_queries_logits.softmax(dim=-1)  # [..., :-1]
#     masks_probs = (
#         masks_queries_logits.sigmoid()
#     )  # [batch_size, num_queries, height, width]

#     # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
#     segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
#     batch_size = class_queries_logits.shape[0]

#     # Resize logits and compute semantic segmentation maps
#     if target_sizes is not None:
#         if batch_size != len(target_sizes):
#             raise ValueError(
#                 "Make sure that you pass in as many target sizes as the batch dimension of the logits"
#             )

#         semantic_segmentation = []
#         for idx in range(batch_size):
#             resized_logits = torch.nn.functional.interpolate(
#                 segmentation[idx].unsqueeze(dim=0),
#                 size=target_sizes[idx],
#                 mode="bilinear",
#                 align_corners=False,
#             )
#             semantic_map = resized_logits[0].argmax(dim=0)
#             semantic_segmentation.append(semantic_map)
#     else:
#         semantic_segmentation = segmentation.argmax(dim=1)
#         semantic_segmentation = [
#             semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])
#         ]

#     return semantic_segmentation
