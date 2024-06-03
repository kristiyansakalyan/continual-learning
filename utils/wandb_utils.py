import torch
from torchvision.transforms.functional import to_pil_image

import wandb
from utils.dataset_utils import ZEISS_CATEGORIES

ZEISS_WITH_BG = ZEISS_CATEGORIES
BG = 0
ZEISS_WITH_BG[BG] = "Background"


def log_table_of_images(
    table: wandb.Table,  # Should have ID and Image columns
    pixel_values: torch.Tensor,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    pred_maps: list[torch.Tensor],
    masks: list[torch.Tensor],
    batch_index: int,
) -> None:
    batch_size = pixel_values.shape[0]
    for INDEX in range(batch_size):
        img = (
            (
                (pixel_values[INDEX, ...].cpu() * pixel_std.reshape(3, 1, 1))
                + pixel_mean.reshape(3, 1, 1)
            )
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        pred_instance = pred_maps[INDEX].cpu().numpy()
        gt_instance = masks[INDEX].cpu().numpy()

        mask_img = wandb.Image(
            to_pil_image(img),
            masks={
                "predictions": {
                    "mask_data": pred_instance,
                    "class_labels": ZEISS_WITH_BG,
                },
                "ground_truth": {
                    "mask_data": gt_instance,
                    "class_labels": ZEISS_WITH_BG,
                },
            },
        )

        img_index = (batch_index * batch_size) + INDEX

        table.add_data(img_index, mask_img)
