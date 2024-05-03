import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (128, 0, 0),  # Maroon
    (128, 128, 0),  # Olive
    (0, 128, 0),  # Green
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (0, 0, 128),  # Navy
    (139, 69, 19),  # Saddle Brown
    (255, 69, 0),  # Red-Orange
    (0, 100, 0),  # Dark Green
    (139, 0, 139),  # Dark Magenta
    (0, 255, 127),  # Spring Green
    (70, 130, 180),  # Steel Blue
    (220, 20, 60),  # Crimson
    (0, 139, 139),  # Dark Cyan
]

# Reuse colors from CaDISV2
NEW_COLORS = [
    [0, 137, 255],
    [255, 165, 0],
    [255, 156, 201],
    [99, 0, 255],
    [255, 0, 0],
    [255, 0, 165],
    [255, 255, 255],
    [141, 141, 141],
    [255, 218, 0],
    [173, 156, 255],
    [73, 73, 73],
    [250, 213, 255],
    [255, 156, 156],
    [99, 255, 0],
    [157, 225, 255],
    [255, 89, 124],
    [173, 255, 156],
    [255, 60, 0],
    [40, 0, 255],
    [170, 124, 0],
    [188, 255, 0],
    [0, 207, 255],
    [0, 255, 207],
    [188, 0, 255],
    [243, 0, 255],
    [0, 203, 108],
    [252, 255, 0],
    [93, 182, 177],
    [0, 81, 203],
    [211, 183, 120],
    [231, 203, 0],
    [0, 124, 255],
    [10, 91, 44],
    [2, 0, 60],
    [0, 144, 2],
    [133, 59, 59],
]


def overlay_mask_on_image(
    image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.8
) -> None:
    """
    Overlay a multiclass mask on an image using class indices.

    Parameters
    ----------
    image : torch.Tensor
        The input image. Expected shape - (3, H, W)
    mask : torch.Tensor
        The segmentation mask with class indices.
    alpha: : float
        Transparency factory, by default 0.8
    Raises
    ------
    ValueError
        If the input image or mask is not in the correct shape.
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError("Input tensor must have 3 dimensions and 3 channels.")

    if mask.dim() != 2 or mask.shape != image.shape[1:]:
        raise ValueError(
            "Mask dimensions must be 2 and must match the height and width of the image"
        )

    # Convert to numpy
    image = image.permute(1, 2, 0).numpy()
    mask = mask.numpy()

    # Initialize the overlay image
    overlay = np.zeros_like(image)

    # Map each class index in the mask to its corresponding color
    for class_index in np.unique(mask):
        # Skip background
        if class_index == 255:
            continue

        if class_index < len(NEW_COLORS):
            class_mask = mask == class_index
            overlay[class_mask] = NEW_COLORS[class_index % len(COLORS)]

    overlay /= 255.0

    # Blend original and overlay; This results in a little warning regarding
    # the colors but I think that it makes the segmentation easier to see.
    # If you change 1 to 1 - alpha instead, it will resolve the warning
    display_image = cv2.addWeighted(image, 1, overlay, alpha, 0)

    # Display the result using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(display_image)
    plt.axis("off")
    plt.show()


def visualize_tensor_image(image: torch.Tensor) -> None:
    """
    Visualizes a PyTorch tensor image with shape (C, H, W).

    Parameters
    ----------
    image_tensor : torch.Tensor
        The input image tensor, expected shape is (C, H, W).

    Raises
    ------
    ValueError
        If the input image is not in the correct shape.
    """
    if image.dim() != 3 or image.shape[0] not in [1, 3]:
        raise ValueError(
            "Input tensor must have 3 dimensions (C, H, W) and 1 or 3 channels."
        )

    # Check if it's a grayscale image (1 channel)
    if image.shape[0] == 1:
        # Remove the channel dimension since plt.imshow expects (H, W) for grayscale images
        image_np = image.squeeze().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np, cmap="gray")
    else:
        # Convert from (C, H, W) to (H, W, C)
        image_np = image.permute(1, 2, 0).numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)

    plt.axis("off")
    plt.show()
