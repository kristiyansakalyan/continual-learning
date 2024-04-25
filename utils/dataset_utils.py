import json
import os
from typing import Literal, NamedTuple, TypeAlias

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

Split = Literal["train", "val", "test"]


SPLIT_TO_FILE: dict[Split, str] = {
    "train": "training.json",
    "val": "validation.json",
    "test": "test.json",
}

SPLIT_TO_FOLDER: dict[Split, str] = {
    "train": "training",
    "val": "validation",
    "test": "test",
}


class CadisImageDict(NamedTuple):
    """A named tuple to store image data.

    Parameters
    ----------
    path : str
        The file path to the image.
    category_id : list[int]
        List of category IDs for each segment.
    segmentation : list[list[list[float]]]
        List of segmentation polygons.
    """

    path: str
    category_id: list[int]
    segmentation: list[list[list[float]]]


CadisImages: TypeAlias = dict[int, CadisImageDict]


class CadisDataset(Dataset):
    """A dataset class for handling Cadis data in PyTorch.

    Attirbutes
    ----------
    root_folder : str
        The root directory where images and annotations are stored
    split : Split
        The dataset split to use ('train', 'val', 'test').
    img_shape : tuple[int, int, int]
        The shape of the images (height, width, channels).
    imgs : CadisImages
        Dictionary of images with their IDs as keys.
    categories_to_idx : dict[int, int]
        Dictionary mapping category IDs to tensor indices.
    transform : None | transforms.Compose
        A composition of transformations to apply to the images.
    mask_size : tuple[int, int, int]
        The expected size of the output masks.
    """

    def __init__(
        self,
        root_folder: str,
        split: Split = "train",
        img_shape: tuple[int, int, int] = (540, 960, 3),
        transform: None | transforms.Compose = None,
    ):
        """The constructor for the CadisDataset.

        Parameters
        ----------
        root_folder : str
            The root directory where images and split files are stored
        split : Split, optional
            The dataset split to use ('train', 'val', 'test'), by default "train"
        img_shape : tuple[int, int, int], optional
            The shape of the images (height, width, channels), by default (540, 960, 3)
        transform : None | transforms.Compose, optional
            A composition of transformations to apply to the images, by default None
        """
        # TODO: Add default transform
        self.root_folder = root_folder
        self.split = split
        self.img_shape = img_shape

        self.imgs, self.categories_to_idx = self._get_imgs_and_categories()
        self.transform = transform
        self.mask_size = (*self.img_shape[:2], len(self.categories_to_idx))

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[Image.Image, NDArray[bool]]:
        """Retrieves an image and its corresponding mask by index.

        Parameters
        ----------
        idx : int
            The index of the image in the dataset.

        Returns
        -------
        tuple[Image.Image, NDArray[bool]]
            A tuple containing the image and its corresponding mask.
        """
        image_id = list(self.imgs.keys())[idx]
        image_info = self.imgs[image_id]

        # Load image
        img_path = image_info["path"]
        image = Image.open(img_path).convert("RGB")

        # Create mask and convert it to Tensor
        mask = self._create_mask(image_info["segmentation"], image_info["category_id"])
        mask = torch.from_numpy(mask)

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, mask

    def _get_imgs_and_categories(self) -> tuple[CadisImages, dict[int, int]]:
        """Loads image paths and categories from a JSON file based on
        the specified dataset split.

        Returns
        -------
        tuple[CadisImages, dict[int, int]]
            A tuple containing a dictionary of image data and a
            dictionary mapping category IDs to indices.

        Raises
        ------
        FileNotFoundError
            If the specified file was not found
        ValueError
            If there is an error in the provided split
        """
        if not os.path.exists(self.root_folder):
            raise FileNotFoundError(
                f"The specified root folder does not exist: {self.root_folder}"
            )

        # Determine the correct annotation file based on the split
        if self.split not in SPLIT_TO_FILE:
            raise ValueError(
                "Invalid split specified. Expected one of: 'train', 'val', 'test'"
            )

        split_file = os.path.join(self.root_folder, SPLIT_TO_FILE[self.split])

        # Check if the annotation file exists
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"The specified split file does not exist: {split_file}"
            )

        # Load annotations from the file
        with open(split_file, "r") as file:
            split_file = json.load(file)

        # Create dictionary
        images: CadisImages = {
            img["id"]: {
                "path": os.path.join(
                    self.root_folder, SPLIT_TO_FOLDER[self.split], img["file_name"]
                ),
                "category_id": [],
                "segmentation": [],
            }
            for img in split_file["images"]
        }

        # Add segmentations
        for annotation in split_file["annotations"]:
            img_id = annotation["image_id"]
            category_id = annotation["category_id"]
            segmentation = annotation["segmentation"]

            images[img_id]["segmentation"].append(segmentation)
            images[img_id]["category_id"].append(category_id)

        # Extract categories
        categories_to_idx = {
            item["id"]: idx for idx, item in enumerate(split_file["categories"])
        }

        return images, categories_to_idx

    def _create_mask(
        self, polygons: list[list[list[float]]], categories: list[int]
    ) -> NDArray[bool]:
        """Generates a segmentation mask for the given polygons and category indices.

        This method converts segmentation polygons into a binary mask where each category
        is represented in a separate channel of the mask. Each polygon is filled into
        the corresponding category layer of the mask.

        Parameters
        ----------
        polygons : list[list[list[float]]]
            A list of polygons where each polygon is represented as a list of lists.
            Each sublist contains the coordinates of the polygon's vertices.
        categories : list[int]
            A list of category indices corresponding to each polygon. These indices
            determine which layer of the mask each polygon is drawn into.

        Returns
        -------
        NDArray[bool]
            A multi-layered binary mask with the same height and width as the images in
            the dataset and depth equal to the number of categories. Each layer of the mask
            corresponds to a different category, with polygons filled in as True (1).

        Raises
        ------
        ValueError
            If any polygon has an odd number of coordinates, indicating incomplete pairs
            of x and y coordinates, or if the coordinates are not in the expected format.
        """
        H, W, _ = self.mask_size
        mask = np.zeros(self.mask_size, dtype=bool)

        for polygon_list, category in zip(polygons, categories):
            # One category might have multiple polygons
            for flat_coords in polygon_list:
                # Check if the number of coordinates is even
                if len(flat_coords) % 2 != 0:
                    raise ValueError(
                        "The number of coordinates should be even (pairs of x and y)."
                    )

                # Convert flat list to list of (x, y) tuples
                polygon = [
                    (flat_coords[i], flat_coords[i + 1])
                    for i in range(0, len(flat_coords), 2)
                ]

                # Ensure the polygon is closed (first point equals last point)
                if polygon[0] != polygon[-1]:
                    polygon.append(
                        polygon[0]
                    )  # Close the polygon by adding the first point at the end

                # Create a temporary mask for the current category
                temp_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(
                    temp_mask, [np.array(polygon, dtype=np.int32)], 1
                )  # Fill the polygon with '1'

                # Place the temporary mask in the correct position in the full mask array
                mask_idx = self.categories_to_idx[category]
                mask[:, :, mask_idx] |= temp_mask.astype(bool)

        return mask
