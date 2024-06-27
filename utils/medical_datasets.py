import json
import os
from typing import Literal, NamedTuple, TypeAlias

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Mask2FormerImageProcessor

# from utils.common import BG_VALUE
BG_VALUE_0 = 0
Split = Literal["train", "val", "test"]

CADIS_SPLIT_TO_FILE: dict[Split, str] = {
    "train": "training.json",
    "val": "validation.json",
    "test": "test.json",
}

CADIS_SPLIT_TO_FOLDER: dict[Split, str] = {
    "train": "training",
    "val": "validation",
    "test": "test",
}

CATARACT_101_ANNOTATION_FILE = "coco-annotations.json"

CATARACT_1K_FODLERS: dict[str, str] = {
    "annotations": "Annotations/Coco-Annotations",
    "images": "Annotations/Images-and-Supervisely-Annotations",
}

CATARACT_1K_CATEGORIES: dict[str, int] = {
    "Cornea": 1,
    "Katena Forceps": 2,
    "cornea1": 3,
    "Lens Injector": 4,
    "Irrigation-Aspiration": 5,
    "Capsulorhexis Forceps": 6,
    "Spatula": 7,
    "pupil1": 8,
    "Phacoemulsification Tip": 9,
    "Incision Knife": 10,
    "Pupil": 11,
    "Slit Knife": 12,
    "Lens": 13,
    "Capsulorhexis Cystotome": 14,
    "Gauge": 15,
}


class ImageInfo(NamedTuple):
    """A named tuple to store image data.

    Parameters
    ----------
    path : str
        The file path to the image.
    category_id : list[int]
        List of category IDs for each segment.
    segmentation : list[list[list[float]]]
        List of segmentation polygons.
    height : int
        The height of the image.
    width : int
        The width of the image
    """

    path: str
    category_id: list[int]
    segmentation: list[list[list[float]]]
    height: int
    width: int


ImagesDict: TypeAlias = dict[int, ImageInfo]


class BaseSegmentDataset(Dataset):
    """A dataset class for handling Cadis data in PyTorch.

    Attirbutes
    ----------
    root_folder : str
        The root directory where images and annotations are stored
    split : Split | None
        The dataset split to use ('train', 'val', 'test') if available,
        by default None
    imgs : CadisImages
        Dictionary of images with their IDs as keys.
    categories : dict[int, str]
        Dictionary mapping category IDs to category labels.
    categories_to_idx : dict[int, int] | dict[str, dict[int, int]]
        Dictionary mapping category IDs to tensor indices.
    image_transform : transforms.Compose | None, optional
        A list of transformations to be applied on the images, by default None
    mask_transform : transforms.Compose | None, optional
        A list of transformations to be applied on the masks, by default None
    class_mappings: dict[int, int] | None
        Dictionary that maps the classes of the dataset to some other ones
    """

    def __init__(
        self: "BaseSegmentDataset",
        root_folder: str,
        split: Split | None = None,
        image_transform: transforms.Compose | None = None,
        mask_transform: transforms.Compose | None = None,
        class_mappings: dict[int, int] | None = None,
        return_path: True | False = False,
    ):
        """The constructor for the CadisDataset.

        Parameters
        ----------
        root_folder : str
            The root directory where images and split files are stored
        split : Split, optional
            The dataset split to use ('train', 'val', 'test') if available,
            by default None
        image_transform : transforms.Compose | None, optional
            A list of transformations to be applied on the images, by default None
        mask_transform : transforms.Compose | None, optional
            A list of transformations to be applied on the masks, by default None
        """
        self.root_folder = root_folder
        self.split = split
        self.image_transform = image_transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((270, 480))]
        )
        self.mask_transform = mask_transform or transforms.Compose(
            [
                transforms.Resize(
                    (270, 480), interpolation=transforms.InterpolationMode.NEAREST
                )
            ]
        )
        self.class_mappings = class_mappings
        self.return_path = return_path
        # Correct root folder?
        if not os.path.exists(self.root_folder):
            raise FileNotFoundError(
                f"The specified root folder does not exist: {self.root_folder}"
            )

        self.imgs, self.categories_to_idx, self.categories = (
            self.load_images_and_categories()
        )

    def __len__(self: "BaseSegmentDataset") -> int:
        """Returns the number of images in the dataset."""
        return len(self.imgs)

    def __getitem__(
        self: "BaseSegmentDataset", idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves an image and its corresponding mask by index.

        Parameters
        ----------
        idx : int
            The index of the image in the dataset.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image and its corresponding mask, both as torch.Tensor.
        """
        image_id = list(self.imgs.keys())[idx]
        image_info = self.imgs[image_id]

        image = Image.open(image_info.path).convert("RGB")
        image = self.image_transform(image)

        mask = self.create_mask(image_info)
        mask = torch.from_numpy(mask.astype(np.int32))

        if self.mask_transform is not None:
            if mask.ndim == 2:
                mask = mask[None, ...]
            mask = self.mask_transform(mask)

        return image, mask

    def load_images_and_categories(
        self: "BaseSegmentDataset",
    ) -> tuple[ImagesDict, dict[int, int] | dict[str, dict[int, int]], dict[int, str]]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def create_mask(
        self: "BaseSegmentDataset", image_info: ImageInfo
    ) -> NDArray[np.int32]:
        """Generates a segmentation mask for the given polygons and category indices.

        This method converts segmentation polygons into a class-index mask where each
        pixel's value corresponds to the class index of the polygon that covers it.

        Parameters
        ----------
        image_info: ImageInfo
            A dictionary containing the segmentation polygons, the associated categories,
            the heigth and the width of the image.

        Returns
        -------
        NDArray[bool]
            A single-layered 2D array with the same height and width as the images in
            the dataset, where each pixel's value is the class index.


        Raises
        ------
        ValueError
            If any polygon has an odd number of coordinates, indicating incomplete pairs
            of x and y coordinates, or if the coordinates are not in the expected format.
        """
        polygons = image_info.segmentation
        categories = image_info.category_id
        H = image_info.height
        W = image_info.width

        mask = np.full(
            (H, W), BG_VALUE_0, dtype=np.int32
        )  # Only one channel needed, with default class index 0 (not 255 anymore!) (background)

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

                # If class mappings are provided, use them
                # otherwise simply use the class.
                color = self.categories_to_idx[category]
                if self.class_mappings is not None:
                    color = self.class_mappings.get(color, -1)

                # In domain incremental scenario, we do not consider all the
                # categories => skip this class if it's not part of our new classes.
                if color == -1:
                    continue

                # Fill the polygon with the class index
                cv2.fillPoly(
                    mask,
                    [np.array(polygon, dtype=np.int32)],
                    color=color,
                )

        return mask

    def extract_data_from_annot_file(
        self: "BaseSegmentDataset", data: dict, additional_path: str = None
    ) -> tuple[ImagesDict, dict[int, int], dict[int, str]]:
        """Extracts the annotation data from a given file, which
        is in COCO format.

        Parameters
        ----------
        data : dict
            The annotation file in COCO-Format
        additional_path : str, optional
            Additional path to be joined if needed,
            by default None

        Returns
        -------
        tuple[ImagesDict, dict[int, int], dict[int, str]]
            The images, mapping between categories and indicies and
            mapping between categories and the labels.
        """
        path = [self.root_folder]
        if additional_path is not None:
            path.append(additional_path)

        images: ImagesDict = {
            img["id"]: ImageInfo(
                path=os.path.join(*path, img["file_name"]),
                category_id=[],
                segmentation=[],
                height=img["height"],
                width=img["width"],
            )
            for img in data["images"]
        }

        for annotation in data["annotations"]:
            img = images[annotation["image_id"]]
            img.category_id.append(annotation["category_id"])
            img.segmentation.append(annotation["segmentation"])

        categories_to_idx = {
            cat["id"]: idx + 1 for idx, cat in enumerate(data["categories"])
        }

        categories = {idx: cat["name"] for idx, cat in enumerate(data["categories"])}
        # categories[255] = "Background"
        categories[0] = "Background"
        return images, categories_to_idx, categories


class Cataract101Dataset(BaseSegmentDataset):
    """A dataset class for handling the Cataract-101 dataset

    Splits
    ------
    The Cataract-101 dataset does not provide pre-defined splits. You can use
    PyTorch utilities to create a split. Here is an example:

        >>> from torch.utils.data import random_split
        >>> total_size = len(cadis_dataset)
        >>> # You can adapt the sizes
        >>> train_size = int(0.7 * total_size)
        >>> val_size = int(0.15 * total_size)
        >>> test_size = total_size - train_size - val_size
        >>> train_dataset, val_dataset, test_dataset = random_split(cadis_dataset, [train_size, val_size, test_size])

    """

    def load_images_and_categories(
        self: "Cataract101Dataset",
    ) -> tuple[ImagesDict, dict[int, int] | dict[str, dict[int, int]], dict[int, str]]:
        """Loads the images and categories for the Cataract-101 dataset.

        Returns
        -------
        tuple[ImagesDict, dict[int, int] | dict[str, dict[int, int]], dict[int, str]]
            The images, mapping between categories and indicies and
            mapping between categories and the labels.

        Raises
        ------
        FileNotFoundError
            If the specified file was not found
        """
        annotation_file = os.path.join(self.root_folder, CATARACT_101_ANNOTATION_FILE)

        # Check if the annotation file exists
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(
                f"The specified split file does not exist: {annotation_file}"
            )

        with open(annotation_file, "r") as file:
            data = json.load(file)

        return self.extract_data_from_annot_file(data)


class CadisDataset(BaseSegmentDataset):
    """A dataset class for handling the Cadis dataset"""

    def load_images_and_categories(
        self: "CadisDataset",
    ) -> tuple[ImagesDict, dict[int, int] | dict[str, dict[int, int]], dict[int, str]]:
        """Loads the images and categories for the Cadis dataset.

        Returns
        -------
        tuple[ImagesDict, dict[int, int] | dict[str, dict[int, int]], dict[int, str]]
            The images, mapping between categories and indicies and
            mapping between categories and the labels.

        Raises
        ------
        FileNotFoundError
            If the specified file was not found
        """
        # Valid split?
        if self.split not in CADIS_SPLIT_TO_FILE.keys():
            raise ValueError(
                "Invalid split specified. Expected one of: 'train', 'val', 'test'"
            )

        split_annot_file = os.path.join(
            self.root_folder, CADIS_SPLIT_TO_FILE[self.split]
        )

        # Check if the annotation file exists
        if not os.path.exists(split_annot_file):
            raise FileNotFoundError(
                f"The specified split file does not exist: {split_annot_file}"
            )

        # Load annotations from the file
        with open(split_annot_file, "r") as file:
            data = json.load(file)

        return self.extract_data_from_annot_file(
            data, additional_path=CADIS_SPLIT_TO_FOLDER[self.split]
        )


class Cataract1KDataset(BaseSegmentDataset):
    """A dataset class for handling the Cataract-1K dataset"""

    def load_images_and_categories(
        self: "CadisDataset",
    ) -> tuple[ImagesDict, dict[int, int] | dict[str, dict[int, int]], dict[int, str]]:
        """Loads the images and categories for the Cadis dataset.

        Returns
        -------
        tuple[ImagesDict, dict[int, int] | dict[str, dict[int, int]], dict[int, str]]
            The images, mapping between categories and indicies and
            mapping between categories and the labels.

        Raises
        ------
        FileNotFoundError
            If the specified file was not found
        """
        annotations_dir = os.path.join(
            self.root_folder, CATARACT_1K_FODLERS["annotations"]
        )
        cases = [
            case for case in os.listdir(annotations_dir) if case.startswith("case_")
        ]

        images: ImagesDict = {}

        case_cat_to_idx: dict[str, dict[int, int]] = {}

        for case in cases:
            annotation_file_path = os.path.join(
                annotations_dir, case, "annotations/instances.json"
            )
            # Check if the annotation file exists:
            if not os.path.exists(annotation_file_path):
                raise FileNotFoundError(
                    f"The annotation file does not exist: {annotation_file_path}"
                )

            # Load file
            with open(annotation_file_path, "r") as file:
                data = json.load(file)

            # Images path:
            imgs_path = os.path.join(CATARACT_1K_FODLERS["images"], case, "img")

            imgs, _, _ = self.extract_data_from_annot_file(
                data, additional_path=imgs_path
            )

            # Extract the case specific mapping to our categories
            categories = data["categories"]
            cat_to_idx_curr = {
                entry["id"]: CATARACT_1K_CATEGORIES[entry["name"]]
                for entry in categories
            }
            case_cat_to_idx[case.replace("_", "")] = cat_to_idx_curr

            images.update(imgs)

        categories = {
            v: k
            for k, v in zip(
                CATARACT_1K_CATEGORIES.keys(), CATARACT_1K_CATEGORIES.values()
            )
        }
        # categories[255] = "Background"
        categories[0] = "Background"
        return images, case_cat_to_idx, categories

    def __getitem__(
        self: "Cataract1KDataset", idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Overrides the `BaseSegmentDataset` getitem method as it needs to add
        case-specific information.

        Parameters
        ----------
        idx : int
            Index

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Image and mask.
        """
        image_id = list(self.imgs.keys())[idx]
        image_info = self.imgs[image_id]
        case = image_info.path.split("/")[-3].replace("_", "")

        image = Image.open(image_info.path).convert("RGB")
        image = self.image_transform(image)

        mask = self.create_mask(image_info, case)
        mask = torch.from_numpy(mask.astype(np.int32))

        if self.mask_transform is not None:
            if mask.ndim == 2:
                mask = mask[None, ...]
            mask = self.mask_transform(mask).squeeze()
        if self.return_path:
            return image, mask, image_info.path
        else:
            return image, mask

    def create_mask(
        self: "BaseSegmentDataset", image_info: ImageInfo, case: str
    ) -> NDArray[np.int32]:
        """Generates a segmentation mask for the given polygons and category indices.

        This method converts segmentation polygons into a class-index mask where each
        pixel's value corresponds to the class index of the polygon that covers it.

        Parameters
        ----------
        image_info: ImageInfo
            A dictionary containing the segmentation polygons, the associated categories,
            the heigth and the width of the image.
        case: str
            The specific case of the dataset. Categories mapping varry between the
            different cases.

        Returns
        -------
        NDArray[bool]
            A single-layered 2D array with the same height and width as the images in
            the dataset, where each pixel's value is the class index.


        Raises
        ------
        ValueError
            If any polygon has an odd number of coordinates, indicating incomplete pairs
            of x and y coordinates, or if the coordinates are not in the expected format.
        """
        polygons = image_info.segmentation
        categories = image_info.category_id
        H = image_info.height
        W = image_info.width

        mask = np.full(
            (H, W), BG_VALUE_0, dtype=np.int32
        )  # Only one channel needed, with default class index 0 (not 255 anymore!) (background)

        # Define a custom sorting function where Polygons are filled first for
        # cornea () and then p
        def custom_sort(item):
            """
            Define a custom sorting function to make sure Polygons are filled first for
            cornea (id=4) and then pupil (id=3)
            """
            category = item[1]
            # Get correct category first...
            category = self.categories_to_idx[case][category]
            if category == 1 or category == 3:
                return 0  # Iris first
            elif category == 8 or category == 11:
                return 1  # Pupil second
            else:
                # All the rest should be on top
                return category

        sorted_data = sorted(zip(polygons, categories), key=custom_sort)

        for polygon_list, category in sorted_data:
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

                # If class mappings are provided, use them
                # otherwise simply use the class.
                color = self.categories_to_idx[case][category]

                if self.class_mappings is not None:

                    color = self.class_mappings.get(color, -1)

                # In domain incremental scenario, we do not consider all the
                # categories => skip this class if it's not part of our new classes.
                if color == -1:
                    continue

                # Fill the polygon with the class index
                cv2.fillPoly(
                    mask,
                    [np.array(polygon, dtype=np.int32)],
                    color=color,  # self.categories_to_idx[case][category],
                )

        return mask


def parse_video_splits(content: list[str]) -> dict[str, list[str]]:
    training = []
    validation = []
    test = []
    current_list = None

    for line in content:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# Training"):
            current_list = training
        elif line.startswith("# Validation"):
            current_list = validation
        elif line.startswith("# Test"):
            current_list = test
        elif line.startswith("Video"):
            if current_list is not None:
                current_list.append(line)

    return {"train": training, "val": validation, "test": test}


class CadisV2Image(NamedTuple):
    img_path: str
    mask_path: str


class CadisV2Dataset(Dataset):

    def __init__(
        self: "CadisV2Dataset",
        root_folder: str,
        split: Split = "train",
        image_transform: transforms.Compose | None = None,
        mask_transform: transforms.Compose | None = None,
        class_mappings: dict[int, int] | None = None,
        return_path: True | False = False,
    ) -> None:
        self.root_folder = root_folder
        self.split = split
        self.image_transform = image_transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((270, 480))]
        )
        self.mask_transform = mask_transform or transforms.Compose(
            [
                transforms.Resize(
                    (270, 480), interpolation=transforms.InterpolationMode.NEAREST
                )
            ]
        )
        self.class_mappings = class_mappings
        self.return_path = return_path
        # Correct root folder?
        if not os.path.exists(self.root_folder):
            raise FileNotFoundError(
                f"The specified root folder does not exist: {self.root_folder}"
            )

        split_path = os.path.join(self.root_folder, "splits.txt")
        # Split file available?
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"The splits file does not exist: {split_path}")

        with open(split_path, "r") as file:
            data = file.readlines()

        splits = parse_video_splits(data)

        if self.split not in splits:
            raise ValueError(
                f'The provided split is incorrect: {split}; Please choose one of ("train", "val", "test")'
            )

        self.videos = splits[self.split]

        # Extract the path to each image and each label mask
        self.data: list[CadisV2Image] = []

        for video in self.videos:
            imgs_path = os.path.join(self.root_folder, video, "Images")
            labels_path = os.path.join(self.root_folder, video, "Labels")

            images_filenames = os.listdir(imgs_path)
            labels_filenames = os.listdir(labels_path)

            self.data += [
                CadisV2Image(
                    os.path.join(imgs_path, img_filename),
                    os.path.join(labels_path, label_filename),
                )
                for img_filename, label_filename in zip(
                    images_filenames, labels_filenames
                )
            ]

        classes_path = os.path.join(self.root_folder, "classes.csv")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"The classes file does not exist: {classes_path}")

        self.categories = (
            pd.read_csv(classes_path).set_index("Index")["Class"].to_dict()
        )

        # self.categories[255] = "Background"
        self.categories[0] = "Background"

    def __len__(self: "BaseSegmentDataset") -> int:
        """Returns the number of images in the dataset."""
        return len(self.data)

    def __getitem__(
        self: "BaseSegmentDataset", idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves an image and its corresponding mask by index.

        Parameters
        ----------
        idx : int
            The index of the image in the dataset.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image and its corresponding mask, both as torch.Tensor.
        """
        image_info = self.data[idx]

        image = cv2.imread(image_info.img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.image_transform(image)

        mask = cv2.imread(image_info.mask_path, cv2.COLOR_BGR2GRAY)
        mask = (F.to_tensor(mask) * 255).to(torch.int32).squeeze(0)
        if self.class_mappings is not None:
            mapped_mask = torch.full_like(mask, BG_VALUE_0)
            for zeiss_id, values in self.class_mappings.items():
                for cadis_id in values:
                    mapped_mask[mask == cadis_id] = zeiss_id

            mask = mapped_mask

        if self.mask_transform is not None:
            if mask.ndim == 2:
                mask = mask[None, ...]

            mask = self.mask_transform(mask).squeeze()
        if self.return_path:
            return image, mask, image_info.img_path
        else:
            return image, mask


class Mask2FormerDataset(Dataset):

    def __init__(
        self,
        dataset: BaseSegmentDataset | CadisV2Dataset,
        processor: Mask2FormerImageProcessor,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        
        if self.transform is not None:
            img=self.transform(img)
            
        batch = self.processor(
            images=[img],
            segmentation_maps=[mask],
            return_tensors="pt",
            do_rescale=False,
        )

        return batch
