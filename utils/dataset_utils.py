from torch.utils.data import random_split
from torchvision import transforms

from utils.medical_datasets import CadisDataset, Cataract1K

"""
DONE: 
  Domain incremental
    - Slit/IncisionKnife →  Primary knife, Secondary knife → C1: Knife
    - Katena Forceps → Bonn forceps → C2: Bonn forceps
    - Gauge → Hydro. cannula, Rycroft cannula, Visco. cannula, Charleux cannula → C3: Cannula
    - Capsulorhexis Cystotome → Cap. cystotome → C4: Capsulorhexis Cystotome
    - Capsulorhexis forceps → Cap. forceps → C5: Capsulorhexis forceps
    - Phacoemulsifier Tip → Phaco. handpiece → C6: Phacoemulsification handpiece
    - Spatula → Micromanipulator → C7: Micromanipulator
    - Irrigation-Aspiration →  A/I handpiece → C8: I/A handpiece
    - Lens Injector → Lens injector → C9: Lens injector
    - Pupil → Pupil → C10: Pupil
    - Iris → Cornea → C11: iris

TODO: 
  Class incremental
    - Intraocular Lens→ not available → E1: Intraocular Lens
    - not available → Viter. handpiece → E2: Viter. handpiece
    - not available → Suture needle → E3: Suture needle
"""

CADIS_CATEGORIES = {
    1: "Eye Retractors",
    2: "Hydro. Cannula",
    3: "Visco. Cannula",
    4: "Cap. Cystotome",
    5: "Rycroft Cannula",
    6: "Bonn Forceps",
    7: "Primary Knife",
    8: "Phaco. Handpiece",
    9: "Lens Injector",
    10: "A/I Handpiece",
    11: "Secondary Knife",
    12: "Micromanipulator",
    13: "Cap. Forceps",
    14: "Water Sprayer",
    15: "Suture Needle",
    16: "Needle Holder",
    17: "Charleux Cannula",
    18: "Vannas Scissors",
    19: "Viter. Handpiece",
    20: "Mendez Ring",
    21: "Biomarker",
    22: "Marker",
}

ZEISS_TO_CADIS = {
    1: [7, 11],
    2: [6],
    3: [2, 3, 5, 17],
    4: [4],
    5: [13],
    6: [8],
    7: [12],
    8: [10],
    9: [9],
    # Pupil missing
    # Iris missing
}

ZEISS_TO_CADIS_CONT = {
    # We add those !?
    13: [19],
    14: [15],
}

CATARACT1K_CATEGORIES = {
    1: "Cornea",
    2: "Katena Forceps",
    3: "cornea1",
    4: "Lens Injector",
    5: "Irrigation-Aspiration",
    6: "Capsulorhexis Forceps",
    7: "Spatula",
    8: "pupil1",
    9: "Phacoemulsification Tip",
    10: "Incision Knife",
    11: "Pupil",
    12: "Slit Knife",
    13: "Lens",
    14: "Capsulorhexis Cystotome",
    15: "Gauge",
}

ZEISS_TO_CATARACT1K = {
    1: [10, 12],
    2: [2],
    3: [15],
    4: [14],
    5: [6],
    6: [9],
    7: [7],
    8: [5],
    9: [4],
    10: [8, 11],
    11: [1, 3],
}

ZEISS_CATEGORIES = {
    # Default
    1: "Knife",
    2: "Bonn forceps",
    3: "Cannula",
    4: "Capsulorhexis Cystotome",
    5: "Capsulorhexis forceps",
    6: "Phacoemulsification handpiece",
    7: "Micromanipulator",
    8: "I/A handpiece",
    9: "Lens injector",
    10: "Pupil",
    11: "Iris",
    # Class incremental
    12: "Intraocular Lens",
    13: "Viter. handpiece",
    14: "Suture needle",
}


def get_cadis_dataset(
    root_folder: str,
    transform: transforms.Compose | None = None,
    domain_incremental: bool = False,
    class_incremental: bool = False,
) -> list[CadisDataset]:
    """Creates a list of CadisDataset objects for train,
    validation, and test splits from a specified root folder.

    Parameters
    ----------
    root_folder : str
        The directory path where the dataset files are stored.
    transform : transforms.Compose | None, optional
        A list of transformations to be applied on the images, by default None
    domain_incremental : bool, optional
        A flag to determine whether to setup the dataset for domain incremental learning,
        by default False.
    class_incremental : bool, optional
        A flag to indicate whether to configure the dataset for class incremental learning.
        Currently, this feature is not implemented and will raise NotImplementedError
        if set to True, by default False.

    Returns
    -------
    list[CadisDataset]
        A list of CadisDataset instances for each data split ('train', 'val', 'test').
        Each dataset is configured according to the specified parameters.

    Raises
    ------
    NotImplementedError
        Raised if `class_incremental` is True, as class incremental learning setup is not yet implemented.
    """
    class_mappings = None
    if domain_incremental:
        class_mappings = {
            cadis_cat: zeiss_cat
            for zeiss_cat, cadis_list in ZEISS_TO_CADIS.items()
            for cadis_cat in cadis_list
        }
    if class_incremental:
        raise NotImplementedError("Class incremental is not implemented yet.")

    datasets = [
        CadisDataset(
            root_folder=root_folder,
            split=split,
            transform=transform,
            class_mappings=class_mappings,
        )
        for split in ["train", "val", "test"]
    ]
    return datasets


def get_cataract1k_dataset(
    root_folder: str,
    split_ratios: tuple[float, float] = [0.8, 0.1],
    transform: transforms.Compose | None = None,
    domain_incremental: bool = False,
    class_incremental: bool = False,
) -> list[Cataract1K]:
    """Creates a list of CadisDataset objects for train,
    validation, and test splits from a specified root folder
    and split ratios.

    Parameters
    ----------
    root_folder : str
        The directory path where the dataset files are stored.
    split_ratios : tuple[float, float], optional
        The split ratios to be used for the splits ('train', 'val').
        The ratio for test will be infered from the rest, by default [0.8, 0.1]
    domain_incremental : bool, optional
        A flag to determine whether to setup the dataset for domain incremental learning,
        by default False.
    class_incremental : bool, optional
        A flag to indicate whether to configure the dataset for class incremental learning.
        Currently, this feature is not implemented and will raise NotImplementedError
        if set to True, by default False.

    Returns
    -------
    list[CadisDataset]
        A list of CadisDataset instances for each data split ('train', 'val', 'test').
        Each dataset is configured according to the specified parameters.

    Raises
    ------
    NotImplementedError
        Raised if `class_incremental` is True, as class incremental learning setup is not yet implemented.
    """
    class_mappings = None
    if domain_incremental:
        class_mappings = {
            cataract_cat: zeiss_cat
            for zeiss_cat, cataract_list in ZEISS_TO_CATARACT1K.items()
            for cataract_cat in cataract_list
        }
    if class_incremental:
        raise NotImplementedError("Class incremental is not implemented yet.")

    dataset = Cataract1K(
        root_folder=root_folder,
        transform=transform,
        class_mappings=class_mappings,
    )

    train_ratio, val_ratio = split_ratios
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - val_size - train_size

    return random_split(dataset, [train_size, val_size, test_size])
