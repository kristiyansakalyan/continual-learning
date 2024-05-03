import random

import numpy as np
import torch


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


MASK_IGNORE_VALUE = 255
