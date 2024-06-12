import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


class GaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0)):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, tensor):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return TF.gaussian_blur(tensor, self.kernel_size, [sigma])


class RandomChannelRearrangement(torch.nn.Module):
    def __init__(self):
        super(RandomChannelRearrangement, self).__init__()

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input image must be a torch.Tensor, but got {type(img)}")
        channels = img.split(1, dim=0)  # Split channels
        random.shuffle(channels)  # Shuffle channels
        img = torch.cat(channels, dim=0)  # Concatenate shuffled channels
        return img


# ================== Common Augmentations ================== #
"""
TODO: 
We should consider using
    - transforms.AutoAgument
    - transforms.RandAugment
    - transforms.AugMix
    - transforms.TrivialAgumentWide

Furthermore, in the paper "Principles of Forgetting in Domain-Incremental 
Semantic Segmentation in Adverse Weather Conditions", they used:
    
    - AutoAlbument (https://github.com/albumentations-team/autoalbument/tree/master)
    - Color Jittering with randomly rearranging input image channels (they call it distortion)
    - Gaussian Blurring !OR! Gaussian Noise.

"""
train_transforms_noise = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomApply([GaussianNoise(mean=0.0, std=0.1)], p=0.5),
        RandomChannelRearrangement(),
    ]
)
train_transforms_noise_no_distortion = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomApply([GaussianNoise(mean=0.0, std=0.1)], p=0.5),
    ]
)
train_transforms_blur = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 5.0))], p=0.5),
        RandomChannelRearrangement(),
    ]
)

train_transforms_blur_no_distortion = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 5.0))], p=0.5),
    ]
)
train_transforms_color_jitter = transforms.Compose(
    [
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                )
            ],
            p=0.5,
        )
    ]
)
# ========================================================== #
