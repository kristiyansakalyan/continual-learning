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
    
class ColorAugSSDTransform(torch.nn.Module):
    def __init__(
        self,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=0.1,  # hue_delta is normalized to [0, 0.5] in torchvision
    ):
        super().__init__()
        self.brightness_delta = brightness_delta
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high
        self.hue_delta = hue_delta

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)

        img = self.brightness(img)
        
        if random.random() > 0.5:
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            img = self.saturation(img)
            img = self.hue(img)
            img = self.contrast(img)
        return img

    def brightness(self, img):
        if random.random() > 0.5:
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            return TF.adjust_brightness(img, 1 + delta / 255)
        return img

    def contrast(self, img):
        if random.random() > 0.5:
            factor = random.uniform(self.contrast_low, self.contrast_high)
            return TF.adjust_contrast(img, factor)
        return img

    def saturation(self, img):
        if random.random() > 0.5:
            factor = random.uniform(self.saturation_low, self.saturation_high)
            return TF.adjust_saturation(img, factor)
        return img

    def hue(self, img):
        if random.random() > 0.5:
            delta = random.uniform(-self.hue_delta, self.hue_delta)
            return TF.adjust_hue(img, delta)
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
train_transforms_ColorAugSSD = transforms.Compose(
    [
        ColorAugSSDTransform()
    ]
)
# ========================================================== #