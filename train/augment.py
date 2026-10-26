from __future__ import annotations

import random

import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter


class NightLowLightSimulation:
    def __init__(self, p: float = 0.25, gamma_range: tuple[float, float] = (1.5, 3.0)) -> None:
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        gamma = random.uniform(*self.gamma_range)
        lut = [min(255, int(((i / 255.0) ** gamma) * 255.0)) for i in range(256)]
        return img.point(lut * 3)

class MotionBlur:
    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.BoxBlur(1))

class SafeColorJitter:
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.15,
        p: float = 0.5,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        b_factor = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
        c_factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        s_factor = random.uniform(1.0 - self.saturation, 1.0 + self.saturation)
        img = ImageEnhance.Brightness(img).enhance(b_factor)
        img = ImageEnhance.Contrast(img).enhance(c_factor)
        img = ImageEnhance.Color(img).enhance(s_factor)
        return img

def get_train_augmentation() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(size=(128, 128), scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(-10, 10)),
        T.RandomPerspective(distortion_scale=0.1, p=0.2),
        SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, p=0.5),
        NightLowLightSimulation(p=0.25),
        MotionBlur(p=0.2),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2)),
    ])
