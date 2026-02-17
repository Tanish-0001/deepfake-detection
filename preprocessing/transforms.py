"""
Image transformation utilities for training and evaluation.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class TransformConfig:
    """Configuration for image transforms."""
    input_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet mean
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet std
    horizontal_flip_prob: float = 0.5
    use_augmentation: bool = True


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image


class Resize:
    """Resize image to target size."""
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (width, height)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)


class ToFloat:
    """Convert image to float32 and normalize to [0, 1]."""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0


class Normalize:
    """Normalize image with mean and std."""
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return (image - self.mean) / self.std


class BGRtoRGB:
    """Convert BGR to RGB format."""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ToTensor:
    """Convert HWC numpy array to CHW format for PyTorch."""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # HWC to CHW
        return np.transpose(image, (2, 0, 1))


class RandomHorizontalFlip:
    """Randomly flip image horizontally."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return cv2.flip(image, 1)
        return image


class RandomRotation:
    """Randomly rotate image."""
    
    def __init__(self, max_angle: float = 5):
        self.max_angle = max_angle
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


class ColorJitter:
    """Randomly adjust brightness, contrast, saturation."""
    
    def __init__(
        self,
        brightness: float = 0.05,
        contrast: float = 0.05,
        saturation: float = 0.05
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Brightness
        if self.brightness > 0:
            factor = 1 + np.random.uniform(-self.brightness, self.brightness)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Contrast
        if self.contrast > 0:
            factor = 1 + np.random.uniform(-self.contrast, self.contrast)
            mean = np.mean(image)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Saturation
        if self.saturation > 0:
            factor = 1 + np.random.uniform(-self.saturation, self.saturation)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image


class GaussianBlur:
    """Apply Gaussian blur to image."""
    
    def __init__(self, kernel_size: int = 3, p: float = 0.3):
        self.kernel_size = kernel_size
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        return image


class JPEGCompression:
    """Simulate JPEG compression artifacts."""
    
    def __init__(self, quality_range: Tuple[int, int] = (80, 100), p: float = 0.3):
        self.quality_range = quality_range
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            quality = np.random.randint(*self.quality_range)
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return image


def get_train_transforms(config: Optional[TransformConfig] = None) -> Compose:
    """
    Get training transforms with augmentation.
    
    Args:
        config: Transform configuration
        
    Returns:
        Composed transforms
    """
    if config is None:
        config = TransformConfig()
    
    transforms = [
        Resize(config.input_size),
    ]
    
    if config.use_augmentation:
        transforms.extend([
            RandomHorizontalFlip(config.horizontal_flip_prob),
            ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            GaussianBlur(kernel_size=3, p=0.1),
            JPEGCompression(quality_range=(80, 100), p=0.1),
        ])
    
    transforms.extend([
        BGRtoRGB(),
        ToFloat(),
        Normalize(config.mean, config.std),
        ToTensor(),
    ])
    
    return Compose(transforms)


def get_val_transforms(config: Optional[TransformConfig] = None) -> Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        config: Transform configuration
        
    Returns:
        Composed transforms
    """
    if config is None:
        config = TransformConfig()
    
    transforms = [
        Resize(config.input_size),
        BGRtoRGB(),
        ToFloat(),
        Normalize(config.mean, config.std),
        ToTensor(),
    ]
    
    return Compose(transforms)


def get_pytorch_transforms(train: bool = True, config: Optional[TransformConfig] = None):
    """
    Get PyTorch-native transforms using torchvision.
    
    Args:
        train: Whether to get training transforms
        config: Transform configuration
        
    Returns:
        torchvision.transforms.Compose
    """
    import torchvision.transforms as T
    
    if config is None:
        config = TransformConfig()
    
    if train and config.use_augmentation:
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(config.input_size),
            T.RandomHorizontalFlip(config.horizontal_flip_prob),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            T.ToTensor(),
            T.Normalize(mean=config.mean, std=config.std),
        ])
    else:
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(config.input_size),
            T.ToTensor(),
            T.Normalize(mean=config.mean, std=config.std),
        ])
    
    return transforms
