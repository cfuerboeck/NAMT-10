import json
from typing import Dict, Tuple, Union

import cv2
import numpy as np
from torchvision import transforms


class Transform:
    transform = None

    def __call__(self, x):
        return self.transform(x)

    @property
    def config(self) -> Dict:
        raise NotImplementedError


class BaseTransform(Transform):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    @property
    def config(self) -> Dict:
        return {"name": "base"}


class ResizeTransform(Transform):
    def __init__(self, image_dim: int):
        self.image_dim = image_dim
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_dim, image_dim)),
                transforms.ToTensor(),
            ]
        )

    @property
    def config(self) -> Dict:
        return {"name": "resize", "image_dim": self.image_dim}


class RandomAffineTransform(Transform):
    """
    Random slight scaling, rotation and translation of image.
    """

    def __init__(
        self,
        degrees: Tuple[float] = (-15, 15),
        translate: Tuple[float] = (0.05, 0.05),
        scale: Tuple[float] = (0.95, 1.05),
        fill: Union[Tuple[float, float, float], float] = 0,
    ):
        self.transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=degrees, translate=translate, scale=scale, fill=fill
                )
            ]
        )

    @property
    def config(self) -> Dict:
        return {"name": "random_affine"}


class ColorRandomAffineTransform(RandomAffineTransform):
    """
    Adds a blue background to the edges of the shifted image.
    """

    def __init__(self):
        super().__init__(fill=(0.0741, 0.2052, 0.4265))

    @property
    def config(self) -> Dict:
        return {"name": "color_random_affine"}


class ColorTransform(Transform):
    """
    Transform implemented by most CheXpert papers.
    Reads the image and maps the black/white values to a black-blue scale.
    """

    def __init__(self, image_dim: int):
        self.transform = self._transform
        self.image_dim = image_dim

    def _transform(self, x):
        x = np.array(x)
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        # resize and normalize; e.g., ToTensor()
        x = cv2.resize(
            x, dsize=(self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR
        )
        x = x / 255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        x = (x - __mean__) / __std__
        x = x.transpose((2, 0, 1)).astype(np.float32)
        return x

    @property
    def config(self) -> Dict:
        return {"name": "color", "image_dim": self.image_dim}


class CropAndColorTransform(Transform):
    """
    Transform implemented by most CheXpert papers.
    Reads the image and maps the black/white values to a black-blue scale.
    """

    def __init__(self, image_dim: int):
        self.transform = self._transform
        self.image_dim = image_dim

    def _transform(self, x):
        x = np.array(x)
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        x = crop_black_borders(x)

        # resize and normalize; e.g., ToTensor()
        x = cv2.resize(
            x, dsize=(self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR
        )
        x = x / 255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        x = (x - __mean__) / __std__
        x = x.transpose((2, 0, 1)).astype(np.float32)
        return x

    @property
    def config(self) -> Dict:
        return {"name": "crop_and_color", "image_dim": self.image_dim}


class XrayTransform(Transform):
    """
    Transforms images into grayscale tensors of size [1, image_dim, image_dim]
    """

    def __init__(self, image_dim: int):
        self.image_dim = image_dim
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((image_dim, image_dim)),
                transforms.ToTensor(),
            ]
        )

    @property
    def config(self) -> Dict:
        return {"name": "xray", "image_dim": self.image_dim}


def crop_black_borders(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image > image.mean() * 0.01)
    return image[
        np.min(y_nonzero) : np.max(y_nonzero), np.min(x_nonzero) : np.max(x_nonzero)
    ]
