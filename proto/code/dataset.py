import os
import numpy as np
from pathlib import Path
from typing import Callable, List

import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import tensor_transform
from .utils import ON_KAGGLE


DATA_ROOT = Path('../input/aptos-train-dataset/aptos-train-images' if ON_KAGGLE else './data')
IMG_SIZE = 256


class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, debug: bool = True):
        super().__init__()
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._debug = debug

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_transform_image(
            item, self._root, self._image_transform, debug=self._debug)
        target = torch.tensor(self._df.loc[idx, 'diagnosis'])
        return image, target


class TTADataset:
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, tta: int):
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._tta = tta

    def __len__(self):
        return len(self._df) * self._tta

    def __getitem__(self, idx):
        item = self._df.iloc[idx % len(self._df)]
        image = load_transform_image(item, self._root, self._image_transform)
        return image, item.diagnosis


def load_transform_image(
        item, root: Path, image_transform: Callable, debug: bool = False):
    image = load_image(item, root)
    image = image_transform(image)
    if debug:
        image.save('_debug.png')
    return tensor_transform(image)


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id_code}'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


"""
def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id_code}'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), IMG_SIZE / 10), -4, 128)
    return Image.fromarray(image)
"""

"""
thanks to https://www.kaggle.com/ratthachat/aptos-simple-preprocessing-decoloring-cropping
"""


def crop_image1(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        h, w, _ = img.shape
        img1 = cv2.resize(crop_image1(img[:, :, 0]), (w, h))
        img2 = cv2.resize(crop_image1(img[:, :, 1]), (w, h))
        img3 = cv2.resize(crop_image1(img[:, :, 2]), (w, h))
        img[:, :, 0] = img1
        img[:, :, 1] = img2
        img[:, :, 2] = img3
        return img
