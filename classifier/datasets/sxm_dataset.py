import os
import glob
import math
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from tarfile import TarError
from PIL import Image
from scipy import stats
from scipy.signal import convolve2d
from torchvision import transforms as T

from torch.utils.data import Dataset, Subset, DataLoader
from atomai.utils import extract_patches, create_multiclass_lattice_mask

from nanoscopy import spm


class SXM_Dataset(torch.utils.data.Dataset):
    def __init__(
        self, root, paths=None, filename_filter="", transforms=None, resize=False
    ):
        self.root = root
        self.target_size = 512
        self.resize = resize
        self.paths = (
            paths
            if paths is not None
            else spm.io.read_dir(root, filename_filter=filename_filter)
        )
        print(f"{len(self.paths)} sxm files found in {self.root}")

        self.transforms = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                #        T.Resize(size=(512, 512))
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # get path for this index
        path = self.paths[idx]

        # get image for this index
        sxm = spm.read(path)[0]
        Z_fwd = sxm.fwd()

        Z = spm.correct(
            [Z_fwd], terrace=False, poly=True, equalize=False, rescale=False
        )[0]

        image = spm.correct(
            [Z_fwd],
            terrace=False,
            poly=True,
            equalize=True,
            rescale=True,
        )[0]

        # optionally rezie image if needed
        if self.resize:
            resY, resX = image.size
            if resX != self.target_size:
                image = image.resize((self.target_size, self.target_size))
                Z = cv2.resize(
                    Z,
                    dsize=(self.target_size, self.target_size),
                    interpolation=cv2.INTER_CUBIC,
                )

        # Set the target dict
        target = {}
        target["index"] = idx
        target["sxm"] = sxm
        target["Z"] = Z

        # apply any supplied transforms
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def _get_transform(self):
        transforms = [
            T.ToPILImage(),
            T.ToTensor(),
            #        T.Resize(size=(512, 512))
        ]

        return T.Compose(transforms)
