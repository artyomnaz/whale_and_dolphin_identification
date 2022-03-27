import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from util import pad_and_resize_image


class WhaleAndDolphinDataset(nn.Module):
    """Whale and dolphin Dataset class
    """

    def __init__(self, dataset_path, df_path, image_size=128, transform=None, is_train=True, balanced=True, balance_amount=10):
        """initialization

        Args:
            dataset_path (str): path to dataset images
            df_path (str): path to dataset description csv file
            image_size (int, optional): output image size. Defaults to 128.
            transform (torchvision.transforms, optional): transformations and augmentations. Defaults to None.
            is_train (bool, optional): train/test flag. Defaults to True.
        """
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.df_path = df_path
        self.transform = transform
        self.is_train = is_train
        self.df = pd.read_csv(self.df_path)

        if balanced:
            labels_dict = {}
            for i in tqdm(range(len(self.df)), 'Prepare counts of items'):
                individual_key = self.df.iloc[i].individual_key
                if individual_key in labels_dict:
                    labels_dict[individual_key] += 1
                else:
                    labels_dict[individual_key] = 1

            counts = dict.fromkeys(labels_dict.keys(), 0)
            drop_rows = []
            for i in tqdm(range(len(self.df)), 'Drop a lot of lines'):
                individual_key = self.df.iloc[i].individual_key

                if counts[individual_key] >= balance_amount:
                    drop_rows.append(i)

                counts[individual_key] += 1

            self.df = self.df.drop(drop_rows).reset_index()

            new_rows = []

            for i in tqdm(range(len(self.df)), 'Create new duplicate lines'):
                individual_key = self.df.iloc[i].individual_key

                if counts[individual_key] < balance_amount:
                    new_rows_amount = balance_amount - counts[individual_key]
                    new_indexes = list(
                        self.df[self.df.individual_key == individual_key].index)

                    if len(new_indexes) < new_rows_amount:
                        new_indexes = new_indexes * balance_amount

                    new_indexes = new_indexes[:new_rows_amount]

                    for new_index in new_indexes:
                        new_rows.append(new_index)
                    counts[individual_key] += new_rows_amount

            self.df = self.df.append(
                [self.df.iloc[new_rows]], ignore_index=True)

    def __len__(self):
        """Dataset length

        Returns:
            _type_: int
        """
        return len(self.df)

    def __getitem__(self, idx):
        """Get i-th element of dataset

        Args:
            idx (int): index

        Returns:
            _type_: dict
        """

        # fix current row
        row = self.df.iloc[idx]

        # get image path
        image_path = os.path.join(self.dataset_path, row.image)

        # open and preprocess an image
        image = pad_and_resize_image(image_path, image_size=self.image_size)

        # apply transformations
        if self.transform is not None:
            image = self.transform(image=np.array(image))['image']

        # to tensor
        if image.shape[2] != self.image_size:
            image = image.transpose((2, 0, 1))

        out_dict = {}
        out_dict['image'] = image

        if self.is_train:
            out_dict['label'] = row.individual_key

        return out_dict


def get_train_transform():
    """train transforms

    Returns:
        _type_: torchvision.transforms
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=0.15,
                           rotate_limit=60,
                           p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2,
            sat_shift_limit=0.2,
            val_shift_limit=0.2,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
    return transform


def get_test_transform():
    """test transforms

    Returns:
        _type_: torchvision.transforms
    """

    transform = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225], 
                                       max_pixel_value=255.0, p=1.0),
                           ToTensorV2()], p=1.)
    return transform
