import os

import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms

from util import pad_and_resize_image


class WhaleAndDolphinDataset(nn.Module):
    """Whale and dolphin Dataset class
    """

    def __init__(self, dataset_path, df_path, image_size=128, transform=None, is_train=True):
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

        self.image_paths = [os.path.join(self.dataset_path, image_name) for image_name in os.listdir(
            self.dataset_path) if 'ipynb' not in image_name]

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
        image_path = self.image_paths[idx]

        # open and preprocess an image
        image = pad_and_resize_image(image_path, image_size=self.image_size)

        # apply transformations
        if self.transform is not None:
            image = self.transform(image)

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
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    return transform


def get_test_transform():
    """test transforms

    Returns:
        _type_: torchvision.transforms
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    return transform
