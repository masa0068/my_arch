# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.memory_loaders = []
        self.train_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets, memory_dataset: datasets, 
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """

    #train_maskとtest_maskを作成して、setting.i（現在のタスクの開始クラス番号）から setting.i + setting.N_CLASSES_PER_TASK（タスクに含まれるクラスの数）までの範囲にデータを制限
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    
    #debug========================
    # print(f"Task {setting.i // setting.N_CLASSES_PER_TASK + 1}:")
    # print(f"Train mask sum: {np.sum(train_mask)}")
    # print(f"Test mask sum: {np.sum(test_mask)}")
    # print(f"Memory mask sum: {np.sum(train_mask)}")
    # print(f"Current i: {setting.i}")
    # print(f"Class range: {setting.i} - {setting.i + setting.N_CLASSES_PER_TASK - 1}")
    # if np.sum(train_mask) == 0 or np.sum(test_mask) == 0 :
    #     raise ValueError("No samples found for the specified class range. Please check `setting.i` and `N_CLASSES_PER_TASK` values.")
    #debug=========================

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    memory_dataset.data = memory_dataset.data[train_mask]
    memory_dataset.targets = np.array(memory_dataset.targets)[train_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.train.batch_size, shuffle=True, num_workers=4) #setting.args.train.batch_size=256
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.train.batch_size, shuffle=False, num_workers=4)
    memory_loader = DataLoader(memory_dataset,
                              batch_size=setting.args.train.batch_size, shuffle=False, num_workers=4)

    setting.test_loaders.append(test_loader)
    setting.train_loaders.append(train_loader)
    setting.memory_loaders.append(memory_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK

    return train_loader, memory_loader, test_loader


def store_masked_label_loaders(train_dataset: datasets, test_dataset: datasets, memory_dataset: datasets, 
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.labels) >= setting.i,
        np.array(train_dataset.labels) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.labels) >= setting.i,
        np.array(test_dataset.labels) < setting.i + setting.N_CLASSES_PER_TASK)
    
    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.labels)[train_mask]
    test_dataset.targets = np.array(test_dataset.labels)[test_mask]

    memory_dataset.data = memory_dataset.data[train_mask]
    memory_dataset.targets = np.array(memory_dataset.labels)[train_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.train.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.train.batch_size, shuffle=False, num_workers=4)
    memory_loader = DataLoader(memory_dataset,
                              batch_size=setting.args.train.batch_size, shuffle=False, num_workers=4)

    setting.test_loaders.append(test_loader)
    setting.train_loaders.append(train_loader)
    setting.memory_loaders.append(memory_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, memory_loader, test_loader




def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
