import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import coloredlogs

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
from abc import ABC

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from key2med.data.datasets import BaseDataset, CheXpertDataset, StudyDataset
from key2med.utils.transforms import (
    BaseTransform,
    ColorRandomAffineTransform,
    ColorTransform,
    RandomAffineTransform,
    ResizeTransform,
    Transform,
    XrayTransform,
)


class ADataLoader(ABC):
    """
    Abstract base class for all dataloaders.
    Defines the train, validate and test dataloaders with default None.
    """

    @property
    def train(self):
        return None

    @property
    def validate(self):
        return None

    @property
    def test(self):
        return None

    @property
    def n_train_batches(self):
        if self.train is None:
            return 0
        return len(self.train)

    @property
    def n_validate_batches(self):
        if self.validate is None:
            return 0
        return len(self.validate)

    @property
    def n_test_batches(self):
        if self.test is None:
            return 0
        return len(self.test)


class BaseDataLoader(ADataLoader):
    """
    Basic dataloader class.
    To be called with a dataset to be split or already split datasets.
    Creates torch dataloaders for the provided datasets.
    """

    def __init__(
        self,
        dataset: BaseDataset = None,
        valid_size: float = 0.1,
        test_size: float = 0.1,
        train_dataset: BaseDataset = None,
        valid_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
        batch_size: int = None,
        collate_function: Optional[Callable] = None,
        n_workers: int = 1,
    ):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.collate_function = collate_function

        self.train_set_size = 0
        self.__train_loader = None
        self.valid_set_size = 0
        self.__valid_loader = None
        self.test_set_size = 0
        self.__test_loader = None

        if all([x is None for x in [train_dataset, valid_dataset, test_dataset]]):
            self.init_split_from_dataset(dataset, valid_size, test_size)
        else:
            self.init_split_from_split(train_dataset, valid_dataset, test_dataset)

    @staticmethod
    def all_are_none(*args):
        return all([x is None for x in args])

    def init_split_from_dataset(self, dataset, valid_size, test_size):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split_valid = int(np.floor(valid_size * dataset_size))
        split_test = split_valid + int(np.floor(test_size * dataset_size))
        np.random.shuffle(indices)

        train_indices = indices[split_test:]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        self.train_set_size = len(train_sampler)
        self.__train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            collate_fn=self.collate_function,
            sampler=train_sampler,
        )
        print(f"{self.train_set_size:,} samples for training")

        valid_indices = indices[:split_valid]
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
        self.valid_set_size = len(valid_sampler)
        self.__valid_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            collate_fn=self.collate_function,
            sampler=valid_sampler,
        )
        print(f"{self.valid_set_size:,} samples for validation")

        test_indices = indices[split_valid:split_test]
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
        self.test_set_size = len(test_sampler)
        self.__test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            collate_fn=self.collate_function,
            sampler=test_sampler,
        )
        print(f"{self.test_set_size:,} samples for validation")

    def init_split_from_split(self, train_dataset, valid_dataset, test_dataset):
        if train_dataset is not None:
            self.train_set_size = len(train_dataset)
            self.__train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_function,
                shuffle=True,
                num_workers=self.n_workers,
            )
            print(f"{self.train_set_size:,} samples for training")
        else:
            self.train_set_size = 0
            self.__train_loader = None

        if valid_dataset is not None:

            self.valid_set_size = len(valid_dataset)
            self.__valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_function,
                shuffle=True,
                num_workers=self.n_workers,
            )
            print(f"{self.valid_set_size:,} samples for validation")
        else:
            self.valid_set_size = 0
            self.__valid_loader = None

        if test_dataset is not None:
            self.test_set_size = len(test_dataset)
            self.__test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_function,
                shuffle=True,
                num_workers=self.n_workers,
            )
            print(f"{self.test_set_size:,} samples for testing")
        else:
            self.test_set_size = 0
            self.__test_loader = None

    @property
    def train(self):
        return self.__train_loader

    @property
    def validate(self):
        return self.__valid_loader

    @property
    def test(self):
        return self.__test_loader


class CheXpertDataLoader(BaseDataLoader):
    """
    Dataloader for CheXpert dataset.
    When called with a valid path to the chexpert dataset, initializes
    the training and validation split given by chexpert.
    Most arguments given to init are passed on to chexpert datasets.
    """

    def __init__(
        self,
        data_path,
        batch_size: int = 16,
        img_resize: int = 128,
        channels: int = 3,
        transform: Transform = None,
        uncertainty_upper_bound=0.5,
        uncertainty_lower_bound=0.5,
        one_labels: List[int] = None,
        splits: List[str] = "train_valid",
        uncertain_to_one: List[str] = None,
        uncertain_to_zero: List[str] = None,
        do_random_transform: bool = True,
        min_age: int = None,
        max_age: int = None,
        sex_values: List[str] = None,
        frontal_lateral_values: List[str] = None,
        ap_pa_values: List[str] = None,
        label_filter: Union[str, List[int]] = "full",
        use_upsampling: bool = False,
        upsample_labels: List[str] = None,
        fix_labels_by_hierarchy: bool = False,
        n_workers=1,
        max_size=None,
        in_memory=False,
        use_cache: bool = False,
        rank: int = None,
        world_size: int = None,
        collate_function: Optional[Callable] = None,
        plot_stats: bool = True,
    ):
        """

        :param data_path: Path to chexpert dataset.
        :param batch_size: Batch size for both training and validation.
        :param img_resize: Dimension of image as input to the model.
        :param channels: Number of image channels. CheXpert are B/W-images by default, one channel. Most pretrained vision models
                         expect 3 channels for color. By default, this class implements a transform that copies the grayscale values
                         from one channel to 3 channels.
        :param transform: Transform class object to transform the data before training. By default resizes the image to img_resize.
        :param uncertainty_upper_bound: Uncertain values (if not in uncertain_to_one or uncertain_to_zero) are mapped onto an interval
                                        between uncertainty_upper_bound and uncertainty_lower_bound.
                                        To map onto exactly one value set both values the same.
        :param uncertainty_lower_bound: Uncertain values (if not in uncertain_to_one or uncertain_to_zero) are mapped onto an interval
                                        between uncertainty_upper_bound and uncertainty_lower_bound.
                                        To map onto exactly one value set both values the same.
        :param one_labels: Labels that have to be one, or the datapoint is filtered. List of integers, refers to the list of labels AFTER
                           filtering by label_filter.
        :param uncertain_to_one: Labels for which uncertain values should be mapped onto 1.0. List of string with the label name.
        :param uncertain_to_zero: Labels for which uncertain values should be mapped onto 0.0. List of string with the label name.
        :param do_random_transform: Boolean, do a random transform when loading a training image.
        :param min_age: Minimum age of person in xray.
        :param max_age: Maximum age of person in xray.
        :param sex_values: One or multiple of ['Unknown', 'Male', 'Female']
        :param frontal_lateral_values: One or multiple of ['Frontal', 'Lateral']
        :param ap_pa_values: One or multiple of ['', 'AP', 'PA', 'LL', 'RL']
        :param label_filter: Set to 'competition' to only train and evaluate on the competition classes:
                             'Edema', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion'
        :param n_worker: Number of processes for the torch dataloader
        :param max_size: Limit for number of images in training and validation each. For debugging.
        :param in_memory: Load the entire dataset into memory before starting the training.
        :param use_cache: Find or create a cache file for the datasets and load from there.
        """

        self.image_dim = img_resize
        self.channels = channels
        self.uncertainty_upper_bound = uncertainty_upper_bound
        self.uncertainty_lower_bound = uncertainty_lower_bound
        self.do_random_transform = do_random_transform
        train_dataset = self.init_dataset(
            data_path=data_path,
            split="train",
            transform=transform or self.transform,
            random_transform=self.random_transform,
            channels=self.channels,
            splits=splits,
            max_size=max_size,
            use_cache=use_cache,
            in_memory=in_memory,
            uncertain_to_one=uncertain_to_one,
            uncertain_to_zero=uncertain_to_zero,
            min_age=min_age,
            max_age=max_age,
            sex_values=sex_values,
            frontal_lateral_values=frontal_lateral_values,
            ap_pa_values=ap_pa_values,
            label_filter=label_filter,
            use_upsampling=use_upsampling,
            upsample_labels=upsample_labels,
            uncertainty_upper_bound=self.uncertainty_upper_bound,
            uncertainty_lower_bound=self.uncertainty_lower_bound,
            fix_labels_by_hierarchy=fix_labels_by_hierarchy,
            one_labels=one_labels,
            plot_stats=plot_stats,
            rank=rank,
            world_size=world_size,
        )

        valid_dataset = self.init_dataset(
            data_path=data_path,
            split="valid",
            transform=transform or self.transform,
            random_transform=None,
            channels=self.channels,
            splits=splits,
            max_size=max_size,
            use_cache=use_cache,
            in_memory=in_memory,
            uncertain_to_one=uncertain_to_one,
            uncertain_to_zero=uncertain_to_zero,
            min_age=min_age,
            max_age=max_age,
            sex_values=sex_values,
            frontal_lateral_values=frontal_lateral_values,
            ap_pa_values=ap_pa_values,
            label_filter=label_filter,
            use_upsampling=False,
            upsample_labels=upsample_labels,
            uncertainty_upper_bound=self.uncertainty_upper_bound,
            uncertainty_lower_bound=self.uncertainty_lower_bound,
            fix_labels_by_hierarchy=False,
            one_labels=one_labels,
            plot_stats=plot_stats,
            rank=0,
            world_size=1,
        )

        test_dataset = self.init_dataset(
            data_path=data_path,
            split="test",
            transform=transform or self.transform,
            random_transform=None,
            channels=self.channels,
            splits=splits,
            max_size=max_size,
            use_cache=use_cache,
            in_memory=in_memory,
            uncertain_to_one=uncertain_to_one,
            uncertain_to_zero=uncertain_to_zero,
            min_age=min_age,
            max_age=max_age,
            sex_values=sex_values,
            frontal_lateral_values=frontal_lateral_values,
            ap_pa_values=ap_pa_values,
            label_filter=label_filter,
            use_upsampling=False,
            upsample_labels=upsample_labels,
            uncertainty_upper_bound=self.uncertainty_upper_bound,
            uncertainty_lower_bound=self.uncertainty_lower_bound,
            fix_labels_by_hierarchy=False,
            one_labels=one_labels,
            plot_stats=plot_stats,
            rank=0,
            world_size=1,
        )

        self.index_to_label = train_dataset.index_to_label
        self.collate_function = collate_function or self.default_collate_function

        super().__init__(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            collate_function=self.collate_function,
            n_workers=n_workers,
        )

    def init_dataset(self, *args, **kwargs):
        # descriptor of all splits in the dataloader, "train_valid_test" or "train_valid"
        splits = kwargs.get("splits")
        # this dataset split
        split = kwargs.get("split")
        if splits == "train_valid_test":
            # all three splits are used, return Dataset for train valid and test
            return CheXpertDataset(*args, **kwargs)
        # we habe only train and valid, return None for test set
        if split in ["train", "valid"]:
            return CheXpertDataset(*args, **kwargs)
        return None

    @property
    def transform(self):
        return ColorTransform(self.image_dim)

    @property
    def random_transform(self):
        if self.do_random_transform:
            return RandomAffineTransform()
        else:
            return None

    @property
    def default_collate_function(self):
        return None

    @property
    def label_dim(self):
        if self.train is not None:
            return self.train.dataset.label_dim
        if self.validate is not None:
            return self.validate.dataset.label_dim
        if self.test is not None:
            return self.test.dataset.label_dim

    @property
    def imratio(self):
        return self.train.dataset.imratio

    @property
    def imratios(self):
        imratios = []
        for key, value in self.train.dataset.imratios.items():
            imratios.append(value[2])
        return imratios


class ColorCheXpertDataLoader(CheXpertDataLoader):
    """
    CheXpert dataset as used in most papers.
    Applies a transform that does not copy the B/W-values from the image onto
    3 channels, but colors them black-blue.
    Also sets a blue background for the filler in the random transformation.
    Same init arguments as CheXpertDataLoader
    """

    @property
    def transform(self):
        return ColorTransform(self.image_dim)

    @property
    def random_transform(self):
        if self.do_random_transform:
            return ColorRandomAffineTransform()
        else:
            return None


class StudiesDataLoader(ColorCheXpertDataLoader):
    def __init__(self, *args, upsample_study: bool = None, **kwargs):
        self.upsample_study = upsample_study
        super(StudiesDataLoader, self).__init__(*args, **kwargs)

    def init_dataset(self, *args, **kwargs):
        if kwargs.get("split") in ["train", "valid"]:
            return StudyDataset(*args, upsample_study=self.upsample_study, **kwargs)
        # split is 'test'
        return None

    def init_dataset(self, *args, **kwargs):
        # descriptor of all splits in the dataloader, "train_valid_test" or "train_valid"
        splits = kwargs.get("splits")
        # this dataset split
        split = kwargs.get("split")
        if kwargs.get("splits") == "train_valid" and kwargs.get("split") == "test":
            return None
        return StudyDataset(*args, upsample_study=self.upsample_study, **kwargs)

    @property
    def default_collate_function(self):
        return collate_studies_function


def collate_patients_function(
    batch,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    images = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    lens = [img.shape[0] for img in images]
    return images, labels, lens


def collate_studies_function(
    batch,
) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
    batch_images = []
    batch_labels = []
    batch_metadata = []
    for item in batch:
        if isinstance(item, list):
            # item with more than one image
            images = torch.stack([x[0] for x in item])
            labels = item[0][1]
            # here we ignore the ap_pa values for now! add them later!
            metadata = torch.stack([x[2][0] for x in item])
        else:
            # item is one image, label, metadata tuple
            images = item[0].unsqueeze(0)
            labels = item[1]
            # again we ignore the ap_pa values
            metadata = item[2][0].unsqueeze(0)
        batch_images.append(images)
        batch_labels.append(labels)
        batch_metadata.append(metadata)
    return batch_images, batch_labels, batch_metadata
