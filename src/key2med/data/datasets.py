import csv
import glob
import json
import logging
import os
import random
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import coloredlogs
import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm as tqdm_writer

from key2med.data import ImagePath, Label
from key2med.data.patients import Patient, Study
from key2med.utils.helper import get_disk_usage, get_file_size, hash_dict
from key2med.utils.logging import tqdm
from key2med.utils.plotting import text_histogram
from key2med.utils.transforms import BaseTransform, RandomAffineTransform, Transform

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DatasetIndex = int
ImageIndex = int


class BaseDataset(Dataset):
    """
    Basic dataset class, mostly implements reading data into memory and caching.
    """

    def __init__(
        self,
        data_path: str,
        in_memory: bool = False,
        max_size: int = None,
        use_cache: bool = False,
    ):
        """

        :param data_path: Path to data. Cache path is in subdirectory data_path/cache/
        :param in_memory: Boolean, load all data into memory before training.
        :param max_size: Limit on number of data. For debugging.
        :param use_cache: Find or create a cache path for faster loading next run.
        """
        self.max_size = max_size
        self.in_memory = in_memory

        self.data_file = None
        cache_path = self.get_cache_path(data_path) if use_cache else None
        if cache_path is not None and os.path.isfile(cache_path):
            logger.info(f"Reading data from cache file {cache_path}")
            self.check_data_file(cache_path)
            self.data_file = h5py.File(cache_path, "r")

        self.items = None
        if self.in_memory:
            logger.info(f"Reading all data into memory.")
            all_image_indices: List[ImageIndex] = self.get_all_image_indices()
            # load all relevant pictures into memory
            self.items: Dict[ImageIndex:Any] = {
                i: self.load_item(i) for i in tqdm(all_image_indices)
            }
            # self.items = [x for x in tqdm(self, desc="Loading all data into memory")]
            if self.data_file is not None:
                logger.info(
                    f"Closing cache file {cache_path} after loading all data into memory."
                )
                self.data_file.close()
                self.data_file = None

        if cache_path is not None and not os.path.isfile(cache_path):
            logger.info(f"No cache file found. Writing all data to cache.")
            dir_path, total, used, free = get_disk_usage(cache_path)
            logger.info(
                f"Current disk usage in {dir_path}:\n"
                f"Total: {total}GB\n"
                f"Used: {used}GB\n"
                f"Free: {free}GB"
            )
            self.cache_data(cache_path)
            dir_path, total, used, free = get_disk_usage(cache_path)
            logger.info(
                f"Written {get_file_size(cache_path)}GB to {cache_path}.\n"
                f"Current disk usage in {dir_path}:\n"
                f"Total: {total}GB\n"
                f"Used: {used}GB\n"
                f"Free: {free}GB"
            )

    @property
    def index_mapping(self) -> Dict[DatasetIndex, List[ImageIndex]]:
        raise NotImplementedError

    def get_all_image_indices(self) -> List[ImageIndex]:
        return [i for i in range(len(self))]

    def get_cache_path(self, data_path: str) -> str:
        """
        Get path to cache file in a subdirectory of data_path.
        Use config and max_size to create cache file name.
        :param data_path: Path to data. Cache path is in subdirectory data_path/cache/
        :return: str Path to cache file
        """
        if self.max_size is not None:
            # dataset is restricted by max_size
            # check if a cache for the full dataset exists
            logger.info(
                f"Dataset has max_size {self.max_size}. Checking if the full dataset is cached."
            )
            full_dataset_config = self.config.copy()
            full_dataset_config["max_size"] = None
            full_dataset_hash = self.hash_config(full_dataset_config)
            full_dataset_cache_path = os.path.join(
                data_path, "cache", full_dataset_hash
            )
            if os.path.isfile(full_dataset_cache_path):
                logger.info(
                    f"Full dataset is cached at {full_dataset_cache_path}. Reading from full dataset cache."
                )
                return full_dataset_cache_path
            logger.info(
                f"No cache for full dataset. Proceeding with caching dataset of size {self.max_size}"
            )
        # no cache for full dataset or no max_size
        return os.path.join(data_path, "cache", self.hash)

    @staticmethod
    def check_data_file(cache_path: str):
        """
        Reads one datapoint from the cache file and prints the data shape,
        to make sure the cache file is not empty and the shape of the data
        is correct.
        :param cache_path:
        :return:  None
        :raises: IndexError, KeyError If the cache file is empty or there are other
                 issues.
        """
        file = h5py.File(cache_path, "r")
        try:
            item = file["data"][0]
        except (IndexError, KeyError) as e:
            logger.info(f"No cached data found in file {cache_path}")
            raise e
        logger.info(f"Found data of size {item.shape} in file {cache_path}")

    def __len__(self) -> int:
        """
        Length of dataset. Either number of items (self._len implemented by child class)
        or maximum size if given.
        :return:
        """
        if self.max_size is not None:
            return min(self._len(), self.max_size)
        return self._len()

    def _len(self) -> int:
        """
        To be implemented in child class
        :return: int Number of items in dataset.
        """
        raise NotImplementedError

    def __getitem__(self, index: DatasetIndex) -> Any:
        """
        Basic function to get one item from an index.
        Raises an IndexError if the index is larger than the dataset, because
        torch dataloaders need this to know when the dataset is over.
        Loads data from self.items if in_memory was set True.
        :param index: int
        :return: Any Item
        """
        if index >= len(self):
            raise IndexError
        indices: List[ImageIndex] = self.index_mapping[index]
        items: List[Any] = [self._getitem(index) for index in indices]
        items = [self.item_transformation(item) for item in items]
        if len(items) == 1:
            return items[0]
        return items

    def load_item(self, index: ImageIndex) -> Any:
        if self.items is not None:
            return self.items[index]
        if self.data_file is not None:
            return self.data_file["data"][index]
        return self._load_item(index)

    def _load_item(self, index: ImageIndex) -> Any:
        """
        Function implemented by child classes.
        Loads image when no in_memory or cache available.
        :param index:
        :return:
        """
        raise NotImplementedError

    def item_transformation(self, item: Any) -> Any:
        """
        Transformation to be applied directly before training.
        For example a random transformation.
        To be overwritten by child class. Here: do nothin, return the item.
        :param item: Any Item
        :return: Any Transformed item.
        """
        return item

    def cache_data(self, cache_path):
        """
        Function to cache the data of the dataset. To be implemented by child class.
        :param cache_path: str path to cache file.
        :return: None
        """
        return NotImplementedError

    def _getitem(self, index: ImageIndex) -> Any:
        """
        Function to load one item if self.items is (still) None.
        To  be implemented by child class.
        :param index: int
        :return: Any Item
        """
        raise NotImplementedError

    @property
    def config(self) -> Dict:
        """
        Config for caching. To be implemented by child class.
        :return: Dict Config
        """
        raise NotImplementedError

    @property
    def hash(self) -> str:
        """
        Calculate hash of dataset object by converting the config dict to a
        string and hashing the string.
        :return: str hash string
        """
        return self.hash_config(self.config)

    @staticmethod
    def hash_config(config: Dict) -> str:
        """
        Calculate hash of dataset object by converting the config dict to a
        string and hashing the string.
        :return: str hash string
        """
        return hash_dict(config)


class CheXpertDataset(BaseDataset, object):
    """
    Basic dataset for the CheXpert images.
    From the default directory structure, reads the labels,
    filters by patient information and loads the images.
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        transform: Transform = None,
        channels: int = 1,
        uncertainty_upper_bound: float = 0.5,
        uncertainty_lower_bound: float = 0.5,
        splits: str = "train_valid",
        one_labels: List[int] = None,
        label_filter: Union[str, List[int]] = "full",
        sex_values: List[str] = None,
        min_age: int = None,
        max_age: int = None,
        ap_pa_values: List[str] = None,
        frontal_lateral_values: List[str] = None,
        random_transform: Transform = None,
        uncertain_to_one: List[str] = None,
        uncertain_to_zero: List[str] = None,
        upsample_labels: List[str] = None,
        use_upsampling: bool = False,
        fix_labels_by_hierarchy: bool = False,
        use_cache: bool = False,
        in_memory: bool = False,
        max_size: int = None,
        plot_stats: bool = True,
        rank: int = None,
        world_size: int = None,
    ):
        """

        :param data_path: Path to chexpert dataset.
        :param channels: Number of image channels. CheXpert are B/W-images by default, one channel. Most pretrained vision models
                         expect 3 channels for color. By default, this class implements a transform that copies the grayscale values
                         from one channel to 3 channels.
        :param split: 'train' or 'valid'. Loads the corresponding set of images.
        :param transform: Transform class object to transform the data before training. Specific transform usually given by dataloader class.
        :param uncertainty_upper_bound: Uncertain values (if not in uncertain_to_one or uncertain_to_zero) are mapped onto an interval
                                        between uncertainty_upper_bound and uncertainty_lower_bound.
                                        To map onto exactly one value set both values the same.
        :param uncertainty_lower_bound: Uncertain values (if not in uncertain_to_one or uncertain_to_zero) are mapped onto an interval
                                        between uncertainty_upper_bound and uncertainty_lower_bound.
                                        To map onto exactly one value set both values the same.
        :param one_labels: Labels that have to be one, or the datapoint is filtered. List of integers, refers to the list of labels BEFOR
                           filtering by label_filter.
        :param uncertain_to_one: Labels for which uncertain values should be mapped onto 1.0. List of string with the label name.
        :param uncertain_to_zero: Labels for which uncertain values should be mapped onto 0.0. List of string with the label name.
        :param random_transform: Transform to be applied when loading the image from memory for training.
                                 Given by the dataloader to the training set, not to the validation set.
        :param min_age: Minimum age of person in xray.
        :param max_age: Maximum age of person in xray.
        :param sex_values: One or multiple of ['Unknown', 'Male', 'Female']
        :param frontal_lateral_values: One or multiple of ['Frontal', 'Lateral']
        :param ap_pa_values: One or multiple of ['', 'AP', 'PA', 'LL', 'RL']
        :param label_filter: Set to 'competition' to only train and evaluate on the competition classes:
                             'Edema', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion'
        :param max_size: Limit for number of images in training and validation each. For debugging.
        :param in_memory: Load the entire dataset into memory before starting the training.
        :param use_cache: Find or create a cache file for the datasets and load from there.
        :param plot_stats: Boolean. Plot stats on the labels to the command line after reading?
        """
        self.rank = rank or 0
        self.world_size = world_size or 1

        self.dataset_indices = None
        self.uncertainty_upper_bound = uncertainty_upper_bound
        self.uncertainty_lower_bound = uncertainty_lower_bound
        self.label_filter = self.init_label_filter(label_filter)

        self.uncertain_to_one, self.uncertain_to_zero = self.init_uncertain_mapping(
            uncertain_to_one, uncertain_to_zero
        )
        self.one_labels = one_labels or []
        self.min_age = min_age or 0
        self.max_age = max_age or 100
        self.sex_values = sex_values or ["Unknown", "Male", "Female"]
        self.ap_pa_values = ap_pa_values or ["", "AP", "PA", "LL", "RL"]
        self.frontal_lateral_values = frontal_lateral_values or ["Frontal", "Lateral"]
        self.upsample_labels = upsample_labels or []

        self.data_path = data_path
        self.max_size = max_size
        self.channels = channels

        assert splits in ["train_valid_test", "train_valid"]
        if splits == "train_valid_test":
            assert split in ["train", "valid", "test"]
        if splits == "train_valid":
            assert split in ["train", "valid"]
        self.splits = splits
        self.split = split
        self._index_mapping = None
        self.one_labels_in_class = True
        self.split_keys: Optional[Set[ImagePath]] = None

        self.label_data, self.index_to_label = self.read_data()
        self.label_data_all, self.index_to_label_all = (
            self.label_data,
            self.index_to_label,
        )
        self.label_to_index_all = {
            label: i for i, label in enumerate(self.index_to_label_all)
        }

        self.label_to_index = {label: i for i, label in enumerate(self.index_to_label)}
        self.image_index_to_path: List[ImagePath] = list(self.label_data.keys())
        self.path_to_image_index = {
            image_path: image_index
            for image_index, image_path in enumerate(self.image_index_to_path)
        }

        # we now define split keys, i.e. image paths that belong to the train or valid split
        # if we split the original train into train and valid
        if self.splits == "train_valid_test" and self.split in ["train", "valid"]:
            self.split_keys = self.split_image_paths(self.label_data, self.split)

        if fix_labels_by_hierarchy:
            self.fix_labels_by_hierarchy(default_hierarchy)

        self.transform = transform or self.default_transform
        self.random_transform = random_transform

        # list of image indices needed in this dataset. only image indices
        # in here are loaded into memory!
        self.dataset_indices: List[ImageIndex] = self.filter_indices()

        if use_upsampling:
            self.upsample_data()
        if self.world_size > 1:
            gpu_split_size = int(len(self) / self.world_size)
            gpu_split_indices = list(
                range(self.rank * gpu_split_size, (self.rank + 1) * gpu_split_size)
            )
            self.dataset_indices = [self.dataset_indices[i] for i in gpu_split_indices]

        if plot_stats:
            self.plot_label_stats()

        # Mapping back from an image to filtered indices
        # Question: What index must be called to get image_X.jpg?
        # Answer: Map image to original index, map original index to filtered index
        self.path_to_dataset_index: Dict[ImagePath, DatasetIndex] = {
            self.image_index_to_path[image_index]: dataset_index
            for dataset_index, image_index in enumerate(self.dataset_indices)
        }

        super().__init__(data_path, in_memory, max_size, use_cache)

    def split_image_paths(
        self,
        label_data: Dict[ImagePath, Dict],
        split: str,
        valid_size: float = 0.1,
        seed: int = 1,
    ):
        all_keys = list(label_data.keys())
        train_keys, valid_keys = train_test_split(
            all_keys, test_size=valid_size, random_state=seed, shuffle=True
        )
        return set(train_keys) if split == "train" else set(valid_keys)

    def read_data(self) -> Tuple[Dict[ImagePath, Dict], List[str]]:
        if self.splits == "train_valid_test":
            return self.read_data_train_valid_test()
        else:
            return self.read_data_train_valid()

    def read_data_train_valid_test(self) -> Tuple[Dict[ImagePath, Dict], List[str]]:
        if self.split in ["train", "valid"]:
            label_data, index_to_label = self.read_label_csv(
                os.path.join(self.data_path, "train.csv")
            )
            # label_data = self.split_data(label_data, split=self.split, seed=1)
        elif self.split == "test":
            label_data, index_to_label = self.read_label_csv(
                os.path.join(self.data_path, "valid.csv")
            )
        else:
            raise NotImplementedError
        return label_data, index_to_label

    def read_data_train_valid(self) -> Tuple[Dict[ImagePath, Dict], List[str]]:
        if self.split == "train":
            label_data, index_to_label = self.read_label_csv(
                os.path.join(self.data_path, "train.csv")
            )
        elif self.split == "valid":
            label_data, index_to_label = self.read_label_csv(
                os.path.join(self.data_path, "valid.csv")
            )
        else:
            raise NotImplementedError
        return label_data, index_to_label

    @property
    def index_mapping(self) -> Dict[DatasetIndex, List[ImageIndex]]:
        if self._index_mapping is None:
            self._index_mapping = {
                dataset_index: [image_index]
                for dataset_index, image_index in enumerate(self.dataset_indices)
            }
        return self._index_mapping

    def get_all_image_indices(self) -> List[ImageIndex]:
        return self.dataset_indices

    def _len(self):
        """
        If images are filtered, return number of filtered images.
        Else return number of all images.
        :return:
        """
        if self.dataset_indices is None:
            return len(self.image_index_to_path)
        return len(self.dataset_indices)

    def upsample_data(self):
        upsampled_indices = []
        for label in self.upsample_labels:
            label_index = self.label_to_index[label]
            for image_index in self.dataset_indices:
                row = self.label_data[self.image_index_to_path[image_index]]
                if row["labels"][label_index] == 1.0:
                    upsampled_indices.append(image_index)
        self.dataset_indices.extend(upsampled_indices)

    def fix_labels_by_hierarchy(self, hierarchy):
        # if not self.one_labels_in_class:
        for image_path, data in self.label_data_all.items():
            labels = data["labels"]
            self.label_data_all[image_path]["labels"] = fix_vector(
                hierarchy, self.label_to_index_all, labels
            )
        # else:
        # for image_path, data in self.label_data.items():
        #     labels = data['labels']
        #     self.label_data[image_path]['labels'] = fix_vector(hierarchy, self.label_to_index, labels)

    def item_transformation(
        self, item: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random transformation to the image before passing to training.
        :param item: Image-Label pair
        :return:
        """
        image = item[0]
        if self.random_transform is not None:
            image = self.random_transform(image)
        return image, *item[1:]

    def filter_indices(self) -> List[int]:
        """
        Filter images by fontal/lateral, one_labels, patient age, patient sex, ap_pa values.
        Returns a list of
        :return: List[int] Filtered indices, i.e. images that stay in the dataset.
        """
        if self.one_labels and isinstance(self.one_labels[0], str):
            one_indices = [self.label_to_index_all[label] for label in self.one_labels]
        else:
            one_indices = self.one_labels
        filtered_indices: List[int] = []

        for image_index, image_path in enumerate(self.image_index_to_path):
            if self.split_keys is not None and image_path not in self.split_keys:
                continue
            label_entry = self.label_data[image_path]
            if label_entry["front_lateral"] not in self.frontal_lateral_values:
                continue
            if one_indices and not any(
                [label_entry["labels"][label_index] == 1 for label_index in one_indices]
            ):
                continue
            if label_entry["sex"] not in self.sex_values:
                continue
            if not self.min_age <= label_entry["age"] <= self.max_age:
                continue
            if label_entry["ap_pa"] not in self.ap_pa_values:
                continue
            filtered_indices.append(image_index)
        return filtered_indices

    def init_label_filter(self, label_filter) -> Optional[List[int]]:
        """
        Create label filter from description.
        'competition': 5 classes evaluated in the CheXpert competition.
        'full': All 14 labels
        Other labels in order:
            - 0: No Finding
            - 1: Enlarged Cardiom.
            - 2: Cardiomegaly
            - 3: Lung Lesion
            - 4: Lung Opacity
            - 5: Edema
            - 6: Consolidation
            - 7: Pneumonia
            - 8: Atelectasis
            - 9: Pneumothorax
            - 10: Pleural Effusion
            - 11: Pleural Other
            - 12: Fracture
            - 13: Support Devices
        :param label_filter: None or 'full' or 'competition' or list of ints
        :return: List[int] Labels to stay in the dataset.
        """
        if label_filter == "full" or label_filter is None:
            return None
        if label_filter == "competition":
            return [2, 5, 6, 8, 10]
        if isinstance(label_filter, list) and isinstance(label_filter[0], int):
            return label_filter
        raise NotImplementedError

    def init_uncertain_mapping(
        self,
        uncertain_to_one: Optional[Union[str, List[str]]],
        uncertain_to_zero: Optional[Union[str, List[str]]],
    ):
        if uncertain_to_one is None:
            uncertain_to_one = []
        elif uncertain_to_one == "best":
            uncertain_to_one = ["Edema", "Atelectasis"]
        if uncertain_to_zero is None:
            uncertain_to_zero = []
        elif uncertain_to_zero == "best":
            uncertain_to_zero = ["Cardiomegaly", "Consolidation", "Pleural Effusion"]
        return uncertain_to_one, uncertain_to_zero

    def plot_label_stats(self):
        """
        Plot stats on the distribution of labels and image metadata to the
        command line.
        :return: None
        """
        image_paths = [self.image_index_to_path[i] for i in self.dataset_indices]
        label_data = [self.label_data[image_path] for image_path in image_paths]

        imratio_message = (
            f'\n\t{"="*10} SPLIT {self.split} {"=" * 10}:\n'
            f"\tTotal images in split:  {len(self):,}\n"
            f"\tTotal imratio in split: {self.imratio:.1%}.\n"
        )

        max_label_length = max([len(label) for label in self.imratios.keys()])
        for label, ratios in self.imratios.items():
            imratio_message += f"\t{label: <{max_label_length + 1}}: {ratios[0]:>7,} positive, {ratios[1]:>7,} negative, {ratios[2]:>7.1%} imratio.\n"
        logger.info(imratio_message)

        all_labels = self.all_labels()
        try:
            text_histogram(
                all_labels, title=f"distribution of all labels, split: {self.split}"
            )
        except Exception as e:
            logger.info(f"Can not plot stats on data labels: {str(e)}")

        all_sex_values = [data["sex"] for data in label_data]
        try:
            text_histogram(
                all_sex_values, title=f"distribution of sex values, split: {self.split}"
            )
        except Exception as e:
            logger.info(f"Can not plot stats on sex values: {str(e)}")

        all_age_values = [data["age"] for data in label_data]
        try:
            text_histogram(
                all_age_values, title=f"distribution of age values, split: {self.split}"
            )
        except Exception as e:
            logger.info(f"Can not plot stats on age values: {str(e)}")

        all_front_lateral_values = [data["front_lateral"] for data in label_data]
        try:
            text_histogram(
                all_front_lateral_values,
                title=f"distribution of front_lateral values, split: {self.split}",
            )
        except Exception as e:
            logger.info(f"Can not plot stats on front_lateral values: {str(e)}")

        all_ap_pa_values = [data["ap_pa"] for data in label_data]
        try:
            text_histogram(
                all_ap_pa_values,
                title=f"distribution of ap_pa values, split: {self.split}",
            )
        except Exception as e:
            logger.info(f"Can not plot stats on ap_pa values: {str(e)}")

    def cache_data(self, cache_path):
        """
        Function called by the parent class BaseDataset.
        Calls function convert_dataset_to_h5 to cache the images to disk.
        Also writes a config to the same directory.
        :param cache_path: Path to cache file.
        :return:
        """
        os.makedirs(os.path.join(self.data_path, "cache"), exist_ok=True)
        convert_dataset_to_h5(
            dataset=self, h5_image_path=cache_path, h5_label_path=None, batch_size=1000
        )
        json.dump(self.config, open(cache_path + ".json", "w"))

    @property
    def config(self) -> Dict:
        """
        Creates a config dict containing the datapath, the split,
        the config of the transform and the max size.
        All other kwargs in the init are related to labels and to not need
        to be considered for caching the images.
        :return: Dict Class Config
        """
        return {
            "chexpert_path": self.data_path,
            "split": self.config_split,
            "transform": self.transform.config,
            "max_size": self.max_size,
        }

    @property
    def config_split(self):
        if self.splits == "train_valid_test":
            if self.split in ["train", "valid"]:
                return "train"
            return "valid"
        return self.split

    def _getitem(self, index: DatasetIndex):
        """
        Maps the index from [0, len(self.filtered_indices)] onto
        the corresponding index in [0, len(self.image_paths)].
        Then proceeds to load the image and fix the number of color
        channels if there is a mismatch.
        :param index: int Index in [0, len(self.filtered_indices)]
        :return:
        """
        labels = self.get_labels(index)
        image = self.load_item(index)
        return image, labels

    def get_labels(self, index):
        image_path = self.image_index_to_path[index]
        labels = self.label_data[image_path]["labels"]
        labels = self.convert_labels_live(labels)
        return labels

    def load_item(self, index: ImageIndex):
        """
        Function to load an item from raw png
        :param index:
        :return:
        """
        image_path = self.image_index_to_path[index]
        image = self.read_image(image_path)
        image = torch.tensor(image)
        image = self.fix_image_channels(image)
        return image

    def fix_image_channels(self, image):
        image_channels = image.shape[-3]
        if image_channels == 1 and self.channels == 3:
            image = torch.cat(self.channels * [image], dim=-3)
        if image_channels == 3 and self.channels == 1:
            image = image[0].unsqueeze(0)
        return image

    @property
    def imratio(self):
        """
        Ratio of all 1.0 labels to all labels.
        Used for AUC optimization.
        :return: float Ratio
        """
        all_labels = self.all_labels()
        return np.sum(all_labels) / len(all_labels)

    @property
    def imratios(self) -> Dict:
        imratios: Dict[str, Tuple[int, int, float]] = {}
        all_datas = [
            self.label_data[self.image_index_to_path[index]]
            for index in self.dataset_indices
        ]
        for index, label in enumerate(self.index_to_label):
            values = [data["labels"][index].item() for data in all_datas]
            counts = Counter(values)
            positive = counts[1.0]
            negative = counts[0.0]
            imratios[label] = (positive, negative, positive / (positive + negative))
        return imratios

    @property
    def label_dim(self) -> Optional[int]:
        """
        Number of labels in the dataset.
        Default CheXpert 14, competition mode 5.
        :return: int Number of labels
        """
        return self[0][1].shape[-1]

    @property
    def image_dim(self):
        """
        Dimension (square) of image.
        For models pretrained on imagenet usually 224.
        :return:
        """
        return self[0][0].shape[1]

    def read_label_csv(self, file) -> Tuple[Dict, List[str]]:
        """
        Read CheXpert label csv.
        :param file: Path to .csv
        :return: Tuple[Dict, List[str]] Label data dictionary and list of label names.
        """
        data: Dict[ImagePath, Dict] = {}
        with open(file, "r") as f:
            reader = csv.reader(f)
            label_names = next(reader)[5:]
            if self.label_filter is not None and self.one_labels_in_class:
                label_names = [label_names[i] for i in self.label_filter]
            logger.info(f"Found labels in {file}: {label_names}")
            for index, row in tqdm(
                enumerate(reader), desc=f"Reading label csv file {file}"
            ):
                image_path = self.absolute_path_from_relative(row[0], self.data_path)
                data[image_path] = self.read_row(row)
                if self.max_size is not None and index > self.max_size:
                    break
        return data, label_names

    def read_row(self, row) -> Dict[str, Any]:
        """
        Read a single row in the label csv. Convert the labels.
        :param row: List[str]
        :return: Dict
        """
        return {
            "sex": row[1],
            "age": int(row[2]),
            "front_lateral": row[3],
            "ap_pa": row[4],
            "labels": self.convert_labels_initial(row[5:]),
        }

    def all_labels(self):
        """
        Get all labels in the entire dataset as a flattened list.
        :return: List[float] All label values.
        """
        all_datas = [
            self.label_data[self.image_index_to_path[index]]
            for index in self.dataset_indices
        ]
        all_labels = [
            label
            for data in all_datas
            for label in self.convert_labels_live(data["labels"]).tolist()
        ]
        return all_labels

    def convert_labels_initial(
        self, labels: List[str]
    ) -> Union[List[float], torch.Tensor]:
        """
        Label conversion while reading the label file.
        Only done once before training. No random transformations here.

        Possible labels:
        '1.0': positive
        '0.0': negative
        '-1.0': uncertain
        '': no mention

        :param labels: Labels from row of .csv file. As strings.
        :return: torch.Tensor or list of floats. Initially converted labels.
        """
        convert = {"1.0": 1.0, "0.0": 0.0, "": 0.0, "-1.0": -1.0}
        labels = torch.FloatTensor([convert[x] for x in labels])

        if self.one_labels_in_class:
            if self.label_filter is not None:
                labels = labels[self.label_filter]
        return labels

    def convert_labels_live(
        self, labels: Union[List[float], torch.Tensor]
    ) -> torch.Tensor:
        """
        Label conversion during training.
        Called every time _getitem is called. Should be used for random conversion,
        so that the datapoint is converted differently every epoch.

        :param labels: Labels converted in the first step.
        :return: torch.Tensor or list of floats. Labels to be passed to training.
        """

        assert len(labels) == len(self.index_to_label)
        labels = torch.tensor(
            [
                self.convert_uncertain_labels(label, self.index_to_label[i])
                for i, label in enumerate(labels)
            ]
        )
        return labels

    def convert_uncertain_labels(self, label: float, label_name: str) -> float:
        """
        Convert uncertain labels. Either by the uncertain_to_one or uncertain_to_zero lists,
        random mapping onto an interval or mapping onto a single value.
        :param label:
        :param label_name:
        :return:
        """
        if label != -1.0:
            return label

        if label_name in self.uncertain_to_zero:
            return 0.0
        if label_name in self.uncertain_to_one:
            return 1.0
        if self.uncertainty_upper_bound != self.uncertainty_lower_bound:
            return random.uniform(
                self.uncertainty_lower_bound, self.uncertainty_upper_bound
            )
        return self.uncertainty_lower_bound

    def absolute_path_from_relative(self, relative_path, chexpert_path):
        """
        Helper function to map image paths onto the absolute path on the file system.
        """
        return os.path.abspath(os.path.join(chexpert_path, "..", relative_path))

    def read_image(self, image_path):
        """
        Open and transform an image from a given path.
        :param image_path: Path to image.jpg
        :return: Array with image.
        """
        img = Image.open(os.path.join(image_path))
        if self.transform is not None:
            img = self.transform(img)
        return img

    @property
    def default_transform(self):
        """
        When not given another transform in init, an image is only converted
        to a tensor after reading from disk.
        :return:
        """
        return BaseTransform()


class StudyDataset(CheXpertDataset):
    def __init__(self, *args, upsample_study, **kwargs):
        self.upsample_study = upsample_study
        self.studies = None

        self.studies_split = kwargs.get("split")
        self.studies_splits = kwargs.get("splits")

        if self.studies_splits == "train_valid_test" and self.studies_split in [
            "train",
            "valid",
        ]:
            kwargs["split"] = "train"
            kwargs["splits"] = "train_valid"

        super().__init__(*args, **kwargs)

    @property
    def index_mapping(self) -> Dict[DatasetIndex, List[ImageIndex]]:
        if self._index_mapping is None:
            self._index_mapping = {}
            for dataset_index, study in enumerate(self.studies):
                self._index_mapping[dataset_index] = [
                    self.path_to_image_index[path] for path in study
                ]
        return self._index_mapping

    def read_data(self) -> Tuple[Dict[ImagePath, Dict], List[str]]:
        if self.splits == "train_valid_test":
            label_data, index_to_label = self.read_data_train_valid_test()
        else:
            label_data, index_to_label = self.read_data_train_valid()
        self.studies = self.studies_from_paths(list(label_data.keys()))
        self.studies = self.split_studies(self.studies)
        if self.upsample_study:
            self.studies = self.upsample_studies(self.studies)
        # parent class will not override split_keys because we set their self.splits to 'train_valid'
        self.split_keys = set([path for study in self.studies for path in study])
        return label_data, index_to_label

    def studies_from_paths(self, image_paths) -> List[List[ImagePath]]:
        patients = self.patients_from_paths(image_paths)
        studies: List[List[ImagePath]] = []
        for patient in patients:
            for study_index, study in enumerate(patient.studies):
                studies.append(study.images)
        return studies

    def split_studies(self, studies, valid_size=0.1, seed=1):
        if self.studies_splits == "train_valid":
            return studies
        if self.studies_split == "test":
            return studies
        train_indices, valid_indices = train_test_split(
            list(range(len(studies))),
            test_size=valid_size,
            random_state=seed,
            shuffle=True,
        )
        indices = train_indices if self.studies_split == "train" else valid_indices
        return [studies[i] for i in indices]

    def upsample_studies(self, studies: List[Study]) -> List[Study]:
        upsampled_studies = studies.copy()
        for study in studies:
            for img in study:
                if img[-11:] == "lateral.jpg":
                    upsampled_studies.append(study)
                    break
        return upsampled_studies

    def patients_from_paths(self, paths) -> List[Patient]:
        """
        From a list of image paths, create Patient object.
        Sort each image path by study and patient and merge into Patient objects
        """
        patient_studies_to_paths = defaultdict(list)
        for path in paths:
            patient_study = "/".join(self.split_path(path, self.split))
            patient_studies_to_paths[patient_study].append(path)
        patients_to_studies = defaultdict(list)
        for patient_study, paths in patient_studies_to_paths.items():
            patient, study = patient_study.split("/")
            patients_to_studies[patient].append(
                Study(
                    name=study,
                    images=sorted(paths, key=lambda x: self.study_number_from_path(x)),
                )
            )
        patients = []
        for patient, studies in patients_to_studies.items():
            patients.append(
                Patient(name=patient, studies=sorted(studies, key=lambda x: x.number))
            )
        return patients

    def study_number_from_path(self, path):
        return int(path.split("study")[-1].split("/")[0])

    @staticmethod
    def split_path(path: str, split: str = "train") -> Tuple[str, str]:
        if split == "test":
            split = "valid"
        _, path = path.split(split)
        _, patient, study, filename = path.split("/")
        return patient, study

    def __len__(self):
        if self.studies is None:
            return super().__len__()
        return len(self.studies)

    def _getitem(self, index):
        labels = self.get_labels(index)
        image = self.load_item(index)
        metadata = self.get_metadata(index)
        return image, labels, metadata

    def get_metadata(self, index):
        image_path = self.image_index_to_path[index]
        data = self.label_data[image_path]
        frontal_lateral = 0 if data["front_lateral"] == "Frontal" else 1
        ap_pa = 0 if data["ap_pa"] == "AP" else 1
        metadata = torch.tensor([frontal_lateral, ap_pa])
        return metadata


def convert_chexpert_to_h5(
    chexpert_path,
    h5_image_path,
    h5_label_path: str = None,
    split: str = "train",
    transform=None,
    max_size: int = None,
    batch_size: int = 1000,
) -> None:
    dataset = CheXpertDataset(
        data_path=chexpert_path, transform=transform, max_size=max_size, split=split
    )
    convert_dataset_to_h5(dataset, h5_image_path, h5_label_path, batch_size)


def convert_dataset_to_h5(
    dataset, h5_image_path, h5_label_path, batch_size: int = 1000
):
    logger.info(f"Starting cache process for file {h5_image_path}.")
    if h5_label_path is not None:
        logger.info(f"Writing labels into file {h5_label_path}.")
    logger.info(f"Size of dataset: {len(dataset)} images.")
    logger.info(
        f"Size of images: {(dataset.channels, dataset.image_dim, dataset.image_dim)}"
    )
    logger.info(
        f"Writing {len(dataset) // batch_size + 1} batches of {batch_size} images each."
    )

    f_images = h5py.File(h5_image_path, "w")
    if h5_label_path is not None:
        f_labels = h5py.File(h5_label_path, "w")
    else:
        f_labels = None
    for i in tqdm(range(0, len(dataset), batch_size)):
        indices = range(i, min(len(dataset), i + batch_size))
        data_batch = [dataset[j] for j in indices]
        #images = torch.stack([item[0] for item in data_batch])
        #dirty fix SN, sometimes item is not tuple(im,label) but list [tuple(im,label)]
        images = torch.stack([item[0][0] if isinstance(item, list) else item[0] for item in data_batch])
        
        if i == 0:
            print(images.shape)
            print(dataset.image_dim)
            f_images.create_dataset(
                "data",
                data=images,
                chunks=True,
                maxshape=(None,)
                + (dataset.channels, dataset.image_dim, dataset.image_dim),
            )
        else:
            f_images["data"].resize(f_images["data"].shape[0] + images.shape[0], axis=0)
            f_images["data"][-images.shape[0] :] = images

        if f_labels is not None:
            # labels = torch.stack([item[1] for item in data_batch])
            #dirty fix SN, sometimes item is not tuple(im,label) but list [tuple(im,label)]
            labels = torch.stack([item[0][1] if isinstance(item, list) else item[1] for item in data_batch])
            if i == 0:
                f_labels.create_dataset(
                    "data",
                    data=labels,
                    chunks=True,
                    maxshape=(None,) + (dataset.label_dim,),
                )
            else:
                f_labels["data"].resize(
                    (f_labels["data"].shape[0] + labels.shape[0]), axis=0
                )
                f_labels["data"][-labels.shape[0] :] = labels

        tqdm_writer.write(
            f'Wrote {len(indices)} items to file. Size of saved dataset: {f_images["data"].shape, f_labels["data"].shape if f_labels is not None else ""}'
        )
    f_images.close()
    if f_labels is not None:
        f_labels.close()


def main():
    pass


if __name__ == "__main__":
    main()
