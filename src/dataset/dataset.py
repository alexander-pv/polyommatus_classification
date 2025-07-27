import copy
import random
from abc import abstractmethod, ABC
from collections import Counter
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset


def split_balanced(
        df: pd.DataFrame, train: float, test: float, val: float, seed: int
) -> dict:
    """
    Split dataset in balanced manner.
    It may be several objects per ID, so the task is also to prevent data leak of the same object into different subsets.
    :param df:           Dataframe for balanced split
    :param train:        Train fraction
    :param test:         Test fraction
    :param val:          Val fraction
    :param seed:         Random seed
    :return: dict with data subsets: train, test, val, all
    """
    random.seed(seed)
    np.random.seed(seed)

    if train + test + val != 1:
        raise ValueError(f"Incorrect train, test, val fractions: {train, test, val}")

    ids_dict = {}

    class_names = sorted(df["class"].unique().tolist())
    df_unique_object_ids = df.groupby(by=["class"])["id"].unique().reset_index()
    df_unique_object_ids["count"] = df_unique_object_ids["id"].apply(lambda x: len(x))

    for c in class_names:
        class_object_ds = copy.copy(
            df_unique_object_ids[df_unique_object_ids["class"] == c]["id"]
            .iloc[0]
            .tolist()
        )
        np.random.shuffle(class_object_ds)
        ids_dict.update({c: class_object_ds})

    splitted_samples = dict()
    ids_dict = dict(sorted(ids_dict.items(), key=lambda x: len(x[1])))
    for c, id_list in ids_dict.items():
        n_of_ids = len(id_list)

        train_ids = id_list[0: int(n_of_ids * train)]
        val_ids = id_list[
                  int(n_of_ids * train): int(n_of_ids * train) + int(n_of_ids * val)
                  ]
        test_ids = id_list[int(n_of_ids * train) + int(n_of_ids * test):]

        if len(train_ids) == 0:
            logger.warning(
                f"Can't  gather train samples for class {c}. Review data preparation."
            )

        if len(val_ids) == 0 or len(test_ids) == 0:
            logger.info(
                f"Can't split dataset for class {c}. "
                f"Number of samples: {n_of_ids}. Applying simple split into 3 chunks."
            )
            train_ids, val_ids, test_ids = np.array_split(id_list, 3)
            train_ids = train_ids.tolist()
            val_ids = val_ids.tolist()
            test_ids = test_ids.tolist()

        splitted_samples.update(
            {c: {"train": train_ids, "val": val_ids, "test": test_ids}}
        )

    subsets = {"train": [], "test": [], "val": [], "testval": []}
    for c in class_names:
        for subset_name in ("train", "test", "val"):
            if len(splitted_samples[c][subset_name]) == 0:
                logger.warning(f"Subset {subset_name} has no samples for class {c}")
            subsets[subset_name].extend(splitted_samples[c][subset_name])
    subsets["testval"] = [*subsets["test"], *subsets["val"]]
    subsets["all"] = [*subsets["train"], *subsets["testval"]]

    logger.info(f"Subset train: {len(subsets['train'])}")
    logger.info(f"Subset test: {len(subsets['test'])}")
    logger.info(f"Subset val: {len(subsets['val'])}")
    logger.info(f"Subset testval: {len(subsets['testval'])}")
    logger.info(f"Subset all: {len(subsets['all'])}")
    return subsets


class LycaenidaeDataset(Dataset, ABC):
    def __init__(
            self,
            metadata_path: str,
            view: Literal["top", "bottom"],
            subset: Literal["train", "val", "test", "testval", "all"],
            min_images_per_class: int,
            train_size: float,
            val_size: float,
            test_size: float,
            seed: int,
            transform: callable or None = None,
            model_transform: callable or None = None,
            device: str or torch.device = torch.device("cpu"),
    ):
        """
        Lycaenidae dataset for torch framework
        :param metadata_path:         The path to the metadata dataframe file
        :param view:                  Lycaenidae view: top or bottom
        :param subset:                Subset name: train/val/test/testval
        :param min_images_per_class:  Minimum images per class, classes with less than min_images_per_class are dropped
        :param train_size:            The fraction of initial data for train
        :param val_size:              The fraction of initial data for val
        :param test_size:             The fraction of initial data for test
        :param seed:                  Random seed
        :param transform:             The image transformation pipe for augmentation
        :param model_transform:       The image transformation pipe for the specific pretrained model
        :param device:                Computing device
        """
        super().__init__()
        self.df_metadata = pd.read_csv(metadata_path)
        self.view = view
        self.subset = subset
        self.min_images_per_class = min_images_per_class
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

        self.transform = transform
        self.model_transform = model_transform
        self.device = device
        self._logger_info = {}
        logger.warning(f"Torch device: {self.device}")
        logger.warning(f"Subset: {subset}. View: {view}")
        self.df_subset_metadata = self.prepare_subset()
        self.class_labels = {
            name: i
            for i, name in enumerate(sorted(self.df_subset_metadata["class"].unique()))
        }
        self.class_labels_inv = {i: name for name, i in self.class_labels.items()}
        logger.warning(f"[{self.subset}]Class labels: {self.class_labels}")
        self.n_classes = len(self.class_labels.keys())

    def name_to_label(self, name: str) -> int:
        return self.class_labels[name]

    def label_to_name(self, label: int) -> str:
        return self.class_labels_inv[label]

    def get_subset_ids(self, df: pd.DataFrame) -> list:
        """
        :param df:  metadata table
        :return: list of subset ids
        """
        subsets = split_balanced(
            df=df,
            train=self.train_size,
            val=self.val_size,
            test=self.test_size,
            seed=self.seed,
        )
        return subsets[self.subset]

    def get_class_weights(self) -> dict:
        """
        Prepare a dictionary of class weights used for batch sampling.
        :return: dict of class weights
        """
        class_counter = Counter(self.df_subset_metadata["class"].values)
        dataset_size = self.__len__()
        class_weights = {}
        for class_label, count in class_counter.items():
            class_weights.update({class_label: 1 - count / dataset_size})
        return class_weights

    def get_data_weights(self) -> np.ndarray:
        """
        Returns weights for all dataset. For custom batch sampler.
        :return: array of weights per each class
        """
        class_weights = self.get_class_weights()
        class_labels = self.df_subset_metadata["class"].values
        data_weights = np.array([class_weights[label] for label in class_labels])
        return data_weights

    def prepare_subset(self):
        """
        Each specimen can have a, b, and c views.
        Thus, we must prepare subsets so that they do not contain the same image from different views.
        :return:
        """
        # Take specific view
        df_metadata_view = self.df_metadata[self.df_metadata["view"] == self.view]

        # Filter underrepresented classes
        # Note, that we can have two images per same observation.
        # That is why filtration uses deduplication to count observations
        df_deduplicated_obs = df_metadata_view[["id", "class"]].drop_duplicates()
        initial_classes = df_deduplicated_obs["class"].unique().tolist()
        initial_counter = Counter(df_deduplicated_obs["class"].values)
        self.final_counter = {
            name: count
            for name, count in initial_counter.items()
            if count >= self.min_images_per_class
        }
        self.final_classes = [x for x in self.final_counter.keys()]
        # Filter metadata table
        df_metadata_view = df_metadata_view[
            df_metadata_view["class"].isin(self.final_classes)
        ]
        # # Get necessary subset based on object IDs
        self.subset_object_ids = self.get_subset_ids(df_metadata_view)
        # Filter metadata table for specific object IDs
        df_metadata_view = df_metadata_view[
            df_metadata_view["id"].isin(self.subset_object_ids)
        ]
        df_metadata_view = df_metadata_view.reset_index(drop=True)

        logger.info(f"[{self.subset}]Initial classes: {initial_classes}")
        logger.info(f"[{self.subset}]Classes after filtration: {self.final_classes}")
        logger.info(
            f"[{self.subset}]Dropped classes: {set(initial_counter) - set(self.final_classes)}"
        )
        logger.info(
            f"[{self.subset}]Unique object IDs for {self.subset}: {len(self.subset_object_ids)}"
        )
        logger.info(
            f"[{self.subset}]Number of images for {self.subset}: {df_metadata_view.shape[0]}"
        )

        logger.info(
            f"[{self.subset}]Class counter for total dataset: {self.final_counter}"
        )
        logger.info(
            f"[{self.subset}]Class counter for {self.subset}: {Counter(df_metadata_view['class'])}"
        )
        self._logger_info.update(
            {
                f"[{self.subset}]Initial classes": initial_classes,
                f"[{self.subset}]Classes after filtration": self.final_classes,
                f"[{self.subset}]Dropped classes": set(initial_counter)
                                                   - set(self.final_classes),
                f"[{self.subset}]Unique object IDs for {self.subset}": len(
                    self.subset_object_ids
                ),
                f"[{self.subset}]Number of images for {self.subset}": df_metadata_view.shape[
                    0
                ],
            }
        )

        return df_metadata_view

    def __len__(self) -> int:
        return self.df_subset_metadata.shape[0]

    def _getitem(self, idx: int) -> tuple[np.ndarray, int, str]:
        """
        Get image by internal id without any transform
        :param idx:
        :return: image, label, img_path
        """
        sample = self.df_subset_metadata.iloc[idx]
        img_path, class_name = sample["filepath"], sample["class"]
        label = self.name_to_label(class_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, label, img_path

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        pass


class LycaenidaeDatasetCls(LycaenidaeDataset):

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label, _ = self._getitem(idx)
        if self.transform:
            img = self.transform(image=img)["image"]
        input_tensor = torch.as_tensor(img, dtype=torch.float32) / 255
        input_tensor = input_tensor.permute(
            (2, 0, 1)
        ).contiguous()  # Channel order HxWxC -> CxHxW
        input_tensor = self.model_transform(input_tensor)
        return input_tensor, label
