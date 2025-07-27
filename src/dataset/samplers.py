import random
from collections import defaultdict
from typing import Any, Generator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler
from torch.utils.data import WeightedRandomSampler
from loguru import logger

class CategoryBalancedSampler(Sampler):

    def __init__(
        self, data: pd.DataFrame, sample_column: str, batch_size: int, seed: int = 42
    ):
        """
        Custom batch sampler to ensure each batch has close to even distribution based on some category.

        :param data: Dataset to sample from.
        :param sample_column: Name of the column in `data` which is used for batch sampling.
        :param batch_size: Batch size.
        :param seed: Seed for the sampler.
        """
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.data = data
        self.batch_size = batch_size
        self.category_to_indices = defaultdict(list)
        self.sample_column = sample_column
        self._n_of_observations = 0

        # Group dataset indices by category
        # Note, that indices in data should be unique and reset
        if len(np.unique(data.index)) != data.shape[0]:
            raise ValueError(f"DataFrame index should be unique")

        for idx, row in data.iterrows():
            self.category_to_indices[row[self.sample_column]].append(idx)
            self._n_of_observations += 1

        # Ensure all categories have the same number of samples per batch
        self.num_categories = len(self.category_to_indices)
        self.samples_per_category = self.batch_size // self.num_categories

        # If batch size is not divisible by number of categories, raise an error
        if self.batch_size % self.num_categories != 0:
            raise ValueError(
                f"Batch size must be divisible by the number of categories ({self.num_categories})"
            )

        # Create a list of category labels
        self.category_labels = list(self.category_to_indices.keys())

        self.indices_per_category = {
            label: self.category_to_indices[label] for label in self.category_labels
        }
        self._n_of_batches_sent = 0

    def shuffle_data(self) -> None:
        """
        Shuffle indices within each category
        :return: None
        """
        for k, _ in self.indices_per_category.items():
            np.random.shuffle(self.indices_per_category[k])

    def __iter__(self) -> Generator[list[Any], Any, None]:
        batch = []
        while True:
            self.shuffle_data()
            # Create one batch per category
            for label in self.category_labels:
                selected_cat_indices = self.indices_per_category[label][
                    : self.samples_per_category
                ]
                batch.extend(selected_cat_indices)

            self._n_of_batches_sent += 1
            yield batch
            batch = []

            # We stop if the number of created batches equals to the number of calculated batches
            # It means that some data for underrepresented categories may occur more often during one epoch
            if self._n_of_batches_sent == self.__len__():
                self._n_of_batches_sent = 0
                break

    def __len__(self) -> int:
        return self._n_of_observations // self.batch_size


SAMPLERS = {
    "weighted": WeightedRandomSampler,
    "cbalanced": CategoryBalancedSampler,
}
