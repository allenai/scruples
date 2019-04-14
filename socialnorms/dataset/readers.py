"""Dataset readers for socialnorms."""

import json
import os
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple)

from torch.utils.data import Dataset


class SocialNormsDataset(Dataset):
    """A PyTorch ``Dataset`` class for socialnorms.

    Iterating through this dataset returns ``(id, feature, label)``
    triples.

    Attributes
    ----------
    SPLITS : List[str]
        A constant listing the names of the dataset's splits.

    Parameters
    ----------
    data_dir : str, required
        The directory containing the socialnorms dataset.
    split : str, required
        The split to read into the class. Must be one of ``"train"``,
        ``"dev"``, or ``"test"``.
    transform : Optional[Callable], optional (default=None)
        A transformation to apply to the title, text string
        tuples. If ``None``, no transformation is applied.
    label_transform : Optional[Callable], optional (default=None)
        A transformation to apply to the labels. The labels are passed
        in as strings ("YTA", "NTA", "ESH", "NAH", and "INFO"). If
        ``None``, no transformation is applied.
    """
    SPLITS = ["train", "dev", "test"]

    def __init__(
            self,
            data_dir: str,
            split: str,
            transform: Optional[Callable] = None,
            label_transform: Optional[Callable] = None
    ) -> None:
        super().__init__()

        if split not in self.SPLITS:
            raise ValueError(
                f'split must be one of {", ".join(self.SPLITS)}.')

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.label_transform = label_transform

        self.ids, self.features, self.labels = self._read_data()

    def _read_data(self) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        """Return the instance ids, features and labels for the split.

        Read in the dataset files from disk, and return the instance ids
        as a list of strings, the features as a list of
        ``(title, text)`` string pairs and the labels as a list of strings.

        Returns
        -------
        List[str]
            The IDs for each instance in the dataset.
        List[Tuple[str, str]]
            A list of the pairs of strings containing the title and
            text for each dataset instance.
        List[str]
            The labels for the instances.
        """
        ids, features, labels = [], [], []

        split_path = os.path.join(self.data_dir, f'{self.split}.jsonl')
        with open(split_path, 'r') as split_file:
            for ln in split_file:
                row = json.loads(ln)
                ids.append(row['id'])
                features.append((row['title'], row['text']))
                labels.append(row.get('label'))

        return ids, features, labels

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, key: int) -> Tuple[str, Any, Any]:
        id_ = self.ids[key]
        feature = self.features[key]
        label = self.labels[key]

        if self.transform:
            feature = self.transform(feature)

        if self.label_transform:
            label = self.label_transform(label)

        return id_, feature, label
