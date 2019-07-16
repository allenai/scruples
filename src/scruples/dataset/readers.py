"""Dataset readers for scruples."""

import json
import os
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple)

import pandas as pd
from torch.utils.data import Dataset

from .. import settings


# main classes

class ScruplesCorpus:
    """A class for reading the scruples corpus for sklearn.

    Attributes
    ----------
    SPLITS : List[str]
        A constant listing the names of the dataset's splits.
    train : Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]
        A tuple of the form ``(ids, features, labels, label_scores)``
        containing the training data. ``ids`` is a pandas ``Series``
        with the ID of each data point, ``features`` is a pandas
        ``DataFrame`` with the title and text of each data point, and
        ``labels`` is a pandas ``Series`` with the label of each data
        point, and ``label_scores`` is a pandas ``DataFrame`` with a
        column for each label and integers in the column for the number
        of comments that expressed that label.
    dev : Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]
        A tuple of the form ``(ids, features, labels, label_scores)``
        containing the dev data. ``ids`` is a pandas ``Series`` with the
        ID of each data point, ``features`` is a pandas ``DataFrame``
        with the title and text of each data point, ``labels`` is a
        pandas ``Series`` with the label of each data point, and
        ``label_scores`` is a pandas ``DataFrame`` with a column for
        each label and integers in the column for the number of comments
        that expressed that label.
    test : Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]
        A tuple of the form ``(ids, features, labels, label_scores)``
        containing the test data. ``ids`` is a pandas ``Series`` with
        the ID of each data point, ``features`` is a pandas
        ``DataFrame`` with the title and text of each data point,
        ``labels`` is a pandas ``Series`` with the label of each data
        point and ``label_scores`` is a pandas ``DataFrame`` with a
        column for each label and integers in the column for the number
        of comments that expressed that label.

    See `Parameters`_ for more attributes.

    Parameters
    ----------
    data_dir : str, required
        The directory in which the dataset is stored.
    """
    SPLITS = [split['name'] for split in settings.SPLITS]

    def __init__(
            self,
            data_dir: str
    ) -> None:
        super().__init__()

        self.data_dir = data_dir

        # read split data and bind it to the instance
        for split in self.SPLITS:
            split_path = os.path.join(
                data_dir,
                settings.CORPUS_FILENAME_TEMPLATE.format(split=split))
            split_data = pd.read_json(split_path, lines=True)

            ids_features_labels_and_label_scores = (
                split_data['id'],
                split_data[['title', 'text']],
                split_data['label'],
                pd.DataFrame(split_data['label_scores'].tolist())
            )

            setattr(self, split, ids_features_labels_and_label_scores)


class ScruplesCorpusDataset(Dataset):
    """A PyTorch ``Dataset`` class for the scruples corpus.

    Iterating through this dataset returns ``(id, feature, label,
    label_scores)`` tuples.

    Attributes
    ----------
    SPLITS : List[str]
        A constant listing the names of the dataset's splits.
    ids : List[str]
        A list of the instance IDs for the split.
    features : List[Tuple[str, str]]
        A list of ``(title, text)`` tuples for the instances in the
        corpus.
    labels : List[str]
        A list of the labels for the instances in the corpus.
    label_scores : List[Dict[str, int]]
        A list of dictionaries containing the label scores for the
        instances in the corpus.

    Parameters
    ----------
    data_dir : str, required
        The directory containing the scruples corpus.
    split : str, required
        The split to read into the class. Must be one of ``"train"``,
        ``"dev"``, or ``"test"``.
    transform : Optional[Callable], optional (default=None)
        A transformation to apply to the title, text string
        tuples. If ``None``, no transformation is applied.
    label_transform : Optional[Callable], optional (default=None)
        A transformation to apply to the labels. The labels are passed
        in as strings ("AUTHOR", "OTHER", "EVERYBODY", "NOBODY", and
        "INFO"). If ``None``, no transformation is applied.
    label_scores_transform : Optional[Callable], optional (default=None)
        A transformation to apply to the label scores. The label scores
        are passed in as a dictionary mapping the label strings
        ("AUTHOR", "OTHER", "EVERYBODY", "NOBODY", "INFO") to the number
        of comments that expressed that label. If ``None``, no
        transformation is applied.
    """
    SPLITS = [split['name'] for split in settings.SPLITS]

    def __init__(
            self,
            data_dir: str,
            split: str,
            transform: Optional[Callable] = None,
            label_transform: Optional[Callable] = None,
            label_scores_transform: Optional[Callable] = None
    ) -> None:
        super().__init__()

        if split not in self.SPLITS:
            raise ValueError(
                f'split must be one of {", ".join(self.SPLITS)}.')

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.label_transform = label_transform
        self.label_scores_transform = label_scores_transform

        self.ids, self.features, self.labels, self.label_scores =\
            self._read_data()

    def _read_data(self) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        """Return the instance ids, features, labels and label scores.

        Read in the dataset files from disk, and return the instance ids
        as a list of strings, the features as a list of
        ``(title, text)`` string pairs, the labels as a list of strings,
        and the label scores as a list of dictionaries.

        Returns
        -------
        List[str]
            The IDs for each instance in the dataset.
        List[Tuple[str, str]]
            A list of the pairs of strings containing the title and
            text for each dataset instance.
        List[Optional[str]]
            The labels for the instances, if the labels are available,
            otherwise each label is represented as ``None``.
        List[Optional[Dict[str, int]]]
            The scores for each label for the instances, if the labels
            scores are available, otherwise each label scores object is
            represented as ``None``.
        """
        ids, features, labels, label_scores = [], [], [], []

        split_path = os.path.join(
            self.data_dir,
            settings.CORPUS_FILENAME_TEMPLATE.format(split=self.split))
        with open(split_path, 'r') as split_file:
            for ln in split_file:
                row = json.loads(ln)
                ids.append(row['id'])
                features.append((row['title'], row['text']))
                labels.append(row.get('label'))
                label_scores.append(row.get('label_scores'))

        return ids, features, labels, label_scores

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, key: int) -> Tuple[str, Any, Any]:
        id_ = self.ids[key]
        feature = self.features[key]
        label = self.labels[key]
        label_scores = self.label_scores[key]

        if self.transform:
            feature = self.transform(feature)

        if self.label_transform:
            label = self.label_transform(label)

        if self.label_scores_transform:
            label_scores = self.label_scores_transform(label_scores)

        return id_, feature, label, label_scores


class ScruplesBenchmark:
    """A class for reading the scruples benchmark for sklearn.

    Attributes
    ----------
    SPLITS : List[str]
        A constant listing the names of the dataset's splits.
    train : Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]
        A tuple of the form ``(ids, features, labels, label_scores)``
        containing the training data. ``ids`` is a pandas ``Series``
        with the ID of each data point, ``features`` is a pandas
        ``DataFrame`` with the descriptions for both actions in the
        instance, ``labels`` is a pandas ``Series`` with the label of
        each instance, and ``label_scores`` is a pandas ``DataFrame``
        with a column for each label and integers in the column for the
        number of annotators that chose that label.
    dev : Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]
        A tuple of the form ``(ids, features, labels, label_scores)``
        containing the dev data. ``ids`` is a pandas ``Series`` with the
        ID of each data point, ``features`` is a pandas ``DataFrame``
        with the descriptions for both actions in the instance,
        ``labels`` is a pandas ``Series`` with the label of each
        instance, and ``label_scores`` is a pandas ``DataFrame`` with a
        column for each label and integers in the column for the number
        of annotators that chose that label.
    test : Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]
        A tuple of the form ``(ids, features, labels, label_scores)``
        containing the test data. ``ids`` is a pandas ``Series`` with
        the ID of each data point, ``features`` is a pandas
        ``DataFrame`` with the descriptions for both actions in the
        instance, ``labels`` is a pandas ``Series`` with the label of
        each instance and ``label_scores`` is a pandas ``DataFrame``
        with a column for each label and integers in the column for the
        number of annotators that chose that label.

    See `Parameters`_ for more attributes.

    Parameters
    ----------
    data_dir : str, required
        The directory in which the dataset is stored.
    """
    SPLITS = [split['name'] for split in settings.SPLITS]

    def __init__(
            self,
            data_dir: str
    ) -> None:
        super().__init__()

        self.data_dir = data_dir

        # read split data and bind it to the instance
        for split in self.SPLITS:
            split_path = os.path.join(
                data_dir,
                settings.BENCHMARK_FILENAME_TEMPLATE.format(split=split))
            rows = []
            with open(split_path, 'r') as split_file:
                for ln in split_file:
                    row = json.loads(ln)
                    rows.append({
                        'id': row['id'],
                        'action0': row['actions'][0]['description'],
                        'action1': row['actions'][1]['description'],
                        'label': row['gold_label'],
                        'label_scores': row['gold_annotations']
                    })
            split_data = pd.DataFrame(rows)

            ids_features_labels_and_label_scores = (
                split_data['id'],
                split_data[['action0', 'action1']],
                split_data['label'],
                pd.DataFrame(split_data['label_scores'].tolist())
            )

            setattr(self, split, ids_features_labels_and_label_scores)


class ScruplesBenchmarkDataset(Dataset):
    """A PyTorch ``Dataset`` class for the scruples benchmark.

    Iterating through this dataset returns ``(id, feature, label,
    label_scores)`` tuples.

    Attributes
    ----------
    SPLITS : List[str]
        A constant listing the names of the dataset's splits.
    ids : List[str]
        A list of the instance IDs for the split.
    features : List[Tuple[str, str]]
        A list of ``(action0, action1)`` tuples for the instances in the
        benchmark.
    labels : List[int]
        A list of the labels for the instances in the benchmark.
    label_scores : List[List[int]]
        A list of lists containing the label scores for the
        instances in the benchmark.

    Parameters
    ----------
    data_dir : str, required
        The directory containing the scruples benchmark.
    split : str, required
        The split to read into the class. Must be one of ``"train"``,
        ``"dev"``, or ``"test"``.
    transform : Optional[Callable], optional (default=None)
        A transformation to apply to the action description string
        tuples. If ``None``, no transformation is applied.
    label_transform : Optional[Callable], optional (default=None)
        A transformation to apply to the labels. The possible labels are
        ``0`` and ``1`` for the first or second action being more
        pronormative, respectively. If ``None``, no transformation is
        applied.
    label_scores_transform : Optional[Callable], optional (default=None)
        A transform to apply to the label scores. The label scores are
        passed in as an array of two integers, each being a count of the
        number of annotators that said the corresponding action was the
        correct answer.
    """
    SPLITS = [split['name'] for split in settings.SPLITS]

    def __init__(
            self,
            data_dir: str,
            split: str,
            transform: Optional[Callable] = None,
            label_transform: Optional[Callable] = None,
            label_scores_transform: Optional[Callable] = None
    ) -> None:
        super().__init__()

        if split not in self.SPLITS:
            raise ValueError(
                f'split must be one of {", ".join(self.SPLITS)}.')

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.label_transform = label_transform
        self.label_scores_transform = label_scores_transform

        self.ids, self.features, self.labels, self.label_scores =\
            self._read_data()

    def _read_data(self) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        """Return the instance ids, features, labels, and label scores.

        Read in the dataset files from disk, and return the instance ids
        as a list of strings, the features as a list of pairs of the
        actions' descriptions, the labels as a list of integers, and the
        label scores as a list of lists of ints.

        Returns
        -------
        List[str]
            The IDs for each instance in the dataset.
        List[Tuple[str, str]]
            A list of pairs of strings containing the two action
            descriptions for each dataset instance.
        List[Optional[str]]
            The labels for the instances, if the labels are available,
            otherwise each label is represented as ``None``.
        List[Optional[List[int]]]
            The label scores for the instances, if the label scores are
            available, otherwise each label score is represented as
            ``None``.
        """
        ids, features, labels, label_scores = [], [], [], []

        split_path = os.path.join(
            self.data_dir,
            settings.BENCHMARK_FILENAME_TEMPLATE.format(split=self.split))
        with open(split_path, 'r') as split_file:
            for ln in split_file:
                row = json.loads(ln)
                ids.append(row['id'])
                features.append((
                    row['actions'][0]['description'],
                    row['actions'][1]['description']
                ))
                labels.append(row.get('gold_label'))
                label_scores.append(row.get('gold_annotations'))

        return ids, features, labels, label_scores

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, key: int) -> Tuple[str, Any, Any]:
        id_ = self.ids[key]
        feature = self.features[key]
        label = self.labels[key]
        label_scores = self.label_scores[key]

        if self.transform:
            feature = self.transform(feature)

        if self.label_transform:
            label = self.label_transform(label)

        if self.label_scores_transform:
            label_scores = self.label_scores_transform(label_scores)

        return id_, feature, label, label_scores
