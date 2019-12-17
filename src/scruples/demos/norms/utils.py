"""Utilities for the ``norms`` demo."""

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple)

import torch


class PredictionDataset(torch.utils.data.Dataset):
    """A PyTorch ``Dataset`` class for prediction.

    Parameters
    ----------
    features : List[Any], required
        The list of features for the dataset's instances.
    transform : Optional[Callable], optional (default=None)
        The transformation to apply to the features. If ``None``, no
        transformation is applied.
    """
    def __init__(
            self,
            features: List[Any],
            transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.features = features
        self.transform = transform

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, key: int) -> Any:
        feature = self.features[key]

        if self.transform:
            feature = self.transform(feature)

        return feature
