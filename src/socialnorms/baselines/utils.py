"""Utilities for baselines on socialnorms."""

import numpy as np
import pandas as pd


# functions

def concat_title_and_text(features: pd.DataFrame) -> np.ndarray:
    """Return the concatenation of the title and text features.

    Parameters
    ----------
    features : pd.DataFrame
        The features for the socialnorms dataset.

    Returns
    -------
    np.ndarray
        The concatentation of the title and text strings separated by a
        newline character, in a numpy array.
    """
    return (features['title'] + '\n' + features['text']).values
