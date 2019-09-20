"""A class representing a collection of scores for the labels."""

from typing import Dict, Optional

import attr

from .. import settings
from . import utils
from .labels import BinarizedLabel, Label


@attr.s(frozen=True, kw_only=True)
class LabelScores:
    """A class representing scores for all the labels.

    Attributes
    ----------
    binary_label_to_score : Dict[BinarizedLabel, int]
        A dictionary mapping each binarized label to its corresponding
        score.
    best_binarized_label : BinarizedLabel
        The overall highest scoring binarized label. Ties are broken
        arbitrarily.
    best_label : Label
        The overall highest scoring label. Ties are broken arbitrarily.
    has_all_zero_binarized_label_scores : bool
        ``True`` if all the binarized label scores are zero.
    has_unique_highest_scoring_binarized_label : bool
        ``True`` if one of the binarized labels scored higher than the
        others.
    has_all_zero_label_scores : bool
        ``True`` if all the label scores are zero.
    has_unique_highest_scoring_label : bool
        ``True`` if one of the labels scored higher than all others.
    is_good : bool
        ``True`` if the label scores are considered good for inclusion
        in the final dataset.

    See `Parameters`_ for additional attributes.

    Parameters
    ----------
    label_to_score : Dict[Label, int]
        A dictionary mapping each label to its corresponding score.
    """
    label_to_score: Dict[Label, int] = attr.ib(
        validator=attr.validators.deep_mapping(
            key_validator=attr.validators.instance_of(Label),
            value_validator=attr.validators.instance_of(int)))

    # computed content properties

    @utils.cached_property
    def binarized_label_to_score(self) -> Dict[BinarizedLabel, int]:
        binarized_label_to_score = {
            binarized_label: 0
            for binarized_label in BinarizedLabel
        }
        for label, score in self.label_to_score.items():
            binarized_label = BinarizedLabel.binarize(label)
            if binarized_label is not None:
                binarized_label_to_score[binarized_label] += score

        return binarized_label_to_score

    @utils.cached_property
    def best_binarized_label(self) -> Optional[BinarizedLabel]:
        return max(
            self.binarized_label_to_score.items(),
            key=lambda t: t[1]
        )[0]

    @utils.cached_property
    def best_label(self) -> Label:
        return max(
            self.label_to_score.items(),
            key=lambda t: t[1]
        )[0]

    # computed properties for identifying good label scores

    @utils.cached_property
    def has_all_zero_binarized_label_scores(self) -> bool:
        return all(
            v == 0
            for v in self.binarized_label_to_score.values()
        )

    @utils.cached_property
    def has_unique_highest_scoring_binarized_label(self) -> bool:
        max_score = max(self.binarized_label_to_score.values())

        return 1 == sum(
            v == max_score
            for v in self.binarized_label_to_score.values()
        )

    @utils.cached_property
    def has_all_zero_label_scores(self) -> bool:
        return all(v == 0 for v in self.label_to_score.values())

    @utils.cached_property
    def has_unique_highest_scoring_label(self) -> bool:
        max_score = max(self.label_to_score.values())

        return 1 == sum(v == max_score for v in self.label_to_score.values())

    @utils.cached_property
    def is_good(self) -> bool:
        # N.B. place cheaper predicates earlier so short-circuiting can
        # avoid evaluating more expensive predicates.
        return (
            not self.has_all_zero_binarized_label_scores
            and self.has_unique_highest_scoring_binarized_label
            and not self.has_all_zero_label_scores
            and self.has_unique_highest_scoring_label
        )
