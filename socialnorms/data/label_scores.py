"""A class representing a collection of scores for the labels."""

from typing import Dict

import attr

from . import utils
from .labels import Label


@attr.s(frozen=True, kw_only=True)
class LabelScores:
    """A class representing scores for all the labels.

    Attributes
    ----------
    best_label : Label
        The overall highest scoring label. Ties are broken arbitrarily.
    is_all_zero : bool
        ``True`` if all the scores are zero.
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
    def best_label(self) -> Label:
        return max(self.label_to_score.items(), key=lambda t: t[1])[0]

    # computed properties for identifying good label scores

    @utils.cached_property
    def is_all_zero(self) -> bool:
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
            not self.is_all_zero
            and self.has_unique_highest_scoring_label
        )
