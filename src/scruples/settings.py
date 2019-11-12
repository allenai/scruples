"""Constants and settings."""


# subreddit related constants

AUTO_MODERATOR_NAME = 'AutoModerator'
"""The name of the AutoModerator bot."""


# dataset parameters

# splits

SPLITS = [
    {
        'name': 'train',
        'size': None
    },
    {
        'name': 'dev',
        'size': 2500
    },
    {
        'name': 'test',
        'size': 2500
    }
]
"""Definitions for the various dataset splits.

A ``'size'`` of ``None`` signifies that the all the data that is not
used in the other splits should go to that split.
"""
# N.B. This variable is the single-source-of-truth for the splits, their
# names and their sizes (in terms of the number of posts used to create
# them). If this variable is modified, make sure to update the doc
# strings for ``ScruplesCorpus``, ``ScruplesCorpusDataset``.
#
# Note that the resource splits will have a different number of
# instances than the number of posts used to create them, in
# general. So, the 'size' attribute doesn't necessarily give the size of
# the resource splits.

# corpus settings

CORPUS_FILENAME_TEMPLATE = '{split}.scruples-corpus.jsonl'
"""A template string for the corpus's split filenames."""

POSTS_FILENAME = 'all.scruples-posts.jsonl'
"""The filename for the file containing all the posts."""


# resource settings

PROPOSALS_FILENAME_TEMPLATE = '{split}.scruples-proposals.jsonl'
"""A template string for the resource proposals' split filenames."""

RESOURCE_FILENAME_TEMPLATE = '{split}.scruples-actions.jsonl'
"""A template string for the resource's split filenames."""

N_ANNOTATORS_FOR_GOLD_LABELS = 5
"""The number of annotators to use for creating the gold labels."""

MIN_AGREEMENT = 5
"""The minimum number of gold annotators required to agree.

The minimum number of gold annotators required to agree for the instance
to be considered non-controversial (and thus be included in the final
dataset.
"""

N_ANNOTATORS_FOR_HUMAN_PERFORMANCE = 5
"""The number of annotators to use for evaluating human performance."""

N_INSTANCES_PER_HIT = 20
"""The number of instances annotated in a single HIT."""


# evaluation

LOSS_TYPES = [
    'xentropy-hard',
    'xentropy-soft',
    'xentropy-full',
    'dirichlet-multinomial'
]
"""The different loss types for deep baseline models.

``"xentropy-hard"`` uses cross-entropy on the hard labels derived from
the plurality.

``"xentropy-soft"`` uses cross-entropy against soft labels derived from
averaging the individual labels together.

``"xentropy-full"`` uses the full negative log-likelihood objective with
all of the annotations. So, unlike ``"xentropy-soft"``, it doesn't
average the annotations then compute the cross-entropy, but simply sums
the contributions from each label. This loss is equivalent to
``"xentropy-soft"`` in the case where each instance has the same number
of annotations.

``"dirichlet-multinomial"`` uses a dirichlet-multinomial likelihood
where the model is predicting the parameters for the dirichlet
distribution as part of the hierarchical model.
"""


# output and logging

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(name)s: %(message)s'
"""The format string for logging."""

TQDM_KWARGS = {
    'ncols': 72,
    'leave': False
}
"""Key-word arguments for tqdm progress bars."""
