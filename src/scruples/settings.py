"""Constants and settings."""

import os


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
to be considered non-controversial.
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


# demos

def _coerce_if_not_none(value, type_):
    if value is None:
        return None

    return type_(value)


NORMS_ACTIONS_BASELINE = _coerce_if_not_none(
    os.environ.get('SCRUPLES_NORMS_ACTIONS_BASELINE'),
    str)
"""The baseline to use for predicting the actions in the norms demo.

This constant should be one of the keys from
``scruples.baselines.resource.FINE_TUNE_LM_BASELINES`` and should correspond to
the ``NORMS_ACTIONS_MODEL`` setting.
"""

NORMS_ACTIONS_MODEL = _coerce_if_not_none(
    os.environ.get('SCRUPLES_NORMS_ACTIONS_MODEL'),
    str)
"""The path to the model directory to use for predicting the actions.

The path to the fine-tuned Dirichlet-multinomial likelihood model to use when
predicting the actions in the ``norms`` demo. The chosen directory should be
the result of calling ``.save_pretrained`` on the model instance. See the
transformers_ library for more details.

.. _transformers: https://github.com/huggingface/transformers
"""

NORMS_CORPUS_BASELINE = _coerce_if_not_none(
    os.environ.get('SCRUPLES_NORMS_CORPUS_BASELINE'),
    str)
"""The baseline to use for predicting the corpus in the norms demo.

This constant should be one of the keys from
``scruples.baselines.corpus.FINE_TUNE_LM_BASELINES`` and should correspond to
the ``NORMS_CORPUS_MODEL`` setting.
"""

NORMS_CORPUS_MODEL = _coerce_if_not_none(
    os.environ.get('SCRUPLES_NORMS_CORPUS_MODEL'),
    str)
"""The path to the model directory to use for predicting the corpus.

The path to the fine-tuned Dirichlet-multinomial likelihood model to use when
predicting the corpus in the ``norms`` demo. The chosen directory should be the
result of calling ``.save_pretrained`` on the model instance. See the
transformers_ library for more details.
"""

NORMS_PREDICT_BATCH_SIZE = _coerce_if_not_none(
    os.environ.get('SCRUPLES_NORMS_PREDICT_BATCH_SIZE'),
    int)
"""The batch size to use for predictions in the ``norms`` demo."""

NORMS_GPU_IDS = _coerce_if_not_none(
    os.environ.get('SCRUPLES_NORMS_GPU_IDS'),
    str)
"""The GPU IDs to use for making predictions.

The GPU IDs to use when making predictions in the ``norms`` demo. In the
environment variable specifying this configuration, the GPU IDs should be
separated by commas (i.e., ``"0,1,2"``).
"""


# output and logging

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(name)s: %(message)s'
"""The format string for logging."""

TQDM_KWARGS = {
    'ncols': 72,
    'leave': False
}
"""Key-word arguments for tqdm progress bars."""
