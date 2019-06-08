"""Constants and settings."""


# subreddit related constants

AUTO_MODERATOR_NAME = 'AutoModerator'
"""The name of the AutoModerator bot."""


# dataset parameters

# corpus split filenames

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
# strings for ``SocialnormsCorpus``, ``SocialnormsCorpusDataset``.

CORPUS_FILENAME_TEMPLATE = '{split}.socialnorms-corpus.jsonl'
"""A template string for the corpus's split filenames."""


# output and logging

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(name)s: %(message)s'
"""The format string for logging."""

TQDM_KWARGS = {
    'ncols': 72,
    'leave': False
}
"""Key-word arguments for tqdm progress bars."""
