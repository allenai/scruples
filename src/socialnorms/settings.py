"""Constants and settings."""


# subreddit related constants

AUTO_MODERATOR_NAME = 'AutoModerator'
"""The name of the AutoModerator bot."""


# output and logging

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(name)s: %(message)s'

TQDM_KWARGS = {
    'ncols': 72,
    'leave': False
}
"""Key-word arguments for tqdm progress bars."""
