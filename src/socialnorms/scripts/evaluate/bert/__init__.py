"""Scripts for evaluating BERT."""

import logging

import click

from . import (
    predict,
    train)


logger = logging.getLogger(__name__)


# main function

@click.group()
def bert():
    """Evaluate BERT."""
    pass


# register subcommands to the command group

subcommands = [
    predict.predict,
    train.train
]

for subcommand in subcommands:
    bert.add_command(subcommand)
