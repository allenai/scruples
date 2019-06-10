"""Scripts for making the different components of socialnorms."""

import logging

import click

from . import corpus


logger = logging.getLogger(__name__)


# main function

@click.group()
def make():
    """Make different components of socialnorms."""
    pass


subcommands = [
    corpus.corpus
]

for subcommand in subcommands:
    make.add_command(subcommand)
