"""Scripts for making the different components of scruples."""

import logging

import click

from . import (
    resource,
    corpus)


logger = logging.getLogger(__name__)


# main function

@click.group()
def make():
    """Make different components of scruples."""
    pass


subcommands = [
    resource.resource,
    corpus.corpus
]

for subcommand in subcommands:
    make.add_command(subcommand)
