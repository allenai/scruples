"""Scripts for making the scruples benchmark."""

import logging

import click

from . import (
    dataset,
    proposals)


logger = logging.getLogger(__name__)


# main function

@click.group()
def benchmark():
    """Make different components of the scruples benchmark."""
    pass


subcommands = [
    dataset.dataset,
    proposals.proposals
]

for subcommand in subcommands:
    benchmark.add_command(subcommand)
