"""Scripts for making the socialnorms benchmark."""

import logging

import click

from . import (
    dataset,
    prepared_data,
    proposals)


logger = logging.getLogger(__name__)


# main function

@click.group()
def benchmark():
    """Make different components of the socialnorms benchmark."""
    pass


subcommands = [
    dataset.dataset,
    prepared_data.prepared_data,
    proposals.proposals
]

for subcommand in subcommands:
    benchmark.add_command(subcommand)
