"""Scripts for making the scruples resource."""

import logging

import click

from . import (
    dataset,
    proposals)


logger = logging.getLogger(__name__)


# main function

@click.group()
def resource():
    """Make different components of the scruples resource."""
    pass


subcommands = [
    dataset.dataset,
    proposals.proposals
]

for subcommand in subcommands:
    resource.add_command(subcommand)
