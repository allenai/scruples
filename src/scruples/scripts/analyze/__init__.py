"""Scripts for performing analyses."""

import logging

import click

from . import (
    corpus,
    resource)


logger = logging.getLogger(__name__)


# main function

@click.group()
def analyze():
    """Run an analysis."""
    pass


# register subcommands to the command group

subcommands = [
    corpus.corpus,
    resource.resource
]

for subcommand in subcommands:
    analyze.add_command(subcommand)
