"""Scripts for evaluating models on scruples."""

import logging

import click

from . import (
    resource,
    corpus)


logger = logging.getLogger(__name__)


# main function

@click.group()
def evaluate():
    """Evaluate models on scruples."""
    pass


# register subcommands to the command group

subcommands = [
    resource.resource,
    corpus.corpus
]

for subcommand in subcommands:
    evaluate.add_command(subcommand)
