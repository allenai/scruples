"""Scripts for evaluating models on socialnorms."""

import logging

import click

from . import (
    benchmark,
    corpus)


logger = logging.getLogger(__name__)


# main function

@click.group()
def evaluate():
    """Evaluate models on socialnorms."""
    pass


# register subcommands to the command group

subcommands = [
    benchmark.benchmark,
    corpus.corpus
]

for subcommand in subcommands:
    evaluate.add_command(subcommand)
