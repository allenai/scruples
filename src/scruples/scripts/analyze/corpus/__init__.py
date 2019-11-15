"""Scripts for analyzing the scruples corpus."""

import logging

import click

from . import (
    extractions,
    human_performance,
    oracle_performance,
    predictions,
    statistics)


logger = logging.getLogger(__name__)


# main function

@click.group()
def corpus():
    """Analyze the scruples corpus."""
    pass


# register subcommands to the command group

subcommands = [
    extractions.extractions,
    human_performance.human_performance,
    oracle_performance.oracle_performance,
    predictions.predictions,
    statistics.statistics
]

for subcommand in subcommands:
    corpus.add_command(subcommand)
