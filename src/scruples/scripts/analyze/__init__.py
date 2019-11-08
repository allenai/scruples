"""Scripts for performing analyses."""

import logging

import click

from . import (
    corpus,
    corpus_human_performance,
    extractions,
    latent_traits,
    oracle_performance,
    predictions)


logger = logging.getLogger(__name__)


# main function

@click.group()
def analyze():
    """Run an analysis."""
    pass


# register subcommands to the command group

subcommands = [
    corpus.corpus,
    corpus_human_performance.corpus_human_performance,
    extractions.extractions,
    latent_traits.latent_traits,
    oracle_performance.oracle_performance,
    predictions.predictions
]

for subcommand in subcommands:
    analyze.add_command(subcommand)
