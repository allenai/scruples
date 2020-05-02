"""Scripts for analyzing the scruples resource."""

import logging

import click

from . import (
    human_performance,
    latent_traits,
    oracle_performance,
    predictions,
    verbs)


logger = logging.getLogger(__name__)


# main function

@click.group()
def resource():
    """Analyze the scruples resource."""
    pass


# register subcommands to the command group

subcommands = [
    human_performance.human_performance,
    latent_traits.latent_traits,
    oracle_performance.oracle_performance,
    predictions.predictions,
    verbs.verbs
]

for subcommand in subcommands:
    resource.add_command(subcommand)
