"""Scripts for evaluating models on socialnorms."""

import logging

import click

from . import (
    bert,
    ml_baselines)


logger = logging.getLogger(__name__)


# main function

@click.group()
def evaluate():
    """Evaluate models on socialnorms."""
    pass


# register subcommands to the command group

subcommands = [
    bert.bert,
    ml_baselines.ml_baselines
]

for subcommand in subcommands:
    evaluate.add_command(subcommand)
