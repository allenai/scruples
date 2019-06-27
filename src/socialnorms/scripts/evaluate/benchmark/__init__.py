"""Scripts for evaluating models on the socialnorms benchmark."""

import logging

import click

from . import (
    predict_lm,
    run_shallow,
    train_lm)


logger = logging.getLogger(__name__)


# main function

@click.group()
def benchmark():
    """Evaluate baseline models on the socialnorms benchmark."""
    pass


# register subcommands to the command group

subcommands = [
    predict_lm.predict_lm,
    run_shallow.run_shallow,
    train_lm.train_lm
]

for subcommand in subcommands:
    benchmark.add_command(subcommand)
