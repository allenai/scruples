"""Scripts for evaluating models on the scruples benchmark."""

import logging

import click

from . import (
    predict_lm,
    run_shallow,
    tune_lm)


logger = logging.getLogger(__name__)


# main function

@click.group()
def benchmark():
    """Evaluate baseline models on the scruples benchmark."""
    pass


# register subcommands to the command group

subcommands = [
    predict_lm.predict_lm,
    run_shallow.run_shallow,
    tune_lm.tune_lm
]

for subcommand in subcommands:
    benchmark.add_command(subcommand)
