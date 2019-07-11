"""Scripts for evaluating models on the scruples corpus."""

import logging

import click

from . import (
    predict_lm,
    run_shallow,
    tune_lm)


logger = logging.getLogger(__name__)


# main function

@click.group()
def corpus():
    """Evaluate baseline models on the scruples corpus."""
    pass


# register subcommands to the command group

subcommands = [
    predict_lm.predict_lm,
    run_shallow.run_shallow,
    tune_lm.tune_lm
]

for subcommand in subcommands:
    corpus.add_command(subcommand)
