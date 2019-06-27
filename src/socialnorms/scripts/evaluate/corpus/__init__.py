"""Scripts for evaluating models on the socialnorms corpus."""

import logging

import click

from . import (
    predict_lm,
    run_shallow,
    train_lm)


logger = logging.getLogger(__name__)


# main function

@click.group()
def corpus():
    """Evaluate baseline models on the socialnorms corpus."""
    pass


# register subcommands to the command group

subcommands = [
    predict_lm.predict_lm,
    run_shallow.run_shallow,
    train_lm.train_lm
]

for subcommand in subcommands:
    corpus.add_command(subcommand)
