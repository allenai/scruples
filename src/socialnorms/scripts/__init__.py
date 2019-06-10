"""Scripts for socialnorms."""

import logging

import click

from . import (
    analyze,
    evaluate,
    make)
from .. import utils


logger = logging.getLogger(__name__)


# main function

@click.group()
@click.option(
    '--verbose',
    is_flag=True,
    help='Set the log level to DEBUG.')
def socialnorms(verbose: bool) -> None:
    """The command line interface for socialnorms."""
    utils.configure_logging(verbose=verbose)


# register subcommands to the command group

subcommands = [
    analyze.analyze,
    evaluate.evaluate,
    make.make
]

for subcommand in subcommands:
    socialnorms.add_command(subcommand)
