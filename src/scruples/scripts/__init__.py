"""Scripts for scruples."""

import logging

import click

from . import (
    analyze,
    demo,
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
def scruples(verbose: bool) -> None:
    """The command line interface for scruples."""
    utils.configure_logging(verbose=verbose)


# register subcommands to the command group

subcommands = [
    analyze.analyze,
    demo.demo,
    evaluate.evaluate,
    make.make
]

for subcommand in subcommands:
    scruples.add_command(subcommand)
