"""Scripts for demos."""

import logging

import click

from . import scoracle


logger = logging.getLogger(__name__)


# main function

@click.group()
def demo():
    """Run a demo's server."""
    pass


# register subcommands to the command group

subcommands = [
    scoracle.scoracle
]

for subcommand in subcommands:
    demo.add_command(subcommand)
