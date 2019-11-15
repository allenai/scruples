"""Scripts for analyzing the scruples resource."""

import logging

import click

from . import latent_traits


logger = logging.getLogger(__name__)


# main function

@click.group()
def resource():
    """Analyze the scruples resource."""
    pass


# register subcommands to the command group

subcommands = [
    latent_traits.latent_traits
]

for subcommand in subcommands:
    resource.add_command(subcommand)
