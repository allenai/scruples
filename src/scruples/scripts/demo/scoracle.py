"""Run the scoracle demo's server."""

import logging

import click
from gevent.pywsgi import WSGIServer

from scruples.demos.scoracle.app import app


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--expose', is_flag=True,
    help='Whether to expose the server to the internet, or run on localhost.')
@click.option(
    '--port', type=int, default=5000,
    help='The port on which to serve the demo. Defaults to 5000.')
def scoracle(expose, port):
    """Serve the scoracle demo."""
    ip = '0.0.0.0' if expose else '127.0.0.1'

    logger.info(f'Running scoracle server on http://{ip}:{port}/')

    WSGIServer((ip, port), app).serve_forever()
