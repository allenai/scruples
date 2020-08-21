"""Run the norms demo's server."""

import logging

import click
from gevent.pywsgi import WSGIServer

from ... import settings
from ...demos.norms.app import (
    app,
    get_device,
    load_model)


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--expose', is_flag=True,
    help='Whether to expose the server to the internet, or run on localhost.')
@click.option(
    '--port', type=int, default=5050,
    help='The port on which to serve the demo. Defaults to 5050.')
def norms(
        expose: bool,
        port: int
) -> None:
    """Serve the norms demo.

    In order to run this server, you must set the following environment
    variables:

      \b
      SCRUPLES_NORMS_ACTIONS_BASELINE   : The baseline to use for the resource
        (actions).
      SCRUPLES_NORMS_ACTIONS_MODEL      : The path to the saved pretrained
        model to use for predicting the actions.
      SCRUPLES_NORMS_CORPUS_BASELINE    : The baseline to use for the corpus.
      SCRUPLES_NORMS_CORPUS_MODEL       : The path to the saved pretrained
        model to use for predicting the corpus.
      SCRUPLES_NORMS_PREDICT_BATCH_SIZE : The batch size to use for
        prediction.
      SCRUPLES_NORMS_GPU_IDS            : A comma separated list of GPU IDs to
        use. If none are provided, then the CPU will be used instead.

    """
    # load the device and model

    get_device()

    logger.info('Loading the Actions model.')
    load_model(dataset='resource',)

    logger.info('Loading the Corpus model.')
    load_model(dataset='corpus')

    # start the server

    ip = '0.0.0.0' if expose else '127.0.0.1'

    logger.info(f'Running norms server on http://{ip}:{port}/')

    WSGIServer((ip, port), app).serve_forever()
