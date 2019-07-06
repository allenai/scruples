"""Prepare data for being annotated on Mechanical Turk."""

import json
import logging

import click

from .... import settings


logger = logging.getLogger(__name__)


@click.command()
@click.argument(
    'data_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
def prepared_data(
        data_path: str,
        output_path: str
) -> None:
    """Prepare data for being annotated on Mechanical Turk.

    Prepare the data at DATA_PATH for being annotated on Mechanical
    Turk, writing the results to OUTPUT_PATH.
    """
    logger.info(f'Reading dataset instances from {data_path}.')

    with click.open_file(data_path, 'r') as data_file:
        instances = [json.loads(ln) for ln in data_file]

    logger.info(f'Writing prepared data to {output_path}.')

    with click.open_file(output_path, 'w') as output_file:
        for i in range(0, len(instances), settings.N_INSTANCES_PER_HIT):
            output_file.write(json.dumps({
                'instances': instances[i:i+settings.N_INSTANCES_PER_HIT]
            }) + '\n')
