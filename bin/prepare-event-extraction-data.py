"""Prepare data for the event extraction HIT."""

import html
import json
import logging

import click
import ftfy
import spacy
import tqdm

from socialnorms import settings, utils


logger = logging.getLogger(__name__)


# main function

@click.command()
@click.argument(
    'data_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'output_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option(
    '--verbose', is_flag=True,
    help='Set the log level to DEBUG.')
def prepare_event_extraction_data(
        data_path: str,
        output_path: str,
        verbose: bool
) -> None:
    """Create data for the event extraction HIT.

    Read in the data from DATA_PATH, then convert them to data for
    seeding the event extraction HIT, writing the result to OUTPUT_PATH.
    """
    utils.configure_logging(verbose=verbose)

    nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

    logger.info(f'Reading data from {data_path}.')
    with click.open_file(data_path, 'r') as data_file:
        rows = []
        for ln in tqdm.tqdm(data_file, **settings.TQDM_KWARGS):
            rows.append(json.loads(ln))

    logger.info(f'Writing transformed data to {output_path}.')
    with click.open_file(output_path, 'w') as output_file:
        for row in tqdm.tqdm(rows, **settings.TQDM_KWARGS):
            output_file.write(json.dumps({
                'id': row['id'],
                'text': '@%@'.join(
                    html.escape(t.text)
                    for t in nlp(ftfy.fix_text(row['text']))
                )
            }) + '\n')


if __name__ == '__main__':
    prepare_event_extraction_data()
