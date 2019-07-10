"""Make the final version of the socialnorms benchmark.

This script takes the raw annotation data from MTurk and creates the
final labeled version of the dataset.
"""

import collections
import json
import logging
import random

import click

from .... import settings


logger = logging.getLogger(__name__)


@click.command()
@click.argument(
    'proposals_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'annotations_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'instances_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument(
    'judgments_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option(
    '--min-agreement', type=int, default=2,
    help='The minimum number of annotators (out of'
        f' {settings.N_ANNOTATORS_FOR_GOLD_LABELS}) required to agree'
         ' for the instance to be included in the dataset.')
def dataset(
        proposals_path: str,
        annotations_path: str,
        instances_path: str,
        judgments_path: str,
        min_agreement: int
) -> None:
    """Create the socialnorms benchmark.

    Create the socialnorms benchmark, reading the proposed dataset
    instances from PROPOSALS_PATH, the MTurk annotations from
    ANNOTATIONS_PATH and then writing the resulting dataset instances to
    INSTANCES_PATH along with the individual annotator's judgments to
    JUDGMENTS_PATH.

    The annotations provided at ANNOTATIONS_PATH should be the results
    from running "amti extract tabular" command with a JSON Lines output
    format on the HIT batches. The proposals should be the data provided
    to the HITs for annotation.
    """
    if not min_agreement <= settings.N_ANNOTATORS_FOR_GOLD_LABELS:
        raise ValueError(
            '--min-agreement must be less than or equal to'
            ' the number of annotators for the gold labels'
           f' ({settings.N_ANNOTATORS_FOR_GOLD_LABELS}).')

    logger.info(f'Reading dataset instances from {proposals_path}')

    instances = []
    with click.open_file(proposals_path, 'r') as proposals_file:
        for ln in proposals_file:
            row = json.loads(ln)
            for instance in row['instances']:
                instances.append(instance)

    logger.info(f'Reading annotations from {annotations_path}.')

    instance_id_to_annotations = collections.defaultdict(list)
    with click.open_file(annotations_path, 'r') as annotations_file:
        for ln in annotations_file:
            annotation = json.loads(ln)

            # parse the responses from the HIT.

            # Each HIT has a number of form elements named
            # ``instance-$idx`` and ``action-$idx`` where ``$idx`` is
            # the index of the item in the form.
            instance_ids = [None for _ in range(settings.N_INSTANCES_PER_HIT)]
            action_ids = [None for _ in range(settings.N_INSTANCES_PER_HIT)]
            for key, value in annotation.items():
                if key.startswith('instance'):
                    _, idx = key.split('-')
                    idx = int(idx)
                    instance_ids[idx] = value
                elif key.startswith('action'):
                    _, idx = key.split('-')
                    idx = int(idx)
                    action_ids[idx] = value
                else:
                    pass

            # index the annotations by their instance IDs
            for instance_id, action_id in zip(instance_ids, action_ids):
                if instance_id is None and action_id is None:
                    logger.warning(
                         'Found a HIT with fewer than'
                        f' {settings.N_INSTANCES_PER_HIT} instances.')
                    continue
                elif instance_id is not None and action_id is None:
                    logger.error(
                        f'Found an instance ID ({instance_id}) without'
                         ' an annotated action ID.')
                    continue
                elif instance_id is None and action_id is not None:
                    logger.error(
                        f'Found an action ID ({action_id}) without an'
                         ' associated instance ID.')
                    continue

                instance_id_to_annotations[instance_id].append(
                    (annotation['WorkerId'], action_id))

    logger.info(f'Assembling the datset from the annotations.')

    n_expected_annotators = settings.N_ANNOTATORS_FOR_GOLD_LABELS \
        + settings.N_ANNOTATORS_FOR_HUMAN_PERFORMANCE
    dropped_instances = 0
    dataset = []
    judgments = []
    for instance in instances:
        action_id_to_idx = {
            action['id']: idx
            for idx, action in enumerate(instance['actions'])
        }
        annotations = instance_id_to_annotations[instance['id']]
        if len(annotations) != n_expected_annotators:
            logger.error(
                f'Found {len(annotations)} annotations for'
                f' {instance["id"]}, expected {n_expected_annotators}.'
                 ' Skipping.')
            continue

        # split the annotations into the gold label and human
        # performance annotations
        random.shuffle(annotations)

        gold_annotations = annotations[
            :settings.N_ANNOTATORS_FOR_GOLD_LABELS]
        human_perf_annotations = annotations[
            settings.N_ANNOTATORS_FOR_GOLD_LABELS:]

        # compute the gold annotations / label

        instance['gold_annotations'] = [
            0 for action in instance['actions']
        ]
        for _, action_id in gold_annotations:
            instance['gold_annotations'][
                action_id_to_idx[action_id]] += 1

        gold_action_idx, gold_action_score = max(
            enumerate(instance['gold_annotations']),
            key=lambda t: t[1])

        if gold_action_score < min_agreement:
            dropped_instances += 1
            continue

        instance['gold_label'] = gold_action_idx

        # compute the human performance annotations / label

        instance['human_perf_annotations'] = [
            0 for action in instance['actions']
        ]
        for _, action_id in human_perf_annotations:
            instance['human_perf_annotations'][
                action_id_to_idx[action_id]] += 1

        human_action_idx, _ = max(
            enumerate(instance['human_perf_annotations']),
            key=lambda t: t[1])

        instance['human_perf_label'] = human_action_idx

        dataset.append(instance)
        for worker_id, action_id in gold_annotations:
            judgments.append({
                'instance_id': instance['id'],
                'annotation_type': 'gold',
                'annotator_id': worker_id,
                'label': action_id_to_idx[action_id]
            })
        for worker_id, action_id in human_perf_annotations:
            judgments.append({
                'instance_id': instance['id'],
                'annotation_type': 'human',
                'annotator_id': worker_id,
                'label': action_id_to_idx[action_id]
            })

    logger.info(f'Writing the dataset to {instances_path}.')

    with click.open_file(instances_path, 'w') as output_file:
        for instance in dataset:
            output_file.write(json.dumps(instance) + '\n')

    logger.info(f'Writing annotator judgments to {judgments_path}.')

    with click.open_file(judgments_path, 'w') as judgments_file:
        for judgment in judgments:
            judgments_file.write(json.dumps(judgment) + '\n')

    logger.info(
        f'Finished writing output. Dropped {dropped_instances} instances.')
