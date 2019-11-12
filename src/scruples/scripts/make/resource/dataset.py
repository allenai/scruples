"""Make the final version of the scruples resource.

This script takes the raw annotation data from MTurk and creates the
final labeled version of the dataset.
"""

import collections
import copy
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
    'hits_path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument(
    'instances_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument(
    'judgments_path',
    type=click.Path(exists=False, file_okay=True, dir_okay=False))
def dataset(
        proposals_path: str,
        hits_path: str,
        instances_path: str,
        judgments_path: str
) -> None:
    """Create the scruples resource.

    Create the scruples resource, reading the proposed dataset
    instances from PROPOSALS_PATH, the MTurk HITs from HITS_PATH and
    then writing the resulting dataset instances to INSTANCES_PATH along
    with the individual annotator's judgments to JUDGMENTS_PATH.

    The HIT data provided at HITS_PATH should be the results from
    running the "amti extract tabular" command with a JSON Lines output
    format on the HIT batches. The proposals should be the data provided
    to the HITs for annotation.
    """
    logger.info(f'Reading dataset instances from {proposals_path}')

    proposals = []
    with click.open_file(proposals_path, 'r') as proposals_file:
        for ln in proposals_file:
            row = json.loads(ln)
            for proposal in row['instances']:
                proposals.append(proposal)

    logger.info(f'Reading HITs from {hits_path}.')

    instance_id_to_annotations = collections.defaultdict(list)
    with click.open_file(hits_path, 'r') as hits_file:
        for ln in hits_file:
            assignment = json.loads(ln)

            # parse the responses in the HIT's assignment.

            # Each assignment has a number of form elements named
            # ``instance-$idx`` and ``action-$idx`` where ``$idx`` is
            # the index of the item in the form.
            instance_ids = [None for _ in range(settings.N_INSTANCES_PER_HIT)]
            action_ids = [None for _ in range(settings.N_INSTANCES_PER_HIT)]
            for key, value in assignment.items():
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
                        'Found an item with neither an instance ID nor'
                        ' an action ID.')
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
                    (assignment['WorkerId'], action_id))

    logger.info(f'Assembling the datset from the annotations.')

    n_expected_annotators = (
        settings.N_ANNOTATORS_FOR_GOLD_LABELS
        + settings.N_ANNOTATORS_FOR_HUMAN_PERFORMANCE)
    controversial_instances = 0
    instances = []
    judgments = []
    for proposal in proposals:
        instance = copy.deepcopy(proposal)

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

        # add a label for controversiality to the instance

        instance['controversial'] = gold_action_score < settings.MIN_AGREEMENT

        if instance['controversial']:
            controversial_instances += 1

        # gather the instance's information in the proper places

        instances.append(instance)

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

    logger.info(f'Writing the instances to {instances_path}.')

    with click.open_file(instances_path, 'w') as instances_file:
        for instance in instances:
            instances_file.write(json.dumps(instance) + '\n')

    logger.info(f'Writing annotator judgments to {judgments_path}.')

    with click.open_file(judgments_path, 'w') as judgments_file:
        for judgment in judgments:
            judgments_file.write(json.dumps(judgment) + '\n')

    logger.info(
        f'Finished writing output. {controversial_instances} instances'
        f' were found to be controversial.')
