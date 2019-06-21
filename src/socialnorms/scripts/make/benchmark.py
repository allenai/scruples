"""Make the socialnorms benchmark from the corpus.

This script takes in the socialnorms corpus and creates a benchmark
dataset of ranked action pairs.
"""

import collections
import json
import logging
import os
import random
from typing import List

import click
import networkx as nx
import numpy as np
import tqdm

from ... import settings, utils
from ...data.action import Action


logger = logging.getLogger(__name__)


@click.command()
@click.argument(
    'corpus_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'benchmark_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument(
    'splits', nargs=-1,
    type=click.Choice([split['name'] for split in settings.SPLITS]))
@click.option(
    '--margin', type=float, default=0.7,
    help='The desired minimum-margin of the normativity scores between'
         ' any paired actions in the benchmark.')
@click.option(
    '--threshold', type=float, default=0.99,
    help="The minimum expected probability that two paired actions'"
         ' normativity scores differ by at least --margin, in order to'
         ' be included in the benchmark.')
@click.option(
    '--rounds', type=int, default=5,
    help='The number of rounds of maximum matchings to run. Each round,'
         ' a maximum matching is computed and those edges are added to'
         ' the benchmark. Then, those edges are removed from the action'
         ' graph for the next round.')
def benchmark(
        corpus_dir: str,
        benchmark_dir: str,
        splits: List[str],
        margin: float,
        threshold: float,
        rounds: int,
) -> None:
    """Create the socialnorms benchmark and write it to BENCHMARK_DIR

    Read in the socialnorms corpus from CORPUS_DIR and then for each
    split passed in the final argument:

      1. Create a graph from all of the actions where if the actions have
         normativity scores p1 and p2, then each edge has weight:

             max { P(p1 - p2 > margin), P(p2 - p1 > margin) }

      2. Prune any edge with a weight lower than some threshold.

      3. Compute a maximum weight matching for the graph.

      4. Repeat this procedure for a number of rounds.

    The resulting action pairs are gathered into a dataset and written
    to BENCHMARK_DIR.
    """
    # Create the output directory
    os.makedirs(benchmark_dir)

    # Iterate over each split of the corpus, creating the benchmark data
    # for each.
    for split in settings.SPLITS:
        if split['name'] not in splits:
            # skip any split that does not need processing
            continue

        # Step 1: Read in the corpus.
        logger.info('Reading the corpus.')

        id_to_action = {}
        corpus_split_path = os.path.join(
            corpus_dir,
            settings.CORPUS_FILENAME_TEMPLATE.format(split=split['name']))
        with open(corpus_split_path, 'r') as corpus_split_file:
            for ln in tqdm.tqdm(corpus_split_file, **settings.TQDM_KWARGS):
                row = json.loads(ln)
                if row['action'] is not None:
                    action = Action(**row['action'])
                    if action.is_good:
                        id_to_action[row['id']] = action

        # Step 2: Fit a beta prior for the pronormative /
        # contranormative scores on the actions, using maximum
        # likelihood in a beta-binomial model.
        logger.info('Fitting a beta prior to the action scores.')

        # observations (success / failure pairs)
        ss = np.array([
            action.pronormative_score
            for action in id_to_action.values()
        ])
        fs = np.array([
            action.contranormative_score
            for action in id_to_action.values()
        ])

        # compute the MLE for the beta parameters
        a, b = utils.estimate_beta_binomial_parameters(
            successes=ss, failures=fs)

        logger.info(
            f'Found parameters a: {a:.5f} and b: {b:.5f} for the learned'
            f' prior.')

        # Step 3: Create a fully connected graph with the edge weight
        # between action i and action j being:
        #
        #     max { P(p_i - p_j > margin), P(p_j - p_i > margin) }
        #
        # where p_i and p_j are the probabilities that someone would
        # vote pronormative on actions i and j, respectively.
        logger.info('Contructing the action graph.')

        # bucket each pair of actions by their scores, to avoid
        # performing redundant computations
        params_to_action_pairs = collections.defaultdict(list)
        for i, id1 in enumerate(id_to_action.keys()):
            for j, id2 in enumerate(id_to_action.keys()):
                if i <= j:
                    # To only process distinct pairs (and not waste time
                    # comparing actions with themselves), require that
                    # i > j.
                    continue

                # canonicalize the order in which the pairs are
                # evaluated
                (a1, b1, id1), (a2, b2, id2) = sorted([
                    (a + ss[i], b + fs[i], id1),
                    (a + ss[j], b + fs[j], id2)
                ])

                # bucket the pair by their parameters
                params_to_action_pairs[(a1, b1, a2, b2)].append((id1, id2))

        graph = nx.Graph()
        graph.add_nodes_from(id_to_action.keys())
        for params, action_pairs in tqdm.tqdm(
                params_to_action_pairs.items(),
                **settings.TQDM_KWARGS
        ):
            a1, b1, a2, b2 = params

            w_12 = utils.prob_p1_greater_than_p2(
                a1=a1, b1=b1, a2=a2, b2=b2, margin=margin)
            w_21 = utils.prob_p1_greater_than_p2(
                a1=a2, b1=b2, a2=a1, b2=b1, margin=margin)

            # check if the weight is too low either way
            if max(w_12, w_21) < threshold:
                continue

            if w_12 > w_21:
                for id1, id2 in action_pairs:
                    graph.add_edge(id1, id2, weight=w_12, direction=(id1, id2))
            else:
                for id1, id2 in action_pairs:
                    graph.add_edge(id2, id1, weight=w_21, direction=(id2, id1))

        for _ in range(rounds):
            # Step 4: Compute a maximum weight matching from the graph.
            logger.info('Computing a maximum weight matching.')

            matching = nx.algorithms.matching.max_weight_matching(graph)

            # Step 5: Write the benchmark to disk.
            benchmark_split_path = os.path.join(
                benchmark_dir,
                settings.BENCHMARK_FILENAME_TEMPLATE.format(split=split['name']))
            with open(benchmark_split_path, 'a') as benchmark_split_file:
                for match in tqdm.tqdm(matching, **settings.TQDM_KWARGS):
                    id1, id2 = graph.edges[match]['direction']

                    # remove the matching edge from the graph in case we
                    # want to run more matching rounds
                    graph.remove_edge(*match)

                    action1 = id_to_action[id1]
                    action2 = id_to_action[id2]

                    shuffled_actions = [(id1, action1), (id2, action2)]
                    random.shuffle(shuffled_actions)

                    instance = {
                        'id': utils.make_id(),
                        'actions': [
                            {
                                'id': id_,
                                'description': action.description
                            }
                            for id_, action in shuffled_actions
                        ],
                        'label': [
                            id_ == id1
                            for id_, _ in shuffled_actions
                        ].index(True)
                    }

                    benchmark_split_file.write(json.dumps(instance) + '\n')
