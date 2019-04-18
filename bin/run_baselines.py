"""Run baseline models on socialnorms."""

import collections
import json
import logging
import os
import pickle
from typing import List

import click
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
import tqdm

from socialnorms import settings, utils
from socialnorms.baselines import BASELINES
from socialnorms.baselines.metrics import METRICS
from socialnorms.dataset.readers import SocialNorms


logger = logging.getLogger(__name__)


# main function

@click.command()
@click.argument(
    'DATA_DIR',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'RESULTS_DIR',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument(
    'SPLITS', type=click.Choice(SocialNorms.SPLITS), nargs=-1)
@click.option(
    '--metric',
    type=click.Choice(METRICS.keys()),
    default='matthews_corrcoef',
    help='The metric to use for hyper-parameter tuning. Defaults to'
         ' matthews_corrcoef.')
@click.option(
    '--n-iter', type=int, default=256,
    help='The number of iterations of Bayesian optimization to run when'
         ' tuning baseline hyper-parameters. Defaults to 256.')
@click.option(
    '--n-points', type=int, default=8,
    help='The number of points to evaluate in parallel during Bayesian'
         ' optimization. Defaults to 8.')
@click.option(
    '--n-folds', type=int, default=4,
    help='The number of cross-validation folds to use. Defaults to 4.')
@click.option(
    '--n-jobs', type=int, default=0,
    help='The number of parallel processes to use for tuning'
         ' hyper-parameters. At most n_folds * n_points processses can'
         ' be used at a given time. If 0, then the same number of'
         ' processes as CPUs will be used. Defaults to 0.')
@click.option(
    '--verbose', is_flag=True,
    help='Set the log level to DEBUG.')
def run_baselines(
        data_dir: str,
        results_dir: str,
        splits: List[str],
        metric: str,
        n_iter: int,
        n_points: int,
        n_folds: int,
        n_jobs: int,
        verbose: bool
) -> None:
    """Train baselines on socialnorms and report performance on SPLITS.

    Train baseline models on socialnorms, reading the dataset from
    DATA_DIR, and writing trained models, logs, and other results to
    RESULTS_DIR.
    """
    # configure logging

    utils.configure_logging(verbose=verbose)


    # manage paths

    os.makedirs(results_dir)
    model_paths = {}
    metrics_paths = collections.defaultdict(dict)
    predictions_paths = collections.defaultdict(dict)
    for baseline, _, _ in BASELINES:
        os.makedirs(os.path.join(results_dir, baseline))
        model_paths[baseline] = os.path.join(
            results_dir, baseline, 'model.pkl')
        for split in splits:
            os.makedirs(os.path.join(results_dir, baseline, split))
            metrics_paths[baseline][split] = os.path.join(
                results_dir, baseline, split, 'metrics.json')
            predictions_paths[baseline][split] = os.path.join(
                results_dir, baseline, split, 'predictions.jsonl')


    # load the data

    socialnorms = SocialNorms(data_dir=data_dir)


    # run the baselines

    for baseline, Model, hyper_parameter_space in tqdm.tqdm(
            BASELINES, **settings.TQDM_KWARGS):
        ids, features, labels = socialnorms.train

        # tune hyper-parameters and train the model

        if hyper_parameter_space:
            model = BayesSearchCV(
                Model,
                hyper_parameter_space,
                scoring=make_scorer(
                    score_func=METRICS[metric][1],
                    **METRICS[metric][2]),
                n_iter=n_iter,
                n_points=n_points,
                cv=n_folds,
                n_jobs=n_jobs or os.cpu_count(),
                refit=True)
        else:
            model = Model
        model.fit(features, labels)

        # save the model

        with open(model_paths[baseline], 'wb') as model_file:
            pickle.dump(model, model_file)

        # run evaluation on the splits

        for split in splits:
            ids, features, labels = getattr(socialnorms, split)

            predictions = model.predict(features)
            probabilities = model.predict_proba(features)

            with open(metrics_paths[baseline][split], 'w') as metrics_file:
                json.dump(
                    {
                        key: metric(
                            y_true=labels,
                            y_pred=probabilities
                              if scorer_kwargs['needs_proba']
                              else predictions)
                        for key, (_, metric, scorer_kwargs) in METRICS.items()
                    },
                    metrics_file)

            with open(predictions_paths[baseline][split], 'w')\
                 as predictions_file:
                for id_, probs, prediction in zip(
                        ids, probabilities, predictions
                ):
                    predictions_file.write(
                        json.dumps({
                            'id': id_,
                            'label': prediction,
                            'label_scores': {
                                class_: prob
                                for class_, prob
                                in zip(model.classes_, probs)
                            }
                        }) + '\n')


if __name__ == '__main__':
    run_baselines()
