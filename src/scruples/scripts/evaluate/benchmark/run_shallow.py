"""Run shallow baseline models on the scruples benchmark."""

import collections
import json
import logging
import os
from typing import List

import click
import dill
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
import tqdm

from .... import settings, baselines
from ....baselines.metrics import METRICS
from ....dataset.readers import ScruplesBenchmark


logger = logging.getLogger(__name__)


# main function

@click.command()
@click.argument(
    'data_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument(
    'results_dir',
    type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.argument(
    'splits', type=click.Choice(ScruplesBenchmark.SPLITS), nargs=-1)
@click.option(
    '--metric',
    type=click.Choice(METRICS.keys()),
    default='log_loss',
    help='The metric to use for hyper-parameter tuning. Defaults to'
         ' log_loss.')
@click.option(
    '--n-iter', type=int, default=128,
    help='The number of iterations of Bayesian optimization to run when'
         ' tuning baseline hyper-parameters. Defaults to 128.')
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
def run_shallow(
        data_dir: str,
        results_dir: str,
        splits: List[str],
        metric: str,
        n_iter: int,
        n_points: int,
        n_folds: int,
        n_jobs: int
) -> None:
    """Evaluate shallow baselines on the scruples benchmark.

    Train shallow baseline models on the scruples benchmark, reading
    the dataset from DATA_DIR, and writing trained models, logs, and
    other results to RESULTS_DIR. Performance is reported for each split
    provided as an argument.
    """
    # Step 1: Manage and construct paths.

    logger.info('Creating the results directory.')

    os.makedirs(results_dir)
    model_paths = {}
    metrics_paths = collections.defaultdict(dict)
    predictions_paths = collections.defaultdict(dict)
    for baseline in baselines.benchmark.SHALLOW_BASELINES.keys():
        os.makedirs(os.path.join(results_dir, baseline))
        model_paths[baseline] = os.path.join(
            results_dir, baseline, 'model.pkl')
        for split in splits:
            os.makedirs(os.path.join(results_dir, baseline, split))
            metrics_paths[baseline][split] = os.path.join(
                results_dir, baseline, split, 'metrics.json')
            predictions_paths[baseline][split] = os.path.join(
                results_dir, baseline, split, 'predictions.jsonl')

    # Step 2: Load the data.

    logger.info(f'Loading the data from {data_dir}.')

    dataset = ScruplesBenchmark(data_dir=data_dir)

    # Step 3: Run the baselines.

    logger.info('Running the baselines.')

    for baseline, (Model, hyper_parameter_space) in tqdm.tqdm(
            baselines.benchmark.SHALLOW_BASELINES.items(),
            **settings.TQDM_KWARGS
    ):
        # tune the hyper-parameters and train the model
        ids, features, labels, label_scores = dataset.train
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
                n_jobs=os.cpu_count() if n_jobs == 0 else n_jobs,
                refit=True)
        else:
            model = Model
        model.fit(features, labels)

        # Step 4: Save the model.

        with open(model_paths[baseline], 'wb') as model_file:
            dill.dump(model, model_file)

        # Step 5: Run evaluation on the splits.

        for split in splits:
            ids, features, labels, label_scores = getattr(dataset, split)

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
                            'label': prediction.tolist(),
                            'label_scores': probs.tolist()
                        }) + '\n')
