""" Hyperparameter optimization using Optuna """

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from optuna import create_study
from optuna.storages import RDBStorage
from sklearn import model_selection
from tqdm import tqdm

from .utils import (OptunaObjective, init_db, run_final_classification,
                    run_optuna_optimization, save_json)
from ..constants import MODEL_NAME_MAPPING


def hpo_cross_validate(X, y, metrics, cv, classifier, num_processes=1, db_url=None, num_trials=100, **kwargs):
    k_fold = model_selection.StratifiedKFold(n_splits=cv, shuffle=True)

    outer_cv_results = {}

    for run_num, (train_indexes, test_indexes) in enumerate(tqdm(k_fold.split(X, y), total=cv, desc='Outer CV')):
        x_train = np.asarray(
            [X.iloc[train_index, :].values.tolist()
             for train_index in train_indexes]
        )
        x_test = np.asarray(
            [X.iloc[test_index, :].values.tolist()
             for test_index in test_indexes]
        )
        y_train = np.asarray(
            [y[train_index]
             for train_index in train_indexes]
        )
        y_test = np.asarray(
            [y[test_index]
             for test_index in test_indexes]
        )

        objective = OptunaObjective(x_train, y_train, metrics, cv, classifier, **kwargs)
        directions = ['maximize'] * len(metrics)

        ############################################################################
        # Speed comparison for 100 trials
        # 1 process: 218.6 seconds
        # 2 processes: 169.9 seconds
        # 5 processes: 80.45 seconds
        ############################################################################

        if num_processes <= 1:
            study = create_study(directions=directions)
            run_optuna_optimization(study, num_trials, objective)
        else:
            if db_url is None:
                raise ValueError('Database URL must be provided for parallel optimization')

            init_db(db_url=db_url)
            storage = RDBStorage(url=db_url + '/optuna')

            study = create_study(
                directions=directions,
                storage=storage
            )

            number_of_trials = -(-num_trials // num_processes)  # Round up division

            Parallel(n_jobs=num_processes)(
                delayed(run_optuna_optimization)(
                    study,
                    number_of_trials,
                    objective
                )
                for _ in range(num_processes)
            )

        # TODO: Add support for multiple metrics
        best_trial = max(study.best_trials, key=lambda trial: trial.values[0])
        cv_results = run_final_classification(x_train, y_train, x_test, y_test, metrics, best_trial.params, 'RandomForest')
        outer_cv_results[run_num] = cv_results

    return outer_cv_results


def do_classification(
        data: pd.DataFrame,
        model_name: str,
        out_dir: str,
        validation_cv: int,
        scoring_metrics: List[str],
        rand_labels: bool,
        num_processes: int,
        mysql_url: Optional[str],
        num_trials: int,
        **kwargs
) -> Dict[str, Any]:
    """Perform classification on embeddings generated from previous step.

    :param data: Dataframe containing the embeddings
    :param model_name: model that should be used for cross validation
    :param out_dir: Path to the output directory
    :param validation_cv: Number of cross validation steps
    :param scoring_metrics: Scoring metrics tested during cross validation
    :param rand_labels: Boolean variable to indicate if labels must be randomized to check for ML stability
    :param num_processes: Number of processes to use for parallelization
    :param mysql_url: URL for the MySQL database to store Optuna results
    :param num_trials: Number of trials to run for hyperparameter optimization
    :arg kwargs: Additional named arguments to be passed to the classifier
    :return: Dictionary containing the cross validation results
    """

    # Separate embeddings from labels in data
    labels = list(data['label'].values)
    data = data.drop(columns='label')

    if rand_labels:
        np.random.shuffle(labels)

    # Perform cross validation with optuna based hyperparameter optimization
    cv_results = hpo_cross_validate(
        X=data,
        y=labels,
        metrics=scoring_metrics,
        cv=validation_cv,
        classifier=MODEL_NAME_MAPPING[model_name],
        num_processes=num_processes,
        db_url=mysql_url,
        num_trials=num_trials,
        **kwargs
    )

    save_json(results=copy.deepcopy(cv_results), out_dir=out_dir)

    return cv_results
