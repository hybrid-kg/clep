# -*- coding: utf-8 -*-

"""Wrap machine-learning classifiers for clepp."""

import json
import logging
import sys
from collections import defaultdict
from typing import Dict, List, Any, Callable, Tuple

import click
import numpy as np
import pandas as pd
from clepp import constants
from sklearn import linear_model, svm, ensemble, model_selection, multiclass, metrics, preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def do_classification(
        data: pd.DataFrame,
        model_name: str,
        optimizer: str,
        out_dir: str,
        validation_cv: int,
        scoring_metrics: List[str],
        rand_labels: bool,
        *args
) -> Dict[str, Any]:
    """Perform classification on embeddings generated from previous step.

    :param data: Dataframe containing the embeddings
    :param model_name: model that should be used for cross validation
    :param optimizer: Optimizer used to optimize the classification
    :param out_dir: Path to the output directory
    :param validation_cv: Number of cross validation steps
    :param scoring_metrics: Scoring metrics tested during cross validation
    :param rand_labels: Boolean variable to indicate if labels must be randomized to check for ML stability
    :arg args: Custom arguments to the estimator model
    :return Dictionary containing the cross validation results

    """
    # Get classifier user arguments
    model, optimizer_cv = get_classifier(model_name, *args)

    # Separate embeddings from labels in data
    labels = data['label'].values
    data = data.drop(columns='label')

    if rand_labels:
        np.random.shuffle(labels)

    if len(np.unique(labels)) > 2:
        multi_roc_auc = metrics.make_scorer(multiclass_score_func, metric_func=metrics.roc_auc_score)
        optimizerCV = get_optimizer(optimizer, model, model_name, optimizer_cv, multi_roc_auc)
        optimizerCV.fit(data, labels)

        # Run cross validation over the given model for multiclass classification
        logger.debug("Doing multiclass classification")
        cv_results = _do_multiclass_classification(
            estimator=optimizerCV,
            x=data,
            y=labels,
            cv=validation_cv,
            scoring=scoring_metrics,
            return_estimator=True,
        )
    else:
        optimizerCV = get_optimizer(optimizer, model, model_name, optimizer_cv, 'roc_auc')
        optimizerCV.fit(data, labels)

        # Run cross validation over the given model
        logger.debug(f"Running binary cross validation")
        cv_results = model_selection.cross_validate(
            estimator=optimizerCV,
            X=data,
            y=labels,
            cv=model_selection.StratifiedKFold(n_splits=validation_cv, shuffle=True),
            scoring=scoring_metrics,
            return_estimator=True,
        )

    _save_json(results=cv_results, out_dir=out_dir)

    return cv_results


def _do_multiclass_classification(estimator: BaseEstimator, x: pd.DataFrame, y: pd.Series, cv: int, scoring: List[str],
                                  return_estimator: bool = True) -> Dict[str, Any]:
    """Do multiclass classification using OneVsRest classifier.

    :param estimator: estimator/classifier that should be used for cross validation
    :param x: Pandas dataframe containing the sample & feature data
    :param y: Pandas series containing the sample's class information
    :param cv: Number of cross validation splits to be carried out
    :param scoring: List of scoring metrics tested during cross validation
    :param return_estimator: Boolean value to indicate if the estimator used should returned in the results
    """
    unique_labels = list(np.unique(y))
    logger.debug(f"unique_labels:\n {unique_labels}")

    n_classes = len(unique_labels)
    logger.debug(f"n_classes:\n {n_classes}")

    cv_results = defaultdict(list)

    # Make k-fold splits for cross validations
    k_fold = model_selection.StratifiedKFold(n_splits=cv, shuffle=True)
    logger.debug(f"k_fold Classifier:\n {k_fold}")

    # Split the data and the labels
    for run_num, (train_indexes, test_indexes) in enumerate(k_fold.split(x, y)):
        logger.debug(f"\nCurrent Run number: {run_num}\n")
        # Make a One-Hot encoding of the classes
        y = preprocessing.label_binarize(y, classes=unique_labels)

        x_train = np.asarray(
            [x.iloc[train_index, :].values.tolist()
             for train_index in train_indexes]
        )
        x_test = np.asarray(
            [x.iloc[test_index, :].values.tolist()
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
        logger.debug(f"Counter y_train:\n {np.unique(y_train, axis=0, return_counts=True)} \nCounter y_test: \n"
                     f"{np.unique(y_test, axis=0, return_counts=True)}")

        # Make a multiclass classifier for the given estimator
        clf = multiclass.OneVsRestClassifier(estimator)
        logger.debug(f"clf:\n {clf}")

        # Fit and predict using the multiclass classifier
        y_fit = clf.fit(x_train, y_train)
        logger.debug(f"y_fit:\n {y_fit}")

        y_pred = y_fit.predict(x_test)
        logger.debug(f"y_pred:\n {y_pred}\n\n")
        logger.debug(f"y_true:\n {y_test}\n\n")

        if return_estimator:
            cv_results['estimator'].append(clf.estimator)

        # For the multiclass metric find the score and add it to cv_results.
        # TODO: Add other Scorers from sklearn
        for metric in scoring:
            if metric == 'roc_auc':
                roc_auc = _multiclass_metric_evaluator(
                    metric_func=metrics.roc_auc_score,
                    n_classes=n_classes,
                    y_test=y_test,
                    y_pred=y_pred
                )
                cv_results['test_roc_auc'].append(roc_auc)

            elif metric == 'f1':
                f1 = _multiclass_metric_evaluator(
                    metric_func=metrics.f1_score,
                    n_classes=n_classes,
                    y_test=y_test,
                    y_pred=y_pred,
                    average='binary',
                )
                cv_results['test_f1_micro'].append(f1)

            elif metric == 'f1_micro':
                f1_micro = _multiclass_metric_evaluator(
                    metric_func=metrics.f1_score,
                    n_classes=n_classes,
                    y_test=y_test,
                    y_pred=y_pred,
                    average='micro',
                )
                cv_results['test_f1_micro'].append(f1_micro)

            elif metric == 'f1_macro':
                f1_macro = _multiclass_metric_evaluator(
                    metric_func=metrics.roc_auc_score,
                    n_classes=n_classes,
                    y_test=y_test,
                    y_pred=y_pred,
                    average='macro',
                )
                cv_results['test_f1_macro'].append(f1_macro)

            elif metric == 'f1_weighted':
                f1_weighted = _multiclass_metric_evaluator(
                    metric_func=metrics.f1_score,
                    n_classes=n_classes,
                    y_test=y_test,
                    y_pred=y_pred,
                    average='weighted',
                )
                cv_results['test_f1_weighted'].append(f1_weighted)

            elif metric == 'accuracy':
                accuracy = _multiclass_metric_evaluator(
                    metric_func=metrics.accuracy_score,
                    n_classes=n_classes,
                    y_test=y_test,
                    y_pred=y_pred,
                )
                cv_results['test_accuracy'].append(accuracy)

            else:
                click.echo('The passed metric has not been defined in the code for multiclass classification.')
                sys.exit()
        logger.debug(f"cv_results:\n {cv_results}")

    return cv_results


def _multiclass_metric_evaluator(metric_func: Callable[..., float], n_classes: int, y_test: np.ndarray,
                                 y_pred: np.ndarray, **kwargs) -> float:
    """Calculate the average metric for multiclass classifiers."""
    metric = 0

    for label in range(n_classes):
        metric += metric_func(y_test[:, label], y_pred[:, label], **kwargs)
    metric /= n_classes

    return metric


def get_classifier(model_name: str, *args) -> Tuple[BaseEstimator, StratifiedKFold]:
    """Retrieve the appropriate classifier from sci-kit learn based on the arguments."""
    cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True)

    if model_name == 'logistic_regression':
        model = linear_model.LogisticRegression(*args, solver='lbfgs')

    elif model_name == 'elastic_net':
        # Logistic regression with elastic net penalty & equal weightage to l1 and l2
        model = linear_model.LogisticRegression(*args, penalty='elasticnet', solver='saga')

    elif model_name == 'svm':
        model = svm.SVC(*args, gamma='scale')

    elif model_name == 'random_forest':
        model = ensemble.RandomForestClassifier(*args)

    elif model_name == 'gradient_boost':
        model = XGBClassifier(*args)

    else:
        raise ValueError(
            f'The entered model "{model_name}", was not found. Please check that you have chosen a valid model.'
        )

    return model, cv


def get_optimizer(
        optimizer: str,
        estimator,
        model,
        cv: StratifiedKFold,
        scorer
):
    if optimizer == 'grid_search':
        param_grid = constants.get_param_grid(model)
        return model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scorer)
    elif optimizer == 'random_search':
        param_dist = constants.get_param_dist(model)
        return model_selection.RandomizedSearchCV(estimator=estimator, param_distributions=param_dist, cv=cv,
                                                  scoring=scorer)
    elif optimizer == 'bayesian_search':
        param_space = constants.get_param_space(model)
        return BayesSearchCV(estimator=estimator, search_spaces=param_space, cv=cv, scoring=scorer)
    else:
        raise ValueError(f'Unknown optimizer, {optimizer}.')


def multiclass_score_func(y, y_pred, metric_func, **kwargs):
    classes = np.unique(y)
    n_classes = len(classes)

    if n_classes == 2:
        return metric_func(y, y_pred)

    y = preprocessing.label_binarize(y, classes=classes)
    y_pred = preprocessing.label_binarize(y_pred, classes=classes)

    metric = 0

    for label in range(n_classes):
        metric += metric_func(y[:, label], y_pred[:, label], **kwargs)

    metric /= n_classes

    return metric


def _save_json(results: Dict[str, Any], out_dir: str) -> None:
    """Save the cross validation results as a json file."""
    for key in results.keys():
        # Check if the result is a numpy array, if yes convert to list
        if isinstance(results[key], np.ndarray):
            results[key] = results[key].tolist()

        # Check if the results are numpy float values, if yes skip it
        elif isinstance(results[key][0], np.float):
            continue

        elif isinstance(results[key][0], list):
            continue

        # Check if the key is an estimator and convert it into a JSON Serializable object.
        # Also Check if it an estimator wrapper like OneVsRest classifier.
        else:
            results[key] = [
                classifier.get_params()
                if 'estimator' not in classifier.get_params()
                else classifier.get_params()['estimator'].get_params()
                for classifier in results[key]
            ]

    with open(f'{out_dir}/cross_validation_results.json', 'w') as out:
        json.dump(results, out, indent=4)
