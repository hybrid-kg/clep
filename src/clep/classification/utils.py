# -*- coding: utf-8 -*-

"""Wrap machine-learning classifiers for clep."""

import json
import logging
import sys
from collections import defaultdict
from typing import Dict, List, Any, Callable, Union, cast
import copy

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.typing as pat
import optuna
from optuna import Trial
from optuna.study import Study
from sklearn.base import BaseEstimator
from sklearn import model_selection, multiclass, metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sqlalchemy
from xgboost import XGBClassifier


optuna.logging.set_verbosity(optuna.logging.WARN)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


class OptunaObjective:
    def __init__(self, X, y, metric, cv, classifier, **kwargs):
        self.X = X
        self.y = y
        self.metric = metric
        self.cv = cv
        self.classifier = classifier
        self.classifier_params = kwargs

    def __call__(self, trial):
        # self.classifier = trial.suggest_categorical('classifier', ['SVM', 'ElasticNet', 'LogisticRegression', 'RandomForest', 'GradientBoosting'])
        # Define the hyperparameters to optimize based on the suggested classifier
        clf, suggested_params = self._get_clf(trial)
        if len(np.unique(self.y)) > 2:
            cv_results = _do_multiclass_classification(
                estimator=clf,
                x=self.X,
                y=self.y,
                cv=self.cv,
                scoring=self.metric
            )
        else:
            cv_results = model_selection.cross_validate(
                estimator=clf,
                X=self.X,
                y=self.y,
                cv=model_selection.StratifiedKFold(n_splits=self.cv, shuffle=True),
                scoring=self.metric,
                return_estimator=True
            )

        for key, value in cv_results.items():
            # print(f'{key}: {type(value)}')

            if key == 'estimator':
                serializable_value = suggested_params
                serializable_value['estimator'] = self.classifier
            elif isinstance(value, np.ndarray):
                serializable_value = value.tolist()

            trial.set_user_attr(key, serializable_value)

        metric_scores = self._get_metric(cv_results)

        return metric_scores

    def _get_clf(self, trial):
        clf = None
        clf_params = {}
        match self.classifier:
            case 'LogisticRegression':
                C = trial.suggest_float('C', 1e-6, 1e+6, log=True)

                clf_params = {'C': C}
                clf = LogisticRegression(C=C, solver='lbfgs', max_iter=5000, **self.classifier_params)
            case 'ElasticNet':
                C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

                clf_params = {'C': C, 'l1_ratio': l1_ratio}
                clf = LogisticRegression(C=C, solver='saga', penalty='elasticnet', l1_ratio=l1_ratio, max_iter=5000, **self.classifier_params)
            case 'RandomForest':
                n_estimators = trial.suggest_int('n_estimators', 100, 1000)
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])   # ! auto does not work

                clf_params = {'n_estimators': n_estimators, 'max_features': max_features}
                clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, **self.classifier_params)
            case 'GradientBoosting':
                learning_rate = trial.suggest_float('learning_rate', 0.0, 1.0)
                subsample = trial.suggest_float('subsample', 0.1, 0.9)
                max_depth = trial.suggest_int('max_depth', 0, 10)
                min_child_weight = trial.suggest_int('min_child_weight', 0, 25)

                clf_params = {'learning_rate': learning_rate, 'subsample': subsample, 'max_depth': max_depth, 'min_child_weight': min_child_weight}
                clf = XGBClassifier(learning_rate=learning_rate, subsample=subsample, max_depth=max_depth, min_child_weight=min_child_weight, **self.classifier_params)
            case 'SVM':
                C = trial.suggest_float('C', 1e-3, 1e+3, log=True)
                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
                degree = trial.suggest_int('degree', 1, 5)

                clf_params = {'C': C, 'kernel': kernel, 'degree': degree}
                clf = SVC(C=C, kernel=kernel, degree=degree, **self.classifier_params)
            case _:
                raise ValueError("Invalid classifier specified.")

        return clf, clf_params

    def _get_metric(self, cv_results):
        results = []
        for metric in self.metric:
            if f'test_{metric}' in cv_results:
                results.append(np.mean(cv_results[f'test_{metric}']))

        return tuple(results)


def run_optuna_optimization(study: Study, n_trials: int, objective: OptunaObjective):
    study.optimize(cast(Callable[[Trial], float], objective), n_trials=n_trials, show_progress_bar=True)


def run_final_classification(x_train, y_train, x_test, y_test, metrics, best_params, classifier):
    # classifier = best_params.pop('classifier')
    cv_results = defaultdict(list)
    clf = _get_classifier(classifier)

    n_classes = len(np.unique(y_train))

    y_fit = clf.fit(x_train, y_train)
    y_pred = y_fit.predict(x_test)

    if len(np.unique(y_train)) > 2:
        cv_results = _multiclass_scorer(metrics, cv_results, y_test, y_pred, n_classes)
    else:
        cv_results = _binary_scorer(metrics, cv_results, y_test, y_pred)

    return cv_results


def init_db(db_url, db_name='optuna'):
    engine = sqlalchemy.create_engine(db_url)
    connection = engine.connect()
    connection.execute(sqlalchemy.text('commit'))
    connection.execute(sqlalchemy.text(f'CREATE DATABASE IF NOT EXISTS {db_name}'))
    connection.close()


def save_json(results: Dict[str, Any], out_dir: str) -> None:
    """Save the cross validation results as a json file."""
    for key in results.keys():
        # Check if the result is a numpy array, if yes convert to list
        if isinstance(results[key], np.ndarray):
            results[key] = results[key].tolist()

        # Check if the results are numpy float values, if yes skip it
        elif isinstance(results[key][0], np.floating):
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


def _do_multiclass_classification(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: Union[pat.Series[str | int | float], List[Any]],
    cv: int,
    scoring: List[str],
    return_estimator: bool = True
) -> Dict[str, Any]:
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
        y = preprocessing.label_binarize(y, classes=unique_labels)  # type: ignore

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
            cv_results['estimator'].append(clf.estimators_)

        cv_results = _multiclass_scorer(scoring, cv_results, y_test, y_pred, n_classes)

        # For the multiclass metric find the score and add it to cv_results.

        logger.debug(f"cv_results:\n {cv_results}")

    return cv_results


def _multiclass_scorer(scoring: List[str], cv_results: Dict[str, Any], y_test: npt.NDArray[Any], y_pred: npt.NDArray[Any], n_classes: int) -> Dict[str, Any]:
    cv_results = copy.deepcopy(cv_results)
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

    return cv_results


def _binary_scorer(scoring: List[str], cv_results: Dict[str, Any], y_test: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> Dict[str, Any]:
    cv_results = copy.deepcopy(cv_results)
    for metric in scoring:
        if metric == 'roc_auc':
            roc_auc = metrics.roc_auc_score(y_test, y_pred)
            cv_results['test_roc_auc'].append(roc_auc)

        elif metric == 'f1':
            f1 = metrics.f1_score(y_test, y_pred)
            cv_results['test_f1'].append(f1)

        elif metric == 'f1_micro':
            f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
            cv_results['test_f1_micro'].append(f1_micro)

        elif metric == 'f1_macro':
            f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
            cv_results['test_f1_macro'].append(f1_macro)

        elif metric == 'f1_weighted':
            f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
            cv_results['test_f1_weighted'].append(f1_weighted)

        elif metric == 'accuracy':
            accuracy = metrics.accuracy_score(y_test, y_pred)
            cv_results['test_accuracy'].append(accuracy)

        else:
            click.echo('The passed metric has not been defined in the code for binary classification.')
            sys.exit()

    return cv_results


def _multiclass_metric_evaluator(
    metric_func: Callable[..., Union[float, np.float16, np.float32, np.float64, npt.NDArray[Any]]],
    n_classes: int,
    y_test: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    **kwargs: str
) -> float | npt.NDArray[Any]:
    """Calculate the average metric for multiclass classifiers."""
    metric = 0.0

    for label in range(n_classes):
        metric += metric_func(y_test[:, label], y_pred[:, label], **kwargs)
    metric /= n_classes

    if isinstance(metric, np.ndarray):
        return metric

    return float(metric)


def _get_classifier(model_name: str, best_params: Dict[Any, Any] = {}) -> Union[LogisticRegression, RandomForestClassifier, XGBClassifier, SVC]:
    """Retrieve the appropriate classifier from sci-kit learn based on the arguments."""
    match model_name:
        case 'LogisticRegression':
            clf = LogisticRegression(solver='lbfgs', max_iter=5000, **best_params)
        case 'ElasticNet':
            clf = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=5000, **best_params)
        case 'RandomForest':
            clf = RandomForestClassifier(**best_params)
        case 'GradientBoosting':
            clf = XGBClassifier(**best_params)
        case 'SVM':
            clf = SVC(**best_params)
        case _:
            raise ValueError("Invalid classifier specified.")

    return clf
