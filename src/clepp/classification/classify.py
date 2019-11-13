# -*- coding: utf-8 -*-

"""Wrap machine-learning classifiers for clepp."""

import json
import sys
from collections import defaultdict
from typing import Dict

import click
import numpy as np
import seaborn as sns
from clepp import constants
from sklearn import linear_model, svm, ensemble, model_selection, multiclass, metrics, preprocessing


def do_classification(data, model_name, out_dir, cv, scoring_metrics, title, *args) -> Dict:
    """Perform classification on embeddings generated from previous step.

    :param data: Dataframe containing the embeddings
    :param model_name: model that should be used for cross validation
    :param out_dir: Path to the output directory
    :param cv: Number of cross validation steps
    :param scoring_metrics: Scoring metrics tested during cross validation
    :param title: Title of the Boxplot
    :arg args: Custom arguments to the estimator model
    :return Dictionary containing the cross validation results
    """
    # Get classifier user arguments
    model = get_classifier(model_name, *args)

    # Separate embeddings from labels in data
    labels = data['label']
    data = data.drop(columns='label')

    # Check if we are conducting multiclass classification, if so replace f1 with weighted f1 and remove roc_auc in-case
    if len(np.unique(labels)) > 2:
        # Run cross validation over the given model for multiclass classification
        cv_results = _do_multiclass_classification(
            estimator=model,
            x=data,
            y=labels,
            cv=cv,
            scoring=scoring_metrics,
            return_estimator=True,
        )
    else:
        # Run cross validation over the given model
        cv_results = model_selection.cross_validate(
            estimator=model,
            X=data,
            y=labels,
            cv=cv,
            scoring=scoring_metrics,
            return_estimator=True,
        )

    _save_json(cv_results=cv_results, out_dir=out_dir)

    if title == '':
        title_as_list = [
            param
            for param in out_dir.split('/')
            if param.lower() not in ['classification', 'nrl']
        ]

        title = f'Box Plot for {",".join(title_as_list)}'
        title.replace(',', '\n')

    _plot(cv_results=cv_results, title=title, out_dir=out_dir)

    return cv_results


def _do_multiclass_classification(estimator, x, y, cv, scoring, return_estimator=True):
    """Do multiclass classification using OneVsRest classifier."""
    n_classes = len(np.unique(y))
    cv_results = defaultdict(list)

    # Map classes to integer values and store as a dictionary
    class2label_mapping = dict((key, value) for key, value in zip(np.unique(y), range(n_classes)))

    # Make a One-Hot encoding of the classes
    y = preprocessing.label_binarize(y.map(class2label_mapping), classes=range(n_classes))

    # Make k-fold splits for cross validations
    k_fold = model_selection.KFold(n_splits=cv, shuffle=True)

    # Split the data and the labels
    for train_indexes, test_indexes in k_fold.split(x, y):
        x_train = x.iloc[train_indexes]
        x_test = x.iloc[test_indexes]
        y_train = np.asarray([y[train_index] for train_index in train_indexes])
        y_test = np.asarray([y[test_index] for test_index in test_indexes])

        # Make a multiclass classifier for the given estimator
        clf = multiclass.OneVsRestClassifier(estimator)

        # Fit and predict using the multiclass classifier
        y_score = clf.fit(x_train, y_train).predict(x_test)

        if return_estimator:
            cv_results['estimator'].append(clf)

        # For the multiclass metric find the score and add it to cv_results.
        # TODO: Add other Scorers from sklearn and abstract this by making a function to average the metrics.
        for metric in scoring:
            if metric == 'roc_auc':
                roc_auc = 0

                for label in range(n_classes):
                    roc_auc += metrics.roc_auc_score(y_test[:, label], y_score[:, label])
                roc_auc /= n_classes

                cv_results['test_roc_auc'].append(roc_auc)

            elif metric == 'f1':
                f1 = 0

                for label in range(n_classes):
                    f1 += metrics.f1_score(y_test[:, label], y_score[:, label], average='binary')
                f1 /= n_classes

                cv_results['test_f1_micro'].append(f1)

            elif metric == 'f1_micro':
                f1_micro = 0

                for label in range(n_classes):
                    f1_micro += metrics.f1_score(y_test[:, label], y_score[:, label], average='micro')
                f1_micro /= n_classes

                cv_results['test_f1_micro'].append(f1_micro)

            elif metric == 'f1_macro':
                f1_macro = 0

                for label in range(n_classes):
                    f1_macro += metrics.f1_score(y_test[:, label], y_score[:, label], average='macro')
                f1_macro /= n_classes

                cv_results['test_f1_micro'].append(f1_macro)

            elif metric == 'f1_weighted':
                f1_weighted = 0

                for label in range(n_classes):
                    f1_weighted += metrics.f1_score(y_test[:, label], y_score[:, label], average='weighted')
                f1_weighted /= n_classes

                cv_results['test_f1_micro'].append(f1_weighted)

            elif metric == 'accuracy':
                accuracy = 0

                for label in range(n_classes):
                    accuracy += metrics.accuracy_score(y_test[:, label], y_score[:, label])
                accuracy /= n_classes

                cv_results['test_f1_micro'].append(accuracy)

            else:
                click.echo('The passed metric has not been defined in the code for multiclass classification.')
                sys.exit()

    return cv_results


def get_classifier(model_name: str, *args):
    """Retrieve the appropriate classifier from sci-kit learn based on the arguments."""
    if model_name == 'logistic_regression':
        return linear_model.LogisticRegression(*args, solver='lbfgs')

    elif model_name == 'elastic_net':
        return linear_model.LogisticRegression(*args, penalty='elasticnet', l1_ratio=0.5, solver='saga')

    elif model_name == 'svm':
        return svm.SVC(*args, gamma='scale')

    elif model_name == 'random_forrest':
        return ensemble.RandomForestClassifier(*args)

    raise ValueError(
        f'The entered model "{model_name}", was not found. Please check that you have chosen a valid model.'
    )


def _save_json(cv_results, out_dir) -> None:
    """Save the cross validation results as a json file."""
    for key in cv_results.keys():
        # Check if the result is a numpy array, if yes convert to list
        if isinstance(cv_results[key], np.ndarray):
            cv_results[key] = cv_results[key].tolist()

        # Check if the results are numpy float values, if yes skip it
        elif isinstance(cv_results[key][0], np.float):
            continue

        # Check if the key is an estimator and convert it into a JSON Serializable object.
        # Also Check if it an estimator wrapper like OneVsRest classifier.
        else:
            cv_results[key] = [
                classifier.get_params()
                if 'estimator' not in classifier.get_params()
                else classifier.get_params()['estimator'].get_params()
                for classifier in cv_results[key]
            ]

    with open(f'{out_dir}/cross_validation_results.json', 'w') as out:
        json.dump(cv_results, out, indent=4)


def _plot(cv_results, title, out_dir) -> None:
    """Plot the cross validation results as a boxplot."""
    non_metrics = ['estimator', 'fit_time', 'score_time']

    # Get only the scoring metrics and remove the fit_time, estimator and score_time.
    scoring_metrics = [
        metric
        for metric in cv_results
        if metric not in non_metrics
    ]

    metrics_for_plot = [
        constants.METRIC_TO_LABEL[metric.split('test_')[1]]
        for metric in scoring_metrics
    ]

    # Get the data from for each scoring metric used from the cross validation results
    data = [
        list(cv_results[scores])
        for scores in scoring_metrics
    ]

    # Increase the default font size by a degree of 1.2
    sns.set(font_scale=1.2)
    sns_plot = sns.boxplot(data=data)

    sns_plot.set(
        ylabel='Score',
        title=title,
        xticklabels=metrics_for_plot,
    )

    sns_plot.figure.savefig(f'{out_dir}/boxplot.png')
