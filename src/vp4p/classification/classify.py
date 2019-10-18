# -*- coding: utf-8 -*-

"""Wrap Machine-Learning Classifiers for vp4p."""

import json

import numpy as np
import seaborn as sns
from sklearn import linear_model, svm, ensemble, model_selection


def do_classification(data, model_name, out_dir, cv, metrics, title, *args) -> dict:
    """Perform Classification on embeddings generated from previous step.

    :param data: Dataframe containing the embeddings
    :param model_name: model that should be used for cross validation
    :param out_dir: Path to the output directory
    :param cv: Number of cross validation steps
    :param metrics: Scoring metrics tested during cross validation
    :param title: Title of the Boxplot
    :arg args: Custom arguments to the estimator model
    :return Dictionary containing the cross validation results
    """
    # Get classifier user arguments
    model = get_classifier(model_name, *args)

    # Separate the embedding from the labels in the data
    # TODO: patients and label should be and argument and they should be the default ones
    labels = data['label']
    data = data.drop(columns=['patients', 'label'])

    # Run cross validation over the given model
    cv_results = model_selection.cross_validate(
        estimator=model,
        X=data,
        y=labels,
        cv=cv,
        scoring=metrics,
        return_estimator=True,
    )

    _save_json(cv_results=cv_results, out_dir=out_dir)

    _plot(cv_results=cv_results, title=title, out_dir=out_dir)

    return cv_results


def get_classifier(model_name: str, *args):
    """Wrapper to get the appropriate classifier from arguments."""
    if model_name == 'logistic_regression':
        return linear_model.LogisticRegression(*args, solver='lbfgs')

    elif model_name == 'elastic_net':
        return linear_model.LogisticRegression(*args, penalty='elasticnet', l1_ratio=0.5, solver='saga')

    elif model_name == 'svm':
        return svm.SVC(*args, gamma='scale')

    elif model_name == 'random_forrest':
        return ensemble.RandomForestClassifier(*args)

    raise ValueError(
        f'The entered model "{model_name}", was not found. Please check that you have chonen a valid model.'
    )


def _save_json(cv_results, out_dir) -> None:
    """Save the cross validation results as a json file."""
    for key in cv_results.keys():
        if isinstance(cv_results[key], np.ndarray):
            cv_results[key] = cv_results[key].tolist()

        else:
            cv_results[key] = [
                classifier.get_params()
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
        for metric in cv_results if metric not in non_metrics
    ]

    # Get the data from for each scoring metric used from the cross validation results
    data = [
        list(cv_results[scores])
        for scores in scoring_metrics
        ]

    # Increase the default font size by a degree of 1.2
    sns.set(font_scale=1.2)
    sns_plot = sns.boxplot(data=data)

    if title == '':
        title = f'Box Plot of Scoring Metrics: {str(scoring_metrics)}\n'

    sns_plot.set(
        xlabel='Scoring Metrics',
        ylabel='Score',
        title=title,
        xticklabels=scoring_metrics,
        )

    # TODO: Default name but the user should be able to choose another name if he wants
    sns_plot.figure.savefig(f'{out_dir}/boxplot.png')
