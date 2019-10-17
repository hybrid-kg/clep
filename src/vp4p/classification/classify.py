# -*- coding: utf-8 -*-

"""Wrap Machine-Learning Classifiers for vp4p."""

import json

import numpy as np
import seaborn as sns
from sklearn import linear_model, svm, ensemble, model_selection


def do_classification(data, model_name, out_dir, title=None, *args):
    """TODO: Please add docstring"""
    # Get classifier user arguments
    model = get_classifier(model_name, *args)

    # TODO: Explain please
    # TODO: patients and label should be and argument and they should be the default ones
    labels = data['label']
    data = data.drop(columns=['patients', 'label'])

    # Run crossvalidation over the given model
    cv_results = model_selection.cross_validate(
        estimator=model,
        X=data,
        y=labels,
        cv=10,  # TODO: This should be an argument of the function
        scoring=['roc_auc', 'accuracy', 'f1'],
        # TODO: The same thing here, this should be an argument and these should be the default values
        return_estimator=True,
    )

    # TODO: This should be its own wrapper function
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

    scoring_metrics = ['test_accuracy', 'test_f1', 'test_roc_auc']

    # TODO: Explain please
    data = [
        list(cv_results[scores])
        for scores in scoring_metrics
    ]

    # TODO: Explain please (although it is pretty obvious what you do
    # TODO all the figure stufs should be its own module
    sns.set(font_scale=1.2)
    sns_plot = sns.boxplot(data=data)

    if not title:
        title = f'Box Plot of Scoring Metrics: {str(scoring_metrics)}\n'

    sns_plot.set(
        xlabel='Scoring Metrics',
        ylabel='Score',
        title=title,
        xticklabels=scoring_metrics,
    )

    # TODO: Default name but the user should be able to choose another name if he wants
    sns_plot.figure.savefig(f'{out_dir}/boxplot.png')

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
