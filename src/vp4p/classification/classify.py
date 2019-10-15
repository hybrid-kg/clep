# -*- coding: utf-8 -*-

"""Wrap Machine-Learning Classifiers for vp4p."""

import json

import seaborn as sns
import sklearn as sk


def do_classification(data, labels, model_name, out_dir, title=None, *args):
    model = get_classifier(model_name, *args)

    cv_results = sk.model_selection.cross_validate(
        estimator=model,
        X=data,
        y=labels,
        cv=10,
        scoring=['roc_auc', 'accuracy', 'f1'],
        return_estimator=True,
        )

    with open(f'{out_dir}/cross_validation_results.json', 'w') as out:
        json.dump(cv_results, out, indent=4)

    scoring_metrics = ['test_accuracy', 'test_f1_micro', 'test_roc_auc']

    data = list()
    for scores in scoring_metrics:
        data.append(list(cv_results[scores]))

    sns.set(font_scale=1.2)
    sns_plot = sns.boxplot(data=data)

    if not title:
        title = f'Box Plot of Scoring Metrics: {str(scoring_metrics)}\n'

    sns_plot.set(xlabel='Scoring Metrics',
                 ylabel='Score',
                 title=title,
                 xticklabels=scoring_metrics)

    sns_plot.figure.savefig(f'{out_dir}/Box-Plot.png')

    return cv_results


def get_classifier(model_name, *args):
    if model_name == 'logistic_regression':
        model = sk.linear_model.LogisticRegression(*args, solver='lbfgs')

    elif model_name == 'elastic_net':
        model = sk.linear_model.LogisticRegression(*args, penalty='elasticnet', l1_ratio=0.5, solver='saga')

    elif model_name == 'svm':
        model = sk.svm.SVC(*args, gamma='scale')

    elif model_name == 'random_forrest':
        model = sk.ensemble.RandomForestClassifier(*args)

    else:
        raise ModuleNotFoundError('The entered model was not found. Please check the model that was inputted')

    return model
