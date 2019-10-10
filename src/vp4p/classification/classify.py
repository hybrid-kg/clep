# -*- coding: utf-8 -*-

"""Wrap Machine-Learning Classifiers for vp4p."""

import sklearn as sk


def do_classification(data, labels, model_name, title=None, *args):

    model = get_classifier(model_name, args)

    cv_results = sk.model_selection.cross_validate(model, data, labels, cv=10, scoring=['roc_auc', 'accuracy', 'f1'])

    return cv_results


def get_classifier(model_name, *args):

    if model_name == 'logistic_regression':
        model = sk.linear_model.LogisticRegression(args, solver='lbfgs')

    elif model_name == 'elastic_net':
        model = sk.linear_model.ElasticNet(args)

    elif model_name == 'svm':
        model = sk.svm.SVC(args, gamma='scale')

    elif model_name == 'random_forrest':
        model = sk.ensemble.RandomForestClassifier(args)

    else:
        raise ModuleNotFoundError('The entered model was not found. Please check the model that was inputted')

    return model
