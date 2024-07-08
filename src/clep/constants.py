# -*- coding: utf-8 -*-

"""This module contains all the constants used in CLEP package."""

import os

from sklearn.metrics import get_scorer_names
from typing import List

MODULE_NAME = 'clep'
DEFAULT_CLEP_DIR = os.path.join(os.path.expanduser('~'), '.clep')
CLEP_DIR = os.environ.get('CLEP_DIRECTORY', DEFAULT_CLEP_DIR)

VALUE_TO_COLNAME = {
    -1: 'negative_relation',
    1: 'positive_relation'
}

scorers: List[str] = list(get_scorer_names())


MODEL_NAME_MAPPING = {
    'logistic_regression': 'LogisticRegression',
    'elastic_net': 'ElasticNet',
    'svm': 'SVM',
    'random_forest': 'RandomForest',
    'gradient_boost': 'GradientBoosting',
}


def get_data_dir() -> str:
    """Ensure the appropriate clep data directory exists for the given module, then returns the file path."""
    os.makedirs(CLEP_DIR, exist_ok=True)
    return CLEP_DIR
