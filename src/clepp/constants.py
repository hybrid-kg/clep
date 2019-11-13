# -*- coding: utf-8 -*-

"""This module contains all the constants used in CLEPP package."""

import os

MODULE_NAME = 'clepp'
DEFAULT_CLEPP_DIR = os.path.join(os.path.expanduser('~'), '.clepp')
CLEPP_DIR = os.environ.get('CLEPP_DIRECTORY', DEFAULT_CLEPP_DIR)

METRIC_TO_LABEL = {
    'f1_weighted': '$F_1$ Weighted',
    'f1': '$F_1$',
    'f1_micro': '$F_1$ Micro',
    'f1_macro': '$F_1$ Macro',
    'recall': 'Recall',
    'precision': 'Precision',
    'jaccard': 'Jaccard Score',
    'roc_auc': 'ROC-AUC',
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Balanced Accuracy',
    'average_precision': 'Average Precision',
}


def get_data_dir() -> str:
    """Ensure the appropriate clepp data directory exists for the given module, then returns the file path."""
    os.makedirs(CLEPP_DIR, exist_ok=True)
    return CLEPP_DIR
