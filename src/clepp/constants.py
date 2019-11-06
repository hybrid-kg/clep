# -*- coding: utf-8 -*-

"""This module contains all the constants used in CLEPP package."""

import os

MODULE_NAME = 'clepp'
DEFAULT_CLEPP_DIR = os.path.join(os.path.expanduser('~'), '.clepp')
CLEPP_DIR = os.environ.get('CLEPP_DIRECTORY', DEFAULT_CLEPP_DIR)


def get_data_dir() -> str:
    """Ensure the appropriate clepp data directory exists for the given module, then returns the file path."""
    os.makedirs(CLEPP_DIR, exist_ok=True)
    return CLEPP_DIR
