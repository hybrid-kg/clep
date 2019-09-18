# -*- coding: utf-8 -*-

"""This module contains all the constants used in vp4p package."""

import os

MODULE_NAME = 'vp4p'
DEFAULT_vp4p_DIR = os.path.join(os.path.expanduser('~'), '.vp4p')
vp4p_DIR = os.environ.get('vp4p_DIRECTORY', DEFAULT_vp4p_DIR)


def get_data_dir() -> str:
    """Ensure the appropriate vp4p data directory exists for the given module, then returns the file path."""
    os.makedirs(vp4p_DIR, exist_ok=True)
    return vp4p_DIR
