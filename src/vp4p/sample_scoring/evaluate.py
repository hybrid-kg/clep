# -*- coding: utf-8 -*-

"""Assess the differences between the 2 pre-processing methods"""

import pandas as pd
import numpy as np
import itertools


def do_ss_evaluation(*args, labels: list):
    """
    This function takes in binned pandas dataframes to compare and find the percentage of similarity and contradiction
    between the single sample scoring functions
    """

    if not all(isinstance(df, pd.DataFrame) for df in args):
        raise TypeError('Make sure every argument, except the labels, is a Pandas DataFrame')
    if not len(labels) == len(args):
        raise IndexError('The number of labels are not equal to the number of dataframes passed.')
    if not all(len(args[0].values.flatten()) == len(df.values.flatten()) for df in args):
        raise ValueError('Make sure every dataframe passed is of the same size/shape')

    similarity = dict()
    vec_len = len(args[0].values.flatten())

    for combo in itertools.combinations(range(len(args)), 2):
        labelcombo = (labels[combo[0]], labels[combo[1]])
        sim = (_similarity(args[combo[0]].values, args[combo[1]].values) / vec_len)
        similarity[labelcombo] = f'{sim:2.2%}'

    return similarity


def _similarity(arr1, arr2):
    return np.sum(arr1 == arr2)
