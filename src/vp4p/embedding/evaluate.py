# -*- coding: utf-8 -*-

"""Assess the differences between the 2 pre-processing methods"""

import pandas as pd
import numpy as np
import itertools


def do_ss_evaluation(data: list, labels: list) -> dict:
    """
    This function takes in binned pandas dataframes to compare and find the percentage of similarity and contradiction
    between the single sample scoring functions
    """

    if not all(isinstance(df, pd.DataFrame) for df in data):
        raise TypeError('Make sure every argument, except the labels, is a Pandas DataFrame')
    if not len(labels) == len(data):
        raise IndexError('The number of labels are not equal to the number of dataframes passed.')
    if not all(len(data[0].values.flatten()) == len(df.values.flatten()) for df in data):
        raise ValueError('Make sure every dataframe passed is of the same size/shape')

    result = dict()

    similarity = dict()
    missense = dict()
    nonsense = dict()

    vec_len = len(data[0].values.flatten())

    for combo in itertools.combinations(range(len(data)), 2):
        labelcombo = f"{labels[combo[0]]} - {labels[combo[1]]}"

        sim = (_similarity(data[combo[0]].values, data[combo[1]].values) / vec_len)
        similarity[labelcombo] = f'{sim:2.2%}'

        mis = (_missense(data[combo[0]].values, data[combo[1]].values) / vec_len)
        missense[labelcombo] = f'{mis:2.2%}'

        non = (_nonsense(data[combo[0]].values, data[combo[1]].values) / vec_len)
        nonsense[labelcombo] = f'{non:2.2%}'

    result['similarity'] = similarity
    result['missense'] = missense
    result['nonsense'] = nonsense

    return result


def _similarity(arr1, arr2):
    return np.sum(arr1 == arr2)


def _missense(arr1, arr2):
    o2z = np.sum(((arr1 == 0) & (arr2 == 1)) | ((arr1 == 1) & (arr2 == 0)))
    no2z = np.sum(((arr1 == 0) & (arr2 == -1)) | ((arr1 == -1) & (arr2 == 0)))

    return o2z + no2z


def _nonsense(arr1, arr2):
    return np.sum(((arr1 == -1) & (arr2 == 1)) | ((arr1 == 1) & (arr2 == -1)))
