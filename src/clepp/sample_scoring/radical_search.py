# -*- coding: utf-8 -*-

"""Carry out Radical search to identify extreme samples in the dataset and give them a single sample score."""

from typing import Callable, Optional, List

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF


def do_radical_search(
        data: pd.DataFrame,
        design: pd.DataFrame,
        control: str = 'Control',
        threshold: float = 2.5,
) -> pd.DataFrame:
    """Finds the samples with extreme feature values based on the feature population.

    :param data: Dataframe containing the gene expression values
    :param design: Dataframe containing the design table for the data
    :param control: label used for representing the control in the design table of the data
    :param threshold: Threshold for choosing patients that are "extreme" w.r.t. the controls.
    :return Dataframe containing the Single Sample scores using radical searching
    """
    # Transpose matrix to get the patients as the rows
    data = data.transpose()

    # Give each label an integer to represent the labels during classification
    label_mapping = {
        key: val
        for val, key in enumerate(np.unique(design['Target']))
    }

    # Make sure the number of rows of transposed data and design are equal
    assert len(data) == len(design)

    # Separate the dataset into controls and samples
    controls = data[list(design.Target == control)]

    # Calculate the empirical cdf for every gene and get the cdf score for the data
    control_ecdf = controls.apply(_ECDF, step=False, extrapolate=True).values
    cdf_score = _apply_func(data, control_ecdf).fillna(0)

    # Create a dataframe initialized with 0's
    df = pd.DataFrame(0, index=data.index, columns=data.columns)

    # Values that are greater than the threshold or lesser than negative threshold are considered as extremes.
    df[cdf_score > 1 - (threshold / 100)] = 1
    df[cdf_score < (threshold / 100)] = -1

    # Add labels to the data samples
    label = design['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)
    df['label'] = label.values

    return df


def _ECDF(
        obs: np.array,
        side: Optional[str] = 'right',
        step: Optional[bool] = True,
        extrapolate: Optional[bool] = False
) -> Callable:
    """Calculate the Empirical CDF of an array and return it as a function.

    :param obs: Observations
    :param side: Defines the shape of the intervals constituting the steps. 'right' correspond to [a, b) intervals
    and 'left' to (a, b
    :param step: Boolean value to indicate if the returned value must be a step function or an continuous based on
    interpolation or extrapolation function
    :param extrapolate: Boolean value to indicate if the continuous must be based on extrapolation
    :return: Empirical CDF as a function
    """

    if step:
        return ECDF(x=obs, side=side)
    else:
        obs = np.array(obs, copy=True)
        obs.sort()

        num_of_obs = len(obs)

        y = np.linspace(1. / num_of_obs, 1, num_of_obs)

        if extrapolate:
            return interp1d(obs, y, bounds_error=False, fill_value="extrapolate")
        else:
            return interp1d(obs, y)


def _apply_func(
        df: pd.DataFrame,
        func_list: List[Callable]
) -> pd.DataFrame:
    """Apply functions from the list (in order) on the respective column.

    :param df: Data on which the functions need to be applied
    :param func_list: List of functions to be applied
    :return: Dataframe which has been processed
    """
    final_df = pd.DataFrame()

    for idx, i in enumerate(df.columns):
        final_df[i] = np.apply_along_axis(func_list[idx], 0, df[i].values)

    return final_df
