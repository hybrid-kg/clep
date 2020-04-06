# -*- coding: utf-8 -*-

"""Carry out Radical search to identify extreme samples in the dataset and give them a single sample score."""

from typing import Callable, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF


def do_radical_search(
        data: pd.DataFrame,
        design: pd.DataFrame,
        control: str = 'Control',
        threshold: float = 2.5,
) -> pd.DataFrame:
    """Finds the samples with extreme feature values based on the control population.

    :param data: Dataframe containing the gene expression values
    :param design: Dataframe containing the design table for the data
    :param control: label used for representing the control in the design table of the data
    :param threshold: Threshold for choosing patients that are "extreme" w.r.t. the controls.
    :return Dataframe containing the Single Sample scores using radical searching
    """
    # Transpose matrix to get the patients as the rows
    data_copy = data.transpose()

    # Give each label an integer to represent the labels during classification
    label_mapping = {
        key: val
        for val, key in enumerate(np.unique(design['Target']))
    }

    # Make sure the number of rows of transposed data and design are equal
    assert len(data_copy) == len(design)

    # Separate the dataset into controls and samples
    controls = data_copy[list(design.Target == control)]

    # Calculate the empirical cdf for every gene and get the cdf score for the data
    control_ecdf = controls.apply(_ECDF, step=False, extrapolate=True).values
    cdf_score = _apply_func(data_copy, control_ecdf).fillna(0)

    # Create a dataframe initialized with 0's
    df = pd.DataFrame(0, index=data_copy.index, columns=data_copy.columns)

    # Values that are greater than the threshold or lesser than negative threshold are considered as extremes.
    upper_thresh = 1 - (threshold / 100)
    lower_thresh = (threshold / 100)

    df = pd.DataFrame(np.where(cdf_score.values > upper_thresh, 1, df.values))
    df = pd.DataFrame(np.where(cdf_score.values < lower_thresh, -1, df.values))

    df.columns = data.index
    df.index = data.columns

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
    and 'left' to (a, b]
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

    new_columns = [index for index, _ in enumerate(df.columns)]
    old_columns = list(df.columns)

    df.columns = new_columns

    for idx, i in enumerate(tqdm(df.columns, desc='Searching for radicals: ')):
        final_df[i] = np.apply_along_axis(func_list[idx], 0, df[i].values)

    final_df.columns = old_columns

    return final_df
