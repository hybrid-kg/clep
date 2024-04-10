# -*- coding: utf-8 -*-

"""Carry out Z-Score based single sample DE analysis."""

from typing import List

import numpy as np
import pandas as pd


def do_z_score(
        data: pd.DataFrame,
        design: pd.DataFrame,
        control: str = 'Control',
        threshold: float = 2.0,
) -> pd.DataFrame:
    """Carry out Z-Score based single sample DE analysis.

    :param data: Dataframe containing the gene expression values
    :param design: Dataframe containing the design table for the data
    :param control: label used for representing the control in the design table of the data
    :param threshold: Threshold for choosing patients that are "extreme" w.r.t. the controls.
    :return: Dataframe containing the Single Sample scores using Z_Scores
    """
    # Check if the control variable is as per the R Naming standards
    assert control[0].isalpha(), "Please pass the control indicator contains atleast 1 alphabet."

    # Transpose matrix to get the patients as the rows
    data = data.transpose()

    # Give each label an integer to represent the labels during classification
    label_mapping = {
        key: val
        for val, key in enumerate(np.unique(design['Target']))
    }

    # Make sure the number of rows of transposed data and design are equal
    assert len(data) == len(design)

    # Extract the controls from the dataset
    controls = data[list(design.Target == control)]

    # Calculate the "Z Score" of each individual patient
    mean = controls.mean(axis=0)
    std = controls.std(axis=0)
    z_scores = (data - mean) / std

    out_z_scores = z_scores.copy()

    # Values that are greater than the 2 sigma or lesser than negative 2 sigma are considered as extremes

    out_z_scores[z_scores > threshold] = 1
    out_z_scores[z_scores < -threshold] = -1

    # Values between upper and lower limit are assigned 0
    out_z_scores[(z_scores < threshold) & (z_scores > -threshold)] = 0

    df = pd.DataFrame(data=out_z_scores, index=data.index, columns=data.columns)

    label = design['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)

    output_df = df.apply(_bin).copy()

    output_df['label'] = label.values

    return output_df


def _bin(row: pd.Series[int]) -> List[int]:
    """Replace values greater than 0 as 1 and lesser than 0 as -1."""
    return [
        1 if (val > 0) else (-1 if (val < 0) else 0)
        for val in row
    ]
