# -*- coding: utf-8 -*-

"""Carry out Z-Score based single sample DE analysis."""

# Research question: how much does a single disease patient differ from the distribution of normal ones? In that case you can calculate a z-score for each disease patient relative to normal ones.

import numpy as np
import pandas as pd


def do_z_score(data: pd.DataFrame, design: pd.DataFrame, control: str = 'Control') -> pd.DataFrame:
    """Carry out Z-Score based single sample DE analysis.

    :param data: Dataframe containing the gene expression values
    :param design: Dataframe containing the design table for the data
    :param control: label used for representing the control in the design table of the data
    :return Dataframe containing the Single Sample scores using Z_Scores
    """
    # Check if the control variable is as per the R Naming standards
    assert control[0].isalpha()

    # Transpose matrix to get the patients as the rows
    data = data.transpose()

    # Give each label an integer to represent the labels during classification
    label_mapping = dict(zip(np.unique(design['Target']), range(len(np.unique(design['Target'])))))

    # Make sure the number of rows of transposed data and design are equal
    assert len(data) == len(design)

    # Separate the dataset into controls and samples
    controls = data[list(design.Target == control)]
    samples = data[list(design.Target != control)]

    # Calculate the "Z Score" of each individual patient
    control_mean = controls.mean(axis=0)
    control_std = controls.std(axis=0)
    z_scores = (samples - control_mean) / control_std

    out_z_scores = np.where(np.abs(z_scores) > 1, z_scores, 0)

    df = pd.DataFrame(data=out_z_scores, index=data.index[:len(samples)], columns=data.columns)

    label = design[design['Target'] != control]['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)

    df['label'] = label

    return df
