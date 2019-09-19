# -*- coding: utf-8 -*-

"""Z-Score based single sample DE analysis."""

import numpy as np
import pandas as pd


def do_z_score(data: pd.DataFrame, design: pd.DataFrame, control: str = 'Control'):
    data = data.transpose()

    temp = np.zeros((1, len(data.columns)))
    for ind in design[design.Target == control].index.values:
        temp = np.vstack([temp, data.iloc[ind, :]])
    controls = temp[1:]

    temp = np.zeros((1, len(data.columns)))
    for ind in design[design.Target != control].index.values:
        temp = np.vstack([temp, data.iloc[ind, :]])
    samples = temp[1:]

    control_mean = controls.mean(axis=0)
    control_std = controls.std(axis=0)
    z_scores = (samples - control_mean) / control_std

    return pd.DataFrame(data=z_scores, index=data.index[:len(samples)], columns=data.columns)
