# -*- coding: utf-8 -*-

"""Embed Patients binarily based on if the absolute expression passes a threshold value."""

# TODO: Binning limma and Z_Score in this file

from typing import List

import pandas as pd


def do_binning(data: pd.DataFrame) -> pd.DataFrame:
    """Perform binning on the given dataframe.

    :param data: Dataframe containing the single sample scores from limma or Z_scores
    :return Dataframe containing binned scores
    """
    label = data['label']
    data = data.drop(columns='label')

    output = data.apply(_bin).copy()

    output['label'] = label

    return output


def _bin(row: pd.Series) -> List[int]:
    """Replace values greater than 0 as 1 and lesser than 0 as -1."""
    return [
        1 if (val > 0) else (-1 if (val < 0) else 0)
        for val in row
    ]
