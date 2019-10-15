# -*- coding: utf-8 -*-

"""Vectorize Patients binarily based on if the absolute expression passes a threshold value."""

# TODO: Binning limma and Z_Score in this file

import pandas as pd


def do_binning(data: pd.DataFrame) -> pd.DataFrame:
    output = data.apply(_bin).copy()
    return output


def _bin(row):
    bin_data = [1 if (val > 0) else (-1 if (val < 0) else 0) for val in row]
    return bin_data
