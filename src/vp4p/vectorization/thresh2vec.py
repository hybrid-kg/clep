# -*- coding: utf-8 -*-

"""Vectorize Patients binarily based on if the absolute expression passes a threshold value."""

# TODO: Thresholding for Z-Score


def do_thresh2vec(data):
    output = data.apply(_thresh).copy()
    return output


def _thresh(row):
    """"""
    bin_data = [1 if (val > 0) else(-1 if (val < 0) else 0) for val in row]
    return bin_data
