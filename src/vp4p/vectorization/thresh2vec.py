# -*- coding: utf-8 -*-

"""Vectorize Patients binarily based on if the absolute expression passes a threshold value."""

# TODO: Thresholding for Z-Score

def do_thresh2vec(data):
    output = data.apply(lambda row: [1 if (val > 0) else(-1 if (val < 0) else 0) for val in row]).copy()
    return output
