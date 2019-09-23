# -*- coding: utf-8 -*-

"""Vectorize Patients binarily based on if the absolute expression passes a threshold value."""


def do_thresh2vec(data):
    data.apply(lambda x: 1 if x > 0 else(-1 if x < 0 else 0))
    return data
