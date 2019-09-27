# -*- coding: utf-8 -*-

"""Vectorizing the gene data & pathway using Network Representation Learning."""

import pandas as pd
from typing import TextIO
import itertools as itt


# TODO: the input of this method should be a csv with three columns (subject, relationship, object)
# 1. Check if it is a BEL file, if so -> convert to csv file (ask charlie how to do it)
# 2. Check that csv file has the right input (Ask Rana)
# 3. Using the gene expression files, you add new edges for each patients with its corresponding genes (if they exist in the BEl graph)
# depending on the threshold (default FC 2.0). that should be an argument
# 4. Plug that merged network into PyKEEN (the arguments should specify the type of model and its arguments)
# 5. Returns you the predicted links ranked by likelihood but what we really want are the patient (nodes) vectors to do the clustering/prediction


def do_nrl(data: pd.DataFrame, edge_out: TextIO) -> None:
    _make_edgelist(data, edge_out)


def _make_edgelist(data, edge_out):
    for patient, gene, value in pd.melt(data, id_vars=['patients']).values:
        if value == 1:
            relation = 'positiveCorrelation'
        elif value == -1:
            relation = 'negativeCorrelation'
        else:
            continue
        print(patient, f'HGNC:{gene}', {'relation': relation}, sep='\t', file=edge_out)

