# -*- coding: utf-8 -*-

"""Vectorizing the gene data & pathway using Network Representation Learning."""

import pandas as pd
from typing import TextIO
import pickle
import numpy as np


# TODO: the input of this method should be a csv with three columns (subject, relationship, object)
# 1. Check if it is a BEL file, if so -> convert to csv file (ask charlie how to do it)
# 2. Check that csv file has the right input (Ask Rana)
# 3. Using the gene expression files, you add new edges for each patients with its corresponding genes (if they exist in the BEl graph)
# depending on the threshold (default FC 2.0). that should be an argument
# 4. Plug that merged network into PyKEEN (the arguments should specify the type of model and its arguments)
# 5. Returns you the predicted links ranked by likelihood but what we really want are the patient (nodes) vectors to do the clustering/prediction


def do_nrl(data: pd.DataFrame, design: pd.DataFrame, edge_out: TextIO, edge_out_num: TextIO, label_edge) -> None:
    _make_edgelist(data, design, edge_out, edge_out_num, label_edge)


def _make_edgelist(data, design, edge_out, edge_out_num, label_edge):
    label2num_mapping = dict(zip(np.unique(design['Target']), range(len(np.unique(design['Target'])))))
    node2num_mapping = dict(zip(data['patients'], range(len(data['patients']))))
    node2num_mapping.update(
        dict(zip(data.columns[1:], range(len(data['patients']), len(data.columns[1:]) + len(data['patients']))))
        )
    with open('node2num_mapping.pkl', 'wb') as pkl_file:
        pickle.dump(node2num_mapping, pkl_file)
    corr=[]
    for patient, gene, value in pd.melt(data, id_vars=['patients']).values:
        if value == 1:
            relation = 'positiveCorrelation'
        elif value == -1:
            relation = 'negativeCorrelation'
        else:
            continue
        corr.append(patient)

        print(patient, f'HGNC:{gene}', {'relation': relation}, sep='\t', file=edge_out)

        print(node2num_mapping[patient], node2num_mapping[gene], sep=' ', file=edge_out_num)

    for idx in design.index:
        try:
            if design.at[idx, 'FileName'] in corr:
                print(node2num_mapping[design.at[idx, 'FileName']], label2num_mapping[design.at[idx, 'Target']], sep=' ',
              file=label_edge)
        except KeyError:
            continue
