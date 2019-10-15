# -*- coding: utf-8 -*-

"""Vectorize the gene data & pathway using Network Representation Learning."""

import pickle
from typing import TextIO

import numpy as np
import pandas as pd


# TODO: the input of this method should be a csv with three columns (subject, relationship, object)
# 1. Check if it is a BEL file, if so -> convert to csv file (ask charlie how to do it)
# 2. Check that csv file has the right input (Ask Rana)
# 3. Using the gene expression files, you add new edges for each patients with its corresponding genes (if they exist in the BEl graph)
# depending on the threshold (default FC 2.0). that should be an argument
# 4. Plug that merged network into PyKEEN (the arguments should specify the type of model and its arguments)
# 5. Returns you the predicted links ranked by likelihood but what we really want are the patient (nodes) vectors to do the clustering/prediction


def do_nrl(data: pd.DataFrame, design: pd.DataFrame, edge_out: TextIO, edge_out_num: TextIO, label_edge, control) -> \
        None:
    """Carry out Network-Representation Learning for the given pandas dataframe."""

    labels = design[design['Target'] != control]
    _make_edgelist(data, labels, edge_out, edge_out_num, label_edge)


def _make_edgelist(data, design, edge_out, edge_out_num, label_edge):
    label2num_mapping = dict(zip(np.unique(design['Target']), range(len(np.unique(design['Target'])))))
    pat2num_mapping = dict(zip(data['patients'], range(len(data['patients']))))
    max_val = pat2num_mapping[max(pat2num_mapping, key=lambda i: pat2num_mapping[i])]
    gene2num_mapping = dict(zip(data.columns[1:], range(max_val + 1, len(data.columns[1:]) + max_val + 1)))

    with open('pat2num_mapping.pkl', 'wb') as pat_file, open('gene2num_mapping.pkl', 'wb') as gene_file:
        pickle.dump(pat2num_mapping, pat_file)
        pickle.dump(gene2num_mapping, gene_file)

    corr = []
    for patient, gene, value in pd.melt(data, id_vars=['patients']).values:
        if value == 1:
            relation = 'positiveCorrelation'
        elif value == -1:
            relation = 'negativeCorrelation'
        else:
            continue
        corr.append(patient)

        print(patient, relation, f'HGNC:{gene}', sep='\t', file=edge_out)

        print(pat2num_mapping[patient], gene2num_mapping[gene], sep=' ', file=edge_out_num)

    for idx in design.index:
        try:
            if design.at[idx, 'FileName'] in corr:
                print(pat2num_mapping[design.at[idx, 'FileName']], label2num_mapping[design.at[idx, 'Target']], sep=' ',
                      file=label_edge)
        except KeyError:
            continue
