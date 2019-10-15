# -*- coding: utf-8 -*-

"""Vectorize the gene data & pathway using Network Representation Learning."""

import pickle

import numpy as np
import pandas as pd


# TODO: Implement BioNEV


def do_nrl(data: pd.DataFrame, design: pd.DataFrame, out, control) -> None:
    """Carry out Network-Representation Learning for the given pandas dataframe."""

    labels = design[design['Target'] != control]
    with open(f'{out}/text.edgelist', 'w') as text_out, \
            open(f'{out}/number.edgelist', 'w') as num_out, \
            open(f'{out}/label.edgelist', 'w') as label_out:
        pat2num_mapping, gene2num_mapping, label2num_mapping = _make_edgelist(data, labels, text_out,
                                                                              num_out, label_out)

    with open(f'{out}/pat2num_mapping.pkl', 'wb') as pat_file, \
            open(f'{out}/gene2num_mapping.pkl', 'wb') as gene_file, \
            open(f'{out}/label_mapping.pkl', 'wb') as label_file:
        pickle.dump(pat2num_mapping, pat_file)
        pickle.dump(gene2num_mapping, gene_file)
        pickle.dump(label2num_mapping, label_file)


def _make_edgelist(data, design, edge_out, edge_out_num, label_edge):
    label2num_mapping = dict(zip(np.unique(design['Target']), range(len(np.unique(design['Target'])))))
    pat2num_mapping = dict(zip(data['patients'], range(len(data['patients']))))
    max_val = pat2num_mapping[max(pat2num_mapping, key=lambda i: pat2num_mapping[i])]
    gene2num_mapping = dict(zip(data.columns[1:], range(max_val + 1, len(data.columns[1:]) + max_val + 1)))

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

    return pat2num_mapping, gene2num_mapping, label2num_mapping
