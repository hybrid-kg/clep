# -*- coding: utf-8 -*-

"""Embed the gene data & pathway using Network Representation Learning."""

import pickle
from typing import TextIO, Dict, Tuple

import numpy as np
import pandas as pd
from bionev.embed_train import embedding_training


def do_nrl(data: pd.DataFrame, kg_data: pd.DataFrame, out, method) -> None:
    """Carry out Network-Representation Learning for the given pandas dataframe.

    :param data: Dataframe containing the data to be embedded
    :param kg_data: Dataframe containing the knowledge graph
    :param out: Path to the output directory
    :param method: model that should be used for creating the embedding
    """

    # Get labels
    label = data['label']
    data = data.drop(columns='label')

    # Output edge-list
    with open(f'{out}/data.edgelist', 'w') as data_edge:
        pat_mapping, gene_mapping, label_mapping = _make_data_edgelist(data, label, data_edge)

        joint_mapping = {**pat_mapping, **gene_mapping}

        kg_mapping = _make_kg_edgelist(kg_data, joint_mapping, data_edge)

    # Pickle patient, gene & knowledge graph to numerical representation dictionaries
    with open(f'{out}/mapping.pkl', 'wb') as mapping_file:
        mapping = {**joint_mapping, **kg_mapping}
        pickle.dump(mapping, mapping_file)

    # Generate embeddings using the model
    embeddings = _gen_embedding(input_path=f'{out}/data.edgelist',
                                method=method,
                                model_out=f'{out}/model.gz',
                                embeddings_out=f'{out}/raw.embedding',
                                word2vec_model_out=f'{out}/word2vec.pkl')

    # Initialize lists for output dataframe
    label_col = list()
    vectors = list()
    index = list()

    # Create inverse of the sample mapping
    inv_pat_map = {v: k for k, v in pat_mapping.items()}

    # Loop over the embeddings to find the sample nodes
    for node in embeddings.keys():
        if int(node) in label_mapping.keys():
            label_col.append(label_mapping[int(node)])
            vectors.append(embeddings[node])
            index.append(inv_pat_map[int(node)])

    # Create an output dataframe with the embeddings and labels for the samples
    out_df = pd.DataFrame(index=index, data=vectors)
    out_df['label'] = label_col

    out_df.to_csv(f'{out}/embedding.tsv', sep='\t')


def _make_data_edgelist(data, label, data_edge) -> Tuple[Dict[str: int], Dict[str: int], Dict[str: int]]:
    """Create an edgelist for the patient data."""
    # Create a mapping from every samples to an unique node ID for the node representation
    pat_mapping = dict((key, val) for val, key in enumerate(np.unique(data['patients'])))

    # Get the max node ID so as to continue the mapping from that number
    max_val = _get_max_dict_val(pat_mapping)

    # Create a mapping for all the genes starting from the max node ID
    gene_mapping = dict(zip(data.columns[1:], range(max_val + 1, len(data.columns[1:]) + max_val + 1)))

    # Only the single sample expression values that are not either up or down regulated are considered for the embedding
    valid_patients = []
    for patient, gene, value in pd.melt(data, id_vars=['patients']).values:
        if value == 0:
            continue
        valid_patients.append(patient)

        print(pat_mapping[patient], gene_mapping[gene], sep=' ', file=data_edge)

    # Add the valid samples with their corresponding labels to a mapping
    label_mapping = dict()
    for idx in data.index:
        if data.at[idx, 'patients'] in valid_patients:
            label_mapping[pat_mapping[data.at[idx, 'patients']]] = label[idx]

    return pat_mapping, gene_mapping, label_mapping


def _make_kg_edgelist(kg_data, joint_mapping, data_edge: TextIO) -> Dict[str: int]:
    """Create an edgelist for the knowledge graph data."""
    kg_mapping = dict()

    # Get the max node ID for the joint mapping of the samples and the genes
    max_val = _get_max_dict_val(joint_mapping)

    # Loop over the 1st (source) and the 3rd (target) columns to add their values to the knowledge graph mapping or
    # use the ID from the joint mapping if it already exists after normalization of the values.
    for col in [kg_data.iloc[:, 0], kg_data.iloc[:, 2]]:
        for val_raw in col:
            val = val_raw.split(':')[1] if len(val_raw.split(':')) > 1 else val_raw.split(':')[0]

            if val in joint_mapping.keys():
                kg_mapping[val_raw] = joint_mapping[val]

            else:
                max_val += 1
                kg_mapping[val_raw] = max_val

    # Append the source to target mapping to the main data edgelist
    for idx in kg_data.index:
        print(kg_mapping[kg_data.iat[idx, 0]], kg_mapping[kg_data.iat[idx, 2]], sep=' ', file=data_edge)

    return kg_mapping


def _get_max_dict_val(dictionary: dict) -> int:
    """Get the maximum value from a key, value pair in a given dictionary."""
    return dictionary[max(dictionary, key=lambda value: dictionary[value])]


def _gen_embedding(
        *,
        input_path,
        method,
        embeddings_out,
        model_out,
        word2vec_model_out,
        dimensions: int = 300,
        number_walks: int = 8,
        walk_length: int = 8,
        window_size: int = 4,
        p: float = 1.5,
        q: float = 2.1,
        alpha: float = 0.1,
        beta: float = 4,
        epochs: int = 5,
        kstep: int = 4,
        order: int = 3,
        weighted: bool = False,
):
    """Generate a NRL embedding using the given model."""
    model = embedding_training(
        train_graph_filename=input_path,
        method=method,
        dimensions=dimensions,
        number_walks=number_walks,
        walk_length=walk_length,
        window_size=window_size,
        p=p,
        q=q,
        alpha=alpha,
        beta=beta,
        epochs=epochs,
        kstep=kstep,
        order=order,
        weighted=weighted,
    )

    # Save the model
    model.save_model(model_out)

    # Save the embeddings
    model.save_embeddings(embeddings_out)

    # Get the embeddings
    if method == 'LINE':
        embeddings = model.get_embeddings_train()
    else:
        # Save the Word2Vec model in-case Deepwalk or Node2Vec was used
        model.word2vec.save(word2vec_model_out)
        embeddings = model.get_embeddings()

    return embeddings
