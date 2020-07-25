# -*- coding: utf-8 -*-

"""Embed patients with the biomedical entities (genes and metabolites) using Knowledge graph embedding."""
import os
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pykeen.hpo.hpo import hpo_pipeline
from pykeen.models.base import Model
from pykeen.pipeline import pipeline_from_path
from pykeen.triples import TriplesFactory


def do_kge(
        edgelist: pd.DataFrame,
        design: pd.DataFrame,
        out: str,
        model_config: Dict,
        return_patients: Optional[bool] = True,
        train_size: Optional[float] = 0.8,
        validation_size: Optional[float] = 0.1
) -> pd.DataFrame:
    """Carry out KGE on the given data.

    :param edgelist: Dataframe containing the patient-feature graph in edgelist format
    :param design: Dataframe containing the design table for the data
    :param out: Output folder for the results
    :param model_config: Configuration file for the KGE models, in JSON format.
    :param return_patients: Flag to indicate if the final data should contain only patients or even the features
    :param train_size: Size of the training data for KGE ranging from 0 - 1
    :param validation_size: Size of the validation data for KGE ranging from 0 - 1. It must be lower than training size
    :return: Dataframe containing the embedding from the KGE
    """
    unique_nodes = edgelist[~edgelist['label'].isna()].drop_duplicates('source')

    label_mapping = {patient: label for patient, label in zip(unique_nodes['source'], unique_nodes['label'])}

    edgelist = edgelist.drop(columns='label')

    # Split the edgelist into training, validation and testing data
    train, validation, test = _weighted_splitter(
        edgelist=edgelist,
        train_size=train_size,
        validation_size=validation_size
    )

    train.to_csv(f'{out}/train.edgelist', sep='\t', index=False, header=False)
    validation.to_csv(f'{out}/validation.edgelist', sep='\t', index=False, header=False)
    test.to_csv(f'{out}/test.edgelist', sep='\t', index=False, header=False)

    create_inverse_triples = False  # In a second HPO configuration, this can be set to true
    training_factory = TriplesFactory(
        path=f'{out}/train.edgelist',
        create_inverse_triples=create_inverse_triples,
    )
    validation_factory = TriplesFactory(
        path=f'{out}/validation.edgelist',
        create_inverse_triples=create_inverse_triples,
    )
    testing_factory = TriplesFactory(
        path=f'{out}/test.edgelist',
        create_inverse_triples=create_inverse_triples,
    )

    run_optimization(
        dataset=(training_factory, validation_factory, testing_factory),
        model_config=model_config,
        out_dir=out
    )

    best_model = run_pipeline(
        dataset=(training_factory, validation_factory, testing_factory),
        out_dir=out
    ).model

    # Get the embedding as a numpy array
    embedding_values = _model_to_numpy(best_model)

    # Create columns as component names
    embedding_columns = [f'Component_{i}' for i in range(1, embedding_values.shape[1] + 1)]

    # Get the nodes of the training triples as index
    node_list = list(best_model.triples_factory.entity_to_id.keys())
    embedding_index = sorted(node_list, key=lambda x: best_model.triples_factory.entity_to_id[x])

    embedding = pd.DataFrame(data=embedding_values, columns=embedding_columns, index=embedding_index)

    if return_patients:
        # TODO: Use clustering before classification to see if embeddings are already good enough
        embedding = embedding[embedding.index.isin(design['FileName'])]

        for index in embedding.index:
            embedding.at[index, 'label'] = label_mapping[index]

    return embedding


def _weighted_splitter(
        edgelist: pd.DataFrame,
        train_size: Optional[float] = 0.8,
        validation_size: Optional[float] = 0.1
) -> Tuple[pd.DataFrame, ...]:
    """Split the given edgelist into training, validation and testing sets on the basis of the ratio of relations.

    :param edgelist: Edgelist in the form of (Source, Relation, Target)
    :param train_size: Size of the training data
    :param validation_size: Size of the training data
    :return: Tuple containing the train, validation & test splits
    """
    # Validation size is the size of the percentage of the remaining data (i.e. If required validation size is 10% of
    # the original data & training size is 80% then the new validation size is 50% of the data without the training
    # data. The similar calculation is done for training size, hence it is always 1
    validation_size = validation_size / (1 - train_size)
    test_size = 1

    # Get the unique relations in the network
    unique_relations = sorted(edgelist['relation'].unique())

    data = edgelist.drop_duplicates().copy()

    split = []
    # Split the data to get training, validation and test samples
    for frac_size in [train_size, validation_size, test_size]:
        frames = []
        # Random sampling of the data for every type of relation
        for relation in unique_relations:
            temp = data[data['relation'] == relation].sample(frac=frac_size)

            data = data[~data.index.isin(temp.index)]

            frames.append(temp)
        # Join all the different relations in one dataframe
        split.append(pd.concat(frames, ignore_index=True, sort=False))

    return tuple(split)


def _model_to_numpy(
        model: Model
) -> np.array:
    """Retrieve embedding from the models as a numpy array."""
    return model.entity_embeddings.weight.detach().cpu().numpy()


def run_optimization(dataset: Tuple[TriplesFactory, TriplesFactory, TriplesFactory], model_config: Dict, out_dir: str):
    """Run HPO."""
    training_factory, testing_factory, validation_factory = dataset

    # Define HPO pipeline
    hpo_results = hpo_pipeline(
        dataset=None,
        training_triples_factory=training_factory,
        testing_triples_factory=testing_factory,
        validation_triples_factory=validation_factory,
        model=model_config["model"],
        model_kwargs=model_config["model_kwargs"],
        model_kwargs_ranges=model_config["model_kwargs_ranges"],
        loss=model_config["loss_function"],
        loss_kwargs=model_config["loss_kwargs"],
        loss_kwargs_ranges=model_config["loss_kwargs_ranges"],
        regularizer=model_config["regularizer"],
        optimizer=model_config["optimizer"],
        optimizer_kwargs=model_config["optimizer_kwargs"],
        optimizer_kwargs_ranges=model_config["optimizer_kwargs_ranges"],
        training_loop=model_config["training_loop"],
        training_kwargs=model_config["training_kwargs"],
        training_kwargs_ranges=model_config["training_kwargs_ranges"],
        negative_sampler=model_config["negative_sampler"],
        negative_sampler_kwargs=model_config["negative_sampler_kwargs"],
        negative_sampler_kwargs_ranges=model_config["negative_sampler_kwargs_ranges"],
        stopper=model_config["stopper"],
        stopper_kwargs=model_config["stopper_kwargs"],
        evaluator=model_config["evaluator"],
        evaluator_kwargs=model_config["evaluator_kwargs"],
        evaluation_kwargs=model_config["evaluation_kwargs"],
        n_trials=model_config["n_trials"],
        timeout=model_config["timeout"],
        metric=model_config["metric"],
        direction=model_config["direction"],
        sampler=model_config["sampler"],
        pruner=model_config["pruner"],
    )

    optimization_dir = os.path.join(out_dir, 'pykeen_results_optim')
    if not os.path.isdir(optimization_dir):
        os.makedirs(optimization_dir)

    hpo_results.save_to_directory(optimization_dir)

    return None


def run_pipeline(
        dataset: Tuple[TriplesFactory, TriplesFactory, TriplesFactory],
        out_dir: str
):
    """Run Pipeline."""
    training_factory, testing_factory, validation_factory = dataset

    config_path = os.path.join(out_dir, 'pykeen_results_optim', 'best_pipeline', 'pipeline_config.json')
    pipeline_results = pipeline_from_path(
        path=config_path,
        training_triples_factory=training_factory,
        testing_triples_factory=testing_factory,
        validation_triples_factory=validation_factory,
    )

    best_pipeline_dir = os.path.join(out_dir, 'pykeen_results_final')
    if not os.path.isdir(best_pipeline_dir):
        os.makedirs(best_pipeline_dir)

    pipeline_results.save_to_directory(best_pipeline_dir, save_replicates=True)

    return pipeline_results
