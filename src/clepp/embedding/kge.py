# -*- coding: utf-8 -*-

"""Embed patients with the biomedical entities (genes and metabolites) using Knowledge graph embedding."""

from pykeen.pipeline import pipeline
from pykeen.hpo.hpo import hpo_pipeline
from pykeen.datasets import DataSet
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.models.base import Model
import pandas as pd
import numpy as np

from typing import Tuple, Optional


def do_kge(
    edgelist: pd.DataFrame,
    design: pd.DataFrame,
    out: str,
    return_patients: Optional[bool] = True,
    model: Optional[str] = 'TransE',
    train_size: Optional[float] = 0.8,
    validation_size: Optional[float] = 0.1
):
    # labels = edgelist[3]
    # edgelist.drop(columns=3, inplace=True)
    # Split the edgelist into training, validation and testing data
    train, validation, test = _weighted_splitter(
        edgelist=edgelist,
        train_size=train_size,
        validation_size=validation_size
    )

    train.to_csv(f'{out}/train.edgelist', sep='\t', index=False, header=False)
    validation.to_csv(f'{out}/validation.edgelist', sep='\t', index=False, header=False)
    test.to_csv(f'{out}/test.edgelist', sep='\t', index=False, header=False)

    # Create triples for training and testing TODO: Add validation once implemented
    # training_triples = TriplesFactory(
    #     path=f'{out}/train.edgelist'
    # )
    # testing_triples = TriplesFactory(
    #     path=f'{out}/test.edgelist',
    #     entity_to_id=training_triples.entity_to_id,
    #     relation_to_id=training_triples.relation_to_id
    # )

    kge_dataset = DataSet(
        training_path=f'{out}/train.edgelist',
        validation_path=f'{out}/validation.edgelist',
        testing_path=f'{out}/test.edgelist'
    )

    # Run the pipeline with the given model
    # pipeline_result = pipeline(
    #     model=model,
    #     training_triples_factory=training_triples,
    #     testing_triples_factory=testing_triples,
    # )

    pipeline_result = hpo_pipeline(
        model=model,
        dataset=kge_dataset,
        n_trials=30
    )

    best_model = pipeline_result.study.best_trial.user_attrs['model']

    # Get the embedding as a numpy array
    embedding_values = _model_to_numpy(best_model)

    # Create 50 column names
    embedding_columns = [f'Component_{i}' for i in range(1, 51)]

    # Get the nodes of the training triples as index
    node_list = list(best_model.training_triples.entity_to_id.keys())
    embedding_index = sorted(node_list, key=lambda x: best_model.training_triples.entity_to_id[x])

    embedding = pd.DataFrame(data=embedding_values, columns=embedding_columns, index=embedding_index)

    if return_patients:
        # TODO: Use clustering before classification to see if embeddings are already good enough
        return embedding[embedding.index.isin(design['FileName'])]
    else:
        return embedding


def _weighted_splitter(
        edgelist: pd.DataFrame,
        train_size: Optional[float] = 0.8,
        validation_size: Optional[float] = 0.1
) -> Tuple[pd.DataFrame, ...]:
    """ Split the given edgelist into training, validation and testing sets on the basis of the ratio of relations

    :param edgelist: Edgelist in the form of (Source, Relation, Target)
    :param train_size: Size of the training data
    :param validation_size: Size of the training data
    :return:
    """
    # Validation size is the size of the percentage of the remaining data (i.e. If required validation size is 10% of
    # the original data & training size is 80% then the new validation size is 50% of the data without the training
    # data. The similar calculation is done for training size, hence it is always 1
    validation_size = validation_size / (1 - train_size)
    test_size = 1

    # Get the unique relations in the network
    unique_relations = np.unique(edgelist[1])

    data = edgelist.copy()

    split = []
    # Split the data to get training, validation and test samples
    for frac_size in [train_size, validation_size, test_size]:
        frames = []
        # Random sampling of the data for every type of relation
        for relation in unique_relations:
            temp = data[data[1] == relation].sample(frac=frac_size)

            data = data[~data.index.isin(temp.index)]

            frames.append(temp)
        # Join all the different relations in one dataframe
        split.append(pd.concat(frames, ignore_index=True, sort=False))

    return tuple(split)


def _model_to_numpy(
        model: Model
) -> np.array:
    return model.entity_embeddings.weight.detach().cpu().numpy()
