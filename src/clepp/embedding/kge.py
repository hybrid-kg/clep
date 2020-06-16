# -*- coding: utf-8 -*-

"""Embed patients with the biomedical entities (genes and metabolites) using Knowledge graph embedding."""
from pykeen.hpo.hpo import hpo_pipeline
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
# from pykeen.datasets import DataSet
from pykeen.models.base import Model
import pandas as pd
import numpy as np

from typing import Tuple, Optional


def do_kge(
    edgelist: pd.DataFrame,
    design: pd.DataFrame,
    out: str,
    model: str,
    return_patients: Optional[bool] = True,
    train_size: Optional[float] = 0.8,
    validation_size: Optional[float] = 0.1
):
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

    # kge_dataset = DataSet(
    #     training_path=f'{out}/train.edgelist',
    #     validation_path=f'{out}/validation.edgelist',
    #     testing_path=f'{out}/test.edgelist'
    # )

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

    optimal_params = run_optimization(
        dataset=(training_factory, validation_factory, testing_factory),
        model=model,
        out_dir=out
    ).study.best_params

    best_model = run_pipeline(
        dataset=(training_factory, validation_factory, testing_factory),
        model=model,
        optimal_params=optimal_params,
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
    return model.entity_embeddings.weight.detach().cpu().numpy()


def run_optimization(dataset: Tuple[TriplesFactory, TriplesFactory, TriplesFactory], model: str, out_dir: str):
    """Run HPO."""
    training_factory, testing_factory, validation_factory = dataset
    # Define model
    model_kwargs = dict(
        automatic_memory_optimization=True
    )
    model_kwargs_ranges = dict(
        embedding_dim=dict(
            type='int',
            low=6,
            high=7,
            scale='power_two',
        )
    )

    # Define Training Loop
    training_loop = 'owa'

    # Define optimizer
    optimizer = 'adam'
    optimizer_kwargs = dict(
        weight_decay=0.0
    )
    optimizer_kwargs_ranges = dict(
        lr=dict(
            type='float',
            low=0.001,
            high=0.1,
            sclae='log',
        )
    )

    # Define Loss Fct.
    loss_function = 'NSSALoss'
    loss_kwargs = dict()
    loss_kwargs_ranges = dict(
        margin=dict(
            type='float',
            low=1,
            high=30,
            q=2.0,
        ),
        adversarial_temperature=dict(
            type='float',
            low=0.1,
            high=1.0,
            q=0.1,
        )
    )

    # Define Regularizer
    regularizer = 'NoRegularizer'

    # Define Negative Sampler
    negative_sampler = 'BasicNegativeSampler'
    negative_sampler_kwargs = dict()
    negative_sampler_kwargs_ranges = dict(
        num_negs_per_pos=dict(
            type='int',
            low=1,
            high=10,
            q=1,
        )
    )

    # Define Evaluator
    evaluator = 'RankBasedEvaluator'
    evaluator_kwargs = dict(
        filtered=True,
    )
    evaluation_kwargs = dict(
        batch_size=None  # searches for maximal possible in order to minimize evaluation time
    )

    # Define Training Arguments
    training_kwargs = dict(
        num_epochs=1000,
        label_smoothing=0.0,
    )
    training_kwargs_ranges = dict(
        batch_size=dict(
            type='int',
            low=7,
            high=9,
            scale='power_two',
        )
    )

    # Define Early Stopper
    stopper = 'early'
    stopper_kwargs = dict(
        frequency=50,
        patience=2,
        delta=0.002,
    )

    # Define Optuna Related Parameters
    n_trials = 100
    timeout = 86400
    metric = 'hits@10'
    direction = 'maximize'
    sampler = 'random'
    pruner = 'nop'

    # Define HPO pipeline
    hpo_results = hpo_pipeline(
        dataset=None,
        training_triples_factory=training_factory,
        testing_triples_factory=testing_factory,
        validation_triples_factory=validation_factory,
        model=model,
        model_kwargs=model_kwargs,
        model_kwargs_ranges=model_kwargs_ranges,
        loss=loss_function,
        loss_kwargs=loss_kwargs,
        loss_kwargs_ranges=loss_kwargs_ranges,
        regularizer=regularizer,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        optimizer_kwargs_ranges=optimizer_kwargs_ranges,
        training_loop=training_loop,
        training_kwargs=training_kwargs,
        training_kwargs_ranges=training_kwargs_ranges,
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
        negative_sampler_kwargs_ranges=negative_sampler_kwargs_ranges,
        stopper=stopper,
        stopper_kwargs=stopper_kwargs,
        evaluator=evaluator,
        evaluator_kwargs=evaluator_kwargs,
        evaluation_kwargs=evaluation_kwargs,
        n_trials=n_trials,
        timeout=timeout,
        metric=metric,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    hpo_results.save_to_directory(f'{out_dir}/pykeen_results_optim')

    return hpo_results


def run_pipeline(
        dataset: Tuple[TriplesFactory, TriplesFactory, TriplesFactory],
        model: str,
        optimal_params: dict,
        out_dir: str
):
    """Run Pipeline."""
    training_factory, testing_factory, validation_factory = dataset
    # Define model
    model_kwargs = dict(
        automatic_memory_optimization=True,
        embedding_dim=optimal_params['model.embedding_dim']
    )

    # Define Training Loop
    training_loop = 'owa'

    # Define optimizer
    optimizer = 'adam'
    optimizer_kwargs = dict(
        weight_decay=0.0,
        lr=optimal_params['optimizer.lr']
    )

    # Define Loss Fct.
    loss_function = 'NSSALoss'
    loss_kwargs = dict(
        margin=optimal_params['loss.margin'],
        adversarial_temperature=optimal_params['loss.adversarial_temperature']
    )

    # Define Regularizer
    regularizer = 'NoRegularizer'

    # Define Negative Sampler
    negative_sampler = 'BasicNegativeSampler'
    negative_sampler_kwargs = dict(
        num_negs_per_pos=optimal_params['negative_sampler.num_negs_per_pos']
    )

    # Define Evaluator
    evaluator = 'RankBasedEvaluator'
    evaluator_kwargs = dict(
        filtered=True,
    )
    evaluation_kwargs = dict(
        batch_size=None  # searches for maximal possible in order to minimize evaluation time
    )

    # Define Training Arguments
    training_kwargs = dict(
        num_epochs=1000,
        label_smoothing=0.0,
        batch_size=optimal_params['training.batch_size']
    )

    # Define Early Stopper
    stopper = 'early'
    stopper_kwargs = dict(
        frequency=50,
        patience=2,
        delta=0.002,
    )

    # Define HPO pipeline
    pipeline_results = pipeline(
        dataset=None,
        training_triples_factory=training_factory,
        testing_triples_factory=testing_factory,
        validation_triples_factory=validation_factory,
        model=model,
        model_kwargs=model_kwargs,
        loss=loss_function,
        loss_kwargs=loss_kwargs,
        regularizer=regularizer,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        training_loop=training_loop,
        training_kwargs=training_kwargs,
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
        stopper=stopper,
        stopper_kwargs=stopper_kwargs,
        evaluator=evaluator,
        evaluator_kwargs=evaluator_kwargs,
        evaluation_kwargs=evaluation_kwargs,
    )

    # TODO: Save replicates once pytorch fixes the issue #37703
    pipeline_results.save_to_directory(f'{out_dir}/pykeen_results_final', save_replicates=False)

    return pipeline_results
