# -*- coding: utf-8 -*-

"""Command line interface."""

import json
import logging
import pickle
import warnings
from typing import List, Optional

import click
import numpy as np
import pandas as pd

from .classification import do_classification
from .embedding import (
    do_ss_evaluation, do_graph_gen, do_kge)
from .sample_scoring import (
    do_limma, do_z_score, do_ssgsea, do_radical_search
)
from .constants import scorers

logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """Run clep."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@main.group()
def sample_scoring() -> None:
    """List Single Sample Scoring methods available."""


data_option = click.option(
    '--data',
    help="Path to tab-separated gene expression data file",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
)
design_option = click.option(
    '--design',
    help="Path to tab-separated experiment design file",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
)
output_option = click.option(
    '--out',
    help="Path to the output folder",
    type=click.Path(file_okay=False, dir_okay=True, exists=True),
    required=True,
)
control_option = click.option(
    '--control',
    help="Annotated value for the control samples (must start with an alphabet)",
    type=str,
    required=False,
    default='Control',
    show_default=True,
)


@sample_scoring.command(help='Limma-based Single Sample Scoring')
@data_option
@design_option
@output_option
@click.option(
    '--alpha',
    help="Family-wise error rate",
    type=float,
    required=False,
    default=0.05,
    show_default=True,
)
@click.option(
    '--method',
    help="Method used for testing and adjustment of P-Values",
    type=str,
    required=False,
    default='fdr_bh',
    show_default=True,
)
@control_option
def limma(data: str, design: str, out: str, alpha: float, method: str, control: str) -> None:
    """Perform Single Sample limma based differential gene expression for all samples wrt the controls."""
    click.echo(
        f"Starting Limma Based Single Sample Scoring with {data} & {design} files and "
        f"saving it to {out}/sample_scoring.tsv"
    )

    data_df = pd.read_csv(data, sep='\t', index_col=0)
    design_df = pd.read_csv(design, sep='\t')

    output_df = do_limma(data=data_df, design=design_df, alpha=alpha, method=method, control=control)

    output_df.to_csv(f'{out}/sample_scoring.tsv', sep='\t')

    click.echo(f"Done with limma calculation for {data}")


@sample_scoring.command(help='Z-Score based Single Sample Scoring')
@data_option
@design_option
@output_option
@control_option
@click.option(
    '--threshold',
    help="Threshold for choosing patients that are 'extreme' w.r.t. the controls.\nIf the z_score of a gene is "
         "greater than this threshold the gene is either up or down regulated.",
    type=float,
    required=False,
    default=2.0,
    show_default=True,
)
def z_score(data: str, design: str, out: str, control: str, threshold: float) -> None:
    """Perform Single Sample Z_Score based differential gene expression for all samples wrt the controls."""
    click.echo(
        f"Starting Z-Score Based Single Sample Scoring with {data} & {design} files and "
        f"saving it to {out}/sample_scoring.tsv"
    )

    data_df = pd.read_csv(data, sep='\t', index_col=0)
    design_df = pd.read_csv(design, sep='\t', index_col=0)

    output = do_z_score(data=data_df, design=design_df, control=control, threshold=threshold)
    output.to_csv(f'{out}/sample_scoring.tsv', sep='\t')

    click.echo(f"Done with Z-Score calculation for {data}")


@sample_scoring.command(help='ssGSEA based Single Sample Scoring')
@data_option
@design_option
@output_option
@click.option(
    '--gs',
    help="Path to the .gmt geneset file",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
)
def ssgsea(data: str, design: str, out: str, gs: str) -> None:
    """Perform Single Sample GSEA for all the samples with prior knowledge (geneset)."""
    click.echo(
        f"Starting ssGSEA with {data} & {gs} files and saving it to {out}/sample_scoring.tsv")

    data_df = pd.read_csv(data, sep='\t', index_col=0)
    design_df = pd.read_csv(design, sep='\t', index_col=0)

    label_mapping = dict((key, val) for val, key in enumerate(np.unique(design_df['Target'])))

    single_sample_gsea = do_ssgsea(data_df, gs)

    label = design_df['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)

    single_sample_gsea['label'] = label.values

    single_sample_gsea.to_csv(f'{out}/sample_scoring.tsv', sep='\t')

    click.echo(f"Done with ssGSEA for {data}")


@sample_scoring.command(help='Radical Searching based Single Sample Scoring')
@data_option
@design_option
@output_option
@control_option
@click.option(
    '--threshold',
    help="Percentage of samples considered as 'extreme' on either side of the distribution",
    type=float,
    required=False,
    default=2.5,
    show_default=True,
)
@click.option(
    '-rs',
    '--ret_summary',
    help="Flag to indicate if the edge summary for patients must be created.",
    is_flag=True,
    show_default=True,
)
@click.option(
    '-cb',
    '--control_based',
    help="Run Radical Searching where the scoring is based on the control population instead of entire dataset",
    is_flag=True,
    required=False,
)
def radical_search(
        data: str,
        design: str,
        out: str,
        control: str,
        threshold: float,
        ret_summary: bool,
        control_based: bool
) -> None:
    """Search for samples that are extremes as compared to the samples."""
    assert 0 < threshold <= 100

    click.echo(
        f"Starting radical searching based single sample scoring with {data} & {design} files and "
        f"saving it to {out}/sample_scoring.tsv"
    )

    data_df = pd.read_csv(data, sep='\t', index_col=0)
    design_df = pd.read_csv(design, sep='\t')

    data_df.fillna(0, inplace=True)

    output, summary = do_radical_search(
        data=data_df,
        design=design_df,
        threshold=threshold,
        control=control,
        control_based=control_based
    )
    output.to_csv(f'{out}/sample_scoring.tsv', sep='\t')

    if ret_summary:
        summary.to_csv(f'{out}/radical_summary.tsv', sep='\t')

    click.echo(f"Done with Radical-Score calculation for {data}")


@main.group()
def embedding() -> None:
    """List Vectorization methods available."""


@embedding.command()
@click.option(
    '--data',
    help='Path to a set of binned files',
    type=click.Path(file_okay=True, dir_okay=False, exists=False),
    required=True,
    nargs=2
)
@click.option(
    '--label',
    help='Label for the set of binned files',
    type=str,
    required=True,
    nargs=2
)
def evaluate(data: str, label: str) -> None:
    """Perform Evaluation of the Embeddings."""
    click.echo(f"Starting Evaluation of the following files: \n{data}")

    # Adding individual files to a list
    data_lst = [
        pd.read_csv(file, sep='\t', index_col=0)
        for file in data
    ]

    # Cast string based label argument to a list
    result = do_ss_evaluation(data_lst, list(label))

    click.echo('===============Evaluation Results===============')
    click.echo(json.dumps(result, indent=4, sort_keys=True).replace('{', ' ').replace('}', ' '))
    click.echo('================================================')


@embedding.command()
@data_option
@output_option
@click.option(
    '--method',
    help='The method used to generate the network',
    type=click.Choice(['pathway_overlap', 'interaction_network', 'interaction_network_overlap']),
    required=False,
    default='interaction_network',
    show_default=True,
)
@click.option(
    '--kg',
    help="Path to the Knowledge Graph file in tsv format if Interaction Network method is chosen",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=False,
)
@click.option(
    '--gmt',
    help="Path to the gmt file if Pathway Overlap method is chosen",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=False,
)
@click.option(
    '--network_folder',
    help="Path to the folder containing all the knowledge graph files if Interaction Network Overlap method is chosen",
    type=click.Path(file_okay=False, dir_okay=True, exists=True),
    required=False,
)
@click.option(
    '--intersect_thr',
    help="Threshold to make edges in Pathway Overlap method",
    type=float,
    required=False,
    default=0.1,
    show_default=True,
)
@click.option(
    '-rs',
    '--ret_summary',
    help="Flag to indicate if the edge summary for patients must be created.",
    is_flag=True,
    show_default=True,
)
@click.option(
    '--jaccard_thr',
    help="Threshold to make edges in Interaction Network Overlap method",
    type=float,
    required=False,
    default=0.1,
    show_default=True,
)
def generate_network(
        data: str,
        out: str,
        method: str,
        intersect_thr: float,
        jaccard_thr: float,
        kg: Optional[str] = None,
        gmt: Optional[str] = None,
        ret_summary: bool = False,
        network_folder: Optional[str] = None,
) -> None:
    """Generate Network for the given data."""
    data_df = pd.read_csv(data, sep='\t', index_col=0)
    data_df.rename(columns={'Unnamed: 0': 'patients'}, inplace=True)

    if method == 'pathway_overlap':
        assert gmt is not None
        click.echo(f"Generating {method} based network with {data} & {gmt} and outputting it to {out}")

        graph_results = do_graph_gen(
            data=data_df,
            gmt=gmt,
            network_gen_method=method,
            intersection_threshold=intersect_thr,
            summary=ret_summary
        )

    elif method == 'interaction_network':
        assert kg is not None
        click.echo(f"Generating {method} based network with {data} & {kg} and outputting it to {out}")

        kg_data_df = pd.read_csv(kg, sep='\t', header=None)

        graph_results = do_graph_gen(
            data=data_df,
            kg_data=kg_data_df,
            network_gen_method=method,
            summary=ret_summary
        )

    else:
        assert network_folder is not None
        design = network_folder

        graph_results = do_graph_gen(
            data=data_df,
            folder_path=design,
            network_gen_method=method,
            jaccard_threshold=jaccard_thr,
            summary=ret_summary
        )

    if ret_summary:
        graph_df, summary_data, linked_genes = graph_results

        if isinstance(summary_data, pd.DataFrame):
            summary_data.to_csv(f'{out}/network_summary.tsv', sep='\t')

        with open(f'{out}/linked_genes.pkl', 'wb') as pickle_file:
            pickle.dump(linked_genes, pickle_file)
    else:
        graph_df, _, _ = graph_results
    
    graph_df.to_csv(f'{out}/weighted.edgelist', sep='\t', header=False, index=False)

    click.echo("Done with network generation")


@embedding.command()
@data_option
@design_option
@output_option
@click.option(
    '--all_nodes',
    help='Use this tag to return all nodes (not just patients)',
    is_flag=True,
    show_default=True,
)
@click.option(
    '-m',
    '--model_config',
    help='The configuration file for the model used for knowledge graph embedding in JSON format',
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
)
@click.option(
    '--train_size',
    help='Size of the training data for the knowledge graph embedding model',
    type=float,
    required=False,
    default=0.8,
    show_default=True,
)
@click.option(
    '--validation_size',
    help='Size of the validation data for the knowledge graph embedding model',
    type=float,
    required=False,
    default=0.1,
    show_default=True,
)
@click.option(
    '--raw_embedding',
    help='Flag to indicate if the embedding should be returned as is (default only returns the real part of a complex embedding)',
    is_flag=True,
    show_default=True,
    default=False,
    required=False,
)
def kge(
        data: str,
        design: str,
        out: str,
        model_config: str,
        all_nodes: bool = False,
        train_size: float = 0.8,
        validation_size: float = 0.1,
        raw_embedding: bool = False
) -> None:
    """Perform knowledge graph embedding."""
    with open(model_config, 'r') as config_file:
        config = json.load(config_file)

    click.echo(f"Running {config['model']} based KGE on {data} & outputting it to {out}")

    edgelist_df = pd.read_csv(data, sep='\t', index_col=None, header=None)
    design_df = pd.read_csv(design, sep='\t')

    edgelist_df.columns = pd.Index(['source', 'target', 'relation', 'label'])
    edgelist_df = edgelist_df[['source', 'relation', 'target', 'label']]

    embedding_df = do_kge(
        edgelist=edgelist_df,
        design=design_df,
        out=out,
        return_patients=(not all_nodes),
        model_config=config,
        train_size=train_size,
        validation_size=validation_size,
        complex_embedding=raw_embedding
    )

    embedding_df.to_csv(f'{out}/embedding.tsv', sep='\t')
    click.echo(f"Finished running {config['model']} based KGE")


@main.command()
@data_option
@output_option
@click.option(
    '--model',
    help="Choose a classification model",
    type=click.Choice([
        'logistic_regression',
        'elastic_net',
        'svm',
        'random_forest',
        'gradient_boost'
    ]),
    required=True
)
@click.option(
    '--cv',
    help="Number of cross validation steps",
    type=int,
    required=False,
    default=5,
    show_default=True,
)
@click.option(
    '-m',
    '--metrics',
    help="Metrics that should be tested during cross validation (comma separated)",
    type=click.Choice(scorers),
    required=False,
    multiple=True,
)
@click.option(
    '--randomize',
    help="Randomize sample labels to test the stability of and effectiveness of the machine learning algorithm",
    is_flag=True,
    required=False,
)
@click.option(
    '--num-processes',
    help="Number of processes to use for parallelization",
    type=int,
    required=False,
    default=1,
    show_default=True,
)
@click.option(
    '--mysql-url',
    help="URL for the MySQL database (with username and password) to store the optuna study results",
    type=str,
    required=False,
    default=None,
    show_default=True,
)
@click.option(
    '--study-name',
    help="Name of the Optuna study to store the results",
    type=str,
    required=False,
    default='hpo_study',
    show_default=True,
)
@click.option(
    '--num-trials',
    help="Number of trials to run for hyperparameter optimization",
    type=int,
    required=False,
    default=100,
    show_default=True,
)
def classify(
        data: str,
        out: str,
        model: str,
        cv: int,
        metrics: Optional[List[str]],
        randomize: bool,
        num_processes: int,
        mysql_url: Optional[str],
        study_name: str,
        num_trials: int
) -> None:
    """Perform machine-learning classification."""
    if not metrics:
        metrics = ['roc_auc', 'accuracy', 'f1_micro', 'f1_macro', 'f1']

    if num_processes > 1 and mysql_url is None:
        ctx = click.get_current_context()
        click.echo("Please provide a MySQL URL to store the optuna study results")
        click.echo(ctx.get_help())
        return

    click.echo(
        f"Starting {model} classification with {data} and out-putting to results to {out}/cross_validation_results.json")

    data_df = pd.read_csv(data, sep='\t', index_col=0)

    _ = do_classification(
        data=data_df,
        model_name=model,
        out_dir=out,
        validation_cv=cv,
        scoring_metrics=list(metrics),
        rand_labels=randomize,
        num_processes=num_processes,
        mysql_url=mysql_url,
        study_name=study_name,
        num_trials=num_trials
    )

    click.echo(f"Done with {model} classification")


if __name__ == '__main__':
    main()
