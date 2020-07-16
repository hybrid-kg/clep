# -*- coding: utf-8 -*-

"""Command line interface."""

import json
import logging
import pickle
import warnings
from typing import List, Union, Optional

import click
import numpy as np
import pandas as pd
from sklearn.metrics import SCORERS

from clep.classification import do_classification
from clep.embedding import (
    do_ss_evaluation, do_graph_gen, do_kge)
from clep.sample_scoring import (
    do_limma, do_z_score, do_ssgsea, do_radical_search
)

logger = logging.getLogger(__name__)


@click.group(help='clep')
def main():
    """Run clep."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@main.group()
def sample_scoring():
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

    df = single_sample_gsea.res2d.transpose()

    label = design_df['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)

    df['label'] = label.values

    df.to_csv(f'{out}/sample_scoring.tsv', sep='\t')

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
def embedding():
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

        graph_df = do_graph_gen(
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

        graph_df = do_graph_gen(
            data=data_df,
            kg_data=kg_data_df,
            network_gen_method=method,
            summary=ret_summary
        )

    else:
        assert network_folder is not None
        design = network_folder

        graph_df = do_graph_gen(
            data=data_df,
            folder_path=design,
            network_gen_method=method,
            jaccard_threshold=jaccard_thr,
            summary=ret_summary
        )

    if ret_summary:
        graph_df, summary_data, linked_genes = graph_df

        summary_data.to_csv(f'{out}/network_summary.tsv', sep='\t')

        with open(f'{out}/linked_genes.pkl', 'wb') as pickle_file:
            pickle.dump(linked_genes, pickle_file)

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
    '--model',
    help='The model used for knowledge graph embedding',
    type=click.Choice([
        'ComplEx',
        'ComplExLiteral',
        'ConvE',
        'ConvKB',
        'DistMult',
        'DistMultLiteral',
        'ERMLP',
        'ERMLPE',
        'HolE',
        'KG2E',
        'NTN',
        'ProjE',
        'RESCAL',
        'RGCN',
        'RotatE',
        'SimplE',
        'StructuredEmbedding',
        'TransD',
        'TransE',
        'TransH',
        'TransR',
        'TuckER',
        'UnstructuredModel'
    ]),
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
def kge(
        data: str,
        design: str,
        out: str,
        model: str,
        all_nodes: Optional[bool] = False,
        train_size: Optional[float] = 0.8,
        validation_size: Optional[float] = 0.1
) -> None:
    """Perform knowledge graph embedding."""
    click.echo(f"Running {model} based KGE on {data} & outputting it to {out}")
    edgelist_df = pd.read_csv(data, sep='\t', index_col=None, header=None)
    design_df = pd.read_csv(design, sep='\t')

    edgelist_df.columns = ['source', 'target', 'relation', 'label']
    edgelist_df = edgelist_df[['source', 'relation', 'target', 'label']]

    embedding_df = do_kge(
        edgelist=edgelist_df,
        design=design_df,
        out=out,
        return_patients=(not all_nodes),
        model=model,
        train_size=train_size,
        validation_size=validation_size
    )

    embedding_df.to_csv(f'{out}/embedding.tsv', sep='\t')
    click.echo(f"Finished running {model} based KGE")


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
    '--optimizer',
    help="Optimizer used for classifier.",
    type=click.Choice(['grid_search', 'random_search', 'bayesian_search']),
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
    '--metrics',
    help="Metrics that should be tested during cross validation (comma separated)",
    type=click.Choice(list(SCORERS.keys())),
    required=False,
    multiple=True,
)
@click.option(
    '--randomize',
    help="Randomize sample labels to test the stability of and effectiveness of the machine learning algorithm",
    is_flag=True,
    required=False,
)
def classify(
        data: str,
        out: str,
        model: str,
        optimizer: str,
        cv: int,
        metrics: Union[List[str], None],
        randomize: bool,
) -> None:
    """Perform machine-learning classification."""
    # Ignore bayesian search warning. Waiting for a fix (https://github.com/scikit-optimize/scikit-optimize/issues/302)
    warnings.filterwarnings("ignore", message='The objective has been evaluated at this point before.')

    if not metrics:
        metrics = ['roc_auc', 'accuracy', 'f1_micro', 'f1_macro', 'f1']

    click.echo(
        f"Starting {model} classification with {data} and out-putting to results to {out}/cross_validation_results.json")

    data_df = pd.read_csv(data, sep='\t', index_col=0)

    _ = do_classification(data_df, model, optimizer, out, cv, metrics, randomize)

    click.echo(f"Done with {model} classification")


if __name__ == '__main__':
    main()
