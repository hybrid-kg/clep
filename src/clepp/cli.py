# -*- coding: utf-8 -*-

"""Command line interface."""

import json
import logging
from typing import List, Union, Optional

import click
import numpy as np
import pandas as pd
from sklearn.metrics import SCORERS

from clepp.classification import do_classification
from clepp.embedding import (
    do_binning, do_nrl, do_ss_evaluation,
    do_graph_gen, do_kge)
from clepp.sample_scoring import (
    do_limma, do_z_score, do_ssgsea, do_radical_search
)

logger = logging.getLogger(__name__)


@click.group(help='clepp')
def main():
    """Run clepp."""
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
def radical_search(data: str, design: str, out: str, control: str, threshold: float) -> None:
    """Search for samples that are extremes as compared to the samples."""

    assert 0 < threshold <= 100

    click.echo(
        f"Starting radical searching based single sample scoring with {data} & {design} files and "
        f"saving it to {out}/sample_scoring.tsv"
    )

    data_df = pd.read_csv(data, sep='\t', index_col=0)
    design_df = pd.read_csv(design, sep='\t')

    output = do_radical_search(data=data_df, design=design_df, control=control, threshold=threshold)
    output.to_csv(f'{out}/sample_scoring.tsv', sep='\t')

    click.echo(f"Done with Radical-Score calculation for {data}")


@main.group()
def embedding():
    """List Vectorization methods available."""


@embedding.command()
@data_option
@output_option
def binning(data: str, out: str) -> None:
    """Perform Threshold based Vectorization."""
    click.echo(f"Starting binning with {data}, and out-putting to {out}/binned.tsv")

    data_df = pd.read_csv(data, sep='\t', index_col=0)

    output = do_binning(data=data_df)

    output.to_csv(f'{out}/binned.tsv', sep='\t')

    click.echo(f"Done with binning for {data}")


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
@click.option(
    '--kg',
    help="Path to the Knowledge Graph file in tsv format",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
)
@output_option
@click.option(
    '--method',
    help='The NRL method to train the model',
    type=click.Choice(['DeepWalk', 'node2vec', 'LINE']),
    required=True,
)
def nrl(data: str, kg: str, out: str, method: str) -> None:
    """Perform network representation learning."""
    click.echo(f"Starting {method} based NRL with {data} & {kg} and outputting it to {out}")

    data_df = pd.read_csv(data, sep='\t')
    data_df.rename(columns={'Unnamed: 0': 'patients'}, inplace=True)

    kg_data_df = pd.read_csv(kg, sep='\t')

    do_nrl(data_df, kg_data_df, out, method)

    click.echo(f"Done with NRL")


@embedding.command()
@data_option
@output_option
@click.option(
    '--method',
    help='The method used to generate the network',
    type=click.Choice(['Pathway Overlap', 'Interaction Network', 'Interaction Network Overlap']),
    required=False,
    default='Interaction Network',
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
        network_folder: Optional[str] = None,
) -> None:
    """Generate Network for the given data."""

    data_df = pd.read_csv(data, sep='\t')
    data_df.rename(columns={'Unnamed: 0': 'patients'}, inplace=True)

    if method == 'Pathway Overlap':
        assert gmt is not None
        click.echo(f"Generating {method} based network with {data} & {gmt} and outputting it to {out}")

        graph_df = do_graph_gen(data=data_df, gmt=gmt, network_gen_method=method, intersection_threshold=intersect_thr)

    elif method == 'Interaction Network':
        assert kg is not None
        click.echo(f"Generating {method} based network with {data} & {kg} and outputting it to {out}")

        kg_data_df = pd.read_csv(kg, sep='\t')

        graph_df = do_graph_gen(data=data_df, kg_data=kg_data_df, network_gen_method=method)

    else:
        assert network_folder is not None
        design = network_folder

        graph_df = do_graph_gen(data=data_df, folder_path=design, network_gen_method=method, jaccard_threshold=jaccard_thr)

    graph_df.to_csv(f'{out}/weighted.edgelist', sep='\t', header=False, index=False)

    click.echo(f"Done with network generation")


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
    required=False,
    default='TransE',
    show_default=True,
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
    all_nodes: Optional[bool] = False,
    model: Optional[str] = 'TransE',
    train_size: Optional[float] = 0.8,
    validation_size: Optional[float] = 0.1
) -> None:
    """Perform knowledge graph embedding."""
    edgelist_df = pd.read_csv(data, sep='\t', index_col=None, header=None)
    design_df = pd.read_csv(design, sep='\t')

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
    default=10,
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
    '--title',
    help="Title of the box plot (if not provided a title will be 'Boxplot')",
    type=str,
    required=False,
    default='Boxplot',
    show_default=False,
)
def classify(data: str, out: str, model: str, cv: int, metrics: Union[List[str], None], title: str) -> None:
    """Perform machine-learning classification."""
    if not metrics:
        metrics = ['roc_auc', 'accuracy', 'f1_micro', 'f1_macro', 'f1']

    click.echo(
        f"Starting {model} classification with {data} and out-putting to results to {out}/cross_validation_results.json"
        f" & the plot to {out}/boxplot.png"
    )

    data_df = pd.read_csv(data, sep='\t', index_col=0)

    results = do_classification(data_df, model, out, cv, metrics, title)

    click.echo(results)
    click.echo(f"Done with {model} classification")


if __name__ == '__main__':
    main()
