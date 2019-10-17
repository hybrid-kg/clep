# -*- coding: utf-8 -*-

"""Command line interface."""

import json
import logging
import os

import click
import pandas as pd
from vp4p.classification import do_classification
from vp4p.embedding import (
    do_binning, do_nrl, do_ss_evaluation
    )
from vp4p.sample_scoring import (
    do_limma, do_z_score, do_ssgsea
    )

logger = logging.getLogger(__name__)


@click.group(help='vp4p')
def main():
    """Run vp4p."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@main.group()
def sample_scoring():
    """
        List Single Sample Scoring methods available.
    """


data_option = click.option(
    '--data',
    help="Path to tab-separated gene expression data file",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True
    )
design_option = click.option(
    '--design',
    help="Path to tab-separated experiment design file",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True
    )
output_option = click.option(
    '--out',
    help="Path to the output folder",
    type=click.Path(file_okay=False, dir_okay=True, exists=True),
    required=True
    )
control_option = click.option(
    '--control',
    help="Annotated value for the control samples (Must start with an alphabet)",
    type=str,
    required=False,
    default='Control',
    show_default=True
    )


@sample_scoring.command(help='Limma based Single Sample Scoring')
@data_option
@design_option
@output_option
@click.option(
    '--alpha',
    help="Family-wise error rate",
    type=float,
    required=False,
    default=0.05,
    show_default=True
    )
@click.option(
    '--method',
    help="Method used for testing and adjustment of P-Values",
    type=str,
    required=False,
    default='fdr_bh',
    show_default=True
    )
@control_option
def limma(data, design, out, alpha, method, control: str) -> None:
    click.echo(
        f"Starting Limma Based Single Sample Scoring with {data} & {design} files and saving it to {out}/sample_scoring.tsv")

    data_df = pd.read_csv(data, sep='\t', index_col=0)
    design_df = pd.read_csv(design, sep='\t')

    ctrl_data = data_df.transpose()[list(design_df.Target == control)].transpose()
    sample_data = data_df.transpose()[list(design_df.Target != control)].transpose()

    ctrl_design = design_df[list(design_df.Target == control)]
    sample_design = design_df[list(design_df.Target != control)]

    output_df = pd.DataFrame(columns=sample_data.index, index=sample_data.columns)

    for col_idx, col in enumerate(sample_data.columns):
        data_df = ctrl_data.copy()

        data_df[col] = sample_data.iloc[:, col_idx]
        design_df = ctrl_design.append(sample_design.iloc[col_idx, :], ignore_index=True)

        output = do_limma(data=data_df, design=design_df, alpha=alpha, adjust_method=method)
        output_df.iloc[col_idx, :] = output['logFC'].values.flatten()

    output_df.to_csv(f'{out}/sample_scoring.tsv', sep='\t')

    click.echo(f"Done With limma calculation for {data}")


@sample_scoring.command(help='Z-Score based Single Sample Scoring')
@data_option
@design_option
@output_option
@control_option
def z_score(data, design, out, control) -> None:
    click.echo(
        f"Starting Z-Score Based Single Sample Scoring with {data} & {design} files and saving it to {out}/sample_scoring.tsv")

    data_df = pd.read_csv(data, sep='\t', index_col=0)
    design_df = pd.read_csv(design, sep='\t', index_col=0)

    output = do_z_score(data=data_df, design=design_df, control=control)
    output.to_csv(f'{out}/sample_scoring.tsv', sep='\t')

    click.echo(f"Done With Z-Score calculation for {data}")


@sample_scoring.command(help='ssGSEA based Single Sample Scoring')
@data_option
@output_option
@click.option(
    '--gs',
    help="Path to the .gmt geneset file",
    type=click.Path(file_okay=True, dir_okay=False, exists=False),
    required=True
    )
def ssgsea(data, out, gs) -> None:
    click.echo(
        f"Starting Z-Score Based Single Sample Scoring with {data} & {gs} files and saving it to {out}/sample_scoring.tsv")

    data_df = pd.read_csv(data, sep='\t', index_col=0)

    single_sample_gsea = do_ssgsea(data_df, gs)
    single_sample_gsea.res2d.transpose().to_csv(f'{out}/sample_scoring.tsv', sep='\t')

    click.echo(f"Done With ssgsea for {data}")


@main.group()
def embedding():
    """List Vectorization methods available."""


@embedding.command()
@data_option
@output_option
def binning(data, out) -> None:
    """Performs Threshold based Vectorization."""
    click.echo(f"Starting binning with {data}, and out-putting to {out}/binned.tsv")

    data_df = pd.read_csv(data, sep='\t', index_col=0)

    output = do_binning(data=data_df)

    output.to_csv(f'{out}/binned.tsv', sep='\t')

    click.echo(f"Done With binning for {data}")


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
def evaluate(data, label) -> None:
    """Perform Evaluation of the Embeddings"""
    click.echo(f"Starting Evaluation of the following files: \n{data}")

    data_lst = []
    for file in data:
        data_lst.append(pd.read_csv(file, sep='\t', index_col=0))

    label = list(label)

    result = do_ss_evaluation(data_lst, label)

    click.echo('===============Evaluation Results===============')
    click.echo(json.dumps(result, indent=4, sort_keys=True).replace('{', ' ').replace('}', ' '))
    click.echo('================================================')


@embedding.command()
@data_option
@design_option
@output_option
@control_option
def nrl(data, design, out, control) -> None:
    """Perform Network representation learning."""
    click.echo(f"Starting NRL")

    data_df = pd.read_csv(data, sep='\t')
    data_df.rename(columns={'Unnamed: 0': 'patients'}, inplace=True)

    design_df = pd.read_csv(design, sep='\t')
    do_nrl(data_df, design_df, out, control)

    click.echo(f"Done With NRL")


@main.command()
@data_option
@output_option
@design_option
@control_option
@click.option(
    '--model',
    help="Choose a Classification Model",
    type=click.Choice(['logistic_regression',
                       'elastic_net',
                       'svm',
                       'random_forrest']),
    required=True
    )
def classify(data, design, control, out, model) -> None:
    """Perform Machine-Learning Classification."""

    click.echo(f"Starting NRL")

    data_df = pd.read_csv(data, sep='\t')
    design_df = pd.read_csv(design, sep='\t', index_col=0)

    do_classification(data_df, design_df, control, out, model)

    click.echo(f"Done With NRL")


if __name__ == '__main__':
    main()
