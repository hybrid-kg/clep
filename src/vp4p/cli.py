# -*- coding: utf-8 -*-

"""Command line interface."""

import logging

import click
import pandas as pd

from vp4p.preprocessing import (
    do_limma, do_z_score
)
from vp4p.vectorization import (
    do_row2vec, do_path2vec, do_thresh2vec, do_nrl
)

logger = logging.getLogger(__name__)


@click.group(help='vp4p')
def main():
    """Run vp4p."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@main.group()
def preprocessing():
    """
        List Pre-Processing methods available.\n
        Formats for Data and Design Matrices:\n
        -------------------------------------\n
        Data:\n
        genes | Sample1 | Sample2 | Sample3\n\n

        Design:\n
        IndexCol | FileName | Patient_Annotation\n
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
    help="Path to the output file",
    type=click.Path(file_okay=True, dir_okay=False, exists=False),
    required=True
)


@preprocessing.command(help='Limma based Pre-Processing')
@data_option
@design_option
@output_option
@click.option(
    '--contrasts',
    help="Contrast sets must be separated by ',' and each group in a set must be separated by '-' and the "
         "whole thing must be surrounded in quotes.\nE.g. - 'Group1-Group2, Group3-Group4'.",
    type=str,
    required=False,
    default=[],
    show_default=True
)
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
def limma(data, design, out, contrasts, alpha, method):
    click.echo(f"Starting Limma Based Pre-Processing with {data} & {design} files and saving it to {out}")
    data_df = pd.read_csv(data, sep='\t')

    design_df = pd.read_csv(design, sep='\t')

    contrasts = contrasts.replace(' ', '').split(',')

    output = do_limma(data=data_df, design=design_df, contrasts=contrasts, alpha=alpha, adjust_method=method)
    output.to_csv(out, sep='\t')

    click.echo(f"Done With limma calculation")


@preprocessing.command(help='Z-Score based Pre-Processing')
@data_option
@design_option
@output_option
@click.option(
    '--control',
    help="Annotated value for the control samples (Must start with an alphabet)",
    type=str,
    required=False,
    default='Control',
    show_default=True
)
def z_score(data, design, out, control):
    click.echo(f"Starting Z-Score Based Pre-Processing with {data} & {design} files and saving it to {out}")
    data_df = pd.read_csv(data, sep='\t')
    design_df = pd.read_csv(design, sep='\t')
    output = do_z_score(data=data_df, design=design_df, control=control)
    output.to_csv(out, sep='\t')
    click.echo(f"Done With Z-Score calculation")


@main.group()
def vectorization():
    """List Vectorization methods available."""


@vectorization.command()
def row2vec():
    """Perform row based vectorization"""
    click.echo(f"Starting Row2Vec")
    do_row2vec()
    click.echo(f"Done With Row2Vec")


@vectorization.command()
def path2vec():
    """Perform pathway based vectorization"""
    click.echo(f"Starting Path2Vec")
    do_path2vec()
    click.echo(f"Done With Path2Vec")


@vectorization.command()
def thresh2vec():
    """Perform Threshold based Vectorization"""
    click.echo(f"Starting Thresh2Vec")
    do_thresh2vec()
    click.echo(f"Done With Thresh2Vec")


@vectorization.command()
def nrl():
    """Perform Network representation learning"""
    click.echo(f"Starting NRL")
    do_nrl()
    click.echo(f"Done With NRL")


if __name__ == '__main__':
    main()
