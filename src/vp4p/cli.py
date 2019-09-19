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
        List Pre-Processing methods available.
        Formats for Data and Design Matrices:
        -------------------------------------
        Data:
        -----
        genes | Sample1 | Sample2 | Sample3

        Design:
        -------
        IndexCol | FileName | Patient_Annotation
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


@data_option
@design_option
@preprocessing.command(help='Limma based Pre-Processing')
def limma(data_file, design_file):
    click.echo(f"Starting Limma Based Pre-Processing with {data_file} & {design_file} files")
    data = pd.read_csv(data_file, sep='\t')
    design = pd.read_csv(design_file, sep='\t')
    do_limma(data=data, design=design, alpha=0.05, adjust_method='fdr_bh')
    click.echo(f"Done With limma calculation")


@data_option
@design_option
@preprocessing.command(help='Z-Score based Pre-Processing')
def z_score(data_file, design_file):
    click.echo(f"Starting Z-Score Based Pre-Processing with {data_file} & {design_file} files")
    data = pd.read_csv(data_file, sep='\t')
    design = pd.read_csv(design_file, sep='\t')
    do_z_score(data=data, design=design, control='Control')
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
