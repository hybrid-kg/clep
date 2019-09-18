# -*- coding: utf-8 -*-

"""Command line interface."""

import logging
import click
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

# matrix_option = click.option(
#     '-m', '--matrix',
#     help="path to matrix",
#     type=click.Path(file_okay=True, dir_okay=False, exists=True),
#     required=True
#     )

@main.group()
def preprocessing():
    """List Pre-Processing methods available."""


@preprocessing.command(help='Limma based Pre-Processing')
def limma():
    click.echo(f"Starting Limma Based Pre-Processing")
    do_limma()
    click.echo(f"Done With limma calculation")


@preprocessing.command(help='Z-Score based Pre-Processing')
def z_score():
    click.echo(f"Starting Z-Score Based Pre-Processing")
    do_z_score()
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
