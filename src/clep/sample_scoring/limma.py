# -*- coding: utf-8 -*-

"""Python wrapper for R-based Limma to perform single sample DE analysis."""
import sys
from typing import List

import click
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from statsmodels.stats.multitest import multipletests


def do_limma(data: pd.DataFrame, design: pd.DataFrame, alpha: float, method: str, control: str) -> pd.DataFrame:
    """Perform data manipulation before limma based SS scoring.

    :param data: Dataframe containing the gene expression values
    :param design: Dataframe containing the design table for the data
    :param alpha: Family-wise error rate
    :param method: Method used family-wise error correction
    :param control: label used for representing the control in the design table of the data
    :return: Dataframe containing the Single Sample scores from limma
    """
    label_mapping = {
        key: val
        for val, key in enumerate(np.unique(design['Target']))
    }

    # Get the control patients using the control parameter
    control_filter = (design.Target == control)

    # The data is transposed to get in the same format as the design matrix & then after the filtering transposed
    # again to return to the original format
    ctrl_data = data.transpose()[list(control_filter)].transpose()

    # Here the sample stands for diseased patients
    sample_data = data.transpose()[list(~control_filter)].transpose()

    ctrl_design = design[list(control_filter)]
    sample_design = design[list(~control_filter)]

    # Final output dataframe (patients as rows & genes as columns)
    output_df = pd.DataFrame(columns=sample_data.index, index=sample_data.columns)

    # Loop over every sample to get the individual limma based SS scores
    for col_idx, col in enumerate(sample_data.columns):
        data_df = ctrl_data.copy()

        # Get a single sample from dataframe
        data_df[col] = sample_data.iloc[:, col_idx]

        # Add the design of that sample with the control samples to make the design table
        design_df = ctrl_design.append(sample_design.iloc[col_idx, :], ignore_index=True)  # type: ignore

        output = _limma(data=data_df, design=design_df, alpha=alpha, adjust_method=method)

        # Only store the logFC values
        output_df.iloc[col_idx, :] = output['logFC'].values.flatten()  # type: ignore

    label = sample_design['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)

    output_df = output_df.apply(_bin).copy()

    output_df['label'] = label.values

    return output_df


def _limma(data: pd.DataFrame, design: pd.DataFrame, alpha: float = 0.05,
           adjust_method: str = 'fdr_bh') -> pd.DataFrame:
    """Wrap limma to perform single sample DE analysis."""
    # Import R libraries
    base = importr('base')
    stats = importr('stats')

    try:
        limma = importr('limma')
    except RRuntimeError as e:
        click.echo(e)
        click.echo("Please check if limma package is installed in R. \n If not, follow the instructions from LINK "
                   "HERE.")
        sys.exit(1)

    # Convert data and design pandas dataframes to R dataframes
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)
        r_design = ro.conversion.py2rpy(design)

    # Use the genes index column from data as a R String Vector
    genes = ro.StrVector(
        [
            str(index)
            for index in data.index.tolist()
        ]
    )

    # Create a model matrix using design's Target column using the R formula "~0 + f" to get all the unique factors
    # as columns
    f = base.factor(r_design.rx2('Target'), levels=base.unique(r_design.rx2('Target')))
    form = Formula('~0 + f')
    form.environment['f'] = f
    r_design = stats.model_matrix(form)
    r_design.colnames = base.levels(f)

    # Fit the data to the design using lmFit from limma
    fit = limma.lmFit(r_data, r_design)

    # Make a contrasts matrix with the 1st and the last unique values
    contrast_matrix = limma.makeContrasts(f"{r_design.colnames[0]}-{r_design.colnames[-1]}", levels=r_design)

    # Fit the contrasts matrix to the lmFit data & calculate the bayesian fit
    fit2 = limma.contrasts_fit(fit, contrast_matrix)
    fit2 = limma.eBayes(fit2)

    # topTreat the bayesian fit using the contrasts and add the genelist
    r_output = limma.topTreat(fit2, coef=1, genelist=genes, number=np.Inf)

    # Convert R dataframe to Pandas
    with localconverter(ro.default_converter + pandas2ri.converter):
        output: pd.DataFrame = ro.conversion.rpy2py(r_output)

    # Adjust P value with the provided adjusted method
    output['adj.P.Val'] = multipletests(output['P.Value'], alpha=alpha, method=adjust_method)[1]
    output['logFC'].loc[output['adj.P.Val'] > 0.05] = 0
    output['logFC'].loc[np.abs(output['logFC']) < 1.3] = 0

    return output


def _bin(row: pd.Series[int]) -> List[int]:
    """Replace values greater than 0 as 1 and lesser than 0 as -1."""
    return [
        1 if (val > 0) else (-1 if (val < 0) else 0)
        for val in row
    ]
