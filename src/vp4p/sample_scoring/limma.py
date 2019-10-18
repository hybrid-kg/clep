# -*- coding: utf-8 -*-

"""Python wrapper for R-based Limma to perform single sample DE analysis."""

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from statsmodels.stats.multitest import multipletests


def do_limma(data, design, alpha, method, control: str) -> pd.DataFrame:
    """Perform data manipulation before limma based SS scoring.

    :param data: Dataframe containing the gene expression values
    :param design: Dataframe containing the design table for the data
    :param alpha: Family-wise error rate
    :param method: Method used family-wise error correction
    :param control: label used for representing the control in the design table of the data
    :return Dataframe containing the Single Sample scores from limma
    """
    label_mapping = dict(zip(np.unique(design['Target']), range(len(np.unique(design['Target'])))))

    ctrl_data = data.transpose()[list(design.Target == control)].transpose()
    sample_data = data.transpose()[list(design.Target != control)].transpose()

    ctrl_design = design[list(design.Target == control)]
    sample_design = design[list(design.Target != control)]

    output_df = pd.DataFrame(columns=sample_data.index, index=sample_data.columns)

    # Loop over every sample to get the individual limma based SS scores
    for col_idx, col in enumerate(sample_data.columns):
        data_df = ctrl_data.copy()

        data_df[col] = sample_data.iloc[:, col_idx]
        design_df = ctrl_design.append(sample_design.iloc[col_idx, :], ignore_index=True)

        output = _limma(data=data_df, design=design_df, alpha=alpha, adjust_method=method)
        output_df.iloc[col_idx, :] = output['logFC'].values.flatten()

    label = sample_design['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)

    output_df['label'] = label

    return output_df


def _limma(data: pd.DataFrame, design: pd.DataFrame, alpha: float = 0.05,
           adjust_method: str = 'fdr_bh') -> pd.DataFrame:
    """Wrap limma to perform single sample DE analysis."""
    # Import R libraries
    limma = importr('limma')
    base = importr('base')
    stats = importr('stats')

    # Convert data and design pandas dataframes to R dataframes
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)
        r_design = ro.conversion.py2rpy(design)

    # Use the genes index column from data as a R String Vector
    genes = ro.StrVector([str(index) for index in data.index.tolist()])

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
        output = ro.conversion.rpy2py(r_output)

    # Adjust P value with the provided adjusted method
    output['adj.P.Val'] = multipletests(output['P.Value'], alpha=alpha, method=adjust_method)[1]
    output['logFC'].loc[output['adj.P.Val'] > 0.05] = 0
    output['logFC'].loc[np.abs(output['logFC']) < 1.3] = 0

    return output
