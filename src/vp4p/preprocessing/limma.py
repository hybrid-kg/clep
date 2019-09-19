# -*- coding: utf-8 -*-

"""Python wrapper for R-based Limma to perform single sample DE analysis."""


import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, Formula
import rpy2.robjects as ro
from statsmodels.stats.multitest import multipletests


def do_limma(data: pd.DataFrame, design: pd.DataFrame, contrasts: list = [], alpha: float = 0.05, adjust_method: str = 'fdr_bh'):
    # Import R libraries
    limma = importr('limma')
    base = importr('base')
    stats = importr('stats')
    # Convert data and design pandas dataframes to R dataframes & Use the genes index column from data as a R String Vector
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)
        r_design = ro.conversion.py2rpy(design)
    genes = ro.StrVector(data.index.tolist())

    f = base.factor(r_design.rx2('Target'), levels=base.unique(r_design.rx2('Target')))
    form = Formula('~0 + f')
    form.environment['f'] = f
    r_design = stats.model_matrix(form)
    r_design.colnames = base.levels(f)

    fit = limma.lmFit(r_data, r_design)
    if contrasts:
        contrast_matrix = limma.makeContrasts(*contrasts, levels=r_design)
    else:
        contrast_matrix = limma.makeContrasts(f"{r_design.colnames[0]}-{r_design.colnames[1]}", levels=r_design)
    fit2 = limma.contrasts_fit(fit, contrast_matrix)
    fit2 = limma.eBayes(fit2)
    r_output = limma.topTreat(fit2, coef=1, genelist=genes, number=np.Inf, lfc=1)

    with localconverter(ro.default_converter + pandas2ri.converter):
        output = ro.conversion.rpy2py(r_output)
    output['adj.P.Val'] = multipletests(output['P.Value'], alpha=alpha, method=adjust_method)[1]

    return output
