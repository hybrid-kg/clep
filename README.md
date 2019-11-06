CLEPP - A workflow to generate CLinical Embeddings for Patient Prediction 
============================================================================


The goal of this project is to:
1. Build a workflow to pre-process the gene expression data.
2. Convert the expression data into a vector for prediction using various techniques.

Datasets
--------
Datasets currently used in this repo.

| Number | Dataset | Diseased Patients  | Controls  |
| --| -------------:|:-------------:| -----:|
| #1 | [GSE65682](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65682) | 192 | 33 |

Workflow
--------
![image](https://docs.google.com/drawings/d/e/2PACX-1vT6-VOHbKSqFBjj7mqUR3fjkDCmjRatVZxi0gMfYWZlzXAKHZQgIG8uz2aWCypW5LdI69YojDYG3j0R/pub?w=1319&h=685)

Formats for Data and Design Matrices:
-------------------------------------
Data:

| genes | Sample_1 | ... | Sample_n |
| ----- | -------- | --- | -------- |
| HGNC ID | float | ... | float |

Design:

| FileName | Target |
| -------- | ------ |
| sample expression array file name | annotation of sample |
