<p align="center">
  <img src="docs/source/logo.jpg">
</p>

<h1 align="center">
  Clinical Embeddings for Patient Prediction (CLEPP)
  <img src="https://travis-ci.org/clepp/clepp.svg?branch=master" />
</h1>

CLEPP is workflow containing several methods for generating patient embeddings from *-omics* data.

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

Disclaimer
----------
CLEPP is a scientific software that has been developed in an academic capacity, and thus comes with no warranty or guarantee of maintenance, support, or back-up of data.
