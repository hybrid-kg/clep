<p align="center">
  <img src="docs/source/logo.jpg" height="150">
</p>

<h1 align="center">
  CLEPP
</h1>

<p align="center">
  <img src="https://travis-ci.org/clepp/clepp.svg?branch=master" />
</p>

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
