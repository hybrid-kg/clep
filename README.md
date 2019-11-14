<p style="text-align: center;">
  <img src="docs/source/logo.png" height="150" alt="CLEPP Logo">
</p>

<h1 style="text-align: center;">
  CLEPP
</h1>

CLEPP is workflow containing several methods for generating patient embeddings from *-omics* data.

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
