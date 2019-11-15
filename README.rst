.. image:: docs/source/logo.jpg
   :scale: 1%
   :align: center

Clinical Embeddings for Patient Prediction (CLEPP)
====================================================
.. image:: https://travis-ci.org/clepp/clepp.svg?branch=master
   :target: https://travis-ci.org/clepp/clepp

CLEPP is workflow containing several methods for generating patient embeddings from *-omics* data.

Formats for Data and Design Matrices:
-------------------------------------
Data:

+---------+----------+-----+----------+
| genes   | Sample_1 | ... | Sample_n |
+=========+==========+=====+==========+
| HGNC ID | float    | ... | float    |
+---------+----------+-----+----------+

Design:

+-----------------------------------+----------------------+
| FileName                          | Target               |
+===================================+======================+
| sample expression array file name | annotation of sample |
+-----------------------------------+----------------------+
