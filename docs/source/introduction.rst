.. _intro:

Welcome to CLEP's documentation!
===================================
**Release notes** : https://github.com/hybrid-kg/clep/releases

.. image:: ./logo.jpg
    :align: center

.. raw:: html

   <h1 align="center">
     <img src="https://travis-ci.com/hybrid-kg/clep.svg?branch=master" />
     <img src='https://readthedocs.org/projects/clep/badge/?version=latest' alt='Documentation Status' />
     <img src='https://img.shields.io/github/license/hybrid-kg/clep?color=blue' alt='GitHub License' />
   </h1>

CLEP: A Hybrid Data- and Knowledge- Driven Framework for Generating Patient Representations.
---------------------------------------------------------------------------------------------
CLEP has three main subgroups: ``sample_scoring``, ``embedding``, ``classify``.

1. The ``sample_scoring`` module generates a score for every patient-feature pair.

2. The ``embedding`` module overlays the patients on the prior knowledge in-order generate a new KG, whose embedding
is generated using KGE models from PyKEEN(Ali, *et al.*,2020).

3. The ``classify`` module classifies the generated embedding model (or any data that is passed to it) using generic
classification models.

General info
-------------
CLEP is a framework that contains novel methods for generating patient representations from any patient level data and its corresponding prior knowledge encoded in a knowledge graph. The framework is depicted in the graphic below

.. image:: ./framework.jpg
    :align: center


Installation
------------

In-order to install CLEP, an installation of R is **required** with a copy of Limma. Once they are installed, you
can install CLEP package from pypi.

.. code:: shell

   # Use pip to install the latest release
   $ python3 -m pip install clep

You may instead want to use the development version from Github, by running

.. code:: shell

   $ python3 -m pip install git+https://github.com/hybrid-kg/clep.git

For contributors, the repository can be cloned from [GitHub](https://github.com/hybrid-kg/clep.git) and installed in editable mode using:

.. code:: shell

   $ git clone https://github.com/hybrid-kg/clep.git
   $ cd clep
   $ python3 -m pip install -e .

Dependency
--------------
- Python 3.6+
- Installation of R

Mandatory
~~~~~~~~~

- Numpy
- Scipy
- Pandas
- Matplotlib
- rpy2 (for limma)
- Limma package from [bioconductor](https://bioconductor.org/packages/release/bioc/html/limma.html)


For API information to use this library, see the :ref:`dev-guide`.

Issues
-------

If you have difficulties using CLEP, please open an issue at our [GitHub](https://github.com/hybrid-kg/clep.git) repository.

Acknowledgements
-----------------

Graphics
~~~~~~~~~

The CLEP logo and framework graphic was designed by Carina Steinborn.

Disclaimer
-----------

CLEP is a scientific software that has been developed in an academic capacity, and thus comes with no warranty or
guarantee of maintenance, support, or back-up of data.

