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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   howto


.. toctree::
   :maxdepth: 3

   cli
   development


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
