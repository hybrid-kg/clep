.. _dev-guide:

Developmental Guide
=====================================

.. module:: clep


Core Module APIs
-----------------

Sample Scoring
~~~~~~~~~~~~~~~

.. autofunction:: clep.sample_scoring.limma.do_limma()


.. autofunction:: clep.sample_scoring.ssgsea.do_ssgsea()


.. autofunction:: clep.sample_scoring.z_score.do_z_score()


.. autofunction:: clep.sample_scoring.radical_search.do_radical_search()


KG Generation
~~~~~~~~~~~~~~

.. autofunction:: clep.embedding.network_generator.do_graph_gen()


KG Embedding
~~~~~~~~~~~~~

.. autofunction:: clep.embedding.kge._weighted_splitter()


.. autofunction:: clep.embedding.kge.do_kge()


Classification
~~~~~~~~~~~~~~~

.. autofunction:: clep.classification.classify.do_classification()

