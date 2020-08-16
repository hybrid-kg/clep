.. _howto:

.. _PyKEEN: https://pykeen.readthedocs.io/en/latest/tutorial/running_hpo.html

How to use CLEP
================

Sample Scoring
----------------

There are 4 main way to score the patient-feature pairs,

1. Linear model fitting using **Limma**
2. ssGSEA
3. Z-Score
4. Radical Searching (eCDF based)

To carry out sample scoring use,

.. code:: shell

   $ clep sample-scoring radical-search --data <DATA_FILE> --design <DESIGN_FILE> \
   --control Control --threshold 2.5 --control_based --ret_summary --out <OUTPUT_DIR>


Data Format
~~~~~~~~~~~~

The format of a standard data file should look like,

+-----------+----------+----------+----------+
|           | Sample_1 | Sample_2 | Sample_3 |
+===========+==========+==========+==========+
| HGNC_ID_1 | 0.354    | 2.568    | 1.564    |
+-----------+----------+----------+----------+
| HGNC_ID_2 | 1.255    | 1.232    | 0.26452  |
+-----------+----------+----------+----------+
| HGNC_ID_3 | 3.256    | 1.5      | 1.5462   |
+-----------+----------+----------+----------+


The format of a design file, for the data given above should look like,

+----------+----------+
| FileName | Target   |
+==========+==========+
| Sample_1 | Abnormal |
+----------+----------+
| Sample_2 | Abnormal |
+----------+----------+
| Sample_3 | Control  |
+----------+----------+

Knowledge Graph Generation
---------------------------

A patient-feature knowledge graph (KG) can be generated using 3 methods,

1. Based on **pathway overlaps** (needs ssGSEA as the scoring functions)
2. Based on user-provided knowledge graph
3. Based on the overlap of multiple user-provided knowledge graph (needs the use of either ssGSEA, if each KG
   represents a distinct pathway, or any other appropriate 3rd party scoring function)

To carry out KG generation use,

.. code:: shell

   $ clep embedding generate-network --data <SCORED_DATA_FILE> --method interaction_network \
   --ret_summary --out <OUTPUT_DIR>


Data Format
~~~~~~~~~~~~

The format of a knowledge graph file for the data given above should be a modified version of edgelist, as shown below,

+-----------+-------------+-----------+
|  Source   | Relation    | Target    |
+===========+=============+===========+
| HGNC_ID_1 | association | HGNC_ID_2 |
+-----------+-------------+-----------+
| HGNC_ID_2 | decreases   | HGNC_ID_3 |
+-----------+-------------+-----------+
| HGNC_ID_3 | increases   | HGNC_ID_1 |
+-----------+-------------+-----------+


Knowledge Graph Embedding
--------------------------

For the generation of an embedding use,

.. code:: shell

   $ clep embedding kge --data <NETWORK_FILE> --design <DESIGN_FILE> \
   --model_config <MODEL_CONFIG.json> --train_size 0.8 --validation_size 0.1 --out <OUTPUT_DIR>


Data Format
~~~~~~~~~~~~

The config file for the KGE model must contain the model name, and other optimization parameters, as shown in the
template below,

.. code-block:: json

   {
     "model": "RotatE",
     "model_kwargs": {
       "automatic_memory_optimization": true
     },
     "model_kwargs_ranges": {
       "embedding_dim": {
         "type": "int",
         "low": 6,
         "high": 9,
         "scale": "power_two"
       }
     },
     "training_loop": "slcwa",
     "optimizer": "adam",
     "optimizer_kwargs": {
       "weight_decay": 0.0
     },
     "optimizer_kwargs_ranges": {
       "lr": {
         "type": "float",
         "low": 0.0001,
         "high": 1.0,
         "scale": "log"
       }
     },
     "loss_function": "NSSALoss",
     "loss_kwargs": {},
     "loss_kwargs_ranges": {
       "margin": {
         "type": "float",
         "low": 1,
         "high": 30,
         "q": 2.0
       },
       "adversarial_temperature": {
         "type": "float",
         "low": 0.1,
         "high": 1.0,
         "q": 0.1
       }
     },
     "regularizer": "NoRegularizer",
     "regularizer_kwargs": {},
     "regularizer_kwargs_ranges": {},
     "negative_sampler": "BasicNegativeSampler",
     "negative_sampler_kwargs": {},
     "negative_sampler_kwargs_ranges": {
       "num_negs_per_pos": {
         "type": "int",
         "low": 1,
         "high": 50,
         "q": 1
       }
     },
     "create_inverse_triples": false,
     "evaluator": "RankBasedEvaluator",
     "evaluator_kwargs": {
       "filtered": true
     },
     "evaluation_kwargs": {
       "batch_size": null
     },
     "training_kwargs": {
       "num_epochs": 1000,
       "label_smoothing": 0.0
     },
     "training_kwargs_ranges": {
       "batch_size": {
         "type": "int",
         "low": 8,
         "high": 11,
         "scale": "power_two"
       }
     },
     "stopper": "early",
     "stopper_kwargs": {
       "frequency": 25,
       "patience": 4,
       "delta": 0.002
     },
     "n_trials": 100,
     "timeout": 129600,
     "metric": "hits@10",
     "direction": "maximize",
     "sampler": "random",
     "pruner": "nop"
   }


For more details on the configuration, check out `PyKEEN`_

Classification
---------------

The classification of any provided data, can be carried out using any of the 5 different machine learning models,

1. **Logistic regression** with l2 regularization
2. Logistic regression with **elastic net** regularization
3. Support Vector Machines
4. Random forest
5. Gradient boosting

The classification also requires the input of the following optimizers,

1. Grid search
2. Random search
3. Bayesian search

For the carrying out the classification use,

.. code:: shell

   $ clep classify --data <EMBEDDING_FILE> --model elastic_net --optimizer grid_search \
   --out <OUTPUT_DIR>


Data Format
~~~~~~~~~~~~

The format of the input file for classification should look like,

+----------+-------------+-------------+-------------+-------+
|          | Component_1 | Component_2 | Component_3 | label |
+==========+=============+=============+=============+=======+
| Sample_1 | 0.48687     | -1.5675     | 1.74140     |   0   |
+----------+-------------+-------------+-------------+-------+
| Sample_2 | -1.48840    | 5.26354     | -0.4435     |   1   |
+----------+-------------+-------------+-------------+-------+
| Sample_3 | -0.41461    | 4.6261      | 8.104       |   0   |
+----------+-------------+-------------+-------------+-------+


For more information on the command line interface, please refer :ref:`cli`.


Programmatic Access
---------------------
CLEP implements an API through which developers can utilise each module available in the CLEP framework. An example
for the usage of the API functions in shown below.

.. code:: python

   import os
   import pandas as pd
   from clep.classification import do_classification

   model = "elastic_net" # Classification Model
   optimizer = "grid_search" # Optimization function for the classification model
   out = os.getcwd() # Output directory
   cv = 10 # Number of cross-validation folds
   metrics = ['roc_auc', 'accuracy', 'f1_micro', 'f1_macro', 'f1'] # Metrics to be analysed in cross-validation
   randomize = False # If the labels in the data must be permuted

   data_df = pd.read_table(data, index_col=0)

   results = do_classification(data_df, model, optimizer, out, cv, metrics, randomize)


For more information on the available API functions, please refer :ref:`dev-guide`.
