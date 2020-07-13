<p align="center">
  <img src="docs/source/logo.jpg">
</p>

<h1 align="center">
  Clinical Embeddings for Patient Prediction (CLEPP)
  <img src="https://travis-ci.org/clepp/clepp.svg?branch=master" />
</h1>

CLEPP is workflow containing several methods for generating patient embeddings from *-omics* data.

Installation:
-------------

The most recent code can be installed from the source on [GitHub](https://github.com/clepp/clepp) with:

```
    $ python3 -m pip install git+https://github.com/clepp/clepp.git
```

For developers, the repository can be cloned from [GitHub](https://github.com/clepp/clepp) and installed in editable mode with:

```
    $ git clone https://github.com/clepp/clepp.git
    $ cd clepp
    $ python3 -m pip install -e .
```

Requirements:
--------------
`clepp` requires the following libraries:

    click==7.0
    pandas==0.25.0
    numpy==1.17.0
    rpy2==3.1.0
    statsmodels==0.10.1
    scikit-learn==0.21.3
    seaborn==0.9.0
    gseapy==0.9.15
    cffi==1.12.3
    bionev@git+https://github.com/seffnet/BioNEV.git@2192ad7
    python-igraph==0.8.0
    pycairo==1.19.1
    xgboost==1.0.2
    tqdm==4.43.0
    pykeen


Command Line Interface:
-----------------------
The following commands can be used directly use from your terminal:

1. **Radical Searching**
The following command finds the extreme samples with extreme feature values based on the control population.


```
$ python3 -m clepp sample-scoring radical-search --data <DATA_FILE> --design <DESIGN_FILE> --out <OUTPUT_DIR>
```

2. **Graph Generation**
The following command generates the patient-gene network based on the method chosen (pathway_overlap
, Interaction_network, Interaction_Network_Overlap).

```
$ python3 -m clepp embedding generate-network --data <PROCESSED_DATA_FILE> --method [pathway_overlap|interaction_network|interaction_network_overlap] --out <OUTPUT_DIR>
```


3. **Knowledge Graph Embedding**
The following command generates the embedding of the network passed to it.

```
$ python3 -m clepp embedding --data <NETWORK_FILE> --design <DESIGN_FILE> --model <PYKEEN_MODEL> --out <OUTPUT_DIR>
```


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


Network file format
-----------------
The graph format CLEPP can handle is a modified version of the Edge List Format. Which looks as follows:

    source1 edgeweight1 target1 source_label
    source2 edgeweight2 target2 source_label
    source3 edgeweight3 target3 source_label

A toy example with three subnetworks:

    1 0.00 2 0
    0 0.88 2 1
    3 1.00 4 1
    5 0.52 7 2
    7 0.52 8 2
    6 0.52 8 2
    0 1.00 3 1
    2 1.00 4 0
    1 1.00 7 0
    4 1.00 6 1
    4 1.00 8 0
    
Please note that node ids must be unique, even if they belong to different subnetworks. By default, ProphTools will use node identifiers, not labels (second column in txt file) as IDs for nodes. Optionally, you can use the ``--labels_as_ids`` parameter to use labels instead. Please note that in this case labels must be unique per node.

Disclaimer
----------
CLEPP is a scientific software that has been developed in an academic capacity, and thus comes with no warranty or guarantee of maintenance, support, or back-up of data.
