<p align="center">
  <img src="docs/source/logo.jpg">
</p>

<h1 align="center">
  CLEP: A Hybrid Framework for Generating Patient Representations
  <br/>
  <img src="https://travis-ci.com/clepp/clep-dev.svg?token=rBPVN4HCkHSUyy66qGmX&branch=kge_implementation" />
</h1>

## Table of contents
* [General info](#general-info)
* [Installation](#installation)
* [Documentation](#documentation)
* [Input data](#input-data-formats)
* [Usage](#usage)
* [Issues](#issues)
* [Disclaimer](#disclaimer)

## General info
CLEP is a framework containing several methods for generating patient embeddings from *-omics* data.

<p align="center">
  <img src="docs/source/framework.jpg">
</p>

## Installation

The most recent code can be installed from the source on [GitHub](https://github.com/clepp/clep) with:

```
$ python3 -m pip install git+https://github.com/clepp/clep.git
```

For developers, the repository can be cloned from [GitHub](https://github.com/clepp/clep) and installed in editable mode with:

```
$ git clone https://github.com/clepp/clep.git
$ cd clep
$ python3 -m pip install -e .
```

## Documentation
Read the [official docs](#documentation) for more information.

## Input data formats

### Data

| Symbol | Sample_1 | Sample_2 | Sample_3 |
| ------ | -------- | -------- | -------- |
| HGNC_ID_1 | 0.354 | 2.568 | 1.564 |
| HGNC_ID_2 | 1.255 | 1.232 | 0.26452 |
| HGNC_ID_3 | 3.256 | 1.5 | 1.5462 |

**Note:** The data must be in a tab separated file format.

### Design

| FileName | Target |
| -------- | ------ |
| Sample_1 | Abnormal |
| Sample_2 | Abnormal |
| Sample_3 | Control |

**Note:** The data must be in a tab separated file format.


### Knowledge graph
The graph format CLEP can handle is a modified version of the Edge List Format. Which looks as follows:

| Source | Relation | Target |
| ------ | -------- | ------ |
| HGNC_ID_1 | association | HGNC_ID_2
| HGNC_ID_2 | decreases | HGNC_ID_3
| HGNC_ID_3 | increases | HGNC_ID_1
    
**Note:** The data must be in a tab separated file format & if your knowledge graph does not have relations between the source and the target, just populate the relation column with "No Relation".


## Usage
**Note:** These are very basic commands for clep, and the detailed options for each command can be found in the [documentation](#documentation)
1. **Radical Searching**
The following command finds the extreme samples with extreme feature values based on the control population.

```
$ python3 -m clep sample-scoring radical-search --data <DATA_FILE> --design <DESIGN_FILE> --control Control --threshold 2.5 -cb -rs --out <OUTPUT_DIR>
```

2. **Graph Generation**
The following command generates the patient-gene network based on the method chosen (Interaction_network).

```
$ python3 -m clep embedding generate-network --data <PROCESSED_DATA_FILE> --method interaction_network  -rs --out <OUTPUT_DIR>
```

3. **Knowledge Graph Embedding**
The following command generates the embedding of the network passed to it.

```
$ python3 -m clep embedding kge --data <NETWORK_FILE> --design <DESIGN_FILE> --model RotatE --train_size 0.8 --validation_size 0.1 --out <OUTPUT_DIR>
```

4. **Classification**
The following command carries out classification on the given data file for a chosen model (Elastic Net) using a chosen optimizer (Grid Search).

```
$ python3 -m clep classify --data <NETWORK_FILE> --model elastic_net --optimizer grid_search --cv 5 --out <OUTPUT_DIR>
```

## Issues
If you have difficulties using CLEP, please open an issue at our [GitHub](https://github.com/clepp/clep) repository.

## Disclaimer
CLEP is a scientific software that has been developed in an academic capacity, and thus comes with no warranty or guarantee of maintenance, support, or back-up of data.
