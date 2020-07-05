# -*- coding: utf-8 -*-

"""Ensemble of methods for network generation."""
from itertools import combinations
from os import listdir
from os.path import isfile, join
from typing import TextIO, Optional, Tuple, Union, Set
import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from clepp.constants import VALUE_TO_COLNAME
from tqdm import tqdm

logger = logging.getLogger(__name__)


def do_graph_gen(
        data: pd.DataFrame,
        network_gen_method: Optional[str] = 'interaction_network',
        gmt: Optional[str] = None,
        intersection_threshold: Optional[float] = 0.1,
        kg_data: Optional[pd.DataFrame] = None,
        folder_path: Optional[str] = None,
        jaccard_threshold: Optional[float] = 0.2,
        summary: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, Set]]:
    information_graph = nx.DiGraph()

    if network_gen_method == 'pathway_overlap':
        with open(gmt, 'r') as geneset:
            information_graph = plot_pathway_overlap(geneset, intersection_threshold)

    elif network_gen_method == 'interaction_network':
        interaction_graph = nx.from_pandas_edgelist(
            df=kg_data,
            source=kg_data.columns[0],
            target=kg_data.columns[2],
            edge_attr=kg_data.columns[1]
        )

        if nx.number_connected_components(interaction_graph) > 1:
            logger.warning(f'The number of connected components in the graph is greater than 1. '
                           f'There are {nx.number_connected_components(interaction_graph)} connected components of size'
                           f', {[len(c) for c in sorted(nx.connected_components(interaction_graph), key=len, reverse=True)]}'
                           f' respectively.')

        information_graph = plot_interaction_network(kg_data)

    elif network_gen_method == 'interaction_network_overlap':
        information_graph = plot_interaction_net_overlap(folder_path, jaccard_threshold)

    if summary:
        final_graph, summary_data, linked_genes = overlay_samples(data, information_graph, summary=True)
    else:
        final_graph = overlay_samples(data, information_graph, summary=False)

    graph_df = nx.to_pandas_edgelist(final_graph)

    graph_df['relation'].fillna('no_change', inplace=True)

    graph_df = graph_df[['source', 'target', 'relation', 'label']]

    if summary:
        return graph_df, summary_data, linked_genes
    else:
        return graph_df


def plot_pathway_overlap(
        geneset: TextIO,
        intersection_threshold: float = 0.1
) -> nx.DiGraph:
    """Plots the overlap/intersection between pathways as a graph based on shared genes."""
    pathway_dict = {
        line.strip().split("\t")[0]: line.strip().split("\t")[2:]
        for line in geneset.readlines()
    }

    pathway_overlap_graph = nx.DiGraph()

    for pathway_1 in tqdm(pathway_dict.keys(), desc='Finding pathway overlap: '):
        for pathway_2 in pathway_dict.keys():
            if pathway_1 == pathway_2:
                continue

            union = list(set().union(pathway_dict[pathway_1], pathway_dict[pathway_2]))
            intersection = list(set().intersection(pathway_dict[pathway_1], pathway_dict[pathway_2]))

            if len(intersection) > (intersection_threshold * len(union)):
                pathway_overlap_graph.add_edge(str(pathway_1), str(pathway_2))

    return pathway_overlap_graph


def plot_interaction_network(
        kg_data: pd.DataFrame
) -> nx.DiGraph:
    """Plots a knowledge graph based on the interaction data."""
    interaction_graph = nx.DiGraph()

    # Append the source to target mapping to the main data edgelist
    for idx in tqdm(kg_data.index, desc='Plotting interaction network: '):
        interaction_graph.add_edge(
            str(kg_data.iat[idx, 0]),
            str(kg_data.iat[idx, 2]),
            relation=str(kg_data.iat[idx, 1])
        )

    return interaction_graph


def plot_interaction_net_overlap(
        folder_path: str,
        jaccard_threshold: float = 0.2
) -> nx.DiGraph:
    """Plots the overlap/intersection between interaction networks as a graph based on shared nodes."""
    graphs = []
    files = [
        f
        for f in listdir(folder_path)
        if isfile(join(folder_path, f)) and f.endswith('.bel')
    ]

    # Get all the interaction network files from the folder and add them as individual graphs to a list
    for filename in tqdm(files, desc='Plotting interaction network: '):
        with open(join(folder_path, filename), 'r') as file:
            graph = nx.DiGraph(name=filename)
            for line in file:
                src, attr, dst = line.split()
                graph.add_edge(src, dst)
                graph[src][dst]['attribute'] = attr
            graphs.append(graph)

    overlap_graph = nx.DiGraph()

    for graph_1, graph_2 in tqdm(combinations(graphs, 2), desc='Finding interaction network overlap: '):
        if _get_jaccard_index(graph_1, graph_2) > jaccard_threshold:
            overlap_graph.add_edge(str(graph_1.graph['name']), str(graph_2.graph['name']))

    return overlap_graph


def _get_jaccard_index(
        graph_1: nx.DiGraph,
        graph_2: nx.DiGraph
) -> float:
    """Calculates the jaccard index between 2 graphs based on pairwise (edges) jaccard index."""
    j = 0
    iterations = 0
    for v in graph_1:
        if v in graph_2:
            n = set(graph_1[v])  # neighbors of v in G
            m = set(graph_2[v])  # neighbors of v in H

            length_intersection = len(n & m)
            length_union = len(n) + len(m) - length_intersection
            j += float(length_intersection) / length_union

            iterations += 1  # To calculate the average

    return j / iterations


def overlay_samples(
        data: pd.DataFrame,
        information_graph: nx.DiGraph,
        summary: bool = False,
) -> Union[nx.DiGraph, Tuple[nx.DiGraph, pd.DataFrame, Set]]:
    """Overlays the data on the information graph by adding edges between patients and information nodes if pairwise
    value is not 0."""
    patient_label_mapping = {patient: label for patient, label in zip(data.index, data['label'])}
    value_mapping = {0: 'no_change', 1: 'up_reg', -1: 'down_reg'}

    overlay_graph = information_graph.copy()

    data_copy = data.drop(columns='label')
    values_data = data_copy.values

    summary_data = pd.DataFrame(0, index=data_copy.index, columns=["positive_relation", "negative_relation"])
    linked_genes = set()

    for index, value_list in enumerate(tqdm(values_data, desc='Adding patients to the network: ')):
        for column, value in enumerate(value_list):
            patient = data_copy.index[index]
            gene = data_copy.columns[column]

            if value == 0:
                continue
            if gene in information_graph.nodes:
                linked_genes.add(gene)
                overlay_graph.add_edge(patient, gene, relation=value_mapping[value],
                                       label=patient_label_mapping[patient])
            if summary:
                summary_data.at[patient, VALUE_TO_COLNAME[value]] += 1

    if summary:
        non_conn_pats = summary_data[(summary_data['positive_relation'] == 0) & (summary_data['negative_relation'] == 0)]

        if len(non_conn_pats) > 0:
            logger.warning(f'{len(non_conn_pats)} samples is/are not connected to any genes.')

        return overlay_graph, summary_data, linked_genes
    else:
        return overlay_graph


def show_graph(graph: nx.DiGraph):
    options = {'font_color': 'g', 'font_size': 17, 'font_weight': 'bold'}

    pos = nx.spring_layout(graph)

    info_nodes = [
        node[0]
        for node in graph.nodes(data='color') if node[1] is not None
    ]
    nx.draw_networkx_nodes(graph, pos, nodelist=info_nodes, node_color='b', **options)

    data_nodes = [
        node[0]
        for node in graph.nodes(data='color') if node[1] is None
    ]
    nx.draw_networkx_nodes(graph, pos, nodelist=data_nodes, node_color='r', **options)

    nx.draw_networkx_edges(graph, pos, **options)

    pos_higher = {}
    y_off = 0.005
    x_off = -0.15

    for k, v in pos.items():
        pos_higher[k] = (v[0] - x_off if v[0] < 0 else v[0] + x_off, v[1] + y_off)
    nx.draw_networkx_labels(graph, pos_higher, **options)

    plt.tight_layout()
    plt.tick_params(axis='y', length=8)
    plt.show()
