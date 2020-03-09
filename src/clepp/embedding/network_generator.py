# -*- coding: utf-8 -*-

"""Calculate the intersection between the pathways & generate an edgelist based upon that."""
from typing import TextIO, Optional

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from os import listdir
from os.path import isfile, join
from igraph import plot


def do_graph_gen(
        data: pd.DataFrame,
        network_gen_method: Optional[str] = 'interaction_network',
        gmt: Optional[str] = None,
        intersection_threshold: Optional[float] = 0.1,
        kg_data: Optional[pd.DataFrame] = None,
        folder_path: Optional[str] = None,
        jaccard_threshold: Optional[float] = 0.2
):
    information_graph = nx.Graph()

    if network_gen_method == 'pathway_overlap':
        with open(gmt, 'r') as geneset:
            information_graph = plot_pathway_overlap(geneset, intersection_threshold)

    elif network_gen_method == 'interaction_network':
        information_graph = plot_interaction_network(kg_data)

    elif network_gen_method == 'interaction_network_overlap':
        information_graph = plot_interaction_net_overlap(folder_path, jaccard_threshold)

    final_graph = overlay_samples(data, information_graph)

    graph_df = nx.to_pandas_edgelist(final_graph)
    graph_df['regulation'].fillna(0.0, inplace=True)

    col_list = list(graph_df)
    col_list[1], col_list[2] = col_list[2], col_list[1]
    graph_df = graph_df.loc[:, col_list]

    return graph_df


def plot_pathway_overlap(
        geneset: TextIO,
        intersection_threshold: float = 0.1
) -> nx.Graph:
    """Plots the overlap/intersection between pathways as a graph based on shared genes."""
    pathway_dict = {
        line.strip().split("\t")[0]: line.strip().split("\t")[2:]
        for line in geneset.readlines()
    }

    pathway_overlap_graph = nx.Graph()

    for pathway_1 in pathway_dict.keys():
        for pathway_2 in pathway_dict.keys():
            if pathway_1 == pathway_2:
                continue

            union = list(set().union(pathway_dict[pathway_1], pathway_dict[pathway_2]))
            intersection = list(set().intersection(pathway_dict[pathway_1], pathway_dict[pathway_2]))

            if len(intersection) > (intersection_threshold * len(union)):
                pathway_overlap_graph.add_edge(pathway_1, pathway_2)

    return pathway_overlap_graph


def plot_interaction_network(
        kg_data: pd.DataFrame
) -> nx.Graph:
    """Plots a knowledge graph based on the interaction data."""
    interaction_graph = nx.Graph()

    # Append the source to target mapping to the main data edgelist
    for idx in kg_data.index:
        interaction_graph.add_edge(kg_data.iat[idx, 0], kg_data.iat[idx, 2])

    return interaction_graph


def plot_interaction_net_overlap(
        folder_path: str,
        jaccard_threshold: float = 0.2
) -> nx.Graph:
    """Plots the overlap/intersection between interaction networks as a graph based on shared nodes."""
    graphs = []
    files = [
        f
        for f in listdir(folder_path)
        if isfile(join(folder_path, f)) and f.endswith('.bel')
    ]

    # Get all the interaction network files from the folder and add them as individual graphs to a list
    for filename in files:
        with open(join(folder_path, filename), 'r') as file:
            graph = nx.Graph(name=filename)
            for line in file:
                src, attr, dst = line.split()
                graph.add_edge(src, dst)
                graph[src][dst]['attribute'] = attr
            graphs.append(graph)

    overlap_graph = nx.Graph()

    for graph_1, graph_2 in combinations(graphs, 2):
        if _get_jaccard_index(graph_1, graph_2) > jaccard_threshold:
            overlap_graph.add_edge(graph_1.graph['name'], graph_2.graph['name'])

    return overlap_graph


def _get_jaccard_index(
        graph_1: nx.Graph,
        graph_2: nx.Graph
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
        information_graph: nx.Graph
) -> nx.Graph:
    """Overlays the data on the information graph by adding edges between patients and information nodes if pairwise
    value is not 0."""
    overlay_graph = information_graph.copy()

    for patient, gene, value in pd.melt(data, id_vars=data.columns[0]).values:
        if value == 0:
            continue
        if gene in list(overlay_graph):
            overlay_graph.add_edge(patient, gene, regulation=value)

    return overlay_graph


def show_graph(graph: nx.Graph):
    options = {'font_color': 'g', 'font_size': 17, 'font_weight': 'bold'}

    pos = nx.spring_layout(graph)

    info_nodes = [
        node[0]
        for node in graph.nodes(data='color') if node[1] is not None
    ]
    nx.draw_networkx_nodes(graph, pos, nodelist=info_nodes,  node_color='b', **options)

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
