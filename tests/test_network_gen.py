# -*- coding: utf-8 -*-

"""Test Network Generator."""

import os
import unittest

import pandas as pd
from clepp.embedding import do_graph_gen
from clepp.sample_scoring import do_radical_search

TEST_FOLDER = os.path.dirname(os.path.realpath(__file__))
RESOURCES_PATH = os.path.join(TEST_FOLDER, 'resources')

DUMMY_DATA = os.path.join(RESOURCES_PATH, 'dummy_exp.txt')
DUMMY_DESIGN = os.path.join(RESOURCES_PATH, 'dummy_targets.txt')
DUMMY_NETWORK = os.path.join(RESOURCES_PATH, 'dummy_network.edgelist')


class TestNetGen(unittest.TestCase):
    """Test Network Generator."""

    def test_network_gen(self):
        """Test Interaction Network Generator."""

        data = pd.read_csv(DUMMY_DATA, sep='\t', index_col=0)
        design = pd.read_csv(DUMMY_DESIGN, sep='\t')
        kg_data = pd.read_csv(DUMMY_NETWORK, sep='\t')

        sample_scores = do_radical_search(data=data, design=design, control='Control', threshold=2.5)

        graph_df = do_graph_gen(
            data=sample_scores,
            kg_data=kg_data,
            network_gen_method='interaction_network',
            summary=False
        )

        self.assertEqual(
            graph_df.source.to_list(),
            ['dummy_protein1', 'patient1', 'patient2', 'patient97', 'patient98', 'patient99', 'patient100']
        )

        self.assertEqual(
            graph_df.target.to_list(),
            ['dummy_protein2', 'dummy_protein1', 'dummy_protein1', 'dummy_protein1', 'dummy_protein1',
             'dummy_protein1', 'dummy_protein1']
        )
