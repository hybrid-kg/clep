# -*- coding: utf-8 -*-

"""Tests for CDF patient incorporation."""

import os
import unittest

import pandas as pd

from clepp.sample_scoring.radical_search import do_radical_search

TEST_FOLDER = os.path.dirname(os.path.realpath(__file__))
DUMMY_DATA = os.path.join(TEST_FOLDER, 'resources', 'dummy_exp.tsv')
DUMMY_LABELS = os.path.join(TEST_FOLDER, 'resources', 'dummy_labels.csv')


class TestCdf(unittest.TestCase):
    """Tests for cdf method."""

    def test_cdf_1(self):
        """Test cdf."""
        # Load dummy data
        df_data = pd.read_csv(DUMMY_DATA, sep='\t', index_col=0)
        df_labels = pd.read_csv(DUMMY_LABELS)

        output = do_radical_search(
            design=df_labels,
            data=df_data,
            threshold=2.5,
            control='Control',
            control_based=False
        )

        """Test dummy protein (normally distributed)"""
        negative_extreme = output[output.dummy_protein1 == -1]
        positive_extreme = output[output.dummy_protein1 == 1]

        self.assertEqual(
            list(positive_extreme.index),
            ['patient98', 'patient99', 'patient100'],
        )

        self.assertEqual(
            list(negative_extreme.index),
            ['patient1', 'patient2'],
        )

        """Test dummy protein 2 (same values everywhere)"""
        negative_extreme = output[output.dummy_protein2_tied == -1]
        positive_extreme = output[output.dummy_protein2_tied == 1]

        # There should not be any edges because all values of this feature are the same
        # for the entire population
        self.assertFalse(
            list(positive_extreme.index),
        )

        self.assertFalse(
            list(negative_extreme.index),
        )

        """Test dummy protein 3 (extremes)"""
        negative_extreme = output[output.dummy_protein3_extreme == -1]
        positive_extreme = output[output.dummy_protein3_extreme == 1]

        self.assertEqual(
            list(positive_extreme.index),
            ['patient96', 'patient97', 'patient98', 'patient99', 'patient100'],
        )

        self.assertEqual(
            list(negative_extreme.index),
            ['patient1', 'patient2'],
        )

    def test_cdf_2(self):
        """Test cdf."""
        # Load dummy data
        df_data = pd.read_csv(DUMMY_DATA, sep='\t', index_col=0)
        df_labels = pd.read_csv(DUMMY_LABELS)

        output = do_radical_search(
            design=df_labels,
            data=df_data,
            threshold=4,
            control='Control',
            control_based=False
        )

        """Test dummy protein (normally distributed)"""
        negative_extreme = output[output.dummy_protein1 == -1]
        positive_extreme = output[output.dummy_protein1 == 1]

        self.assertEqual(
            list(positive_extreme.index),
            # Note patient 96's values is 0.9600000000000001 so it makes the cut
            ['patient96', 'patient97', 'patient98', 'patient99', 'patient100'],
        )

        self.assertEqual(
            list(negative_extreme.index),
            ['patient1', 'patient2', 'patient3', 'patient4'],
        )

        """Test dummy protein 2 (same values everywhere)"""
        negative_extreme = output[output.dummy_protein2_tied == -1]
        positive_extreme = output[output.dummy_protein2_tied == 1]

        # There should not be any edges because all values of this feature are the same
        # for the entire population
        self.assertFalse(
            list(positive_extreme.index),
        )

        self.assertFalse(
            list(negative_extreme.index),
        )

        """Test dummy protein 3 (extremes)"""
        negative_extreme = output[output.dummy_protein3_extreme == -1]
        positive_extreme = output[output.dummy_protein3_extreme == 1]

        self.assertEqual(
            list(positive_extreme.index),
            ['patient96', 'patient97', 'patient98', 'patient99', 'patient100'],
        )

        self.assertEqual(
            list(negative_extreme.index),
            ['patient1', 'patient2', 'patient3', 'patient4'],
        )

        """Test dummy protein 4 (extremes)"""
        negative_extreme = output[output.dummy_protein4_extreme == -1]
        positive_extreme = output[output.dummy_protein4_extreme == 1]

        self.assertEqual(
            list(positive_extreme.index),
            ['patient95', 'patient96', 'patient97', 'patient98', 'patient99', 'patient100'],
        )

        self.assertEqual(
            list(negative_extreme.index),
            ['patient1', 'patient2', 'patient3', 'patient4'],
        )
