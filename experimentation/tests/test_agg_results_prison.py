import unittest
from ..compute_results.compute_res_funcs import calculate_agg_results_all_datasets


class TestModel(unittest.TestCase):

    def setUp(self):
        self.datasets = ['prison']

    def test_results_gpf_parsing_prison(self):
        self.assertTrue(
            calculate_agg_results_all_datasets(self.datasets,
                                               'mase',
                                               path='experimentation/tests/results_probabilistic')[0].shape,
            (270, 6))