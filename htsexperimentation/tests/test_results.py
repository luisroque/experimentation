import unittest
import pandas as pd
from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.config import RESULTS_PATH


class TestModel(unittest.TestCase):
    def setUp(self):
        self.dataset = 'prison'
        self.results = ResultsHandler(path=RESULTS_PATH, dataset=self.dataset, algorithms=['mint', 'gpf'])

    def test_results_load(self):
        res = self.results.load_results_algorithm(algorithm='ets_bu')
        self.assertTrue(res)

    def test_create_df_boxplot(self):
        res = self.results.data_to_boxplot('mase')
        self.assertTrue(isinstance(res, pd.DataFrame))

    def test_create_boxplot(self):
        res = self.results.data_to_boxplot('mase')
        dataset_res = {}
        dataset_res[self.dataset] = res
        res = self.results.boxplot(datasets_err=dataset_res, err='mase')
