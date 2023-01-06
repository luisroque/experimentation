import unittest
import pandas as pd
from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.config import RESULTS_PATH
from htsexperimentation.visualization.plotting import boxplot


class TestModel(unittest.TestCase):
    def setUp(self):
        self.datasets = ["prison", "tourism", "m5", "police"]
        self.results_prison_gpf = ResultsHandler(
            path=RESULTS_PATH, dataset=self.datasets[0], algorithms=["gpf"]
        )
        self.results_tourism_gpf = ResultsHandler(
            path=RESULTS_PATH, dataset=self.datasets[1], algorithms=["gpf"]
        )
        self.results_m5_gpf = ResultsHandler(
            path=RESULTS_PATH, dataset=self.datasets[2], algorithms=["gpf"]
        )
        self.results_police_gpf = ResultsHandler(
            path=RESULTS_PATH, dataset=self.datasets[3], algorithms=["gpf"]
        )

        self.results_prison = ResultsHandler(
            path=RESULTS_PATH,
            dataset=self.datasets[0],
            algorithms=["mint", "gpf_exact", "deepar", "standard_gp", "ets_bu"],
        )
        self.results_tourism = ResultsHandler(
            path=RESULTS_PATH,
            dataset=self.datasets[1],
            algorithms=["mint", "gpf_exact", "deepar", "standard_gp", "ets_bu"],
        )
        self.results_m5 = ResultsHandler(
            path=RESULTS_PATH,
            dataset=self.datasets[2],
            algorithms=["mint", "gpf_exact", "deepar", "standard_gp", "ets_bu"],
        )
        self.results_police = ResultsHandler(
            path=RESULTS_PATH,
            dataset=self.datasets[3],
            algorithms=["mint", "gpf_exact", "deepar", "standard_gp", "ets_bu"],
        )

    def test_results_load(self):
        res = self.results_prison.load_results_algorithm(algorithm="ets_bu")
        self.assertTrue(res)

    def test_create_df_boxplot(self):
        res = self.results_prison.data_to_boxplot("mase")
        self.assertTrue(isinstance(res, pd.DataFrame))

    def test_create_boxplot_gpf_variants(self):
        dataset_res = {}
        dataset_res[self.datasets[0]] = self.results_prison_gpf.data_to_boxplot("mase")
        dataset_res[self.datasets[1]] = self.results_tourism_gpf.data_to_boxplot("mase")
        dataset_res[self.datasets[2]] = self.results_m5_gpf.data_to_boxplot("mase")
        dataset_res[self.datasets[3]] = self.results_police_gpf.data_to_boxplot("mase")
        res = boxplot(datasets_err=dataset_res, err="mase")

    def test_compute_diferences_gpf_variants(self):
        res, algorithms = self.results_prison_gpf.load_results_algorithm(
            algorithm="gpf"
        )
        differences = self.results_prison_gpf.compute_differences(
            base_algorithm="gpfexact", results=res, algorithms=algorithms, err="rmse"
        )
        res = boxplot(datasets_err=differences, err="rmse")

    def test_create_boxplot_all_algorithms(self):
        dataset_res = {}
        dataset_res[self.datasets[0]] = self.results_prison.data_to_boxplot("mase")
        dataset_res[self.datasets[1]] = self.results_tourism.data_to_boxplot("mase")
        dataset_res[self.datasets[2]] = self.results_m5.data_to_boxplot("mase")
        dataset_res[self.datasets[3]] = self.results_police.data_to_boxplot("mase")
        res = boxplot(datasets_err=dataset_res, err="mase")
