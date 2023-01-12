import unittest
import pandas as pd
import pickle

from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.config import RESULTS_PATH
from htsexperimentation.visualization.plotting import boxplot, plot_predictions_hierarchy


class TestModel(unittest.TestCase):
    def setUp(self):
        self.datasets = ["prison", "tourism"]
        data = {}
        for i in range(len(self.datasets)):
            with open(
                    f"./data/data_{self.datasets[i]}.pickle",
                    "rb",
            ) as handle:
                data[i] = pickle.load(handle)

        self.results_prison_gpf = ResultsHandler(
            path=RESULTS_PATH,
            dataset=self.datasets[0],
            algorithms=["gpf"],
            groups=data[0],
        )
        self.results_tourism_gpf = ResultsHandler(
            path=RESULTS_PATH,
            dataset=self.datasets[1],
            algorithms=["gpf"],
            groups=data[1],
        )

        self.results_prison = ResultsHandler(
            path=RESULTS_PATH,
            dataset=self.datasets[0],
            algorithms=["mint", "gpf_exact", "deepar", "standard_gp", "ets_bu"],
            groups=data[0],
        )
        self.results_tourism = ResultsHandler(
            path=RESULTS_PATH,
            dataset=self.datasets[1],
            algorithms=["mint", "gpf_exact", "deepar", "standard_gp", "ets_bu"],
            groups=data[1],
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
        dataset_res[self.datasets[0]] = self.results_prison.data_to_boxplot(
            "mase", output_type="metrics"
        )
        dataset_res[self.datasets[1]] = self.results_tourism.data_to_boxplot(
            "mase", output_type="metrics"
        )
        res = boxplot(datasets_err=dataset_res, err="mase")

    def test_create_plot_hierarchy(self):
        results_hierarchy, results_by_group_element, group_elements = self.results_prison.compute_results_hierarchy(algorithm='gpf')
        plot_predictions_hierarchy(*results_hierarchy, *results_by_group_element, group_elements, self.results_prison.h)
