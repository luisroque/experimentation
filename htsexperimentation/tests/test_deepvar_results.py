import unittest

from htsexperimentation.compute_results.results_handler_aggregator import (
    aggregate_results,
    aggregate_results_boxplot,
    aggregate_results_plot_hierarchy,
    aggregate_results_table,
    boxplot,
)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.datasets = ["tourism"]
        self.results_path = "./results/"
        self.algorithms = ["deepvarhierarchical"]
        self.algorithms_gpf = ["gpf_exact", "gpf_svg"]
        self.datasets_no_version = ["tourism"]
        self.seasonality_map = {"police": 1}

    def test_results_handler_aggregate_boxplot(self):
        res_gpf, res = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms,
        )
        aggregate_results_boxplot(
            datasets=self.datasets, results=res, ylims=[[0, 2], [0, 2], [0, 2]]
        )

    def test_results_handler_aggregate_plot_hierarchy(self):
        res_gpf, res = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms,
        )
        aggregate_results_plot_hierarchy(
            datasets=self.datasets, results=res, algorithm="deepvarhierarchical"
        )
