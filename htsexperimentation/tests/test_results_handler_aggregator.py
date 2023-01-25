import unittest

from htsexperimentation.compute_results.results_handler_aggregator import (
    aggreate_results, aggreate_results_boxplot, aggregate_results_plot_hierarchy
)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.datasets = ["prison", "tourism"]
        self.results_path = './results/'
        self.algorithms = ["mint", "gpf_exact", "deepar"]
        self.algorithms_gpf = ["gpf_exact", "gpf_svg"]

    def test_results_handler_aggregate(self):
        res_gpf, res = aggreate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms
        )
        self.assertTrue(len(res) == 2)

    def test_results_handler_aggregate_boxplot(self):
        res_gpf, res = aggreate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms
        )
        aggreate_results_boxplot(datasets=self.datasets, results=res, ylims=(0, 2))
        aggreate_results_boxplot(datasets=self.datasets, results=res_gpf)

    def test_results_handler_aggregate_plot_hierarchy(self):
        res_gpf, res = aggreate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms
        )
        aggregate_results_plot_hierarchy(datasets=self.datasets, results=res, algorithm='deepar')
        aggregate_results_plot_hierarchy(datasets=self.datasets, results=res, algorithm='mint')
        aggregate_results_plot_hierarchy(datasets=self.datasets, results=res, algorithm='gpf_exact')
        aggregate_results_plot_hierarchy(datasets=self.datasets, results=res_gpf, algorithm='gpf_exact')
        aggregate_results_plot_hierarchy(datasets=self.datasets, results=res_gpf, algorithm='deepar')

