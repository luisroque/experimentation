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
        self.datasets = ["m5"]
        self.results_path = "./results/"
        self.algorithms_gpf_sub = ["gpf_exact", "gpf_exact75", "gpf_exact90"]

        _, self.res_sub = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms=self.algorithms_gpf_sub,
            sampling_dataset=True,
            use_version_to_search=True,
        )

    def test_results_handler_aggregate(self):
        res_sub_df = aggregate_results_table(
            datasets=self.datasets, results=self.res_sub
        )
        self.assertTrue(res_sub_df.shape == (21, 4))
