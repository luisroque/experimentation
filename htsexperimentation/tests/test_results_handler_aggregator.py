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
        self.datasets = ["tourism", "m5", "police"]
        self.results_path = "./results/"
        self.algorithms = ["mint", "gpf_exact", "deepar"]
        self.algorithms_gpf = ["gpf_exact"]
        self.datasets_no_version = ["tourism"]

    def test_results_handler_aggregate(self):
        res_gpf, res = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms,
            use_version_to_search=True,
        )
        self.assertTrue(len(res) == 2)

    def test_results_handler_aggregate_boxplot(self):
        res_gpf, res = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms,
        )
        aggregate_results_boxplot(
            datasets=self.datasets, results=res, ylims=[[0, 10], [0, 2]]
        )
        aggregate_results_boxplot(datasets=self.datasets, results=res_gpf)

    def test_results_handler_boxplot(self):
        res_gpf, res = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms,
        )
        differences = {}
        for dataset in self.datasets:
            results = res_gpf[dataset].compute_error_metrics(metric="rmse")
            differences[dataset] = res_gpf[dataset].calculate_percent_diff(
                base_algorithm="gpf_exact", results=results
            )

        boxplot(
            datasets_err=differences,
            err="rmse",
            ylim=[[-2, 2], [-1, 10], [-1, 10], [-1, 10]],
        )

    def test_results_handler_aggregate_plot_hierarchy(self):
        res_gpf, res = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms,
        )
        aggregate_results_plot_hierarchy(
            datasets=self.datasets,
            results=res,
            algorithm="deepar",
            include_uncertainty=False,
        )
        aggregate_results_plot_hierarchy(
            datasets=self.datasets, results=res, algorithm="mint"
        )
        aggregate_results_plot_hierarchy(
            datasets=self.datasets, results=res, algorithm="gpf_exact"
        )
        aggregate_results_plot_hierarchy(
            datasets=self.datasets, results=res_gpf, algorithm="gpf_exact"
        )

    def test_results_handler_aggregate_table(self):
        _, res = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms_gpf=self.algorithms_gpf,
            algorithms=self.algorithms,
        )

        res_df = aggregate_results_table(self.datasets, res)
        self.assertTrue(
            res_df.loc[
                (res_df.group == "bottom")
                & (res_df.algorithm == "mint")
                & (res_df.dataset == "prison")
            ].value.item()
            < 3
        )
