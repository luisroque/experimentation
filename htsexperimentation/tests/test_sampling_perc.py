import unittest
import pickle

from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.compute_results.results_handler_aggregator import (
    aggregate_results_boxplot,
    aggregate_results_barplot,
    aggregate_results_lineplot,
    aggregate_results,
    aggregate_results_plot_hierarchy,
)
from htsexperimentation.visualization.plotting import (
    boxplot,
)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.datasets = ["tourism", "m5", "police"]
        data = {}
        for i in range(len(self.datasets)):
            with open(
                f"./data/data_{self.datasets[i]}.pickle",
                "rb",
            ) as handle:
                data[i] = pickle.load(handle)

        self.results_path = "./results/"
        self.algorithms = [
            "gpf_exact",
            "gpf_exact50",
            "gpf_exact60",
            "gpf_exact75",
            "gpf_exact90",
            "mint",
            "mint50",
            "mint60",
            "mint75",
            "mint90",
            "deepar",
            "deepar50",
            "deepar60",
            "deepar75",
            "deepar90"
        ]

        self.gpf_algorithms = [
            "gpf_exact",
            "gpf_exact75",
            "gpf_exact90",
        ]

        self.results_prison_gpf = ResultsHandler(
            path=self.results_path,
            dataset=self.datasets[0],
            algorithms=self.algorithms,
            groups=data[0],
        )

        self.results_m5_gpf = ResultsHandler(
            path=self.results_path,
            dataset=self.datasets[2],
            algorithms=self.gpf_algorithms,
            groups=data[2],
        )

    def test_results_load_gpf_exact_correctly(self):
        res = self.results_prison_gpf.load_results_algorithm(
            algorithm="gpf_exact",
            res_type="fitpred",
            res_measure="mean",
        )
        self.assertTrue(res[0].shape == (48, 32))

    def test_compute_differences_gpf_variants(self):
        differences = {}
        results = self.results_prison_gpf.compute_error_metrics(metric="rmse")
        differences[
            self.results_prison_gpf.dataset
        ] = self.results_prison_gpf.calculate_percent_diff(
            base_algorithm="gpf_exact", results=results
        )
        boxplot(datasets_err=differences, err="rmse")

    def test_compute_differences_gpf_variants_m5(self):
        differences = {}
        results = self.results_m5_gpf.compute_error_metrics(metric="rmse")
        differences[
            self.results_m5_gpf.dataset
        ] = self.results_m5_gpf.calculate_percent_diff(
            base_algorithm="gpf_exact", results=results
        )
        boxplot(datasets_err=differences, err="rmse")

    def test_results_handler_aggregate(self):
        _, res_sub = aggregate_results(
            datasets=[self.datasets[0]],
            results_path=self.results_path,
            algorithms=self.algorithms,
            sampling_dataset=True,
        )
        aggregate_results_boxplot(
            datasets=[self.datasets[0]], results=res_sub, ylims=[[0, 10], [0, 2]]
        )

    def test_results_handler_aggregate_lineplot(self):
        _, res_sub = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms=self.algorithms,
            sampling_dataset=True,
        )
        aggregate_results_lineplot(
            datasets=self.datasets,
            results=res_sub,
            ylims=[[-0.1, 1.5], [-0.1, 1], [-0.3, 0.3]],
        )

    def test_aggregate_results_plot_hierarchy_mint(self):
        _, res_sub = aggregate_results(
            datasets=[self.datasets[0]],
            results_path=self.results_path,
            algorithms=self.algorithms,
            sampling_dataset=True,
        )
        aggregate_results_plot_hierarchy(
            datasets=[self.datasets[0]], results=res_sub, algorithm="mint75"
        )

    def test_aggregate_results_plot_hierarchy_gpf(self):
        _, res_sub = aggregate_results(
            datasets=[self.datasets[0]],
            results_path=self.results_path,
            algorithms=self.algorithms,
            sampling_dataset=True,
        )
        aggregate_results_plot_hierarchy(
            datasets=[self.datasets[0]], results=res_sub, algorithm="gpf_exact75"
        )

    def test_results_handler_aggregate_barplot(self):
        _, res_sub = aggregate_results(
            datasets=[self.datasets[0]],
            results_path=self.results_path,
            algorithms=self.algorithms,
            sampling_dataset=True,
        )
        aggregate_results_barplot(
            datasets=[self.datasets[0]], results=res_sub, ylims=[[0, 10], [0, 2]]
        )

    def test_perc_diff(self):
        _, res_sub = aggregate_results(
            datasets=[self.datasets[0]],
            results_path=self.results_path,
            algorithms=self.algorithms,
            sampling_dataset=True,
        )

        differences = {}
        results = res_sub[self.datasets[0]].compute_error_metrics(metric="rmse")
        differences[self.datasets[0]] = res_sub[
            self.datasets[0]
        ].calculate_percent_diff(base_algorithm="gpf_exact", results=results)

        boxplot(
            datasets_err=differences,
            err="rmse",
            ylim=[[-2, 20], [-1, 50]],
        )
