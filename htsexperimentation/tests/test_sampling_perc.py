import unittest
import pickle

from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.compute_results.results_handler_aggregator import (
    aggreate_results,
    aggreate_results_boxplot,
)
from htsexperimentation.visualization.plotting import (
    boxplot,
)


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

        self.results_path = "./results/"
        self.algorithms = ["gpf_exact", "gpf_exact50", "gpf_exact75"]

        self.results_prison_gpf = ResultsHandler(
            path=self.results_path,
            dataset=self.datasets[0],
            algorithms=self.algorithms,
            groups=data[0],
            sampling_dataset=True,
        )

    def test_results_load_gpf_exact_correctly(self):
        res = self.results_prison_gpf.load_results_algorithm(
            algorithm="gpf_exact",
            res_type="fitpred",
            res_measure="mean",
        )
        self.assertTrue(res[0].shape == (48, 32))

    def test_results_load_gpf_subsampled(self):
        res = self.results_prison_gpf.load_results_algorithm(
            algorithm="gpf_exact50",
            res_type="fitpred",
            res_measure="mean",
        )
        self.assertTrue(res[0].shape == (30, 32))

    def test_compute_differences_gpf_variants(self):
        differences = {}
        results = self.results_prison_gpf.compute_error_metrics(metric="rmse")
        differences[
            self.results_prison_gpf.dataset
        ] = self.results_prison_gpf.calculate_percent_diff(
            base_algorithm="gpf_exact", results=results
        )
        boxplot(datasets_err=differences, err="rmse", zeroline=True)

    def test_results_handler_aggregate(self):
        _, res_sub = aggreate_results(
            datasets=[self.datasets[0]],
            results_path=self.results_path,
            algorithms=self.algorithms,
            sampling_dataset=True,
        )
        aggreate_results_boxplot(
            datasets=[self.datasets[0]], results=res_sub, ylims=[[0, 10], [0, 2]]
        )

    def test_perc_diff(self):
        _, res_sub = aggreate_results(
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
            zeroline=True,
        )
