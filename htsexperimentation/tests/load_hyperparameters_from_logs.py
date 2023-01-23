import unittest
import pickle

from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.compute_results.results_handler_aggregator import (
    aggregate_hyperparameter,
    aggreate_results,
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

        self.results_prison_gpf = ResultsHandler(
            path="./results/",
            dataset=self.datasets[0],
            algorithms=["gpf_exact"],
            groups=data[0],
        )

    def test_hypertuning_from_logs(self):
        res = self.results_prison_gpf.load_hyperparameters_logs("gpf_exact", "./logs/")
        self.assertTrue(res["version"] == "0.3.44")

    def test_agg_hypertuning_from_logs(self):
        _, results_handler = aggreate_results(
            self.datasets, results_path="./results/", algorithms=["gpf_exact"]
        )
        res = aggregate_hyperparameter(
            datasets=self.datasets,
            results_handler=results_handler,
            algorithm="gpf_exact",
        )
        self.assertTrue(res[1]["dataset"] == "tourism")
