import unittest
import pickle

from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.visualization.plotting import (
    boxplot,
    plot_predictions_hierarchy,
    plot_mase,
)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.datasets = ["tourism"]
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
            load_transformed=True
        )

    def test_results_load_gpf(self):
        res = self.results_prison_gpf.load_results_algorithm(
            algorithm="gpf_exact",
            res_type="fitpred",
            res_measure="mean",
        )
        self.assertTrue(res)
