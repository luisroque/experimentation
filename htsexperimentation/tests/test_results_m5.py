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
        self.datasets = ["m5"]
        data = {}
        for i in range(len(self.datasets)):
            with open(
                f"./data/data_{self.datasets[i]}.pickle",
                "rb",
            ) as handle:
                data[i] = pickle.load(handle)

        self.results_m5_gpf = ResultsHandler(
            path="./results/",
            dataset=self.datasets[0],
            algorithms=["gpf_exact"],
            groups=data[0],
        )

    def test_create_boxplot_all_algorithms(self):
        dataset_res = {}
        res_m5 = self.results_m5_gpf.compute_error_metrics(metric="mase")

        res_obj_prison = self.results_m5_gpf.dict_to_df(res_m5, "")
        dataset_res['m5'] = self.results_m5_gpf.concat_dfs(res_obj_prison)

        boxplot(datasets_err=dataset_res, err="mase")
