import unittest
import pandas as pd
import pickle

from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.visualization.plotting import (
    boxplot,
    plot_predictions_hierarchy,
    plot_mase,
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
            algorithms=["gpf_exact", "gpf_svg", "gpf_sparse"],
            groups=data[0],
        )
        self.results_tourism_gpf = ResultsHandler(
            path="./results/",
            dataset=self.datasets[1],
            algorithms=["gpf_exact", "gpf_svg", "gpf_sparse"],
            groups=data[1],
        )

        self.results_prison = ResultsHandler(
            path="./results/",
            dataset=self.datasets[0],
            algorithms=["mint", "gpf_exact", "deepar"],
            groups=data[0],
        )
        self.results_tourism = ResultsHandler(
            path="./results/",
            dataset=self.datasets[1],
            algorithms=["mint", "gpf_exact", "deepar"],
            groups=data[1],
        )

    def test_results_load(self):
        res = self.results_prison.load_results_algorithm(
            algorithm="ets_bu",
            res_type="fitpred",
            res_measure="mean",
        )
        self.assertTrue(res)

    def test_compute_differences_gpf_variants_single_dataset(self):
        differences = {}
        results = self.results_prison_gpf.compute_error_metrics(metric="rmse")
        differences[self.results_prison_gpf.dataset] = self.results_prison_gpf.calculate_percent_diff(
            base_algorithm="gpf_exact", results=results
        )
        boxplot(datasets_err=differences, err="rmse")

    def test_compute_differences_gpf_variants(self):
        differences = {}
        results = self.results_prison_gpf.compute_error_metrics(metric="rmse")
        differences[self.results_prison_gpf.dataset] = self.results_prison_gpf.calculate_percent_diff(
            base_algorithm="gpf_exact", results=results
        )
        results = self.results_tourism_gpf.compute_error_metrics(metric="rmse")
        differences[self.results_tourism_gpf.dataset] = self.results_tourism_gpf.calculate_percent_diff(
            base_algorithm="gpf_exact", results=results
        )
        boxplot(datasets_err=differences, err="rmse")

    def test_create_boxplot_all_algorithms(self):
        dataset_res = {}
        res_prison = self.results_prison.compute_error_metrics(metric="mase")
        res_tourism = self.results_tourism.compute_error_metrics(metric="mase")

        res_obj_prison = self.results_prison.dict_to_df(res_prison, "")
        dataset_res['prison'] = self.results_prison.concat_dfs(res_obj_prison)

        res_obj_tourism = self.results_tourism.dict_to_df(res_tourism, "")
        dataset_res['tourism'] = self.results_prison.concat_dfs(res_obj_tourism)

        boxplot(datasets_err=dataset_res, err="mase")

    def test_compute_mase(self):
        (
            results_hierarchy,
            results_by_group_element,
            group_elements,
        ) = self.results_prison.compute_results_hierarchy(algorithm="gpf_exact")
        plot_predictions_hierarchy(
            *results_hierarchy,
            *results_by_group_element,
            group_elements,
            self.results_prison.h,
            "gpf_exact",
        )
        mase_by_group = self.results_prison._compute_metric(
            results_hierarchy, results_by_group_element, group_elements, "mase"
        )
        self.assertTrue(
            list(mase_by_group.keys()) == ["bottom", "top", "state", "gender", "legal"]
        )

    def test_create_plot_hierarchy(self):
        (
            results_hierarchy,
            results_by_group_element,
            group_elements,
        ) = self.results_prison.compute_results_hierarchy(algorithm="mint")
        mase_by_group = self.results_prison._compute_metric(
            results_hierarchy, results_by_group_element, group_elements, "mase"
        )
        plot_mase(mase_by_group)
