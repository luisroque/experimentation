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

        self.results_tourism_gpf = ResultsHandler(
            path="./results/",
            dataset=self.datasets[0],
            algorithms=["gpf_exact", "gpf_svg", "gpf_sparse"],
            groups=data[0],
        )

        self.results_tourism = ResultsHandler(
            path="./results/",
            dataset=self.datasets[0],
            algorithms=["mint", "gpf_exact", "deepar"],
            groups=data[0],
        )

    def test_results_load_gpf_variant(self):
        res = self.results_tourism_gpf.load_results_algorithm(
            algorithm="gpf_sparse",
            res_type="fitpred",
            res_measure="mean",
        )
        self.assertTrue(res)

    def test_compute_differences_gpf_variants_single_dataset(self):
        differences = {}
        results = self.results_tourism_gpf.compute_error_metrics(metric="rmse")
        differences[self.results_tourism_gpf.dataset] = self.results_tourism_gpf.calculate_percent_diff(
            base_algorithm="gpf_exact", results=results
        )
        boxplot(datasets_err=differences, err="rmse")

    def test_compute_differences_gpf_variants(self):
        differences = {}
        results = self.results_tourism_gpf.compute_error_metrics(metric="rmse")
        differences[self.results_tourism_gpf.dataset] = self.results_tourism_gpf.calculate_percent_diff(
            base_algorithm="gpf_exact", results=results
        )
        results = self.results_tourism_gpf.compute_error_metrics(metric="rmse")
        differences[self.results_tourism_gpf.dataset] = self.results_tourism_gpf.calculate_percent_diff(
            base_algorithm="gpf_exact", results=results
        )
        boxplot(datasets_err=differences, err="rmse")

    def test_create_boxplot_all_algorithms(self):
        dataset_res = {}
        res_tourism = self.results_tourism.compute_error_metrics(metric="mase")
        res_tourism = self.results_tourism.compute_error_metrics(metric="mase")

        res_obj_tourism = self.results_tourism.dict_to_df(res_tourism, "")
        dataset_res['tourism'] = self.results_tourism.concat_dfs(res_obj_tourism)

        res_obj_tourism = self.results_tourism.dict_to_df(res_tourism, "")
        dataset_res['tourism'] = self.results_tourism.concat_dfs(res_obj_tourism)

        boxplot(datasets_err=dataset_res, err="mase")

    def test_compute_mase(self):
        (
            results_hierarchy,
            results_by_group_element,
            group_elements,
        ) = self.results_tourism.compute_results_hierarchy(algorithm="gpf_exact")
        plot_predictions_hierarchy(
            *results_hierarchy,
            *results_by_group_element,
            group_elements=group_elements,
            forecast_horizon=self.results_tourism.h,
            algorithm="gpf_exact",
            include_uncertainty=False
        )
        mase_by_group = self.results_tourism._compute_metric_from_results(
            results_hierarchy, results_by_group_element, group_elements, "mase"
        )
        print(list(mase_by_group.keys()))
        self.assertTrue(
            list(mase_by_group.keys()) == ["bottom", "top", "state", "zone", "region", "purpose"]
        )

    def test_create_plot_hierarchy(self):
        (
            results_hierarchy,
            results_by_group_element,
            group_elements,
        ) = self.results_tourism.compute_results_hierarchy(algorithm="mint")
        mase_by_group = self.results_tourism._compute_metric_from_results(
            results_hierarchy, results_by_group_element, group_elements, "mase"
        )
        plot_mase(mase_by_group)

    def test_compute_mase_hierarchy_mint(self):
        (
            results_hierarchy,
            results_by_group_element,
            group_elements,
        ) = self.results_tourism.compute_results_hierarchy(algorithm="mint")
        plot_predictions_hierarchy(
            *results_hierarchy,
            *results_by_group_element,
            group_elements=group_elements,
            forecast_horizon=self.results_tourism.h,
            algorithm="mint",
            include_uncertainty=False
        )
        mase_by_group = self.results_tourism._compute_metric_from_results(
            results_hierarchy, results_by_group_element, group_elements, "mase"
        )
        self.assertTrue(
            list(mase_by_group.keys()) == ['bottom', 'top', 'state', 'zone', 'region', 'purpose']
        )
