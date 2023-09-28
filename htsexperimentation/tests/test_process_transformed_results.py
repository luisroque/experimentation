import unittest
import os
import matplotlib.pyplot as plt
from htsexperimentation.compute_results.results_handler_transformed import (
    compute_aggregated_results_dict,
    create_dataframes_to_plot,
    plot_results_joint,
    concatenate_transformed_dfs,
    plot_radar_transformation_dataset,
)
from htsexperimentation.compute_results.results_handler_aggregator import (
    aggregate_and_save,
    aggregate_results,
    aggregate_results_plot_hierarchy,
)
import seaborn as sns

sns.set()
plt.rcParams.update({"font.size": 8})


class TestModel(unittest.TestCase):
    def setUp(self):
        self.datasets = ["tourism"]
        self.algorithms = ["gpf", "mint", "deepar"]  # , "mint", "deepar"]
        self.err_metric = "mase"

        self.transformations = ["jitter", "scaling", "magnitude_warp", "time_warp"]
        self.versions = ["v0", "v1", "v2", "v3", "v4", "v5"]
        self.samples = ["s0", "s1", "s2"]

        self.results_path = "./results/"

    def test_load_transformed_datasets_mint(self):
        for dataset in self.datasets:
            for algorithm in self.algorithms:
                dict_gpf = compute_aggregated_results_dict(
                    algorithm=algorithm, dataset=dataset, err_metric=self.err_metric
                )

                # Main keys inside dict_gpf
                main_keys = self.transformations

                # Sub-keys inside each main key
                sub_keys = ["orig"] + self.versions

                # Inner keys inside each sub-key
                inner_keys = [self.samples[0]]

                # Establish a reference value for 'all' inside self.err_metric from the first main key, sub-key, and inner key
                reference_value = dict_gpf[main_keys[0]][sub_keys[0]][inner_keys[0]][
                    self.err_metric
                ]["weighted"]

                # Check if the main keys are present
                for key in main_keys:
                    self.assertIn(
                        key,
                        dict_gpf,
                        f"\n\nMain key '{key}' missing in dict_gpf and algorithm {algorithm}",
                    )

                    # Check if the sub-keys are present inside each main key
                    for sub_key in sub_keys:
                        self.assertIn(
                            sub_key,
                            dict_gpf[key],
                            f"\n\nSub key '{sub_key}' missing in dict_gpf['{key}'] and algorithm {algorithm}",
                        )

                        # Check for inner keys
                        for inner_key in inner_keys:
                            self.assertIn(
                                inner_key,
                                dict_gpf[key][sub_key],
                                f"\n\nInner key '{inner_key}' missing in dict_gpf['{key}']['{sub_key}'] and algorithm {algorithm}",
                            )

                            # Check for self.err_metric inside inner key and 'all' inside self.err_metric
                            self.assertIn(
                                self.err_metric, dict_gpf[key][sub_key][inner_key]
                            )
                            self.assertIn(
                                "weighted",
                                dict_gpf[key][sub_key][inner_key][self.err_metric],
                            )

                            if inner_key == "orig":
                                # Compare the value of 'all' inside self.err_metric for consistency across transformations
                                self.assertAlmostEqual(
                                    dict_gpf[key][sub_key][inner_key][self.err_metric][
                                        "weighted"
                                    ],
                                    reference_value,
                                    places=3,
                                )

    def test_plot_transformed_datasets(self):
        for dataset in self.datasets:
            res = [None, None, None]
            to_show = [True, True, True]
            ylims = (-0.5, 6)
            df_res_tourism = list(
                create_dataframes_to_plot(
                    dataset, gpf=True, mint=True, deepar=True, err_metric="mase"
                )
            )

            i = 0
            for idx, ele in enumerate(to_show):
                if ele:
                    res[idx] = df_res_tourism[i]
                    i += 1

            plot_results_joint(dataset, *res, err_metric="mase", ylims=ylims)
            plt.show()

    def test_plot_radar_single_transf(self):
        for dataset in self.datasets:
            df_res_tourism = list(
                create_dataframes_to_plot(
                    dataset, gpf=True, mint=True, deepar=True, err_metric="mase"
                )
            )
            combined_df = concatenate_transformed_dfs(df_res_tourism, err_metric="mase")
            combined_df = combined_df[
                (combined_df.transformation == "jittering")
                & (combined_df.group == "weighted")
            ]
            plot_radar_transformation_dataset(combined_df)

    def test_plot_radar(self):
        for dataset in self.datasets:
            df_res_tourism = list(
                create_dataframes_to_plot(
                    dataset, gpf=True, mint=True, deepar=True, err_metric="mase"
                )
            )
            combined_df = concatenate_transformed_dfs(df_res_tourism, err_metric="mase")
            plot_radar_transformation_dataset(combined_df)

    @staticmethod
    def generate_pickle_filename(
        dataset, algorithm, transformation, version, sample, algo_version
    ):
        return f"metrics_gp_cov_{dataset}_{algorithm}_{transformation}_{version}_{sample}_wwhole_{algo_version}.pickle"

    def test_results_transformed_store_metrics(self):

        for transformation in self.transformations:
            for version in self.versions:
                for sample in self.samples:
                    _, res = aggregate_results(
                        datasets=self.datasets,
                        results_path=self.results_path,
                        algorithms=self.algorithms,
                        load_transformed=True,
                        transformation=transformation,
                        version=version,
                        sample=sample,
                    )

                    aggregate_and_save(
                        datasets=self.datasets,
                        results_path=self.results_path,
                        algorithms=self.algorithms,
                        transformation=transformation,
                        version=version,
                        sample=sample,
                    )

                    for dataset in self.datasets:
                        for algorithm in self.algorithms:
                            algo_version = res[dataset].algorithms_metadata[algorithm][
                                "version"
                            ]
                            expected_filename = self.generate_pickle_filename(
                                dataset,
                                algorithm,
                                transformation,
                                version,
                                sample,
                                algo_version,
                            )
                            expected_filepath = os.path.join(
                                self.results_path, algorithm, expected_filename
                            )

                            self.assertTrue(
                                os.path.exists(expected_filepath),
                                f"Expected file {expected_filepath} not found!",
                            )

    def test_agg_results_transformed(self):

        transformation = "magnitude_warp"
        version = "v5"
        sample = "s0"

        _, res = aggregate_results(
            datasets=self.datasets,
            results_path=self.results_path,
            algorithms=self.algorithms,
            load_transformed=True,
            transformation=transformation,
            version=version,
            sample=sample,
        )

        for algorithm in self.algorithms:
            aggregate_results_plot_hierarchy(
                self.datasets, res, algorithm, include_uncertainty=False
            )
