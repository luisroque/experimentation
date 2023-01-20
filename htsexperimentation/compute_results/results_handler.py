import os
import pickle
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
import re
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError


class ResultsHandler:
    def __init__(
        self,
        algorithms: List[str],
        dataset: str,
        groups: Dict,
        path: str = "../results",
    ):
        """
        Initialize a ResultsHandler instance.

        Args:
            algorithms: A list of strings representing the algorithms to load results for.
            dataset: The dataset to load results for.
            groups: data and metadata from the original dataset
            path: The path to the directory containing the results.
        """
        self.algorithms = algorithms
        self.dataset = dataset
        self.path = path
        self.groups = groups
        self.h = self.groups["h"]
        self.seasonality = self.groups["seasonality"]
        self.n_train = self.groups["train"]["n"]
        self.n = self.groups["predict"]["n"]
        self.s = self.groups["train"]["s"]
        self.n_groups = self.groups["train"]["groups_n"]
        self.y_orig_fitpred = self.groups["predict"]["data_matrix"]
        self.y_orig_pred = self.groups["predict"]["data_matrix"][-self.h :, :]
        self.mase = MeanAbsoluteScaledError(multioutput="raw_values")
        self.algorithms_metadata = {}
        for algorithm in algorithms:
            self.algorithms_metadata[algorithm] = {}
            if (algorithm.split("_")[0] == "gpf") & (len(algorithm) > len("gpf")):
                # this allows the user to load a specific type
                # of a gpf algorithm, e.g. exact, sparse
                self.algorithms_metadata[algorithm]["algo_path"] = algorithm.split("_")[
                    0
                ]
                self.algorithms_metadata[algorithm][
                    "preselected_algo_type"
                ] = algorithm.split("_")[1]
                self.algorithms_metadata[algorithm][
                    "algo_name_output_files"
                ] = f"{self.algorithms_metadata[algorithm]['algo_path'][:-1]}_{self.algorithms_metadata[algorithm]['preselected_algo_type']}"
            else:
                self.algorithms_metadata[algorithm]["algo_path"] = algorithm
                self.algorithms_metadata[algorithm][
                    "algo_name_output_files"
                ] = algorithm
                self.algorithms_metadata[algorithm]["preselected_algo_type"] = ""
            self.algorithms_metadata[algorithm][
                "version"
            ] = self._get_latest_version_algo(algorithm)

    @staticmethod
    def _extract_version(filename):
        pattern = r"_(\d+\.\d+\.\d+)\.pickle"
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None

    def _get_latest_version_algo(self, algorithm):
        versions = []
        for file in [
            path
            for path in os.listdir(
                f"{self.path}{self.algorithms_metadata[algorithm]['algo_path']}"
            )
            if self.dataset in path
            and "orig" in path
            and self.algorithms_metadata[algorithm]["algo_name_output_files"] in path
        ]:
            versions.append(self._extract_version(file))
        if len(versions) > 0:
            versions.sort(reverse=True)
            return versions[0]
        else:
            return None

    @staticmethod
    def _validate_param(param, valid_values):
        if param not in valid_values:
            raise ValueError(f"{param} is not a valid value")

    def load_all_results_algo_list(
        self, algorithms_list: List, res_type: str, res_measure: str
    ) -> Tuple[List, List]:
        results = []
        algorithms = []
        for algorithm in algorithms_list:
            res, algorithm = self.load_results_algorithm(
                algorithm=algorithm,
                res_type=res_type,
                res_measure=res_measure,
                output_type="metrics",
            )
            results.append(res)
            algorithms.append(algorithm)

        results = self._filter_list(results)
        algorithms = self._filter_list(algorithms)

        return results, algorithms

    def load_results_algorithm(
        self, algorithm: str, res_type: str, res_measure: str, output_type: str
    ) -> Tuple[Dict, str]:
        """
        Load results for a given algorithm.

        Args:
            algorithm: The algorithm to load results for.
            res_type: defines the type of results, could be 'fit_pred' to receive fitted values plus
                predictions or 'pred' to only store predictions
            res_measure: defines the measure to store, could be 'mean' or 'std'
            output_type: defines the type of output - 'results' or 'metrics'

        Returns:
            A list of results for the given algorithm.
        """
        algorithm_w_type = ""
        result = dict()
        if output_type == "metrics":
            for file in [
                path
                for path in os.listdir(
                    f"{self.path}{self.algorithms_metadata[algorithm]['algo_path']}"
                )
                if self.dataset in path
                and "orig" in path
                and self.algorithms_metadata[algorithm]["version"] in path
                and output_type in path
            ]:
                result, algorithm_w_type = self._load_procedure(file, algorithm)
        else:
            for file in [
                path
                for path in os.listdir(
                    f"{self.path}{self.algorithms_metadata[algorithm]['algo_path']}"
                )
                if self.dataset in path
                and "orig" in path
                and self.algorithms_metadata[algorithm]["version"] in path
                and res_type in path
                and res_measure in path
                and output_type in path
            ]:
                result, algorithm_w_type = self._load_procedure(file, algorithm)

        return result, algorithm_w_type

    def _load_procedure(self, file, algorithm):
        # get the gp_type and concatenate with gpf_
        match = re.search(r"gp_(.*)_cov", file)
        if match:
            algo_type = match.group(1)
        else:
            algo_type = ""

        if (self.algorithms_metadata[algorithm]["preselected_algo_type"] != "") & (
            self.algorithms_metadata[algorithm]["preselected_algo_type"] == algo_type
        ):
            with open(
                f"{self.path}/{self.algorithms_metadata[algorithm]['algo_path']}/{file}",
                "rb",
            ) as handle:
                result = pickle.load(handle)
                algorithm_w_type = (
                    f"{self.algorithms_metadata[algorithm]['algo_path']}_{algo_type}"
                )

        elif self.algorithms_metadata[algorithm]["preselected_algo_type"] == "":
            with open(
                f"{self.path}/{self.algorithms_metadata[algorithm]['algo_path']}/{file}",
                "rb",
            ) as handle:
                result = pickle.load(handle)
                algorithm_w_type = (
                    f"{self.algorithms_metadata[algorithm]['algo_path']}{algo_type}"
                )

        return result, algorithm_w_type

    @staticmethod
    def _filter_list(data):
        """
        This function receives a list of strings or dicts.
        If it is a list of strings and it has a '' value, it is removed from the list.
        If it is a list of dicts and we have a {} it should be removed from the list.
        """
        return list(filter(lambda x: x != "" and x != {}, data))

    def compute_differences(
        self, base_algorithm: str, results: List[Dict], algorithms: List, err: str
    ) -> Dict:
        base_results = results[algorithms.index(base_algorithm)]
        differences = {}
        differences_all_algos = []
        for algorithm in algorithms:
            if algorithm != base_algorithm:
                curr_results = results[algorithms.index(algorithm)]
                diff = {}
                for metric in base_results:
                    diff[metric] = {}
                    for group in base_results[metric].keys():
                        base_value = base_results[metric][group]
                        curr_value = curr_results[metric][group]
                        diff[metric][group] = (
                            (curr_value - base_value) / base_value * 100
                        )
                diff_processed = self._handle_groups(diff, err)
                df_res_to_plot = self._differences_to_df(diff_processed)
                df_res_to_plot = df_res_to_plot.assign(algorithm=algorithm)
                differences_all_algos.append(df_res_to_plot)
        if not differences_all_algos:
            raise ValueError("There are no metrics files from gpf variants")
        df_all_algos_boxplot = pd.concat(differences_all_algos)
        differences["Difference"] = df_all_algos_boxplot
        return differences

    @staticmethod
    def _differences_to_df(diff: Dict) -> pd.DataFrame:
        rows = []
        # Iterate over the keys of the data (the groups)
        for group, values in diff.items():
            # If the values are a scalar, store them as a single row in the DataFrame
            if isinstance(values, (int, float)):
                rows.append({"group": group, "value": values})
            # If the values are an array, store each value as a separate row in the DataFrame
            else:
                for value in values:
                    rows.append({"group": group, "value": value})

        df = pd.DataFrame(rows)
        return df

    def compute_results_hierarchy(
        self,
        algorithm: str,
        res_type: str = "pred",
        output_type: str = "results",
    ) -> Tuple[Tuple[Dict, Dict, Dict], Tuple[Dict, Dict, Dict], Dict]:
        self._validate_param(res_type, ["fitpred", "pred"])
        results_algo_mean, algorithm_w_type = self.load_results_algorithm(
            algorithm,
            res_measure="mean",
            res_type=res_type,
            output_type=output_type,
        )
        results_algo_std, algorithm_w_type = self.load_results_algorithm(
            algorithm,
            res_measure="std",
            res_type=res_type,
            output_type=output_type,
        )
        y_group = {}
        mean_group = {}
        std_group = {}
        y_group_by_ele = {}
        mean_group_by_ele = {}
        std_group_by_ele = {}
        group_elements_names = {}

        group_element_active = dict()

        y_group["bottom"] = self.y_orig_fitpred
        mean_group["bottom"] = results_algo_mean
        std_group["bottom"] = results_algo_std

        y_group["top"] = np.sum(self.y_orig_fitpred, axis=1)
        mean_group["top"] = np.sum(results_algo_mean, axis=1)
        std_group["top"] = np.sqrt(np.sum(results_algo_std**2, axis=1))

        for group in list(self.groups["predict"]["groups_names"].keys()):
            n_elements_group = self.groups["predict"]["groups_names"][group].shape[
                0
            ]
            group_elements = self.groups["predict"]["groups_names"][group]
            groups_idx = self.groups["predict"]["groups_idx"][group]

            y_group_element = np.zeros((self.n, n_elements_group))
            mean_group_element = np.zeros((self.n, n_elements_group))
            std_group_element = np.zeros((self.n, n_elements_group))

            elements_name = []

            for group_idx, element_name in enumerate(group_elements):
                group_element_active[element_name] = np.where(
                    groups_idx == group_idx, 1, 0
                ).reshape((1, -1))

                y_group_element[:, group_idx] = np.sum(
                    group_element_active[element_name] * self.y_orig_fitpred,
                    axis=1,
                )
                mean_group_element[:, group_idx] = np.sum(
                    group_element_active[element_name] * results_algo_mean,
                    axis=1,
                )
                # The variance of the resulting distribution will be the sum
                # of the variances of the original Gaussian distributions
                std_group_element[:, group_idx] = np.sqrt(
                    np.sum(
                        group_element_active[element_name] * results_algo_std**2,
                        axis=1,
                    )
                )

                elements_name.append(element_name)

            group_elements_names[group] = elements_name
            y_group[group] = np.mean(y_group_element, axis=1)
            y_group_by_ele[group] = y_group_element
            mean_group[group] = np.mean(mean_group_element, axis=1)
            mean_group_by_ele[group] = mean_group_element
            std_group[group] = np.mean(std_group_element, axis=1)
            std_group_by_ele[group] = std_group_element

        return (
            (y_group, mean_group, std_group),
            (
                y_group_by_ele,
                mean_group_by_ele,
                std_group_by_ele,
            ),
            group_elements_names,
        )

    def compute_mase(self, results_hierarchy, results_by_group_element, group_elements):
        mase_by_group = {}
        for group in results_hierarchy[0].keys():
            y_true = results_hierarchy[0][group]
            y_pred = results_hierarchy[1][group]
            mase = self.mase(
                y_true=y_true[-self.h :],
                y_pred=y_pred[-self.h :],
                y_train=y_true[: self.n - self.h],
                sp=self.seasonality,
            )
            mase_by_group[group] = mase
        for group, group_ele in group_elements.items():
            mase_by_element = {}
            for idx, element in enumerate(group_ele):
                y_true = results_by_group_element[0][group][:, idx]
                y_pred = results_by_group_element[1][group][:, idx]
                mase = self.mase(
                    y_true=y_true[-self.h :],
                    y_pred=y_pred[-self.h :],
                    y_train=y_true[: self.n - self.h],
                    sp=self.seasonality,
                )
                mase_by_element[element] = mase
            mase_by_group[group] = mase_by_element
        return mase_by_group

    def store_metrics(
        self,
        algorithm,
        res: Dict[str, Dict[str, Union[float, np.ndarray]]],
    ):
        with open(
            f"{self.path}{self.algorithms_metadata[algorithm]['algo_path']}metrics_"
            f"{self.algorithms_metadata[algorithm]['algo_name_output_files']}_cov_"
            f"{self.dataset}_{self.algorithms_metadata[algorithm]['version']}.pickle",
            "wb",
        ) as handle:
            pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)

    def data_to_boxplot(
        self,
        err: str,
        res_measure: str = "",
        res_type: str = "",
        output_type: str = "metrics",
    ) -> pd.DataFrame:
        """
        Convert data to a format suitable for creating a boxplot.

        Args:
            err: The error metric to use for the boxplot.

        Returns:
            A pandas DataFrame containing the data in a format suitable for creating a
            boxplot.
        """
        dfs = []
        self._validate_param(res_type, ["fitpred", "pred", ""])
        self._validate_param(res_measure, ["mean", "std", ""])
        self._validate_param(output_type, ["results", "metrics", ""])
        for algorithm in self.algorithms:
            results, algorithm_w_type = self.load_results_algorithm(
                algorithm,
                res_measure=res_measure,
                res_type=res_type,
                output_type=output_type,
            )
            if algorithm_w_type:
                res_to_plot = self._handle_groups(results, err)
                # Create a list of tuples, where each tuple is a group-value pair
                data = [
                    (group, value)
                    for group in res_to_plot
                    for value in res_to_plot[group]
                ]
                df_res_to_plot = pd.DataFrame(data, columns=["group", "value"])
                df_res_to_plot = df_res_to_plot.assign(algorithm=algorithm_w_type)
                dfs.append(df_res_to_plot)
        if len(dfs) > 0:
            df_all_algos_boxplot = pd.concat(dfs)
        else:
            df_all_algos_boxplot = None
        return df_all_algos_boxplot

    @staticmethod
    def _handle_groups(result_err: Dict, err: str) -> Dict:
        res_to_plot = {}
        for group, res in result_err[err].items():
            # get the individual results to later plot a distribution
            if "ind" in group:
                group_name = group.split("_")[0].lower()
                res_to_plot[group_name] = res
        return res_to_plot
