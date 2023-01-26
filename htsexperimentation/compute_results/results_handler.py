import os
import pickle
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
import re
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from sklearn.metrics import mean_squared_error


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
            if not self.algorithms_metadata[algorithm]["version"]:
                raise ValueError(
                    f"Please make sure that you have result files for the {algorithm} algorithm"
                )

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

    def compute_error_metrics(self, metric: str = "mase") -> Dict:
        metric_algorithm = {}
        for algorithm in self.algorithms:
            (
                results_hierarchy,
                results_by_group_element,
                group_elements,
            ) = self.compute_results_hierarchy(algorithm=algorithm)

            metric_algorithm[algorithm] = self._compute_metric(
                results_hierarchy,
                results_by_group_element,
                group_elements,
                metric=metric,
            )
        return metric_algorithm

    def load_results_algorithm(
        self, algorithm: str, res_type: str, res_measure: str
    ) -> Tuple[Dict, str]:
        """
        Load results for a given algorithm.

        Args:
            algorithm: The algorithm to load results for.
            res_type: defines the type of results, could be 'fit_pred' to receive fitted values plus
                predictions or 'pred' to only store predictions
            res_measure: defines the measure to store, could be 'mean' or 'std'

        Returns:
            A list of results for the given algorithm.
        """
        algorithm_w_type = ""
        result = dict()
        # get the gp_type and concatenate with gpf_
        match = re.search(r"gpf_(.*)", algorithm)
        if match:
            algo_type = match.group(1)
        else:
            algo_type = ""

        if algorithm in self.algorithms_metadata.keys():
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
                and 'results' in path
                and algo_type in path
            ]:
                result, algorithm_w_type = self._load_procedure(
                    file, algorithm, algo_type
                )

        return result, algorithm_w_type

    def _load_procedure(self, file, algorithm, algo_type):
        if (self.algorithms_metadata[algorithm]["preselected_algo_type"] != "") & (
            self.algorithms_metadata[algorithm]["preselected_algo_type"] == algo_type
        ):
            with open(
                f"{self.path}{self.algorithms_metadata[algorithm]['algo_path']}/{file}",
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

    def load_hyperparameters_logs(self, algorithm, path_to_logs="./logs/s"):
        hyperparameters_dataset_algo = {}
        for file_name in [
            path
            for path in os.listdir(f"{path_to_logs}")
            if self.dataset in path
            and self.algorithms_metadata[algorithm]["version"] in path
            and self.algorithms_metadata[algorithm]["algo_path"] in path
            and "hypertuning" in path
        ]:
            best_hyperparameters = {}
            execution_date = ""
            correct_log_structure = False
            with open(f"{path_to_logs}{file_name}", "r") as f:
                for line in f:
                    if "Algorithm" in line:
                        correct_log_structure = True
                        algorithm = re.search(r"Algorithm: (.*?),", line).group(1)
                        version = re.search(r"Version: (.*?),", line).group(1)
                        dataset = re.search(r"Dataset: (.*?),", line).group(1)
                        match = re.search(
                            r"learning rate = (.*?), weight decay = (.*?), scheduler = (.*?), gamma = (.*?), inducing points percentage = (.*?), patience = (.*?)$",
                            line,
                        )
                        best_hyperparameters = {
                            "learning rate": match.group(1),
                            "weight decay": match.group(2),
                            "scheduler": match.group(3),
                            "gamma": match.group(4),
                            "inducing points percentage": match.group(5),
                            "patience": match.group(6),
                        }
                        execution_date = re.search(r"^(.*?),", line).group(1)
            if correct_log_structure:
                hyperparameters_dataset_algo = {
                    "algorithm": algorithm,
                    "version": version,
                    "dataset": dataset,
                    "best_hyperparameters": best_hyperparameters,
                    "execution_date": execution_date,
                }
        return hyperparameters_dataset_algo

    @staticmethod
    def _filter_list(data):
        """
        This function receives a list of strings or dicts.
        If it is a list of strings and it has a '' value, it is removed from the list.
        If it is a list of dicts and we have a {} it should be removed from the list.
        """
        return list(filter(lambda x: x != "" and x != {}, data))

    def calculate_percent_diff(self, results, base_algorithm):
        percent_diffs = {}
        base_result = results[base_algorithm]
        for algorithm in results.keys():
            if algorithm != base_algorithm:
                percent_diffs[algorithm] = self.percentage_difference_recur(
                    base_result, results[algorithm]
                )
        percent_diffs_dfs = self.dict_to_df(percent_diffs, base_algorithm)
        return percent_diffs_dfs

    def percentage_difference_recur(self, base_dict, dict2):
        """
        Computes the percentage difference between a base_dict and dict2.
        Since we are handling error metrics, a positive number means that the
        dict2 has a higher error than base_dict
        """
        diff = {}
        for key in base_dict:
            if key in dict2:
                if type(base_dict[key]) == dict:
                    diff[key] = self.percentage_difference_recur(base_dict[key], dict2[key])
                else:
                    diff[key] = (dict2[key] - base_dict[key]) / (base_dict[key])
        return diff

    @staticmethod
    def concat_dfs(obj):
        result = pd.concat([df.assign(algorithm=algorithm) for algorithm, df in obj.items()])
        return result

    def dict_to_df(self, data, base_algorithm):
        dict_df_algo = {}
        for algorithm in self.algorithms:
            if algorithm != base_algorithm:
                data_algo = data[algorithm]
                df = pd.DataFrame(
                    columns=["group", "value", "algorithm", "group_element"]
                )
                for key, value in data_algo.items():
                    if type(value) == dict:
                        for k, v in value.items():
                            for i, val in enumerate(v):
                                df = df.append(
                                    {
                                        "group": key,
                                        "value": val,
                                        "algorithm": algorithm,
                                        "group_element": k,
                                    },
                                    ignore_index=True,
                                )
                    else:
                        for i, val in enumerate(value):
                            df = df.append(
                                {
                                    "group": key,
                                    "value": val,
                                    "algorithm": algorithm,
                                    "group_element": "",
                                },
                                ignore_index=True,
                            )
                dict_df_algo[algorithm] = df
        return dict_df_algo

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
        )
        results_algo_std, algorithm_w_type = self.load_results_algorithm(
            algorithm,
            res_measure="std",
            res_type=res_type,
        )
        y_group = {}
        mean_group = {}
        std_group = {}
        y_group_by_ele = {}
        mean_group_by_ele = {}
        std_group_by_ele = {}
        group_elements_names = {}

        group_element_active = dict()
        if algorithm_w_type:
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

    def _compute_metric(
        self, results_hierarchy, results_by_group_element, group_elements, metric="mase"
    ):
        res = None
        metric_by_group = {}
        for group in results_hierarchy[0].keys():
            y_true = results_hierarchy[0][group]
            y_pred = results_hierarchy[1][group]
            if metric == "mase":
                res = self.mase(
                    y_true=y_true[-self.h :],
                    y_pred=y_pred[-self.h :],
                    y_train=y_true[: self.n - self.h],
                    sp=self.seasonality,
                )
            elif metric == "rmse":
                res = mean_squared_error(
                    y_true=y_true[-self.h :],
                    y_pred=y_pred[-self.h :],
                    multioutput="raw_values",
                )
            metric_by_group[group] = res
        for group, group_ele in group_elements.items():
            metric_by_element = {}
            for idx, element in enumerate(group_ele):
                y_true = results_by_group_element[0][group][:, idx]
                y_pred = results_by_group_element[1][group][:, idx]
                if metric == "mase":
                    res = self.mase(
                        y_true=y_true[-self.h :],
                        y_pred=y_pred[-self.h :],
                        y_train=y_true[: self.n - self.h],
                        sp=self.seasonality,
                    )
                elif metric == "rmse":
                    res = mean_squared_error(
                        y_true=y_true[-self.h :],
                        y_pred=y_pred[-self.h :],
                        multioutput="raw_values",
                    )
                metric_by_element[element] = res
            metric_by_group[group] = metric_by_element
        return metric_by_group

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



