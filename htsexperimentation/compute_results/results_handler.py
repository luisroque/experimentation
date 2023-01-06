import os
import pickle
from typing import Dict, List, Tuple

import pandas as pd
import re


class ResultsHandler:
    def __init__(self, algorithms: List[str], dataset: str, path: str = "../results"):
        """
        Initialize a ResultsHandler instance.

        Args:
            algorithms: A list of strings representing the algorithms to load results for.
            dataset: The dataset to load results for.
            path: The path to the directory containing the results.
        """
        self.algorithms = algorithms
        self.dataset = dataset
        self.path = path
        self.algo_path = ""
        self.preselected_algo_type = ""

    def load_results_algorithm(self, algorithm: str) -> Tuple[List, List]:
        """
        Load results for a given algorithm.

        Args:
            algorithm: The algorithm to load results for.

        Returns:
            A list of results for the given algorithm.
        """
        results = []
        algorithms_w_type = []
        self.preselected_algo_type = ""
        if (algorithm.split("_")[0] == "gpf") & (len(algorithm) > len("gpf")):
            # this allows the user to load a specific type
            # of a gpf algorithm, e.g. exact, sparse
            self.algo_path = algorithm.split("_")[0]
            self.preselected_algo_type = algorithm.split("_")[1]
        else:
            self.algo_path = algorithm
        for file in [
            path
            for path in os.listdir(f"{self.path}{self.algo_path}")
            if self.dataset in path and "orig" in path
        ]:
            # get the gp_type and concatenate with gpf_
            match = re.search(r"gp_(.*)_cov", file)
            if match:
                algo_type = match.group(1)
            else:
                algo_type = ""

            if (self.preselected_algo_type != "") & (
                self.preselected_algo_type == algo_type
            ):
                with open(f"{self.path}/{self.algo_path}/{file}", "rb") as handle:
                    results.append(pickle.load(handle))
                    algorithms_w_type.append(f"{self.algo_path}_{algo_type}")
            elif self.preselected_algo_type == "":
                with open(f"{self.path}/{self.algo_path}/{file}", "rb") as handle:
                    results.append(pickle.load(handle))
                    algorithms_w_type.append(f"{self.algo_path}{algo_type}")

        return results, algorithms_w_type

    def compute_differences(self, base_algorithm: str, results: List[Dict], algorithms: List, err: str) -> Dict:
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
                        diff[metric][group] = (curr_value - base_value) / base_value * 100
                diff_processed = self._handle_groups(diff, err)
                df_res_to_plot = self._differences_to_df(diff_processed)
                df_res_to_plot = df_res_to_plot.assign(algorithm=algorithm)
                differences_all_algos.append(df_res_to_plot)
        df_all_algos_boxplot = pd.concat(differences_all_algos)
        differences['Difference'] = df_all_algos_boxplot
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

    def data_to_boxplot(self, err: str) -> pd.DataFrame:
        """
        Convert data to a format suitable for creating a boxplot.

        Args:
            err: The error metric to use for the boxplot.

        Returns:
            A pandas DataFrame containing the data in a format suitable for creating a
            boxplot.
        """
        dfs = []
        for algorithm in self.algorithms:
            results, algorithm_w_type = self.load_results_algorithm(algorithm)
            for result, algo in zip(results, algorithm_w_type):
                res_to_plot = self._handle_groups(result, err)
                # Create a list of tuples, where each tuple is a group-value pair
                data = [
                    (group, value)
                    for group in res_to_plot
                    for value in res_to_plot[group]
                ]
                df_res_to_plot = pd.DataFrame(data, columns=["group", "value"])
                df_res_to_plot = df_res_to_plot.assign(algorithm=algo)
                dfs.append(df_res_to_plot)
        df_all_algos_boxplot = pd.concat(dfs)
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
