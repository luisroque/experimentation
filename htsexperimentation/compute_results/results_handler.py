import os
import pickle
from typing import Dict, List, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        for file in [
            path
            for path in os.listdir(f"{self.path}{algorithm}")
            if self.dataset in path and "orig" in path
        ]:
            with open(f"{self.path}/{algorithm}/{file}", "rb") as handle:
                results.append(pickle.load(handle))

            # get the gp_type and concatenate with gpf_
            match = re.search(r'gp(.*)_cov', file)
            if match:
                algo_type = match.group(1)
            else:
                algo_type = None
            algorithms_w_type.append(f"{algorithm}{algo_type}")
        return results, algorithms_w_type

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
                res_to_plot = {}
                for group, res in result[err].items():
                    # get the individual results to later plot a distribution
                    if "ind" in group:
                        group_name = group.split("_")[0].lower()
                        res_to_plot[group_name] = res
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
    def boxplot(
        datasets_err: Dict[str, pd.DataFrame], err: str, figsize: tuple = (20, 10)
    ):
        """
        Create a boxplot from the given data.

        Args:
            datasets_err: A dictionary mapping dataset names to pandas DataFrames containing
                the data for each dataset in a format suitable for creating a boxplot.
            err: The error metric to use for the boxplot.
            figsize: The size of the figure to create.

        Returns:
            A matplotlib figure containing the boxplot.
        """
        datasets = []
        dfs = []
        for dataset, df in datasets_err.items():
            datasets.append(dataset)
            dfs.append(df)
        n_datasets = len(datasets)
        if n_datasets == 1:
            _, ax = plt.subplots(1, 1, figsize=figsize)
            fg = sns.boxplot(x="group", y="value", hue="algorithm", data=dfs[0], ax=ax)
            ax.set_title(f"{datasets[0]}_{err}", fontsize=20)
            plt.legend()
            plt.show()
        else:
            _, ax = plt.subplots(
                n_datasets // 2 + n_datasets % 2,
                n_datasets // 2 + n_datasets % 2,
                figsize=figsize,
            )
            ax = ax.ravel()
            for i in range(len(datasets)):
                fg = sns.boxplot(
                    x="group", y="value", hue="algorithm", data=dfs[i], ax=ax[i]
                )
                ax[i].set_title(datasets[i], fontsize=20)
            plt.legend()
            plt.show()
