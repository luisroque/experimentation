import pickle
from typing import List, Dict, Tuple, Any
from pandas import json_normalize
import pandas as pd
from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.visualization.plotting import (
    boxplot,
    lineplot,
    barplot,
    plot_predictions_hierarchy,
)
from htsexperimentation.helpers.helper_func import concat_dataset_dfs


def _read_original_data(datasets):
    data = {}
    for dataset in datasets:
        with open(
            f"./data/data_{dataset}.pickle",
            "rb",
        ) as handle:
            data[dataset] = pickle.load(handle)
    return data


def aggregate_results(
    datasets: List[str],
    results_path: str,
    algorithms_gpf: List[str] = None,
    algorithms: List[str] = None,
    sampling_dataset: bool = False,
    use_version_to_search: bool = True,
) -> Tuple[Dict[str, ResultsHandler], Dict[str, ResultsHandler]]:
    """
    Aggregate results from multiple datasets using the specified algorithms.

    Args:
        datasets: A list of dataset names to be processed.
        results_path: The path to the results directory.
        algorithms_gpf: A list of algorithms to use when running the GPF method.
        algorithms: A list of algorithms to use when running the experiments.
        sampling_dataset: A boolean indicating if sampling is to be performed.

    Returns:
        A tuple of two dictionaries containing the results for the GPF method and experiments respectively.
    """
    results_gpf = {}
    results = {}
    i = 0
    data = _read_original_data(datasets)
    for dataset in datasets:
        if algorithms_gpf:
            results_gpf[dataset] = ResultsHandler(
                path=results_path,
                dataset=dataset,
                algorithms=algorithms_gpf,
                groups=data[dataset],
                use_version_to_search=use_version_to_search,
            )
        if algorithms and sampling_dataset:
            results[dataset] = ResultsHandler(
                path=results_path,
                dataset=dataset,
                algorithms=algorithms,
                groups=data[dataset],
                sampling_dataset=sampling_dataset,
                use_version_to_search=use_version_to_search,
            )
        elif algorithms:
            results[dataset] = ResultsHandler(
                path=results_path,
                dataset=dataset,
                algorithms=algorithms,
                groups=data[dataset],
                use_version_to_search=use_version_to_search,
            )
        i += 1

    return results_gpf, results


def _aggregate_results_df(
    datasets: List[str], results: Dict[str, ResultsHandler]
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate results from multiple datasets into a DataFrame.

    Args:
        datasets: A list of dataset names to be processed.
        results: A dictionary of results for each dataset.

    Returns:
        A dictionary containing the results for each dataset in DataFrame format.
    """
    dataset_res = {}
    for dataset in datasets:
        res = results[dataset].compute_error_metrics(metric="mase")
        res_obj = results[dataset].dict_to_df(res, "")
        dataset_res[dataset] = results[dataset].concat_dfs(res_obj)
    return dataset_res


def aggregate_results_boxplot(
    datasets: List[str],
    results: Dict[str, ResultsHandler],
    ylims: List[List[int]] = None,
    figsize: Tuple[int, int] = (20, 10),
    n_cols: int = 2,
) -> None:
    """
    Aggregate results from multiple datasets and plot them in a boxplot.

    Args:
        datasets: A list of dataset names to be processed.
        results: A dictionary of results for each dataset.
        ylims: A tuple of the lower and upper y-axis limits for the plot.
    """
    dataset_res = _aggregate_results_df(datasets, results)

    boxplot(
        datasets_err=dataset_res,
        err="mase",
        ylim=ylims,
        num_cols=n_cols,
        figsize=figsize,
    )


def aggregate_results_lineplot(
    datasets: List[str],
    results: Dict[str, ResultsHandler],
    ylims: List[List[int]] = None,
) -> None:
    """
    Aggregate results from multiple datasets and plot them in a boxplot.

    Args:
        datasets: A list of dataset names to be processed.
        results: A dictionary of results for each dataset.
        ylims: A tuple of the lower and upper y-axis limits for the plot.
    """
    dataset_res = _aggregate_results_df(datasets, results)

    lineplot(datasets_err=dataset_res, err="mase", ylim=ylims)


def aggregate_results_barplot(
    datasets: List[str],
    results: Dict[str, ResultsHandler],
    ylims: List[List[int]] = None,
) -> None:
    """
    Aggregate results from multiple datasets and plot them in a boxplot.

    Args:
        datasets: A list of dataset names to be processed.
        results: A dictionary of results for each dataset.
        ylims: A tuple of the lower and upper y-axis limits for the plot.
    """
    dataset_res = _aggregate_results_df(datasets, results)

    barplot(datasets_err=dataset_res, err="mase", ylim=ylims)


def aggregate_results_table(
    datasets: List[str], results: Dict[str, ResultsHandler]
) -> pd.DataFrame:
    """
    Aggregate results from multiple datasets and return them in a table format.

    Args:
        datasets: A list of dataset names to be processed.
        results: A dictionary of results for each dataset.

    Returns:
        A DataFrame containing the aggregated results.
    """
    dataset_res = _aggregate_results_df(datasets, results)
    res_df = concat_dataset_dfs(dataset_res)
    res_df = (
        res_df.groupby(["group", "algorithm", "dataset"]).mean()["value"].reset_index()
    )
    res_df = res_df.sort_values(by=["dataset", "algorithm", "group"])
    return res_df


def aggregate_results_plot_hierarchy(
    datasets: List[str],
    results: Dict[str, ResultsHandler],
    algorithm: str,
    include_uncertainty: bool = True,
) -> None:
    """
    Aggregate results from multiple datasets and plot them in a hierarchical format.

    Args:
        datasets: A list of dataset names to be processed.
        results: A dictionary of results for each dataset.
        algorithm: The name of the algorithm to use.

    Returns:
        None
    """
    for dataset in datasets:
        (results_hierarchy, results_by_group_element, group_elements,) = results[
            dataset
        ].compute_results_hierarchy(algorithm=algorithm)
        if group_elements:
            plot_predictions_hierarchy(
                *results_hierarchy,
                *results_by_group_element,
                group_elements=group_elements,
                forecast_horizon=results[dataset].h,
                algorithm=algorithm,
                include_uncertainty=include_uncertainty,
            )
