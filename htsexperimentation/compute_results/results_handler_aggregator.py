import pickle
import numpy as np
import os
from typing import List, Dict, Tuple, Any
import pandas as pd
from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.visualization.plotting import (
    boxplot,
    lineplot,
    barplot,
    plot_predictions_hierarchy,
)
from htsexperimentation.helpers.helper_func import concat_dataset_dfs
from copy import deepcopy
import tsaugmentation as tsag
from tsaugmentation.preprocessing import CreateGroups


def create_groups_from_data(dataset_name, transf_data, freq, sample_perc=None):
    if sample_perc:
        dataset = tsag.preprocessing.PreprocessDatasets(
            dataset_name, sample_perc=sample_perc, freq=freq
        )
        groups = dataset.apply_preprocess()
    else:
        groups = CreateGroups(
            dataset_name=dataset_name, freq=freq
        ).read_original_groups()

    vis = tsag.visualization.Visualizer(dataset_name, transf_data=transf_data)

    return groups, vis


def get_aggregate_key_and_freq(dataset_name):
    """
    Returns the aggregate key and frequency corresponding to the specified dataset.
    """
    if dataset_name == "tourism":
        aggregate_key = "state * zone * region * purpose"
        freq = "MS"
    elif dataset_name == "m5":
        aggregate_key = "Department * Category * Store * State * Item"
        freq = "W"
    elif dataset_name == "police":
        aggregate_key = "Crime * Beat * Street * ZIP"
        freq = "D"
    else:
        # Dataset is prison
        aggregate_key = "state * gender * legal"
        freq = "QS"
    return aggregate_key, freq


def _read_original_data(
    datasets: List[str],
    load_transformed: bool = False,
    transformation: str = None,
    version: str = None,
    sample: str = None,
) -> dict:
    data = {}

    for dataset in datasets:
        if load_transformed and version != 'orig':
            version_idx = int(version[1:])
            sample_idx = int(sample[1:])
            _, freq = get_aggregate_key_and_freq(dataset)

            groups, vis = create_groups_from_data(dataset, "whole", freq)

            vis._read_files(f"single_transf_{transformation}")
            groups_transf = deepcopy(groups)
            groups_transf["train"]["data"] = vis.y_new[version_idx, sample_idx][
                : groups_transf["train"]["n"]
            ]
            groups_transf["predict"]["data"] = vis.y_new[version_idx, sample_idx].T
            groups_transf["predict"]["data_matrix"] = vis.y_new[version_idx, sample_idx]

            data[dataset] = groups_transf
        else:
            file_path = f"./data/data_{dataset}.pickle"
            with open(file_path, "rb") as handle:
                data[dataset] = pickle.load(handle)

    return data


def aggregate_results(
    datasets: List[str],
    results_path: str,
    algorithms_gpf: List[str] = None,
    algorithms: List[str] = None,
    sampling_dataset: bool = False,
    use_version_to_search: bool = True,
    load_transformed: bool = False,
    transformation: str = None,
    version: str = None,
    sample: str = None,
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
    data = _read_original_data(
        datasets,
        load_transformed=load_transformed,
        transformation=transformation,
        version=version,
        sample=sample,
    )
    for dataset in datasets:
        if algorithms_gpf:
            results_gpf[dataset] = ResultsHandler(
                path=results_path,
                dataset=dataset,
                algorithms=algorithms_gpf,
                groups=data[dataset],
                use_version_to_search=use_version_to_search,
                load_transformed=load_transformed,
                transformation=transformation,
                version=version,
                sample=sample,
            )
        if algorithms and sampling_dataset:
            results[dataset] = ResultsHandler(
                path=results_path,
                dataset=dataset,
                algorithms=algorithms,
                groups=data[dataset],
                sampling_dataset=sampling_dataset,
                use_version_to_search=use_version_to_search,
                load_transformed=load_transformed,
                transformation=transformation,
                version=version,
                sample=sample,
            )
        elif algorithms:
            results[dataset] = ResultsHandler(
                path=results_path,
                dataset=dataset,
                algorithms=algorithms,
                groups=data[dataset],
                use_version_to_search=use_version_to_search,
                load_transformed=load_transformed,
                transformation=transformation,
                version=version,
                sample=sample,
            )
        i += 1

    return results_gpf, results


def aggregate_results_df(
    datasets: List[str],
    results: Dict[str, ResultsHandler],
    seasonality_map: Dict = None,
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
        res = results[dataset].compute_error_metrics(
            metric="mase", seasonality_map=seasonality_map
        )
        res_obj = results[dataset].dict_to_df(res, "")
        dataset_res[dataset] = results[dataset].concat_dfs(res_obj)
    return dataset_res


def aggregate_results_boxplot(
    datasets: List[str],
    results: Dict[str, ResultsHandler],
    ylims: List[List[int]] = None,
    figsize: Tuple[int, int] = (20, 10),
    n_cols: int = 2,
    seasonality_map: Dict = None,
) -> None:
    """
    Aggregate results from multiple datasets and plot them in a boxplot.

    Args:
        datasets: A list of dataset names to be processed.
        results: A dictionary of results for each dataset.
        ylims: A tuple of the lower and upper y-axis limits for the plot.
    """
    dataset_res = aggregate_results_df(
        datasets, results, seasonality_map=seasonality_map
    )

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
    dataset_res = aggregate_results_df(datasets, results)

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
    dataset_res = aggregate_results_df(datasets, results)

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
    dataset_res = aggregate_results_df(datasets, results)
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
        ].compute_results_hierarchy(
            algorithm=algorithm,
        )
        if group_elements:
            plot_predictions_hierarchy(
                *results_hierarchy,
                *results_by_group_element,
                group_elements=group_elements,
                forecast_horizon=results[dataset].h,
                algorithm=algorithm,
                include_uncertainty=include_uncertainty,
            )


def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def generate_pickle_filename(
    dataset: str,
    algorithm: str,
    transformation: str,
    version: str,
    sample: str,
    algo_version: str
) -> str:
    """
    Generates a filename for a pickle file based on various parameters.

    Args:
        dataset (str): Name of the dataset.
        algorithm (str): Name of the algorithm.
        transformation (str): Transformation applied.
        version (str): Version of the dataset or process.
        sample (str): Sample identifier.
        algo_version (str): Version of the algorithm.

    Returns:
        str: A string representing the filename.
    """
    return f"metrics_gp_cov_{dataset}_{algorithm}_{transformation}_{version}_{sample}_wwhole_{algo_version}.pickle"


def aggregate_and_save(
    datasets: List[str],
    results_path: str,
    algorithms: List[str],
    transformation: str,
    version: str,
    sample: str,
    load_transformed: bool,
):
    """
    Aggregates results for a given set of parameters and saves them into a pickle file.
    Skips saving if the file already exists.

    Args:
        datasets (List[str]): List of dataset names.
        results_path (str): Path to store the results.
        algorithms (List[str]): List of algorithm names.
        transformation (str): Transformation applied.
        version (str): Version of the dataset or process.
        sample (str): Sample identifier.
    """
    res_gpf, res = aggregate_results(
        datasets=datasets,
        results_path=results_path,
        algorithms=algorithms,
        load_transformed=load_transformed,
        transformation=transformation,
        version=version,
        sample=sample,
    )

    dataset_res = aggregate_results_df(datasets, res)

    for dataset in datasets:
        grouped_data = dataset_res[dataset].groupby(["group", "algorithm"]).mean().reset_index()
        for algorithm in algorithms:
            algo_version = res[dataset].algorithms_metadata[algorithm]["version"]
            pickle_filename = generate_pickle_filename(
                dataset, algorithm, transformation, version, sample, algo_version
            )
            full_path = os.path.join(results_path, algorithm, pickle_filename)

            if not os.path.exists(full_path):
                aggregate_results_plot_hierarchy(
                    datasets=[dataset],
                    results=res,
                    algorithm=algorithm,
                    include_uncertainty=False,
                )

                sub_data = grouped_data[grouped_data["algorithm"] == algorithm]
                error_data = sub_data.set_index("group")["value"].to_dict()

                save_to_pickle({"mase": error_data}, full_path)
            else:
                print(f"File {full_path} already exists, skipping computation and saving.")


def get_version_sample_combinations(load_transformed: bool, versions: List[str], samples: List[str]) -> List[Tuple[str, str]]:
    """
    Generates appropriate combinations of versions and samples.

    Args:
        load_transformed (bool): Indicates whether transformed data is being loaded.
        versions (List[str]): List of versions.
        samples (List[str]): List of samples.

    Returns:
        List[Tuple[str, str]]: List of tuples containing version and sample combinations.
    """
    if load_transformed:
        return [(version, sample) for version in versions for sample in samples]
    else:
        # If not loading transformed data, only use 'orig' version and 's0' sample
        return [('orig', 's0')]


def create_metrics_file(
    algorithms: List[str],
    transformations: List[str],
    versions: List[str],
    samples: List[str],
    dataset: str,
    load_transformed: bool,
    results_path: str = "./results/"
):
    """
    Creates metrics files for various combinations of algorithms, transformations,
    versions, samples, and datasets.

    Args:
        algorithms (List[str]): List of algorithms.
        transformations (List[str]): List of transformations.
        versions (List[str]): List of versions.
        samples (List[str]): List of samples.
        dataset (str): dataset.
        results_path (str, optional): Base path for saving results. Defaults to "./results/".
    """
    for transformation in transformations:
        for version, sample in get_version_sample_combinations(load_transformed, versions, samples):
            aggregate_and_save(
                datasets=[dataset],
                results_path=results_path,
                algorithms=algorithms,
                transformation=transformation,
                version=version,
                sample=sample,
                load_transformed=load_transformed
            )
