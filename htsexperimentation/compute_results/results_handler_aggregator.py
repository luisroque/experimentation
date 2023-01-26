import pickle
from typing import Dict, List
from pandas import json_normalize
from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.visualization.plotting import (
    boxplot,
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


def aggreate_results(datasets, results_path, algorithms_gpf=None, algorithms=None):
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
            )
        if algorithms:
            results[dataset] = ResultsHandler(
                path=results_path,
                dataset=dataset,
                algorithms=algorithms,
                groups=data[dataset],
            )
        i += 1

    return results_gpf, results


def _aggreate_results_df(datasets, results):
    dataset_res = {}
    for dataset in datasets:
        res_prison = results[dataset].compute_error_metrics(metric="mase")
        res_obj = results[dataset].dict_to_df(res_prison, "")
        dataset_res[dataset] = results[dataset].concat_dfs(res_obj)
    return dataset_res


def aggreate_results_boxplot(datasets, results, ylims=None):
    dataset_res = _aggreate_results_df(datasets, results)

    boxplot(datasets_err=dataset_res, err="mase", ylim=ylims)


def aggreate_results_table(datasets, results):
    dataset_res = _aggreate_results_df(datasets, results)
    res_df = concat_dataset_dfs(dataset_res)
    res_df = res_df.groupby(['group', 'algorithm', 'dataset']).mean()['value'].reset_index()
    res_df = res_df.sort_values(by=['dataset', 'algorithm', 'group'])
    return res_df


def aggregate_results_plot_hierarchy(datasets, results, algorithm):
    for dataset in datasets:
        (results_hierarchy, results_by_group_element, group_elements,) = results[
            dataset
        ].compute_results_hierarchy(algorithm=algorithm)
        if group_elements:
            plot_predictions_hierarchy(
                *results_hierarchy,
                *results_by_group_element,
                group_elements,
                results[dataset].h,
                algorithm,
            )


def aggregate_hyperparameter(
    datasets: List, results_handler: Dict[str, ResultsHandler], algorithm: str, path_to_logs: str = "./logs/"
):
    hyperparameters = []
    for dataset in datasets:
        hyperparameters.append(
            results_handler[dataset].load_hyperparameters_logs(algorithm, path_to_logs)
        )
    return json_normalize(hyperparameters)