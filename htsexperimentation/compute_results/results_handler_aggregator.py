import pickle
from htsexperimentation.compute_results.results_handler import ResultsHandler
from htsexperimentation.visualization.plotting import boxplot, plot_predictions_hierarchy


def _read_original_data(datasets):
    data = {}
    for dataset in datasets:
        with open(
                f"./data/data_{dataset}.pickle",
                "rb",
        ) as handle:
            data[dataset] = pickle.load(handle)
    return data


def aggreate_results(datasets, results_path, algorithms_gpf, algorithms):
    results_gpf = {}
    results = {}
    i = 0
    data = _read_original_data(datasets)
    for dataset in datasets:
        results_gpf[dataset] = ResultsHandler(
            path=results_path,
            dataset=dataset,
            algorithms=algorithms_gpf,
            groups=data[dataset],
        )
        results[dataset] = ResultsHandler(
            path=results_path,
            dataset=dataset,
            algorithms=algorithms,
            groups=data[dataset],
        )
        i+=1

    return results_gpf, results


def aggreate_results_boxplot(datasets, results):
    dataset_res = {}
    for dataset in datasets:
        dataset_res[dataset] = results[dataset].data_to_boxplot("mase")

    boxplot(datasets_err=dataset_res, err="mase")


def aggregate_results_plot_hierarchy(datasets, results, algorithm):
    for dataset in datasets:
        (
            results_hierarchy,
            results_by_group_element,
            group_elements,
        ) = results[dataset].compute_results_hierarchy(algorithm=algorithm)
        if group_elements:
            plot_predictions_hierarchy(
                *results_hierarchy,
                *results_by_group_element,
                group_elements,
                results[dataset].h,
                algorithm
            )