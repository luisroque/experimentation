from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsaugmentation.preprocessing.subsample_dataset import CreateGroups
from htsexperimentation.compute_results.results_handler import ResultsHandler


def get_original_and_missing_data(
    dataset_name: str, freq: str, sample_perc: float = None
) -> Tuple[Dict, Dict]:
    create_group = (
        CreateGroups(dataset_name, freq, sample_perc) if sample_perc else None
    )
    create_group_original = CreateGroups(dataset_name, freq)

    dataset_missing = create_group.read_subsampled_groups() if create_group else None
    dataset_original = create_group_original.read_original_groups()

    return dataset_original, dataset_missing


def plot_predictions_vs_original(
    path: str,
    dataset_name: str,
    freq: str,
    sample_perc: float = None,
    algorithm: str = "gpf_exact",
    series_to_plot: Optional[List[int]] = None,
) -> None:
    if series_to_plot is None:
        series_to_plot = [1, 2]
    if sample_perc:
        sample_value = str(int(sample_perc * 100))
        algorithm += sample_value
    dataset_original, dataset_missing = get_original_and_missing_data(
        dataset_name, freq, sample_perc=sample_perc
    )
    n_series_to_plot = len(series_to_plot)

    # Fetch results for Gaussian Process algorithm
    results_gpf = ResultsHandler(
        path=path,
        dataset=dataset_name,
        algorithms=[algorithm],
        groups=dataset_original,
    )

    prediction_mean, _ = results_gpf.load_results_algorithm(
        algorithm=algorithm,
        res_measure="mean",
        res_type="fitpred",
    )
    prediction_std, _ = results_gpf.load_results_algorithm(
        algorithm=algorithm,
        res_measure="std",
        res_type="fitpred",
    )

    x_missing = np.array(dataset_missing["train"]["x_values"])
    y_missing = np.array(dataset_missing["train"]["data"])

    x_original = np.array(dataset_original["predict"]["x_values"])

    # Find the  missing x_values
    x_true_missing = np.setdiff1d(x_original[: -dataset_original["h"]], x_missing)
    y_original = np.array(dataset_original["predict"]["data_matrix"])

    x_total = np.array(dataset_original["predict"]["x_values"])

    if n_series_to_plot == 1:
        fig, ax = plt.subplots(figsize=(20, 10))
        axs = [ax]
    else:
        _, axs = plt.subplots(n_series_to_plot, 1, figsize=(20, 10))
        axs = axs.ravel()

    for i, series_idx in enumerate(series_to_plot):
        axs[i].plot(
            x_total[: -dataset_original["h"]],
            y_original[: -dataset_original["h"], series_idx],
            label="Original Data Train",
            linestyle="--",
            color="darkorange",
        )
        axs[i].plot(
            x_total[-dataset_original["h"] :],
            y_original[-dataset_original["h"] :, series_idx],
            label="Original Data Test",
            linestyle="--",
        )

        y_missing_interpolated = np.interp(
            x_total[: -dataset_original["h"]],
            x_missing,
            y_missing[:, series_idx],
        )
        axs[i].plot(
            x_total[: -dataset_original["h"]],
            y_missing_interpolated,
            label="Interpolated Data",
            color="darkred",
        )
        axs[i].scatter(
            x_true_missing,
            np.zeros_like(x_true_missing),
            color="red",
            zorder=5,
            label="Missing Data",
        )
        axs[i].plot(x_total, prediction_mean[:, series_idx], label="GP Mean", color="darkgreen")
        axs[i].fill_between(
            x_total,
            prediction_mean[:, series_idx] - 1.96 * prediction_std[:, series_idx],
            prediction_mean[:, series_idx] + 1.96 * prediction_std[:, series_idx],
            color="darkgreen",
            alpha=0.1,
            label="GP Uncertainty",
        )

        axs[i].set_title(f"Time Series {series_idx + 1}")
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Value")
        axs[i].legend()
        # ax.grid(True)

    plt.suptitle(f"Predictions for {dataset_name} using {algorithm} GPs")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
