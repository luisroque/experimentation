from typing import Dict
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..compute_results.compute_res_funcs import (
    agg_res_bottom_series,
    compute_aggreated_results_dict,
)


def _build_dict_to_plot_hierarchy(
    true_values,
    mean_predictions,
    std_predictions,
    true_values_by_group_ele,
    mean_predictions_by_group_ele,
    std_predictions_by_group_ele,
    group_elements,
):
    groups = list(filter(lambda x: x not in ["bottom", "top"], true_values.keys()))
    dicts_to_plot = []
    for dict_array, dict_array_by_group_ele in zip(
        (true_values, mean_predictions, std_predictions),
        (
            true_values_by_group_ele,
            mean_predictions_by_group_ele,
            std_predictions_by_group_ele,
        ),
    ):
        dicts_to_plot.append(
            {
                "top": dict_array["top"],
                groups[0]: dict_array[groups[0]],
                f"{groups[0]}-{group_elements[groups[0]][0]}": dict_array_by_group_ele[groups[0]][:, 0],
                f"{groups[0]}-{group_elements[groups[0]][1]}": dict_array_by_group_ele[groups[0]][:, 1],
                groups[1]: dict_array[groups[1]],
                f"{groups[1]}-{group_elements[groups[1]][0]}": dict_array_by_group_ele[groups[1]][:, 0],
                f"{groups[1]}-{group_elements[groups[1]][1]}": dict_array_by_group_ele[groups[1]][:, 1],
                "bottom-1": dict_array["bottom"][:, 0],
                "bottom-2": dict_array["bottom"][:, 1],
                "bottom-3": dict_array["bottom"][:, 2],
                "bottom-4": dict_array["bottom"][:, 3],
                "bottom-5": dict_array["bottom"][:, 4],
            }
        )

    return dicts_to_plot[0], dicts_to_plot[1], dicts_to_plot[2]


def plot_predictions_hierarchy(
    true_values,
    mean_predictions,
    std_predictions,
    true_values_by_group_ele,
    mean_predictions_by_group_ele,
    std_predictions_by_group_ele,
    group_elements,
    forecast_horizon,
):
    (
        true_values_to_plot,
        mean_predictions_to_plot,
        std_predictions_to_plot,
    ) = _build_dict_to_plot_hierarchy(
        true_values,
        mean_predictions,
        std_predictions,
        true_values_by_group_ele,
        mean_predictions_by_group_ele,
        std_predictions_by_group_ele,
        group_elements,
    )
    num_keys = len(true_values_to_plot)
    n = true_values_to_plot["top"].shape[0]

    num_cols = 3
    num_rows = (num_keys + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, figsize=(14, 8))

    # If the figure only has one subplot, make it a 1D array
    # so we can iterate over it
    if num_keys == 1:
        axs = [axs]

    axs = axs.ravel()

    for i, group in enumerate(true_values_to_plot):
        true_vals = true_values_to_plot[group]
        mean_preds = mean_predictions_to_plot[group]
        std_preds = std_predictions_to_plot[group]

        mean_preds_fitted = mean_preds[: n - forecast_horizon]
        mean_preds_pred = mean_preds[-forecast_horizon:]

        std_preds_fitted = std_preds[: n - forecast_horizon]
        std_preds_pred = std_preds[-forecast_horizon:]

        axs[i].plot(true_vals, label="True values")
        axs[i].plot(
            range(n - forecast_horizon), mean_preds_fitted, label="Mean fitted values"
        )
        axs[i].plot(
            range(n - forecast_horizon, n), mean_preds_pred, label="Mean predictions"
        )

        # Add the 95% interval to the plot
        axs[i].fill_between(
            range(n - forecast_horizon),
            mean_preds_fitted - 2 * std_preds_fitted,
            mean_preds_fitted + 2 * std_preds_fitted,
            alpha=0.2,
            label="Fitting 95% CI",
        )
        axs[i].fill_between(
            range(n - forecast_horizon, n),
            mean_preds_pred - 2 * std_preds_pred,
            mean_preds_pred + 2 * std_preds_pred,
            alpha=0.2,
            label="Forecast 95% CI",
        )

        axs[i].set_title(f"{group}")
    plt.tight_layout()
    axs[i].legend()
    plt.show()


def plot_compare_err_metric(
    err="mase", dataset="prison", figsize=(20, 10), path="../results_probabilistic"
):
    dict_gpf = compute_aggreated_results_dict(
        algorithm="gpf", dataset=dataset, err_metric=err, path=path
    )
    df_gpf_bottom = agg_res_bottom_series(dict_gpf)
    dict_mint = compute_aggreated_results_dict(
        algorithm="mint", dataset=dataset, err_metric=err, path=path
    )
    df_mint_bottom = agg_res_bottom_series(dict_mint)
    dict_deepar = compute_aggreated_results_dict(
        algorithm="deepar", dataset=dataset, err_metric=err, path=path
    )
    df_deepar_bottom = agg_res_bottom_series(dict_deepar)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax = ax.ravel()
    sns.barplot(x="value", y="group", data=df_gpf_bottom, color="blue", ax=ax[0])
    ax[0].set_title("gpf")
    sns.barplot(x="value", y="group", data=df_mint_bottom, color="darkorange", ax=ax[1])
    ax[1].set_title("mint")
    sns.barplot(x="value", y="group", data=df_deepar_bottom, color="green", ax=ax[2])
    ax[2].set_title("deepar")
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()


def boxplot_error(df_res, datasets, figsize=(20, 10)):
    if len(datasets) == 1:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        fg = sns.boxplot(x="group", y="value", hue="algorithm", data=df_res[0], ax=ax)
        ax.set_title(datasets[0], fontsize=20)
        plt.legend()
        plt.show()
    else:
        _, ax = plt.subplots(
            len(datasets) // 2 + len(datasets) % 2,
            len(datasets) // 2 + len(datasets) % 2,
            figsize=figsize,
        )
        ax = ax.ravel()
        for i in range(len(datasets)):
            fg = sns.boxplot(
                x="group", y="value", hue="algorithm", data=df_res[i], ax=ax[i]
            )
            ax[i].set_title(datasets[i], fontsize=20)
        plt.legend()
        plt.show()


def plot_mase(mase_by_group):
    data = []
    labels = []
    for group, values in mase_by_group.items():
        if type(values) is dict:
            for sub_group, sub_values in values.items():
                data.append(sub_values)
                labels.append(group)
        else:
            data.append(values)
            labels.append(group)
    df = pd.DataFrame(columns=['Value', 'Group'])
    for i, d in enumerate(data):
        for value in d:
            df = df.append({'Value': value, 'Group': labels[i]}, ignore_index=True)
    sns.boxplot(x='Group', y='Value', data=df)
    plt.title('MASE by group')
    plt.show()


def boxplot(datasets_err: Dict[str, pd.DataFrame], err: str, figsize: tuple = (20, 10)):
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
            max((n_datasets-1) // 2 + (n_datasets-1) % 2, 2),
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
