from typing import Dict, List, Tuple
import re
import math

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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
                f"{groups[0]}-{group_elements[groups[0]][0]}": dict_array_by_group_ele[
                    groups[0]
                ][:, 0],
                f"{groups[0]}-{group_elements[groups[0]][1]}": dict_array_by_group_ele[
                    groups[0]
                ][:, 1],
                groups[1]: dict_array[groups[1]],
                f"{groups[1]}-{group_elements[groups[1]][0]}": dict_array_by_group_ele[
                    groups[1]
                ][:, 0],
                f"{groups[1]}-{group_elements[groups[1]][1]}": dict_array_by_group_ele[
                    groups[1]
                ][:, 1],
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
    algorithm,
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
    plt.suptitle(
        f"Results for different groups for the {algorithm} algorithm", fontsize=16
    )
    plt.tight_layout()
    axs[i].legend()
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
    df = pd.DataFrame(columns=["Value", "Group"])
    for i, d in enumerate(data):
        for value in d:
            df = df.append({"Value": value, "Group": labels[i]}, ignore_index=True)
    sns.boxplot(x="Group", y="Value", data=df)
    plt.title("MASE by group")
    plt.show()


def boxplot(
    datasets_err: Dict[str, pd.DataFrame],
    err: str,
    figsize: tuple = (20, 10),
    ylim: List = None,
    zeroline: bool = False,
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
    gp_types = []
    store_gp_types = True
    for dataset, value in datasets_err.items():
        datasets.append(dataset)
        if isinstance(value, dict):
            for gp_type, df in value.items():
                # store only the first gp_type
                if store_gp_types:
                    gp_types.append(gp_type)
                if df is not None:
                    dfs.append(df)
            store_gp_types = False
        else:
            if value is not None:
                dfs.append(value)
    n_datasets = len(datasets)
    num_gp_types_compare = len(gp_types)
    if n_datasets == 1:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        fg = sns.boxplot(
            x="group", y="value", hue="algorithm", data=pd.concat(dfs), ax=ax
        )
        if gp_types:
            ax.set_title(f"{datasets[0]}_{err}", fontsize=20)
        plt.legend()
        if ylim:
            plt.ylim((ylim[0][0], ylim[0][1]))
        plt.show()
    else:
        _, ax = plt.subplots(
            n_datasets // 2 + n_datasets % 2,
            max((n_datasets - 1) // 2 + (n_datasets - 1) % 2, 2),
            figsize=figsize,
        )
        ax = ax.ravel()
        for dataset_idx in range(len(datasets)):
            df_to_concat = []
            ax[dataset_idx].set_title(
                f"{datasets[dataset_idx]}_{err}",
                fontsize=20,
            )
            if zeroline:
                ax[dataset_idx].axhline(y=0, linestyle="--", alpha=0.2, color="black")
            if gp_types:
                for gp_type_idx in range(num_gp_types_compare):
                    gp_type_idx_dataset = (
                        num_gp_types_compare * dataset_idx + gp_type_idx
                    )

                    df_to_concat.append(dfs[gp_type_idx_dataset])
                df_to_plot = pd.concat(df_to_concat)
                fg = sns.boxplot(
                    x="group",
                    y="value",
                    hue="algorithm",
                    data=df_to_plot,
                    ax=ax[dataset_idx],
                )
            else:
                fg = sns.boxplot(
                    x="group",
                    y="value",
                    hue="algorithm",
                    data=dfs[dataset_idx],
                    ax=ax[dataset_idx],
                )
            if ylim:
                ax[dataset_idx].set_ylim((ylim[dataset_idx][0], ylim[dataset_idx][1]))
        plt.legend()
        plt.show()


def _getting_mean_err_per_algorithm(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by grouping it by algorithm and calculating the mean and standard deviation.

    Args:
        data: The input data in a pandas DataFrame.

    Returns:
        A pandas DataFrame containing the preprocessed data.
    """
    df_mean = data.groupby(["algorithm"]).mean()
    df_std = data.groupby(["algorithm"]).std()
    df_mean.reset_index(inplace=True)
    df_std.reset_index(inplace=True)
    df = pd.merge(df_mean, df_std, on="algorithm", suffixes=("_mean", "_std"))
    return df


def _extract_algorithms(data: pd.DataFrame) -> set:
    """
    Extract the set of unique algorithm names from the input data.

    Args:
        data: The input data in a pandas DataFrame.

    Returns:
        A set of unique algorithm names.
    """
    algorithms = set()
    algorithms.update(
        data["algorithm"].apply(lambda x: re.match(r"([^\d]+)", x).group(1))
    )
    return algorithms


def _extract_x_y(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the x and y data for each algorithm from the input data.

    Args:
        data: The input data in a pandas DataFrame.

    Returns:
        A pandas DataFrame containing the extracted x and y data for each algorithm.
    """
    extracted_data = []
    algorithms = _extract_algorithms(data)
    for algorithm in algorithms:
        algorithm_df = data[data["algorithm"].str.startswith(algorithm)]
        if not algorithm_df.empty:
            x = algorithm_df["algorithm"].apply(
                lambda x: int(re.match(r"([^\d]+)(\d+)", x).group(2))
                if re.match(r"([^\d]+)(\d+)", x)
                else 100
            )
            y_mean = algorithm_df["value_mean"]
            y_std = algorithm_df["value_std"]
            extracted_data.append(
                pd.DataFrame(
                    {"x": x, "y_mean": y_mean, "y_std": y_std, "algorithm": algorithm}
                )
            )
    extracted_data = pd.concat(extracted_data)
    extracted_data.sort_values("x", inplace=True)
    return extracted_data


def _plot_lineplot(
    extracted_data: pd.DataFrame,
    err: str,
    ax: plt.Axes,
    zeroline: bool = True,
):
    """
    Plot a lineplot with standard deviation from the extracted data.

    Args:
        extracted_data: A pandas DataFrame containing the extracted x and y data for each algorithm.
        err: The error metric to use for the lineplot.
        ax: The matplotlib axes object to use for the lineplot.
        zeroline: A boolean indicating whether to draw a horizontal line at y=0.

    Returns:
        None.
    """
    # Compute and plot the relative difference
    base_data = extracted_data[extracted_data["x"] == 100]
    markers = [
        "o",
        "s",
        "^",
        "D",
        "X",
        "P",
        "v",
        "*",
        "H",
        "d",
    ]  # A list of marker symbols
    marker_index = 0
    for algorithm in extracted_data["algorithm"].unique():
        base_mean = base_data[base_data["algorithm"] == algorithm]["y_mean"].values[0]
        base_algorithm = base_data[base_data["algorithm"] == algorithm][
            "algorithm"
        ].values[0]
        algorithm_data = extracted_data[extracted_data["algorithm"] == algorithm]
        algorithm_diff = (algorithm_data["y_mean"] - base_mean) / base_mean
        ax.plot(
            algorithm_data["x"],
            algorithm_diff,
            label=f"{algorithm}",
            linewidth=2,
            marker=markers[marker_index],
            markersize=8,
        )
        marker_index = (marker_index + 1) % len(markers)

    ax.set_xlabel("Percentage of Dataset Used", fontsize=16)
    ax.set_ylabel(f"Relative Difference of {err}", fontsize=16)
    ax.legend(fontsize=14)
    # Customize the appearance of the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_color("#666666")
    ax.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False
    )
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.grid(True, linestyle="--", linewidth=0.5)


def _plot_barplot(
    extracted_data: pd.DataFrame,
    err: str,
    ax: plt.Axes,
):
    hue_order = extracted_data["algorithm"].unique()
    yerrs = {
        alg: extracted_data.loc[extracted_data["algorithm"] == alg, "y_std"].values
        for alg in hue_order
    }

    bar_plot = sns.barplot(
        data=extracted_data,
        x="x",
        y="y_mean",
        hue="algorithm",
        ax=ax,
        capsize=0.1,
        hue_order=hue_order,
        ci=None,
        estimator=np.mean,
    )

    for i, (algorithm, bar_container) in enumerate(zip(hue_order, bar_plot.containers)):
        current_yerrs = yerrs[algorithm]
        for bar, yerr in zip(bar_container, current_yerrs):
            bar_plot.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=yerr,
                fmt="none",
                capsize=0.1,
                color="black",
                elinewidth=1,
            )


def lineplot(
    datasets_err: Dict[str, pd.DataFrame],
    err: str,
    figsize: Tuple[int, int] = (20, 10),
    ylim: List[Tuple[float, float]] = None,
    zeroline: bool = True,
):
    """
    Create a lineplot from the given data.

    Args:
        datasets_err: A dictionary mapping dataset names to pandas DataFrames containing
            the data for each dataset in a format suitable for creating a lineplot.
        err: The error metric to use for the lineplot.
        figsize: The size of the figure to create.
        ylim: A list of tuples containing the y-axis limits for each subplot.
        zeroline: A boolean indicating whether to draw a horizontal line at y=0.

    Returns:
        A matplotlib figure containing the lineplot.
    """
    n_datasets = len(datasets_err)
    n_cols = min(n_datasets, 2)
    n_rows = math.ceil(n_datasets / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = np.atleast_2d(axs)

    for i, (dataset, data) in enumerate(datasets_err.items()):
        if data is not None:
            preprocessed_data = _getting_mean_err_per_algorithm(data)
            extracted_data = _extract_x_y(preprocessed_data)
            ax = axs[i // n_cols, i % n_cols]
            ax.set_title(f"{dataset} - {err}", fontsize=30, fontweight="bold")
            _plot_lineplot(extracted_data, err, ax, zeroline)
            if ylim:
                ax.set_ylim((ylim[i][0], ylim[i][1]))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#666666")
            ax.spines["left"].set_color("#666666")
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
            )
            ax.xaxis.label.set_size(25)
            ax.yaxis.label.set_size(25)
            ax.axhline(y=0, linestyle="--", alpha=0.3, color="black")

    fig.tight_layout()
    plt.show()


def barplot(
    datasets_err: Dict[str, pd.DataFrame],
    err: str,
    figsize: Tuple[int, int] = (20, 10),
    ylim: List[Tuple[float, float]] = None,
):
    """
    Create a barplot from the given data.

    Args:
        datasets_err: A dictionary mapping dataset names to pandas DataFrames containing
            the data for each dataset in a format suitable for creating a barplot.
        err: The error metric to use for the barplot.
        figsize: The size of the figure to create.
        ylim: A list of tuples containing the y-axis limits for each subplot.

    Returns:
        A matplotlib figure containing the barplot.
    """
    n_datasets = len(datasets_err)
    n_cols = min(n_datasets, 2)
    n_rows = math.ceil(n_datasets / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = np.atleast_2d(axs)

    for i, (dataset, data) in enumerate(datasets_err.items()):
        if data is not None:
            preprocessed_data = _getting_mean_err_per_algorithm(data)
            extracted_data = _extract_x_y(preprocessed_data)
            ax = axs[i // n_cols, i % n_cols]
            ax.set_title(f"{dataset}_{err}", fontsize=20)
            _plot_barplot(extracted_data, err, ax)
            if ylim:
                ax.set_ylim((ylim[i][0], ylim[i][1]))

    fig.tight_layout()
    plt.show()
