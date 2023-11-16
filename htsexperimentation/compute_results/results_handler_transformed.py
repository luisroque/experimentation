from typing import Generator, List
import pickle
import os
from collections import Iterable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from htsexperimentation.compute_results.results_handler_aggregator import (
    create_metrics_file,
)


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def find(lst, s):
    return [i for i, x in enumerate(lst) if x == s]


def keys_exists(element, *keys):
    """
    Check if *keys (nested) exists in `element` (dict).
    """
    if not isinstance(element, dict):
        raise AttributeError("keys_exists() expects dict as first argument.")
    if len(keys) == 0:
        raise AttributeError("keys_exists() expects at least two arguments, one given.")

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def is_valid_file(path, algorithm, dataset):
    return (
        algorithm in path
        and "metrics" in path
        and dataset in path
        and any(
            sub_string in path
            for sub_string in [
                "exact_",
                "mint_",
                "deepar_",
                "ets_",
                "tft_",
                "deepvarhierarchical_",
            ]
        )
        and "vae" not in path
        and ("whole" in path or "orig" in path)
    )


def parse_filename(file, dataset):
    flatten_file_name = list(flatten([i.split(".") for i in file.split("_")]))[:-4]
    idx_dataset_name = flatten_file_name.index(dataset)

    if "orig" in file:
        sample = flatten_file_name[-2]
        version = flatten_file_name[-3]
        transformation = "_".join(flatten_file_name[idx_dataset_name + 2 : -3])
    else:
        sample = flatten_file_name[-2]
        version = flatten_file_name[-3]
        transformation = "_".join(flatten_file_name[idx_dataset_name + 2 : -3])

    return transformation, version, sample


def compute_weighted_error(results_dict, transformation, version, sample, err_metric):
    upper_levels = sum(
        err
        for group, err in results_dict[transformation][version][sample][
            err_metric
        ].items()
        if group != "bottom"
    ) / (len(results_dict[transformation][version][sample][err_metric]) - 1)
    bottom_err = results_dict[transformation][version][sample][err_metric].get(
        "bottom", 0
    )

    return (upper_levels + bottom_err) / 2


def compute_aggregated_results_dict(
    algorithm, dataset, path="./results", err_metric="mase"
):
    results_dict = {}

    for file in os.listdir(f"{path}/{algorithm}"):
        full_path = f"{path}/{algorithm}/{file}"

        if not is_valid_file(full_path, algorithm, dataset):
            continue

        with open(full_path, "rb") as handle:
            transformation, version, sample = parse_filename(file, dataset)

            results_dict.setdefault(transformation, {})
            results_dict[transformation].setdefault(version, {})
            results_dict[transformation][version].setdefault(sample, {})

            current_result = pickle.load(handle)

            if err_metric in current_result:
                results_dict[transformation][version][sample][err_metric] = {
                    k: v
                    for k, v in current_result[err_metric].items()
                    if not isinstance(v, np.ndarray)
                }

            results_dict[transformation][version][sample][err_metric][
                "weighted"
            ] = compute_weighted_error(
                results_dict, transformation, version, sample, err_metric
            )

    return results_dict


def compute_aggregated_results_df(results_dict, err_metric="rmse"):
    # shape=(transformation, version, sample, metric, dim)
    # metric = mase, rmse
    # dim = bottom, total, state, gender, legal, all

    df = pd.DataFrame.from_dict(
        {
            (i, j, k, l): results_dict[i][j][k][l]
            for i in results_dict.keys()
            for j in results_dict[i].keys()
            for k in results_dict[i][j].keys()
            for l in results_dict[i][j][k].keys()
        },
        orient="index",
    )

    df_err = df[np.in1d(df.index.get_level_values(3), [err_metric])]
    df_clean = df_err.mean(level=(0, 1, 2)).stack(level=0).to_frame()
    df_clean = df_clean.reset_index()
    df_clean.columns = ["transformation", "version", "sample", "group", err_metric]

    return df_clean


def compute_new_metrics(df_clean, err_metric="rmse"):
    # Calculate percentage change
    df = df_clean.copy()
    s = df.set_index(["transformation", "version", "sample", "group"])[
        err_metric
    ].unstack("version")
    df = df.set_index(["transformation", "version", "sample", "group"])
    df_pct = (
        pd.DataFrame(
            s.div(
                s["orig"]
                .groupby(["transformation", "group"])
                .transform(lambda x: x.fillna(x.sum())),
                axis="rows",
            )
            .sub(1)
            .stack()
        )
        .reset_index()
        .set_index(["transformation", "version", "sample", "group"])
    )
    df_pct = df_pct.rename(columns={0: "pct_change"})
    df_updated = df.merge(df_pct, how="left", left_index=True, right_index=True)

    # Calculate relative change
    df = df_clean.copy()
    s = df.set_index(["transformation", "version", "sample", "group"])[
        err_metric
    ].unstack("version")
    df = df.set_index(["transformation", "version", "sample", "group"])
    df_rel = (
        pd.DataFrame(
            s.sub(
                s["orig"]
                .groupby(["transformation", "group"])
                .transform(lambda x: x.fillna(x.sum())),
                axis="rows",
            ).stack()
        )
        .reset_index()
        .set_index(["transformation", "version", "sample", "group"])
    )
    df_rel = df_rel.rename(columns={0: "rel_change"})
    df_new = df_updated.merge(df_rel, how="left", left_index=True, right_index=True)
    df_new = df_new.reset_index()

    return df_new


def create_dataframes_to_plot(
    dataset: str,
    algorithms: List[str],
    transformations: List[str],
    versions: List[str],
    samples: List[str],
    results_path: str,
    load_transformed: bool,
    err_metric: str = "rmse",
    create_metrics_files: bool = False,
) -> Generator[pd.DataFrame, None, None]:
    """
    Creates and yields dataframes for plotting based on the specified error metric
    for each algorithm in the given dataset. It first calls 'create_metrics_file' to
    prepare metrics files for the algorithms.

    Args:
        dataset (str): The dataset for which to compute results.
        algorithms (List[str]): A list of algorithms to process.
        transformations (List[str]): List of transformations to apply.
        versions (List[str]): List of versions to consider.
        samples (List[str]): List of samples to use.
        results_path (str): Path to store the results.
        err_metric (str, optional): The error metric to use. Defaults to 'rmse'.

    Yields:
        Generator[pd.DataFrame, None, None]: A generator that yields a dataframe for each algorithm.
    """

    if create_metrics_files:
        # Preparing metrics file for each algorithm
        create_metrics_file(
            algorithms=algorithms,
            transformations=transformations,
            versions=versions,
            samples=samples,
            dataset=dataset,
            results_path=results_path,
            load_transformed=load_transformed,
        )

    # Compute and yield dataframes for plotting
    for algorithm in algorithms:
        dict_alg = compute_aggregated_results_dict(
            algorithm=algorithm, dataset=dataset, err_metric=err_metric
        )
        df_alg = compute_aggregated_results_df(dict_alg, err_metric=err_metric)

        # Apply specific transformations if needed based on the algorithm
        if algorithm in ["mint", "ets_bu"]:
            df_alg["group"] = df_alg.group.str.lower()

        df_alg_err = compute_new_metrics(df_alg, err_metric=err_metric)
        df_alg_err["algorithm"] = algorithm
        yield df_alg_err


def plot_results(df, err_metric="mase"):
    _, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax = ax.ravel()
    transf = df.transformation.unique()
    transf.sort()
    for i in range(len(transf)):
        fg = sns.lineplot(
            x="version",
            y=err_metric,
            hue="group",
            marker="o",
            data=df.loc[df["transformation"] == transf[i]].sort_values(
                ["version", "group"]
            ),
            ax=ax[i],
        )
        ax[i].set_ylim(0, 6.0)
        ax[i].set_title(transf[i], fontsize=22)


def plot_results_perc_diff(df):
    _, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax = ax.ravel()
    transf = df.transformation.unique()
    transf.sort()
    for i in range(len(transf)):
        fg = sns.lineplot(
            x="version",
            y="pct_change",
            hue="group",
            marker="o",
            data=df.loc[df["transformation"] == transf[i]].sort_values(
                ["version", "group"]
            ),
            ax=ax[i],
        )
        ax[i].set_ylim(-0.5, 2.5)
        ax[i].set_title(transf[i], fontsize=22)


def set_ylim(err_metric, default_ylims=(-3, 10)):
    ylim_mappings = {
        "mase": (0, 4),
        "rmse": (0, 15000),
        "pct_change": default_ylims,
        "rel_change": (-1, 3.5)
    }
    return ylim_mappings.get(err_metric, default_ylims)


def plot_transformation(ax, df, err_metric, ylim, column, algorithm_name):
    transf = df.transformation.unique()
    transf.sort()
    for i, transformation in enumerate(transf):
        sns.lineplot(
            x="version",
            y=err_metric,
            hue="group",
            marker="o",
            data=df.loc[df["transformation"] == transformation].sort_values(["version", "group"]),
            ax=ax[i, column],
        )
        ax[i, column].set_ylim(*ylim)
        if i == 0:
            # Add bold algorithm name and transformation name to the first row
            title = f"$\\mathbf{{{algorithm_name}}}$\n\n{transformation}"
            ax[i, column].set_title(title, fontsize=12)
        else:
            ax[i, column].set_title(transformation, fontsize=12)
        ax[i, column].axhline(y=0, color="grey", linestyle="dashed")
        ax[i, column].get_legend().remove()


def plot_results_joint(d, dataframes, algorithms, err_metric="mase", default_ylims=(-3, 10)):
    ylim = set_ylim(err_metric, default_ylims)
    n_rows = max(len(df.transformation.unique()) for df in dataframes if df is not None)
    n_cols = len(dataframes)

    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(12, 8))

    for col, df in enumerate(dataframes):
        if df is not None:
            plot_transformation(ax, df, err_metric, ylim, col, algorithms[col])

    plt.setp(ax[n_rows - 1, n_cols - 1].legend(loc="upper right"))
    fig.suptitle(d, fontsize=16)

    plt.tight_layout()


def plot_res(d, ylims, to_show=[False, False, False, False]):
    res = [None, None, None, None]
    df_res = list(create_dataframes_to_plot(d, *to_show))

    i = 0
    for idx, ele in enumerate(to_show):
        if ele:
            res[idx] = df_res[i]
            i += 1

    plot_results_joint(d, *res, err_metric="pct_change", ylims=ylims)
    plt.show()


def get_transformation_type(s):
    if "jitter" in s:
        return "jittering"
    elif "magnitude_warp" in s:
        return "magnitude_warping"
    elif "time_warp" in s:
        return "time_warping"
    elif "scaling" in s:
        return "scaling"
    else:
        return "other"


def transform_dataframe(df):
    df = df.copy()
    df["algorithm"] = df["transformation"].str.split("_").str[0]
    df["transformation"] = df["transformation"].apply(get_transformation_type)
    df["rank"] = df.groupby(["version", "sample", "group", "transformation"])[
        "rmse"
    ].rank(method="min", ascending=True)
    return df


def concatenate_transformed_dfs(df_res, err_metric):
    combined_df = pd.concat(df_res)
    combined_df["transformation"] = combined_df["transformation"].apply(
        get_transformation_type
    )

    # Verify that these columns create unique groupings for your data
    combined_df["rank"] = combined_df.groupby(
        ["version", "sample", "group", "transformation"]
    )[err_metric].rank(method="min", ascending=True)

    return combined_df


def process_datasets(list_of_df_res, dataset_names):
    combined_df_list = []
    for df_res, name in zip(list_of_df_res, dataset_names):
        df_transformed = concatenate_transformed_dfs(df_res)
        df_transformed["dataset"] = name
        combined_df_list.append(df_transformed)
    return pd.concat(combined_df_list)


def plot_radar_datasets(combined_df):
    filtered_df = combined_df[
        (combined_df["sample"] == "s0") & (combined_df["group"] == "weighted")
    ].copy()
    filtered_df["transformation"] = (
        filtered_df["transformation"] + "_" + filtered_df["version"]
    )

    # Compute mean and standard deviation of rank for each algorithm and transformation
    mean_ranks = (
        filtered_df.groupby(["algorithm", "transformation"])["rank"]
        .mean()
        .unstack()
        .sort_index(axis=1, ascending=False)
    )
    std_ranks = (
        filtered_df.groupby(["algorithm", "transformation"])["rank"]
        .std()
        .unstack()
        .sort_index(axis=1, ascending=False)
    )

    # Calculate the number of angles we will plot
    num_vars = len(mean_ranks.columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    colors = sns.color_palette("viridis", len(mean_ranks))

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(polar=True)

    # Define a list of markers
    markers = [
        "o",
        "v",
        "*",
        "+",
        ">",
        "s",
        "p",
        "h",
        "H",
        "+",
        "x",
        "D",
        "d",
        "|",
        "_",
    ]

    num_algorithms = len(mean_ranks)
    jitter = np.linspace(-0.03, 0.03, num_algorithms)

    for idx, (
        (algorithm_mean, mean_rank_row),
        (algorithm_std, std_rank_row),
    ) in enumerate(zip(mean_ranks.iterrows(), std_ranks.iterrows())):
        mean_rank_values = mean_rank_row.values.flatten().tolist()
        std_rank_values = std_rank_row.values.flatten().tolist()
        color = colors[idx]
        marker = markers[idx % len(markers)]

        for trans in range(0, len(mean_rank_values), 6):
            # Plot mean rank and standard deviation line
            ax.plot(
                angles[trans : trans + 6],
                mean_rank_values[trans : trans + 6],
                color=color,
                alpha=1,
                marker=marker,
                label=algorithm_mean if trans == 0 else "",
            )

            for angle, mean_rank, std_rank in zip(
                angles[trans : trans + 6],
                mean_rank_values[trans : trans + 6],
                std_rank_values[trans : trans + 6],
            ):
                jittered_angle = angle + jitter[idx]
                ax.plot(
                    [jittered_angle, jittered_angle],
                    [mean_rank - std_rank, mean_rank + std_rank],
                    color=color,
                    alpha=0.5,
                )

            if trans + 6 < len(mean_rank_values):
                ax.plot(
                    angles[trans + 5 : trans + 7],
                    [mean_rank_values[trans + 5 : trans + 7][0], np.nan],
                    color=ax.get_facecolor(),
                    alpha=1,
                )

    ax.set_thetagrids(np.degrees(angles), mean_ranks.columns)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.1, 1.1))
    ax.set_title(
        "Rank of algorithms for each transformation and version \nof parameters for all datasets",
        size=20,
        color="black",
        y=1.1,
    )

    yticks = [i for i in range(5)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(tick)}" for tick in yticks])

    plt.tight_layout()
    plt.show()


def plot_radar_dataset(combined_df):
    filtered_df = combined_df[
        (combined_df["sample"] == "s0") & (combined_df["group"] == "weighted")
    ].copy()
    filtered_df["transformation"] = (
        filtered_df["transformation"] + "_" + filtered_df["version"]
    )

    grouped = (
        filtered_df.groupby(["algorithm", "transformation"])["rank"]
        .mean()
        .unstack()
        .sort_index(axis=1, ascending=False)
    )

    # Calculate the number of angles we will plot
    num_vars = len(grouped.columns)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    colors = sns.color_palette("viridis", len(grouped))

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(polar=True)

    # Define a list of markers
    markers = [
        "o",
        "v",
        "*",
        "+",
        ">",
        "s",
        "p",
        "h",
        "H",
        "+",
        "x",
        "D",
        "d",
        "|",
        "_",
    ]

    for idx, (i, row) in enumerate(grouped.iterrows()):
        values = row.values.flatten().tolist()
        color = colors[idx]
        marker = markers[
            idx % len(markers)
        ]  # Use modulo operation to cycle through markers

        # Loop through each transformation within the group
        for trans in range(0, len(values), 6):
            # We plot up to the 6th value and then start a new line with a gap
            ax.plot(
                angles[trans : trans + 6],
                values[trans : trans + 6],
                color=color,
                alpha=1,
                marker=marker,
                label=i if trans == 0 else "",
            )
            if trans + 6 < len(values):
                # Plot the gap with the same color as background
                ax.plot(
                    angles[trans + 5 : trans + 7],
                    values[trans + 5 : trans + 7],
                    color=ax.get_facecolor(),
                    alpha=1,
                )

    # Add the labels
    ax.set_thetagrids(np.degrees(angles), grouped.columns)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.1, 1.1))
    ax.set_title(
        "Rank of algorithms for each transformation and version \nof parameters for the Tourism dataset",
        size=20,
        color="black",
        y=1.1,
    )

    yticks = [i for i in range(5)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(tick)}" for tick in yticks])

    plt.tight_layout()
    plt.show()


def plot_radar_transformation_dataset(combined_df):
    filtered_df = combined_df[
        (combined_df["sample"] == "s0") & (combined_df["group"] == "weighted")
    ].copy()
    filtered_df["transformation"] = (
        filtered_df["transformation"] + "_" + filtered_df["version"]
    )

    grouped = (
        filtered_df.groupby(["algorithm", "transformation"])["rank"]
        .mean()
        .unstack()
        .sort_index(axis=1, ascending=False)
    )
    # Calculate the number of angles we will plot
    num_vars = len(grouped.columns)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    colors = sns.color_palette("viridis", len(grouped))

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(polar=True)

    # Define a list of markers
    markers = [
        "o",
        "v",
        "*",
        "+",
        ">",
        "s",
        "p",
        "h",
        "H",
        "+",
        "x",
        "D",
        "d",
        "|",
        "_",
    ]

    for idx, (i, row) in enumerate(grouped.iterrows()):
        values = row.values.flatten().tolist()
        color = colors[idx]
        marker = markers[
            idx % len(markers)
        ]  # Use modulo operation to cycle through markers

        # Loop through each transformation within the group
        for trans in range(0, len(values), 7):
            # We plot up to the 6th value and then start a new line with a gap
            ax.plot(
                angles[trans : trans + 7],
                values[trans : trans + 7],
                color=color,
                alpha=1,
                marker=marker,
                label=i if trans == 0 else "",
            )
            if trans + 7 < len(values):
                # Plot the gap with the same color as background
                ax.plot(
                    angles[trans + 6 : trans + 8],
                    values[trans + 6 : trans + 8],
                    color=ax.get_facecolor(),
                    alpha=1,
                )

    # Add the labels
    ax.set_thetagrids(np.degrees(angles), grouped.columns)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.1, 1.1))
    ax.set_title(
        "Rank of algorithms for each version \nof parameters for the Tourism dataset",
        size=20,
        color="black",
        y=1.1,
    )

    yticks = [i for i in range(5)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(tick)}" for tick in yticks])

    plt.tight_layout()
    plt.show()
