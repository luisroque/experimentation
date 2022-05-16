from ..compute_results.compute_res_funcs import (
    compute_aggreated_results_dict,
    agg_res_bottom_series,
)
import seaborn as sns
import matplotlib.pyplot as plt


def plot_compare_err_metric(err="mase", dataset="prison"):
    dict_gpf = compute_aggreated_results_dict(
        algorithm="gpf", dataset=dataset, err_metric=err
    )
    df_gpf_bottom = agg_res_bottom_series(dict_gpf)
    dict_mint = compute_aggreated_results_dict(
        algorithm="mint", dataset=dataset, err_metric=err
    )
    df_mint_bottom = agg_res_bottom_series(dict_mint)
    dict_deepar = compute_aggreated_results_dict(
        algorithm="deepar", dataset=dataset, err_metric=err
    )
    df_deepar_bottom = agg_res_bottom_series(dict_deepar)
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax = ax.ravel()
    sns.barplot(x="value", y="group", data=df_gpf_bottom, color="blue", ax=ax[0])
    ax[0].set_title("gpf")
    sns.barplot(x="value", y="group", data=df_mint_bottom, color="darkorange", ax=ax[1])
    ax[1].set_title("mint")
    sns.barplot(x="value", y="group", data=df_deepar_bottom, color="green", ax=ax[2])
    ax[2].set_title("deepar")
    fig.suptitle(dataset + " - " + err)


def boxplot_error(df_res, err, datasets):
    _, ax = plt.subplots(len(datasets), 1, figsize=(20, 10 * len(datasets)))
    for i in range(len(datasets)):
        fg = sns.boxplot(x="group", y="value", hue="algorithm", data=df_res[i], ax=ax)
        ax.set_title(datasets[i] + " - " + err, fontsize=20)
        plt.legend()
        plt.show()
