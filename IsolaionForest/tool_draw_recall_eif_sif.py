import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
import scipy.stats as sts
import pandas
import pandas as pd
import eif_old as eif_old_class
import scipy.io as sio
import os
import sklearn.metrics as skm


def draw_Recall_from_EIF_result(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    # print(pd_data.loc[:, "label"])
    pd_data_sorted = pd_data.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_label = pd_data_sorted.loc[:, "label"]
    # print(pd_label)
    np_label = np.array(pd_label)
    cumulative_recall = [0]
    cumulative_value = 0
    count_of_anomaly = pd_data_sorted[pd_data_sorted["label"] == 1].shape[0]
    for label in np_label:
        cumulative_value = cumulative_value + label
        cumulative_recall.append(cumulative_value / count_of_anomaly)

    return cumulative_recall

def draw_Recall_from_SIF_result(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    # print(pd_data.loc[:, "label"])
    pd_data_sorted = pd_data.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_label = pd_data_sorted.loc[:, "label"]
    # print(pd_label)
    np_label = np.array(pd_label)
    cumulative_recall = [0]
    cumulative_value = 0
    count_of_anomaly = pd_data_sorted[pd_data_sorted["label"] == 1].shape[0]
    for label in np_label:
        cumulative_value = cumulative_value + label
        cumulative_recall.append(cumulative_value / count_of_anomaly)

    return cumulative_recall


# dataset_names = ["annthyroid", "cardio", "foresttype", "ionosphere","mammography" ,"satellite", "shuttle", "thyroid"]
# dataset_names = ["annthyroid", "cardio",  "ionosphere","mammography" ,"satellite", "shuttle", "thyroid"]
dataset_names = ["smtp","satimage-2","pendigits","speech"]
dataset_types = ["origin", "copula_0.0625", "copula_0.25", "copula_1", "copula_4", "copula_16", "10BIN", "15BIN"]
# parameter for traing the forest
number_of_trees = 500
subsample_size = 256
extensionLevel = "full"

for dataset_name in dataset_names:
    fig_e, ax_e = plt.subplots()
    fig_s, ax_s = plt.subplots()
    for dataset_type in dataset_types:
        cumulative_recall_e = draw_Recall_from_EIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                                        subsample_size=subsample_size,
                                                        extensionLevel=extensionLevel,
                                                        dataset_type=dataset_type)
        cumulative_recall_s = draw_Recall_from_SIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                                        subsample_size=subsample_size,
                                                        extensionLevel=0,
                                                        dataset_type=dataset_type)

        if dataset_type == "10BIN":
            plot_color = "brown"
            plot_linestyle = "--"
        elif dataset_type == "15BIN":
            plot_color = "darkblue"
            plot_linestyle = "--"
        elif dataset_type == "copula_0.0625":
            plot_color = "brown"
            plot_linestyle = "dotted"
        elif dataset_type == "copula_0.25":
            plot_color = "royalblue"
            plot_linestyle = "dotted"
        elif dataset_type == "copula_1":
            plot_color = "yellow"
            plot_linestyle = "dotted"
        elif dataset_type == "copula_4":
            plot_color = "royalblue"
            plot_linestyle = "dashdot"
        elif dataset_type == "copula_16":
            plot_color = "brown"
            plot_linestyle = "dashdot"
        else:
            # origin
            plot_color = "black"
            plot_linestyle = "-"

        ax_e.plot(np.arange(1, np.array(cumulative_recall_e).shape[0] + 1), cumulative_recall_e, label=dataset_type, color=plot_color, linestyle=plot_linestyle)

        ax_s.plot(np.arange(1, np.array(cumulative_recall_s).shape[0] + 1), cumulative_recall_s, label=dataset_type, color=plot_color, linestyle=plot_linestyle)

    ax_e.set_xlabel('Rank')
    ax_e.set_ylabel('Recall')
    ax_e.set_title("Cumulative Recall Curve")
    ax_e.legend()
    ax_s.set_xlabel('Rank')
    ax_s.set_ylabel('Recall')
    ax_s.set_title("Cumulative Recall Curve")
    ax_s.legend()

    if extensionLevel == "full":
        extensionLevel_str = extensionLevel
    else:
        extensionLevel_str = str(extensionLevel)

    plot_path_e = "./plots/" + dataset_name + "-EIF_Cumulative_Recall_Curve-" + str(number_of_trees) + "-" + str(
        subsample_size) + "-" + extensionLevel + ".jpg"
    plot_path_s = "./plots/" + dataset_name + "-SIF_Cumulative_Recall_Curve-" + str(number_of_trees) + "-" + str(
        subsample_size) + "-" + str(0) + ".jpg"

    fig_e.savefig(plot_path_e)
    fig_s.savefig(plot_path_s)
    plt.close(fig_s)
    plt.close(fig_e)


