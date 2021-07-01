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


def draw_Recall_from_result(filename, number_of_trees, subsample_size, extensionLevel, dataset_type, algo_type):

    if algo_type == "SIF":
        path = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
            number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
        pd_data = pd.read_excel(path, index_col=0)
        sorting_ascend = False
    else:
        prob_result_path = "./EIF_SIF_Result/chordalysis_log_prob_result/" + filename + "-" + dataset_type + "-" + algo_type + "_result.xlsx"
        pd_prob_result_data = pd.read_excel(prob_result_path)
        dataset_path = "./datasets/" + filename + "_discretized_" + dataset_type + "_withLabel.csv"
        pd_data = pd.read_csv(dataset_path)
        pd_data["score"] = pd_prob_result_data
        sorting_ascend = True


    # print(pd_data.loc[:, "label"])
    pd_data_sorted = pd_data.sort_values(by="score", ascending=sorting_ascend).reset_index(drop=True)
    pd_label = pd_data_sorted.loc[:, "label"]
    # print(pd_label)
    np_label = np.array(pd_label)
    cumulative_recall = [0]
    cumulative_value = 0
    count_of_anomaly = pd_data_sorted[pd_data_sorted["label"] == 1].shape[0]
    # print(count_of_anomaly)
    # print(filename)
    for label in np_label:
        cumulative_value = cumulative_value + label
        cumulative_recall.append(cumulative_value / count_of_anomaly)

    return cumulative_recall


# dataset_names = ["annthyroid", "cardio", "foresttype", "ionosphere","mammography" ,"satellite", "shuttle", "thyroid"]
# dataset_names = ["annthyroid", "cardio",  "ionosphere","mammography" ,"satellite", "shuttle", "thyroid"]
# dataset_names = ["smtp","satimage-2","pendigits","speech"]
# dataset_names = ["annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid","smtp","satimage-2","pendigits","speech"]
dataset_names = ["foresttype","http"]

dataset_types = ["10BIN", "15BIN"]
algo_types = ["SIF", "log_pseudolikelihood", "ordered_log_prob"]
# parameter for traing the forest
number_of_trees = 500
subsample_size = 256
extensionLevel = 0

for dataset_name in dataset_names:
    fig, ax = plt.subplots()
    for dataset_type in dataset_types:
        for algo_type in algo_types:
            cumulative_recall = draw_Recall_from_result(filename=dataset_name, number_of_trees=number_of_trees,
                                                        subsample_size=subsample_size,
                                                        extensionLevel=extensionLevel,
                                                        dataset_type=dataset_type, algo_type=algo_type)
            if dataset_type == "10BIN":
                plot_color = "brown"
            elif dataset_type == "15BIN":
                plot_color = "darkblue"
            else:
                # origin
                plot_color = "black"

            if algo_type == "log_pseudolikelihood":
                plot_linestyle = "--"
                algo_type_on_plot = "Chordalysis_Pseudolikelihood"
            elif algo_type == "ordered_log_prob":
                plot_linestyle = "dotted"
                algo_type_on_plot = "Chordalysis_Joint_Probability"
            else:
                plot_linestyle = "-"
                algo_type_on_plot = "SIF"

            ax.plot(np.arange(1, np.array(cumulative_recall).shape[0] + 1), cumulative_recall,
                    label= dataset_type+ "-" + algo_type_on_plot, color=plot_color, linestyle=plot_linestyle)

    ax.set_xlabel('Rank')
    ax.set_ylabel('Recall')
    ax.set_title("Cumulative Recall Curve")
    ax.legend()
    if extensionLevel == "full":
        extensionLevel_str = extensionLevel
    else:
        extensionLevel_str = str(extensionLevel)
    plot_path = "./plots/" + dataset_name + "-Chordalysis_Cumulative_Recall_Curve-" + str(number_of_trees) + "-" + str(
        subsample_size) + "-" + extensionLevel_str + ".jpg"

    fig.savefig(plot_path)
    plt.close(fig)
