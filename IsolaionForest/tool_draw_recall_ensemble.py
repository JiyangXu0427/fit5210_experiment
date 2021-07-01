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


def draw_Recall_from_EIF_SIF_ensemble_by_average_rank(filename, number_of_trees, subsample_size, extensionLevel,
                                                      dataset_type):
    path_eif = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data_eif = pd.read_excel(path_eif, index_col=0)
    pd_data_eif["Data_Index"] = np.arange(pd_data_eif.shape[0])

    path_sif = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data_sif = pd.read_excel(path_sif, index_col=0)
    pd_data_sif["Data_Index"] = np.arange(pd_data_sif.shape[0])

    pd_data_sorted_eif = pd_data_eif.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_sorted_sif = pd_data_sif.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_sorted_eif["Rank"] = np.arange(start=1, stop=pd_data_eif.shape[0] + 1, step=1)
    pd_data_sorted_sif["Rank"] = np.arange(start=1, stop=pd_data_sif.shape[0] + 1, step=1)
    average_rank_list = []
    for i in range(pd_data_sorted_eif.shape[0]):
        eif_rank = pd_data_sorted_eif.at[i, "Rank"]
        eif_index = pd_data_sorted_eif.at[i, "Data_Index"]

        sif_row = pd_data_sorted_sif[pd_data_sorted_sif["Data_Index"] == eif_index]
        sif_row_Rank = np.array(sif_row.loc[:, "Rank"])[0]
        average_rank = (eif_rank + sif_row_Rank) / 2
        average_rank_list.append(average_rank)
        # for j in range(pd_data_sorted_sif.shape[0]):
        #     sif_index = pd_data_sorted_sif.at[j, "Data_Index"]
        #     if eif_index == sif_index:
        #         sif_rank = pd_data_sorted_sif.at[j, "Rank"]
        #         average_rank = (eif_rank + sif_rank) / 2
        #         break


    pd_data_sorted_eif["Average_Rank"] = average_rank_list
    pd_data_sorted_eif_by_average_rank = pd_data_sorted_eif.sort_values(by="Average_Rank", ascending=True).reset_index(
        drop=True)
    pd_label = pd_data_sorted_eif_by_average_rank.loc[:, "label"]

    # print(pd_label)
    np_label = np.array(pd_label)
    cumulative_recall = [0]
    cumulative_value = 0
    count_of_anomaly = pd_data_sorted_eif_by_average_rank[pd_data_sorted_eif_by_average_rank["label"] == 1].shape[0]
    for label in np_label:
        cumulative_value = cumulative_value + label
        cumulative_recall.append(cumulative_value / count_of_anomaly)

    return cumulative_recall


def draw_Recall_from_EIF_SIF_ensemble_by_average_score(filename, number_of_trees, subsample_size, extensionLevel,
                                                       dataset_type):
    path_eif = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data_eif = pd.read_excel(path_eif, index_col=0)
    pd_data_eif["Data_Index"] = np.arange(pd_data_eif.shape[0])

    path_sif = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data_sif = pd.read_excel(path_sif, index_col=0)
    pd_data_sif["Data_Index"] = np.arange(pd_data_sif.shape[0])

    pd_data_sorted_eif = pd_data_eif.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_sorted_sif = pd_data_sif.sort_values(by="score", ascending=False).reset_index(drop=True)

    average_score_list = []
    for i in range(pd_data_sorted_eif.shape[0]):
        eif_score = pd_data_sorted_eif.at[i, "score"]
        eif_index = pd_data_sorted_eif.at[i, "Data_Index"]

        sif_row = pd_data_sorted_sif[pd_data_sorted_sif["Data_Index"] == eif_index]
        sif_row_score = np.array(sif_row.loc[:, "score"])[0]
        average_score = (eif_score + sif_row_score) / 2
        average_score_list.append(average_score)

        # for j in range(pd_data_sorted_sif.shape[0]):
        #     sif_index = pd_data_sorted_sif.at[j, "Data_Index"]
        #     if eif_index == sif_index:
        #         sif_score = pd_data_sorted_sif.at[j, "score"]
        #         average_score = (eif_score + sif_score) / 2
        #         break
        # average_score_list.append(average_score)

    pd_data_sorted_eif["Average_Score"] = average_score_list
    pd_data_sorted_eif_by_average_score = pd_data_sorted_eif.sort_values(by="Average_Score",
                                                                         ascending=False).reset_index(drop=True)
    pd_label = pd_data_sorted_eif_by_average_score.loc[:, "label"]

    # print(pd_label)
    np_label = np.array(pd_label)
    cumulative_recall = [0]
    cumulative_value = 0
    count_of_anomaly = pd_data_sorted_eif_by_average_score[pd_data_sorted_eif_by_average_score["label"] == 1].shape[0]
    for label in np_label:
        cumulative_value = cumulative_value + label
        cumulative_recall.append(cumulative_value / count_of_anomaly)

    return cumulative_recall

def draw_roc_ensemble_more_by_average_rank(filename):

    path_eif = './EIF_SIF_Result/ensemble_result/'+ filename +"_Ensemble_result_data.xlsx"
    pd_data = pd.read_excel(path_eif, index_col=0)

    #raw data is sorted by negative ranking ascendingly, so the normal data, which has smaller negative ranking, was on top.
    #here we sort it descendingly, so that the anomaly, which has larger negative ranking will be on top
    pd_data_sorted = pd_data.sort_values(by="Average_Rank", ascending=False).reset_index(drop=True)
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


# dataset_names = ["annthyroid", "cardio", "foresttype", "ionosphere", "mammography", "satellite", "shuttle", "thyroid"]
# dataset_names = ["annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid","smtp","satimage-2","pendigits","speech"]
# dataset_names = ["smtp","satimage-2","pendigits","speech"]
dataset_names = ["foresttype","http","annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid","smtp","satimage-2","pendigits"]

dataset_types = ["origin", "copula_0.0625", "copula_0.25", "copula_1", "copula_4", "copula_16", "10BIN", "15BIN"]
# parameter for traing the forest
number_of_trees = 500
subsample_size = 256
extensionLevel = "full"

for dataset_name in dataset_names:
    print(dataset_name)
    # fig, (ax_e, ax_s) = plt.subplots(ncols=2,nrows=1)
    # fig, ((ax_e, ax_s), (ax_ensemble_rank, ax_ensemble_score)) = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
    fig, (ax_e, ax_s, ax_ensemble_score) = plt.subplots(ncols=3, nrows=1, figsize=(18, 8))

    cumulative_recall_ensemble_rank = draw_roc_ensemble_more_by_average_rank(filename=dataset_name)
    ax_ensemble_score.plot(np.arange(1, np.array(cumulative_recall_ensemble_rank).shape[0] + 1), cumulative_recall_ensemble_rank,  label="ensemble", color="yellow",
                           linestyle="-")


    for dataset_type in dataset_types:
        cumulative_recall_e = draw_Recall_from_EIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                                          subsample_size=subsample_size,
                                                          extensionLevel=extensionLevel,
                                                          dataset_type=dataset_type)
        print(1)
        cumulative_recall_s = draw_Recall_from_SIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                                          subsample_size=subsample_size,
                                                          extensionLevel=0,
                                                          dataset_type=dataset_type)
        print(2)
        # cumulative_recall_ensemble_rank = draw_Recall_from_EIF_SIF_ensemble_by_average_rank(filename=dataset_name,
        #                                                                                     number_of_trees=number_of_trees,
        #                                                                                     subsample_size=subsample_size,
        #                                                                                     extensionLevel=extensionLevel,
        #                                                                                     dataset_type=dataset_type)
        # print(3)
        # cumulative_recall_ensemble_score = draw_Recall_from_EIF_SIF_ensemble_by_average_score(filename=dataset_name,
        #                                                                                     number_of_trees=number_of_trees,
        #                                                                                     subsample_size=subsample_size,
        #                                                                                     extensionLevel=extensionLevel,
        #                                                                                     dataset_type=dataset_type)
        # print(4)
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

        ax_e.plot(np.arange(1, np.array(cumulative_recall_e).shape[0] + 1), cumulative_recall_e, label=dataset_type,
                  color=plot_color, linestyle=plot_linestyle)

        ax_s.plot(np.arange(1, np.array(cumulative_recall_s).shape[0] + 1), cumulative_recall_s, label=dataset_type,
                  color=plot_color, linestyle=plot_linestyle)

        # ax_ensemble_rank.plot(np.arange(1, np.array(cumulative_recall_ensemble_rank).shape[0] + 1), cumulative_recall_ensemble_rank,
        #                       label=dataset_type,
        #                       color=plot_color, linestyle=plot_linestyle)
        # ax_ensemble_score.plot(np.arange(1, np.array(cumulative_recall_ensemble_rank).shape[0] + 1), cumulative_recall_ensemble_rank,
        #                       label=dataset_type,
        #                       color=plot_color, linestyle=plot_linestyle)
        # ax_ensemble_score.plot(np.arange(1, np.array(cumulative_recall_ensemble_score).shape[0] + 1), cumulative_recall_ensemble_score,
        #                       label=dataset_type,
        #                       color=plot_color, linestyle=plot_linestyle)

    ax_e.set_xlabel('Rank')
    ax_e.set_ylabel('Recall')
    ax_e.set_title("EIF Cumulative Recall Curve")
    ax_e.legend()
    ax_s.set_xlabel('Rank')
    ax_s.set_title("SIF Cumulative Recall Curve")
    ax_s.legend()
    # ax_ensemble_rank.set_xlabel('Rank')
    # ax_ensemble_rank.set_ylabel('Recall')
    # ax_ensemble_rank.set_title("SIF_EIF_Ensemble_by_Rank Cumulative Recall Curve")
    # ax_ensemble_rank.legend()
    ax_ensemble_score.set_xlabel('Rank')
    ax_ensemble_score.set_title("Ensemble Cumulative Recall Curve")
    ax_ensemble_score.legend()

    if extensionLevel == "full":
        extensionLevel_str = extensionLevel
    else:
        extensionLevel_str = str(extensionLevel)

    plot_path = "./EIF_SIF_Result/" + dataset_name + "-EIF_SIF_Ensemble_Cumulative_Recall_Curve-" + str(number_of_trees) + "-" + str(
        subsample_size) + "-" + extensionLevel_str + ".jpg"
    fig.savefig(plot_path)
    plt.tight_layout
    plt.close(fig)
