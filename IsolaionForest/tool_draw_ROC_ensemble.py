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

def draw_roc_from_EIF_result(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path =  './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    y_test = np.array(pd_data.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data.loc[:, "score"])
    fpr_e, tpr_e, thresholds_s = skm.roc_curve(y_test, y_score, pos_label=1)
    AUC_ROC_SCORE = skm.auc(fpr_e, tpr_e)

    return fpr_e, tpr_e,AUC_ROC_SCORE


def draw_roc_from_SIF_result(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    y_test = np.array(pd_data.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data.loc[:, "score"])
    fpr_s, tpr_s, thresholds_s = skm.roc_curve(y_test, y_score, pos_label=1)
    AUC_ROC_SCORE = skm.auc(fpr_s, tpr_s)

    return fpr_s, tpr_s,AUC_ROC_SCORE

#
# def draw_roc_ensemble_by_average_rank(filename, number_of_trees, subsample_size, extensionLevel,
#                                                       dataset_type):
#
#     path_eif = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
#         number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
#     pd_data_eif = pd.read_excel(path_eif, index_col=0)
#
#     path_sif = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
#         number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
#     pd_data_sif = pd.read_excel(path_sif, index_col=0)
#
#     pd_data_eif["Data_Index"] = np.arange(pd_data_eif.shape[0])
#     pd_data_sif["Data_Index"] = np.arange(pd_data_sif.shape[0])
#
#     pd_data_sorted_eif = pd_data_eif.sort_values(by="score", ascending=False).reset_index(drop=True)
#     pd_data_sorted_sif = pd_data_sif.sort_values(by="score", ascending=False).reset_index(drop=True)
#
#     pd_data_sorted_eif["Rank"] = np.arange(start=1, stop=pd_data_eif.shape[0] + 1, step=1)
#     pd_data_sorted_sif["Rank"] = np.arange(start=1, stop=pd_data_sif.shape[0] + 1, step=1)
#
#     average_rank_list = []
#     for i in range(pd_data_sorted_eif.shape[0]):
#         eif_rank = pd_data_sorted_eif.at[i, "Rank"]
#         eif_index = pd_data_sorted_eif.at[i, "Data_Index"]
#
#         sif_row = pd_data_sorted_sif[pd_data_sorted_sif["Data_Index"] == eif_index]
#         sif_row_Rank = np.array(sif_row.loc[:, "Rank"])[0]
#         average_rank = (eif_rank + sif_row_Rank) / 2
#         average_rank_list.append(average_rank)
#
#     pd_data_sorted_eif["Average_Rank"] = average_rank_list
#     pd_data_sorted_eif_by_average_rank = pd_data_sorted_eif.sort_values(by="Average_Rank", ascending=True).reset_index(
#         drop=True)
#     confusion = pd_data_sorted_eif_by_average_rank.loc[:, "Confusion_Matrix"]
#
#     TP = confusion[confusion == "TP"].shape[0]
#     FP = confusion[confusion == "FP"].shape[0]
#     TN = confusion[confusion == "TN"].shape[0]
#     FN = confusion[confusion == "FN"].shape[0]
#     TP_cum = 0
#     FP_cum = 0
#     tpr = []
#     fpr = []
#
#     for i in range(pd_data_sorted_eif_by_average_rank.shape[0]):
#         confusion_value = pd_data_sorted_eif.at[i, "Confusion_Matrix"]
#         if confusion_value == "TP":
#             TP_cum = TP_cum + 1
#             tpr.append(TP_cum/(TP+FN))
#             fpr.append(FP_cum/(FP+FN))
#         elif confusion_value == "FP":
#             FP_cum = FP_cum + 1
#             tpr.append(TP_cum / (TP + FN))
#             fpr.append(FP_cum / (FP + FN))
#         else:
#             tpr.append(TP_cum / (TP + FN))
#             fpr.append(FP_cum / (FP + FN))
#
#     AUC_ROC_SCORE = skm.auc(fpr,tpr)
#
#     return fpr,tpr,AUC_ROC_SCORE


def draw_roc_ensemble_by_average_score(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path_eif = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data_eif = pd.read_excel(path_eif, index_col=0)

    path_sif = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data_sif = pd.read_excel(path_sif, index_col=0)

    pd_data_eif["Data_Index"] = np.arange(pd_data_eif.shape[0])
    pd_data_sif["Data_Index"] = np.arange(pd_data_sif.shape[0])

    average_score_list = []
    for i in range(pd_data_eif.shape[0]):
        eif_score = pd_data_eif.at[i, "score"]
        eif_index = pd_data_eif.at[i, "Data_Index"]

        sif_row = pd_data_sif[pd_data_sif["Data_Index"] == eif_index]
        sif_row_score = np.array(sif_row.loc[:, "score"])[0]
        average_score = (eif_score + sif_row_score) / 2
        average_score_list.append(average_score)

    pd_data_eif["Average_Score"] = average_score_list
    pd_data_eif = pd_data_eif.sort_values(by="Average_Score",ascending=False).reset_index(drop=True)
    y_test = np.array(pd_data_eif.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data_eif.loc[:, "Average_Score"])
    fpr, tpr, thresholds = skm.roc_curve(y_test, y_score, pos_label=1)
    AUC_ROC_SCORE = skm.auc(fpr, tpr)

    return fpr, tpr,AUC_ROC_SCORE

def draw_roc_ensemble_more_by_average_rank(filename):

    path_eif = './EIF_SIF_Result/ensemble_result/'+ filename +"_Ensemble_result_data.xlsx"
    pd_data = pd.read_excel(path_eif, index_col=0)

    y_test = np.array(pd_data.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data.loc[:, "Average_Rank"])

    fpr, tpr, thresholds = skm.roc_curve(y_test, y_score, pos_label=1)
    AUC_ROC_SCORE = skm.auc(fpr, tpr)
    print("ensemble AUC" + str(AUC_ROC_SCORE))

    return fpr, tpr, AUC_ROC_SCORE


# dataset_names = ["annthyroid", "cardio", "foresttype", "ionosphere", "mammography", "satellite", "shuttle", "thyroid"]
dataset_names = ["foresttype","http","annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid","smtp","satimage-2","pendigits","speech"]

dataset_types = ["origin", "copula_0.0625", "copula_0.25", "copula_1", "copula_4", "copula_16", "10BIN", "15BIN"]
# parameter for traing the forest
number_of_trees = 500
subsample_size = 256
extensionLevel = "full"

dataset_name_list = []
dataset_type_list = []
algo_list = []
ROC_AUC_list = []

for dataset_name in dataset_names:
    # fig, (ax_e, ax_s) = plt.subplots(ncols=2,nrows=1)
    # fig, ((ax_e, ax_s), (ax_ensemble_rank, ax_ensemble_score)) = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
    fig, (ax_e, ax_s, ax_ensemble_score) = plt.subplots(ncols=3, nrows=1, figsize=(18, 8))

    fpr_ensemble_score, tpr_ensemble_score, AUC_ROC_SCORE_ensemble = draw_roc_ensemble_more_by_average_rank(filename=dataset_name)
    ax_ensemble_score.plot(fpr_ensemble_score, tpr_ensemble_score, label="ensemble", color="yellow",
                           linestyle="-")

    for dataset_type in dataset_types:
        fpr_e, tpr_e ,AUC_ROC_e = draw_roc_from_EIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                                subsample_size=subsample_size,
                                                extensionLevel=extensionLevel,
                                                dataset_type=dataset_type)
        fpr_s, tpr_s ,AUC_ROC_s = draw_roc_from_SIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                                subsample_size=subsample_size,
                                                extensionLevel=0,
                                                dataset_type=dataset_type)

        # fpr_ensemble_rank, tpr_ensemble_rank,AUC_ROC_SCORE_ensemble_rank = draw_roc_ensemble_by_average_rank(filename=dataset_name, number_of_trees=number_of_trees,
        #                                         subsample_size=subsample_size,
        #                                         extensionLevel=extensionLevel,
        #                                         dataset_type=dataset_type)

        # fpr_ensemble_score, tpr_ensemble_score,AUC_ROC_SCORE_ensemble_score = draw_roc_ensemble_by_average_score(filename=dataset_name, number_of_trees=number_of_trees,
        #                                         subsample_size=subsample_size,
        #                                         extensionLevel=extensionLevel,
        #                                         dataset_type=dataset_type)

        # dataset_name_list.append(dataset_name)
        # dataset_type_list.append(dataset_type)
        # algo_list.append("EIF")
        # ROC_AUC_list.append(AUC_ROC_e)
        #
        # dataset_name_list.append(dataset_name)
        # dataset_type_list.append(dataset_type)
        # algo_list.append("SIF")
        # ROC_AUC_list.append(AUC_ROC_s)

        # dataset_name_list.append(dataset_name)
        # dataset_type_list.append(dataset_type)
        # algo_list.append("SIF_EIF_ensemble_rank")
        # ROC_AUC_list.append(AUC_ROC_SCORE_ensemble_rank)

        # dataset_name_list.append(dataset_name)
        # dataset_type_list.append(dataset_type)
        # algo_list.append("SIF_EIF_ensemble_score")
        # ROC_AUC_list.append(AUC_ROC_SCORE_ensemble_score)

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

        ax_e.plot(fpr_e, tpr_e, label=dataset_type, color=plot_color, linestyle=plot_linestyle)
        ax_s.plot(fpr_s, tpr_s, label=dataset_type, color=plot_color, linestyle=plot_linestyle)
        # ax_ensemble_rank.plot(fpr_ensemble_rank, tpr_ensemble_rank, label=dataset_type, color=plot_color, linestyle=plot_linestyle)

    ax_e.set_xlabel('FPR')
    ax_e.set_ylabel('TPR')
    ax_e.set_title("EIF ROC Curve")
    ax_e.legend()

    ax_s.set_xlabel('FPR')
    # ax_s.set_ylabel('TPR')
    ax_s.set_title("SIF ROC Curve")
    ax_s.legend()

    # ax_ensemble_rank.set_xlabel('FPR')
    # ax_ensemble_rank.set_ylabel('TPR')
    # ax_ensemble_rank.set_title("SIF_EIF_ensemble_rank ROC Curve")
    # ax_ensemble_rank.legend()

    ax_ensemble_score.set_xlabel('FPR')
    # ax_s.set_ylabel('TPR')
    ax_ensemble_score.set_title("Ensemble ROC Curve")
    ax_ensemble_score.legend()

    if extensionLevel == "full":
        extensionLevel_str = extensionLevel
    else:
        extensionLevel_str = str(extensionLevel)

    plot_path = "./EIF_SIF_Result/" + dataset_name + "-EIF-SIF-Ensemble_ROC_Curve-" + str(number_of_trees) + "-" + str(
        subsample_size) + "-" + extensionLevel_str + ".jpg"
    fig.savefig(plot_path)
    plt.tight_layout
    plt.close(fig)

# pd_result = pd.DataFrame({ "dataset_name": dataset_name_list, "data_type": dataset_type_list, "algo_name": algo_list, "ROC_AUC": ROC_AUC_list})
# pd_result.to_excel("./EIF_SIF_Result/EIF_SIF_ensemble_ROC_AUC.xlsx")
