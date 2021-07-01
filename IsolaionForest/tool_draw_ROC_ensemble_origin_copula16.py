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

def draw_roc_ensemble_by_average_score(filename, number_of_trees, subsample_size, extensionLevel):
    path_eif_origin = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + "origin" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    path_eif_copula_16 = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + "copula_16" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data_eif_origin = pd.read_excel(path_eif_origin, index_col=0)
    pd_data_eif_copula_16 = pd.read_excel(path_eif_copula_16, index_col=0)

    path_sif_origin = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + "origin" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    path_sif_copula_16 = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + "copula_16" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data_sif_origin = pd.read_excel(path_sif_origin, index_col=0)
    pd_data_sif_copula_16 = pd.read_excel(path_sif_copula_16, index_col=0)

    pd_data_eif_origin["Data_Index"] = np.arange(pd_data_eif_origin.shape[0])
    pd_data_eif_copula_16["Data_Index"] = np.arange(pd_data_eif_copula_16.shape[0])
    pd_data_sif_origin["Data_Index"] = np.arange(pd_data_sif_origin.shape[0])
    pd_data_sif_copula_16["Data_Index"] = np.arange(pd_data_sif_copula_16.shape[0])

    average_score_list = []
    for i in range(pd_data_eif_origin.shape[0]):
        eif_origin_score = pd_data_eif_origin.at[i, "score"]
        eif_origin_index = pd_data_eif_origin.at[i, "Data_Index"]

        eif_copula_16_row = pd_data_eif_copula_16[pd_data_eif_copula_16["Data_Index"] == eif_origin_index]
        eif_copula_16_score = np.array(eif_copula_16_row.loc[:, "score"])[0]

        sif_origin_row = pd_data_sif_origin[pd_data_sif_origin["Data_Index"] == eif_origin_index]
        sif_origin_score = np.array(sif_origin_row.loc[:, "score"])[0]

        sif_copula_16_row = pd_data_sif_copula_16[pd_data_sif_copula_16["Data_Index"] == eif_origin_index]
        sif_copula_16_score = np.array(sif_copula_16_row.loc[:, "score"])[0]


        average_score = (eif_origin_score + eif_copula_16_score + sif_origin_score + sif_copula_16_score) / 4
        average_score_list.append(average_score)

    pd_data_eif_origin["Average_Score"] = average_score_list
    pd_data_eif_origin = pd_data_eif_origin.sort_values(by="Average_Score", ascending=False).reset_index(drop=True)
    y_test = np.array(pd_data_eif_origin.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data_eif_origin.loc[:, "Average_Score"])
    fpr, tpr, thresholds = skm.roc_curve(y_test, y_score, pos_label=1)
    AUC_ROC_SCORE = skm.auc(fpr, tpr)

    return fpr, tpr, AUC_ROC_SCORE


# dataset_names = ["annthyroid", "cardio", "foresttype", "ionosphere", "mammography", "satellite", "shuttle", "thyroid"]
dataset_names = ["annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid","smtp","satimage-2","pendigits","speech"]

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
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))

    fpr_e_origin, tpr_e_origin, AUC_ROC_e_origin = draw_roc_from_EIF_result(filename=dataset_name,
                                                                            number_of_trees=number_of_trees,
                                                                            subsample_size=subsample_size,
                                                                            extensionLevel=extensionLevel,
                                                                            dataset_type="origin")

    fpr_e_copula_16, tpr_e_copula_16, AUC_ROC_e_copula_16 = draw_roc_from_EIF_result(filename=dataset_name,
                                                                                     number_of_trees=number_of_trees,
                                                                                     subsample_size=subsample_size,
                                                                                     extensionLevel=extensionLevel,
                                                                                     dataset_type="copula_16")

    fpr_s_origin, tpr_s_origin, AUC_ROC_s_origin = draw_roc_from_SIF_result(filename=dataset_name,
                                                                            number_of_trees=number_of_trees,
                                                                            subsample_size=subsample_size,
                                                                            extensionLevel=0,
                                                                            dataset_type="origin")

    fpr_s_copula_16, tpr_s_copula_16, AUC_ROC_s_copula_16 = draw_roc_from_SIF_result(filename=dataset_name,
                                                                                     number_of_trees=number_of_trees,
                                                                                     subsample_size=subsample_size,
                                                                                     extensionLevel=0,
                                                                                     dataset_type="copula_16")

    fpr_ensemble_score, tpr_ensemble_score, AUC_ROC_SCORE_ensemble_score = draw_roc_ensemble_by_average_score(
        filename=dataset_name, number_of_trees=number_of_trees, subsample_size=subsample_size,
        extensionLevel=extensionLevel)

    dataset_name_list.append(dataset_name)
    dataset_type_list.append("origin")
    algo_list.append("EIF")
    ROC_AUC_list.append(AUC_ROC_e_origin)

    dataset_name_list.append(dataset_name)
    dataset_type_list.append("copula_16")
    algo_list.append("EIF")
    ROC_AUC_list.append(AUC_ROC_e_copula_16)

    dataset_name_list.append(dataset_name)
    dataset_type_list.append("origin")
    algo_list.append("SIF")
    ROC_AUC_list.append(AUC_ROC_s_origin)

    dataset_name_list.append(dataset_name)
    dataset_type_list.append("copula_16")
    algo_list.append("SIF")
    ROC_AUC_list.append(AUC_ROC_s_copula_16)

    dataset_name_list.append(dataset_name)
    dataset_type_list.append("origin+copulat_16")
    algo_list.append("EIF+SIF")
    ROC_AUC_list.append(AUC_ROC_SCORE_ensemble_score)

    ax.plot(fpr_e_origin, tpr_e_origin, label="EIF_Origin", color="darkblue", linestyle="dotted")
    ax.plot(fpr_e_copula_16, tpr_e_copula_16, label="EIF_Copula16", color="darkblue", linestyle="dashdot")
    ax.plot(fpr_s_origin, tpr_s_origin, label="SIF_Origin", color="brown", linestyle="dotted")
    ax.plot(fpr_s_copula_16, tpr_s_copula_16, label="SIF_Copula16", color="brown",linestyle="dashdot")
    ax.plot(fpr_ensemble_score, tpr_ensemble_score, label="EIF_SIF_Origin_Copula16_ensemble", color="yellow",linestyle="-")

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title("EIF_SIF_Origin_Copula16_Ensemble ROC Curve")
    ax.legend()

    if extensionLevel == "full":
        extensionLevel_str = extensionLevel
    else:
        extensionLevel_str = str(extensionLevel)

    plot_path = "./EIF_SIF_Result/" + dataset_name + "-EIF_SIF_origin_Copula16_Ensemble_ROC_Curve-" + str(number_of_trees) + "-" + str(
        subsample_size) + "-" + extensionLevel_str + ".jpg"
    fig.savefig(plot_path)
    plt.tight_layout
    plt.close(fig)

pd_result = pd.DataFrame({ "dataset_name": dataset_name_list, "data_type": dataset_type_list, "algo_name": algo_list, "ROC_AUC": ROC_AUC_list})
pd_result.to_excel("./EIF_SIF_Result/EIF_SIF_origin_Copula16_ensemble_ROC_AUC.xlsx")
