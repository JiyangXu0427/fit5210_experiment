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

def draw_PR_from_EIF_result(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path =  './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    y_test = np.array(pd_data.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data.loc[:, "score"])

    precision, recall, thresholds_pr = skm.precision_recall_curve(y_test, y_score, pos_label=1)
    PRC_AUC = skm.auc(recall, precision)
    print("PRC_AUC_Score: " + str(PRC_AUC))


    AP = skm.average_precision_score(y_test, y_score)
    print("Average_Precision_Score: " + str(AP))


    return PRC_AUC, AP, precision, recall


def draw_PR_from_SIF_result(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    y_test = np.array(pd_data.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data.loc[:, "score"])
    precision, recall, thresholds_pr = skm.precision_recall_curve(y_test, y_score, pos_label=1)
    PRC_AUC = skm.auc(recall, precision)
    print("PRC_AUC_Score: " + str(PRC_AUC))


    AP = skm.average_precision_score(y_test, y_score)
    print("Average_Precision_Score: " + str(AP))


    return PRC_AUC, AP, precision, recall

def draw_PR_ensemble_by_average_score(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
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
    precision, recall, thresholds_pr = skm.precision_recall_curve(y_test, y_score, pos_label=1)
    PRC_AUC = skm.auc(recall, precision)
    print("PRC_AUC_Score: " + str(PRC_AUC))


    AP = skm.average_precision_score(y_test, y_score)
    print("Average_Precision_Score: " + str(AP))


    return PRC_AUC, AP, precision, recall

def draw_PR_from_Chordalysis_result(filename, dataset_type, algo_type):
    prob_result_path = "./EIF_SIF_Result/chordalysis_log_prob_result/" + filename + "-" + dataset_type + "-" + algo_type + "_result.xlsx"
    pd_prob_result_data = pd.read_excel(prob_result_path)
    dataset_path = "./datasets/" + filename + "_discretized_" + dataset_type + "_withLabel.csv"
    pd_data = pd.read_csv(dataset_path)
    pd_data["score"] = abs(pd_prob_result_data)
    y_test = np.array(pd_data["label"])
    y_score = np.array(pd_data["score"])

    precision, recall, thresholds_pr = skm.precision_recall_curve(y_test, y_score, pos_label=1)
    PRC_AUC = skm.auc(recall, precision)
    print("PRC_AUC_Score: " + str(PRC_AUC))


    AP = skm.average_precision_score(y_test, y_score)
    print("Average_Precision_Score: " + str(AP))


    return PRC_AUC, AP, precision, recall

dataset_names = ["annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid", "smtp",
                 "satimage-2", "pendigits", "speech"]

dataset_types = ["origin", "copula_0.0625", "copula_0.25", "copula_1", "copula_4", "copula_16", "10BIN", "15BIN"]
# parameter for traing the forest
number_of_trees = 500
subsample_size = 256
extensionLevel = "full"

dataset_name_list = []
dataset_type_list = []
algo_list = []
PR_AUC_list = []
AP_list = []

for dataset_name in dataset_names:
    for dataset_type in dataset_types:
        PRC_AUC_e, AP_e, precision_e, recall_e = draw_PR_from_EIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                            subsample_size=subsample_size,
                                            extensionLevel=extensionLevel,
                                            dataset_type=dataset_type)
        PRC_AUC_s, AP_s, precision_s, recall_s = draw_PR_from_SIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                            subsample_size=subsample_size,
                                            extensionLevel=0,
                                            dataset_type=dataset_type)
        PRC_AUC_ensemble, AP_ensemble, precision_ensemble, recall_ensemble = draw_PR_ensemble_by_average_score(filename=dataset_name,
                                                                         number_of_trees=number_of_trees,
                                                                         subsample_size=subsample_size,
                                                                         extensionLevel=extensionLevel,
                                                                         dataset_type=dataset_type)

        dataset_name_list.append(dataset_name)
        dataset_type_list.append(dataset_type)
        algo_list.append("EIF")
        PR_AUC_list.append(PRC_AUC_e)
        AP_list.append(AP_e)

        dataset_name_list.append(dataset_name)
        dataset_type_list.append(dataset_type)
        algo_list.append("SIF")
        PR_AUC_list.append(PRC_AUC_s)
        AP_list.append(AP_s)

        dataset_name_list.append(dataset_name)
        dataset_type_list.append(dataset_type)
        algo_list.append("SIF_EIF_ensemble_score")
        PR_AUC_list.append(PRC_AUC_ensemble)
        AP_list.append(AP_ensemble)


        if dataset_type in ["10BIN", "15BIN"]:
            PRC_AUC_pse, AP_pse, precision_pse, recall_pse = draw_PR_from_Chordalysis_result(filename=dataset_name,
                                                                             dataset_type=dataset_type, algo_type = "log_pseudolikelihood")

            dataset_name_list.append(dataset_name)
            dataset_type_list.append(dataset_type)
            algo_list.append("log_pseudolikelihood")
            PR_AUC_list.append(PRC_AUC_pse)
            AP_list.append(AP_pse)

            PRC_AUC_joint, AP_joint, precision_joint, recall_joint  = draw_PR_from_Chordalysis_result(filename=dataset_name,
                                                                       dataset_type=dataset_type,
                                                                       algo_type="ordered_log_prob")

            dataset_name_list.append(dataset_name)
            dataset_type_list.append(dataset_type)
            algo_list.append("ordered_log_prob")
            PR_AUC_list.append(PRC_AUC_joint)
            AP_list.append(AP_joint)


pd_result = pd.DataFrame({"dataset_name": dataset_name_list, "data_type": dataset_type_list, "algo_name": algo_list,
                          "PR_AUC": PR_AUC_list,"AP":AP_list})
pd_result.to_excel("./EIF_SIF_Result/EIF_SIF_ensemble_Chordalysis_PR_AUC.xlsx")
