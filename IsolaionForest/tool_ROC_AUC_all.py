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
import statistics

def draw_roc_from_EIF_result(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path =  './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    y_test = np.array(pd_data.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data.loc[:, "score"])
    fpr_e, tpr_e, thresholds_s = skm.roc_curve(y_test, y_score, pos_label=1)
    AUC_ROC_SCORE = skm.auc(fpr_e, tpr_e)
    AUC_ROC_SCORE_2 = skm.roc_auc_score(y_test,y_score)
    print(AUC_ROC_SCORE)
    print(AUC_ROC_SCORE_2)

    return AUC_ROC_SCORE

def draw_roc_from_SIF_result(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    y_test = np.array(pd_data.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data.loc[:, "score"])
    fpr_s, tpr_s, thresholds_s = skm.roc_curve(y_test, y_score, pos_label=1)
    AUC_ROC_SCORE = skm.auc(fpr_s, tpr_s)

    return AUC_ROC_SCORE

def draw_roc_from_Chordalysis_result(filename, dataset_type, algo_type):
    prob_result_path = "./EIF_SIF_Result/chordalysis_log_prob_result/" + filename + "-" + dataset_type + "-" + algo_type + "_result.xlsx"
    pd_prob_result_data = pd.read_excel(prob_result_path)
    dataset_path = "./datasets/" + filename + "_discretized_" + dataset_type + "_withLabel.csv"
    pd_data = pd.read_csv(dataset_path)
    pd_data["score"] = abs(pd_prob_result_data)
    y_test = np.array(pd_data["label"])
    y_score = np.array(pd_data["score"])

    AUC_ROC_SCORE2 = skm.roc_auc_score(y_test,y_score)

    return AUC_ROC_SCORE2


def draw_roc_ensemble_by_average_rank(filename):

    path_eif = './EIF_SIF_Result/ensemble_result/'+ filename +"_Ensemble_result_data.xlsx"
    pd_data = pd.read_excel(path_eif, index_col=0)


    y_test = np.array(pd_data.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data.loc[:, "Average_Rank"])

    fpr, tpr, thresholds = skm.roc_curve(y_test, y_score, pos_label=1)
    AUC_ROC_SCORE = skm.auc(fpr, tpr)
    print("ensemble AUC" + str(AUC_ROC_SCORE))

    return AUC_ROC_SCORE



dataset_names = ["foresttype","http","annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid", "smtp",
                 "satimage-2", "pendigits"]

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
    AUC_ROC_SCORE_ensemble_score = draw_roc_ensemble_by_average_rank(filename=dataset_name)
    dataset_name_list.append(dataset_name)
    dataset_type_list.append("ensemble")
    algo_list.append("ensemble")
    ROC_AUC_list.append(AUC_ROC_SCORE_ensemble_score)

    for dataset_type in dataset_types:
        AUC_ROC_e = draw_roc_from_EIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                             subsample_size=subsample_size,
                                             extensionLevel=extensionLevel,
                                             dataset_type=dataset_type)
        AUC_ROC_s = draw_roc_from_SIF_result(filename=dataset_name, number_of_trees=number_of_trees,
                                             subsample_size=subsample_size,
                                             extensionLevel=0,
                                             dataset_type=dataset_type)

        dataset_name_list.append(dataset_name)
        dataset_type_list.append(dataset_type)
        algo_list.append("EIF")
        ROC_AUC_list.append(AUC_ROC_e)

        dataset_name_list.append(dataset_name)
        dataset_type_list.append(dataset_type)
        algo_list.append("SIF")
        ROC_AUC_list.append(AUC_ROC_s)

        if dataset_type in ["10BIN", "15BIN"]:
            AUC_ROC_SCORE_Pseudolikelihood = draw_roc_from_Chordalysis_result(filename=dataset_name,
                                                                        dataset_type=dataset_type, algo_type = "log_pseudolikelihood")

            dataset_name_list.append(dataset_name)
            dataset_type_list.append(dataset_type)
            algo_list.append("log_pseudolikelihood")
            ROC_AUC_list.append(AUC_ROC_SCORE_Pseudolikelihood)

            AUC_ROC_SCORE_likelihood = draw_roc_from_Chordalysis_result(filename=dataset_name,
                                                                              dataset_type=dataset_type,
                                                                              algo_type="ordered_log_prob")

            dataset_name_list.append(dataset_name)
            dataset_type_list.append(dataset_type)
            algo_list.append("ordered_log_prob")
            ROC_AUC_list.append(AUC_ROC_SCORE_likelihood)


pd_result = pd.DataFrame({"dataset_name": dataset_name_list, "data_type": dataset_type_list, "algo_name": algo_list,
                          "ROC_AUC": ROC_AUC_list})
pd_result.to_excel("./EIF_SIF_Result/ROC_AUC_EIF_SIF_Chordalysis_ensemble_more.xlsx")
