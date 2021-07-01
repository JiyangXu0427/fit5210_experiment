import statistics

import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
import scipy.stats as sts
import pandas
import pandas as pd
import eif_old as eif_old_class
import eif as eif_new
import scipy.io as sio
import os
import sklearn.metrics as skm
from sklearn.ensemble import IsolationForest
import math


def calculate_PrecisionAtK_sif(filename, number_of_trees, subsample_size, extensionLevel, dataset_type, k):
    extensionLevel = 0
    path = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    pd_data_sorted_by_score = pd_data.sort_values(by="score", ascending=False).reset_index(drop=True)

    if k >= pd_data_sorted_by_score.shape[0]:
        k = pd_data_sorted_by_score.shape[0]

    if k < 1 and k > 0:
        k = int(pd_data_sorted_by_score.shape[0] * k)
        confusion = pd_data_sorted_by_score.loc[0:k - 1, "label"]
    else:
        confusion = pd_data_sorted_by_score.loc[0:k - 1, "label"]

    TP = confusion[confusion == 1].shape[0]
    FP = confusion[confusion == 0].shape[0]

    precision_at_k = 0

    if TP != 0 or FP != 0:
        precision_at_k = TP / (TP + FP)

    return precision_at_k


def calculate_PrecisionAtK_eif(filename, number_of_trees, subsample_size, extensionLevel, dataset_type, k):
    path = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    pd_data_sorted_by_score = pd_data.sort_values(by="score", ascending=False).reset_index(drop=True)

    if k >= pd_data_sorted_by_score.shape[0]:
        k = pd_data_sorted_by_score.shape[0]

    if k < 1 and k > 0:
        k = int(pd_data_sorted_by_score.shape[0] * k)
        confusion = pd_data_sorted_by_score.loc[0:k - 1, "label"]
    else:
        confusion = pd_data_sorted_by_score.loc[0:k - 1, "label"]

    TP = confusion[confusion == 1].shape[0]
    FP = confusion[confusion == 0].shape[0]

    precision_at_k = 0

    if TP != 0 or FP != 0:
        precision_at_k = TP / (TP + FP)

    return precision_at_k



def calculate_PrecisionAtK_chord(filename, dataset_type, algo_type,k):
    prob_result_path = "./EIF_SIF_Result/chordalysis_log_prob_result/" + filename + "-" + dataset_type + "-" + algo_type + "_result.xlsx"
    pd_prob_result_data = pd.read_excel(prob_result_path)
    dataset_path = "./datasets/" + filename + "_discretized_" + dataset_type + "_withLabel.csv"
    pd_data = pd.read_csv(dataset_path)
    pd_data["score"] = abs(pd_prob_result_data)

    pd_data_sorted_by_score = pd_data.sort_values(by="score", ascending=False).reset_index(drop=True)

    if k >= pd_data_sorted_by_score.shape[0]:
        k = pd_data_sorted_by_score.shape[0]

    if k < 1 and k > 0:
        k = int(pd_data_sorted_by_score.shape[0] * k)
        confusion = pd_data_sorted_by_score.loc[0:k - 1, "label"]
    else:
        confusion = pd_data_sorted_by_score.loc[0:k - 1, "label"]

    TP = confusion[confusion == 1].shape[0]
    FP = confusion[confusion == 0].shape[0]


    precision_at_k = 0

    if TP != 0 or FP != 0:
        precision_at_k = TP / (TP + FP)

    return precision_at_k


def calculate_PrecisionAtK_eif_sif_ensemble_rank(filename,  k):

    path_eif = './EIF_SIF_Result/ensemble_result/' + filename + "_Ensemble_result_data.xlsx"
    pd_data = pd.read_excel(path_eif, index_col=0)

    # raw data is sorted by negative ranking ascendingly, so the normal data, which has smaller negative ranking, was on top.
    # here we sort it descendingly, so that the anomaly, which has larger negative ranking will be on top
    pd_data_sorted_eif_by_average_rank = pd_data.sort_values(by="Average_Rank", ascending=False).reset_index(
        drop=True)


    if k >= pd_data_sorted_eif_by_average_rank.shape[0]:
        k = pd_data_sorted_eif_by_average_rank.shape[0]

    if k < 1 and k > 0:
        k = int(pd_data_sorted_eif_by_average_rank.shape[0] * k)
        confusion = pd_data_sorted_eif_by_average_rank.loc[0:k - 1, "label"]
    else:
        confusion = pd_data_sorted_eif_by_average_rank.loc[0:k - 1, "label"]

    TP = confusion[confusion == 1].shape[0]
    FP = confusion[confusion == 0].shape[0]

    precision_at_k = 0

    if TP != 0 or FP != 0:
        precision_at_k = TP / (TP + FP)

    return precision_at_k


dataset_names = ["foresttype","http","annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid", "smtp",
                 "satimage-2", "pendigits"]

dataset_types = ["origin", "copula_0.0625", "copula_0.25", "copula_1", "copula_4", "copula_16", "10BIN", "15BIN"]
algo_types = ["log_pseudolikelihood", "ordered_log_prob"]

# parameter for traing the forest
number_of_trees = 500
subsample_size = 256
extensionLevel = "full"
# k_list = [0.1, 100, 500, 1000]
k_list = [0.1]
dataset_name_list = []
dataset_type_list = []
precisio_list = []
algo_list = []
k_para_list = []

for dataset_name in dataset_names:
    for k in k_list:
        precision_rank = calculate_PrecisionAtK_eif_sif_ensemble_rank(filename=dataset_name,
                                                                      k=k)
        dataset_name_list.append(dataset_name)
        dataset_type_list.append("ensemble")
        algo_list.append("ensemble")
        precisio_list.append(precision_rank)
        k_para_list.append(k)

        for dataset_type in dataset_types:

                precision_e = calculate_PrecisionAtK_eif(filename=dataset_name, number_of_trees=number_of_trees,
                                                         subsample_size=subsample_size, extensionLevel=extensionLevel,
                                                         dataset_type=dataset_type, k=k)
                dataset_name_list.append(dataset_name)
                dataset_type_list.append(dataset_type)
                algo_list.append("EIF")
                precisio_list.append(precision_e)
                k_para_list.append(k)

                precision_s = calculate_PrecisionAtK_sif(filename=dataset_name, number_of_trees=number_of_trees,
                                                         subsample_size=subsample_size, extensionLevel=0,
                                                         dataset_type=dataset_type, k=k)
                dataset_name_list.append(dataset_name)
                dataset_type_list.append(dataset_type)
                algo_list.append("SIF")
                precisio_list.append(precision_s)
                k_para_list.append(k)

                if dataset_type in ["10BIN", "15BIN"]:
                    precision_pse = calculate_PrecisionAtK_chord(filename=dataset_name, dataset_type=dataset_type, algo_type = "log_pseudolikelihood", k=k)

                    dataset_name_list.append(dataset_name)
                    dataset_type_list.append(dataset_type)
                    algo_list.append("log_pseudolikelihood")
                    precisio_list.append(precision_pse)
                    k_para_list.append(k)

                    precision_chord = calculate_PrecisionAtK_chord(filename=dataset_name,
                                                                   dataset_type=dataset_type,
                                                                   algo_type="ordered_log_prob", k=k)

                    dataset_name_list.append(dataset_name)
                    dataset_type_list.append(dataset_type)
                    algo_list.append("ordered_log_prob")
                    precisio_list.append(precision_chord)
                    k_para_list.append(k)


pd_eif_result = pd.DataFrame(
    {"dataset_name": dataset_name_list, "data_type": dataset_type_list, "algo_name": algo_list, "K": k_para_list,
     "Precision": precisio_list})
pd_eif_result.to_excel("./EIF_SIF_Result/Precision_at_K_IR_EIF_SIF_ensemble_more.xlsx")
