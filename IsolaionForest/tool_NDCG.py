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
import statistics


def calculate_NDCG_sif(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    extensionLevel = 0
    path = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    pd_data_sorted_by_score = pd_data.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_sorted_by_label = pd_data.sort_values(by="label", ascending=False).reset_index(drop=True)
    dcg = calculate_dcg(pd_data_sorted_by_score)
    idcg = calculate_dcg(pd_data_sorted_by_label)
    ndcg = dcg / idcg
    return ndcg, dcg, idcg


def calculate_NDCG_eif(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    path = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data = pd.read_excel(path, index_col=0)
    pd_data_sorted_by_score = pd_data.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_sorted_by_label = pd_data.sort_values(by="label", ascending=False).reset_index(drop=True)
    dcg = calculate_dcg(pd_data_sorted_by_score)
    idcg = calculate_dcg(pd_data_sorted_by_label)
    ndcg = dcg / idcg
    return ndcg, dcg, idcg

def calculate_NDCG_chordalysis_likelihood(filename, dataset_type, algo_type):
    prob_result_path = "./EIF_SIF_Result/chordalysis_log_prob_result/" + filename + "-" + dataset_type + "-" + algo_type + "_result.xlsx"
    pd_prob_result_data = pd.read_excel(prob_result_path)
    dataset_path = "./datasets/" + filename + "_discretized_" + dataset_type + "_withLabel.csv"
    pd_data = pd.read_csv(dataset_path)
    pd_data["score"] = abs(pd_prob_result_data)
    pd_data_sorted_by_score = pd_data.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_sorted_by_label = pd_data.sort_values(by="label", ascending=False).reset_index(drop=True)

    dcg = calculate_dcg(pd_data_sorted_by_score)
    idcg = calculate_dcg(pd_data_sorted_by_label)
    ndcg = dcg / idcg

    return ndcg, dcg, idcg

def calculate_NDCG_eif_sif_ensemble_by_rank(filename):

    path_eif = './EIF_SIF_Result/ensemble_result/'+ filename +"_Ensemble_result_data.xlsx"
    pd_data = pd.read_excel(path_eif, index_col=0)

    #raw data is sorted by negative ranking ascendingly, so the normal data, which has smaller negative ranking, was on top.
    #here we sort it descendingly, so that the anomaly, which has larger negative ranking will be on top
    pd_data_sorted_eif_by_average_rank = pd_data.sort_values(by="Average_Rank", ascending=False).reset_index(
        drop=True)

    pd_data_sorted_by_label = pd_data.sort_values(by="label", ascending=False).reset_index(drop=True)

    dcg = calculate_dcg(pd_data_sorted_eif_by_average_rank)
    idcg = calculate_dcg(pd_data_sorted_by_label)
    ndcg = dcg / idcg
    print(ndcg)
    return ndcg, dcg, idcg

def calculate_dcg(pd_data_sorted):
    dcg = 0
    # print(pd_data_sorted.iloc[:, -5:-1])
    for i in range(pd_data_sorted.shape[0]):
        rank = i + 1
        relevance_value = pd_data_sorted.at[i, "label"]
        gain = relevance_value / math.log2(rank+1)
        dcg = dcg + gain
    return dcg


dataset_names = ["foresttype","http","annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid", "smtp",
                 "satimage-2", "pendigits"]

dataset_types = ["origin", "copula_0.0625", "copula_0.25", "copula_1", "copula_4", "copula_16", "10BIN", "15BIN"]
algo_types = ["log_pseudolikelihood", "ordered_log_prob"]

# parameter for traing the forest
number_of_trees = 500
subsample_size = 256
extensionLevel = "full"

dataset_name_list = []
dataset_type_list = []
algo_list = []
ndcg_list = []
dcg_list = []
idcg_list = []


for dataset_name in dataset_names:
    ndcg_rank, dcg_rank, idcg_rank = calculate_NDCG_eif_sif_ensemble_by_rank(filename=dataset_name)

    dataset_name_list.append(dataset_name)
    dataset_type_list.append("ensemble")
    ndcg_list.append(ndcg_rank)
    dcg_list.append(dcg_rank)
    idcg_list.append(idcg_rank)
    algo_list.append("ensemble")

    for dataset_type in dataset_types:
        ndcg_e, dcg_e, idcg_e = calculate_NDCG_eif(filename=dataset_name, number_of_trees=number_of_trees, subsample_size=subsample_size, extensionLevel=extensionLevel, dataset_type=dataset_type)
        dataset_name_list.append(dataset_name)
        dataset_type_list.append(dataset_type)
        ndcg_list.append(ndcg_e)
        dcg_list.append(dcg_e)
        idcg_list.append(idcg_e)
        algo_list.append("EIF")

        ndcg_s, dcg_s, idcg_s = calculate_NDCG_sif(filename=dataset_name, number_of_trees=number_of_trees,
                                                   subsample_size=subsample_size, extensionLevel=0,
                                                   dataset_type=dataset_type)
        dataset_name_list.append(dataset_name)
        dataset_type_list.append(dataset_type)
        ndcg_list.append(ndcg_s)
        dcg_list.append(dcg_s)
        idcg_list.append(idcg_s)
        algo_list.append("SIF")

        if dataset_type in ["10BIN", "15BIN"]:
            ndcg_pseu, dcg_pseu, idcg_pseu = calculate_NDCG_chordalysis_likelihood(filename=dataset_name,
                                                       dataset_type=dataset_type, algo_type = "log_pseudolikelihood")

            dataset_name_list.append(dataset_name)
            dataset_type_list.append(dataset_type)
            ndcg_list.append(ndcg_pseu)
            dcg_list.append(dcg_pseu)
            idcg_list.append(idcg_pseu)
            algo_list.append("log_pseudolikelihood")

            ndcg_log, dcg_log, idcg_log = calculate_NDCG_chordalysis_likelihood(filename=dataset_name,
                                                                                   dataset_type=dataset_type,
                                                                                   algo_type="ordered_log_prob")

            dataset_name_list.append(dataset_name)
            dataset_type_list.append(dataset_type)
            ndcg_list.append(ndcg_log)
            dcg_list.append(dcg_log)
            idcg_list.append(idcg_log)
            algo_list.append("ordered_log_prob")


pd_NDCF_result = pd.DataFrame({"dataset_name": dataset_name_list, "data_type": dataset_type_list, "algo_name": algo_list, "NDCG": ndcg_list, "DCG": dcg_list, "IDCG": idcg_list})
pd_NDCF_result.to_excel("./EIF_SIF_Result/NDCG_EIF_SIF_Chordalysis_Ensemble_more.xlsx")
