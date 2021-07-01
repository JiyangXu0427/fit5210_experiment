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





def ensemble_by_average_rank(filename, number_of_trees, subsample_size, extensionLevel):

    path_eif = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + "origin" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data_eif_org = pd.read_excel(path_eif, index_col=0)

    path_eif = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + "copula_16" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data_eif_copula_16 = pd.read_excel(path_eif, index_col=0)

    path_eif = './EIF_SIF_Result/EIF_Result/' + filename + "_EIF_Result_Data_" + "15BIN" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(extensionLevel) + ".xlsx"
    pd_data_eif_15BIN = pd.read_excel(path_eif, index_col=0)

    path_sif = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + "origin" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data_sif_org = pd.read_excel(path_sif, index_col=0)

    path_sif = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + "copula_16" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data_sif_copula_16 = pd.read_excel(path_sif, index_col=0)

    path_sif = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + "15BIN" + "-" + str(
        number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
    pd_data_sif_15BIN = pd.read_excel(path_sif, index_col=0)

    #
    # prob_result_path = "./EIF_SIF_Result/chordalysis_log_prob_result/" + filename + "-" + "15BIN" + "-" + "ordered_log_prob" + "_result.xlsx"
    # pd_prob_result_data = pd.read_excel(prob_result_path)
    # dataset_path = "./datasets/" + filename + "_discretized_" + "15BIN" + "_withLabel.csv"
    # pd_data_chordalysis = pd.read_csv(dataset_path)
    # pd_data_chordalysis["score"] = abs(pd_prob_result_data)

    pd_data_eif_org["Data_Index"] = np.arange(pd_data_eif_org.shape[0])
    pd_data_eif_copula_16["Data_Index"] = np.arange(pd_data_eif_copula_16.shape[0])
    pd_data_eif_15BIN["Data_Index"] = np.arange(pd_data_eif_15BIN.shape[0])

    pd_data_sif_org["Data_Index"] = np.arange(pd_data_sif_org.shape[0])
    pd_data_sif_copula_16["Data_Index"] = np.arange(pd_data_sif_copula_16.shape[0])
    pd_data_sif_15BIN["Data_Index"] = np.arange(pd_data_sif_15BIN.shape[0])

    # pd_data_chordalysis["Data_Index"] = np.arange(pd_data_chordalysis.shape[0])
###################

    #ranking by score, the larger score on top, the more likely to be anomaly
    pd_data_eif_org = pd_data_eif_org.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_eif_copula_16 = pd_data_eif_copula_16.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_eif_15BIN = pd_data_eif_15BIN.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_sif_org = pd_data_sif_org.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_sif_copula_16 = pd_data_sif_copula_16.sort_values(by="score", ascending=False).reset_index(drop=True)
    pd_data_sif_15BIN = pd_data_sif_15BIN.sort_values(by="score", ascending=False).reset_index(drop=True)
    # pd_data_chordalysis = pd_data_chordalysis.sort_values(by="score", ascending=False).reset_index(drop=True)

    #data with a smaller ranking is more likely to be anomaly now (1,2,3..) is more likely than (100,101,102..)
    pd_data_eif_org["Rank"] = np.arange(start=1, stop=pd_data_eif_org.shape[0] + 1, step=1)
    pd_data_eif_copula_16["Rank"] = np.arange(start=1, stop=pd_data_eif_copula_16.shape[0] + 1, step=1)
    pd_data_eif_15BIN["Rank"] = np.arange(start=1, stop=pd_data_eif_15BIN.shape[0] + 1, step=1)
    pd_data_sif_org["Rank"] = np.arange(start=1, stop=pd_data_sif_org.shape[0] + 1, step=1)
    pd_data_sif_copula_16["Rank"] = np.arange(start=1, stop=pd_data_sif_copula_16.shape[0] + 1, step=1)
    pd_data_sif_15BIN["Rank"] = np.arange(start=1, stop=pd_data_sif_15BIN.shape[0] + 1, step=1)
    # pd_data_chordalysis["Rank"] = np.arange(start=1, stop=pd_data_chordalysis.shape[0] + 1, step=1)

    average_rank_list = []

    for i in range(pd_data_eif_org.shape[0]):
        eif_rank = pd_data_eif_org.at[i, "Rank"]
        eif_index = pd_data_eif_org.at[i, "Data_Index"]

        eif_row_copula_16 = pd_data_eif_copula_16[pd_data_eif_copula_16["Data_Index"] == eif_index]
        eif_row_15bin = pd_data_eif_15BIN[pd_data_eif_15BIN["Data_Index"] == eif_index]
        sif_row_org = pd_data_sif_org[pd_data_sif_org["Data_Index"] == eif_index]
        sif_row_copula_16 = pd_data_sif_copula_16[pd_data_sif_copula_16["Data_Index"] == eif_index]
        sif_row_15bin = pd_data_sif_15BIN[pd_data_sif_15BIN["Data_Index"] == eif_index]
        # chordalysis_row = pd_data_chordalysis[pd_data_chordalysis["Data_Index"] == eif_index]

        eif_row_copula_16_rank = np.array(eif_row_copula_16.loc[:, "Rank"])[0]
        eif_row_15bin_rank = np.array(eif_row_15bin.loc[:, "Rank"])[0]
        sif_row_org_rank = np.array(sif_row_org.loc[:, "Rank"])[0]
        sif_row_copula_16_rank = np.array(sif_row_copula_16.loc[:, "Rank"])[0]
        sif_row_15bin_rank = np.array(sif_row_15bin.loc[:, "Rank"])[0]
        # chordalysis_row_rank = np.array(chordalysis_row.loc[:, "Rank"])[0]

        #average rank using negative number, so that data with lower rank (1,2,3..) will have larger negative ranking (-1,-2,-3...)
        #So that, when calculating AUC score, which is decreasing the threshold in descending order, the anomaly will be first detected
        average_rank = - statistics.mean(
            [eif_rank, eif_row_copula_16_rank, eif_row_15bin_rank, sif_row_org_rank, sif_row_copula_16_rank,
             sif_row_15bin_rank])

        # average_rank = - statistics.mean([eif_rank,eif_row_copula_16_rank,eif_row_15bin_rank,sif_row_org_rank,sif_row_copula_16_rank,sif_row_15bin_rank,chordalysis_row_rank])
        average_rank_list.append(average_rank)

    pd_data_eif_org["Average_Rank"] = average_rank_list
    #Note, here the negative average rank is sorted ascending, so the normaly data, which has smaller negative ranking will be on top!
    pd_data_sorted_eif_by_average_rank = pd_data_eif_org.sort_values(by="Average_Rank", ascending=True).reset_index(
        drop=True)


    pd_data_sorted_eif_by_average_rank.to_excel('./EIF_SIF_Result/ensemble_result/'+ filename +"_Ensemble_result_data.xlsx")





dataset_names = ["foresttype","http","annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid", "smtp",
                 "satimage-2", "pendigits"]
# parameter for traing the forest
number_of_trees = 500
subsample_size = 256
extensionLevel = "full"


for dataset_name in dataset_names:
   ensemble_by_average_rank(filename=dataset_name, number_of_trees=number_of_trees,
                            subsample_size=subsample_size,
                            extensionLevel=extensionLevel)


