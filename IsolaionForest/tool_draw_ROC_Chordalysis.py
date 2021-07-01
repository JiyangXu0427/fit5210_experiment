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


def draw_ROC(filename, number_of_trees, subsample_size, extensionLevel, dataset_type, algo_type):
    if algo_type == "SIF":
        path = './EIF_SIF_Result/SIF_Result/' + filename + "_SIF_Result_Data_" + dataset_type + "-" + str(
            number_of_trees) + "-" + str(subsample_size) + "-" + str(0) + ".xlsx"
        pd_data = pd.read_excel(path, index_col=0)

    else:
        prob_result_path = "./EIF_SIF_Result/chordalysis_log_prob_result/" + filename + "-" + dataset_type + "-" + algo_type + "_result.xlsx"
        pd_prob_result_data = pd.read_excel(prob_result_path)
        dataset_path = "./datasets/" + filename + "_discretized_" + dataset_type + "_withLabel.csv"
        pd_data = pd.read_csv(dataset_path)
        pd_data["score"] = abs(pd_prob_result_data)

    y_test = np.array(pd_data.loc[:, "label"], dtype="int32")
    y_score = np.array(pd_data.loc[:, "score"])

    # auc_roc_score = skm.roc_auc_score(y_test, y_score)
    # print("ROC_AUC_Score: " + str(auc_roc_score))

    fpr_s, tpr_s, thresholds_s = skm.roc_curve(y_test, y_score, pos_label=1)
    return fpr_s, tpr_s
    # return fpr_s, tpr_s, auc_roc_score


# dataset_names = ["annthyroid", "cardio", "foresttype", "ionosphere", "mammography", "satellite", "shuttle", "thyroid"]
# dataset_names = ["annthyroid", "cardio",  "ionosphere","mammography" ,"satellite", "shuttle", "thyroid"]
# dataset_names = ["annthyroid", "cardio", "ionosphere", "mammography", "satellite", "shuttle", "thyroid","smtp","satimage-2","pendigits","speech"]
dataset_names = ["foresttype","http"]
# dataset_names = ["smtp","satimage-2","pendigits","speech"]
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
            fpr_s, tpr_s = draw_ROC(filename=dataset_name,
                                    number_of_trees=number_of_trees,
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



            ax.plot(fpr_s, tpr_s, label=dataset_type + "-" + algo_type_on_plot, color=plot_color, linestyle=plot_linestyle)
            # result_summary_path = "./plots/" + dataset_name + "_Chordalysis_Result_Summary.txt"
            # with open(result_summary_path, 'a') as opened_file:
            #     opened_file.write("Data_Type: " + dataset_type)
            #     opened_file.write("\n")
            #     opened_file.write("Algo_type: " + algo_type)
            #     opened_file.write("\n")
            #     opened_file.write("ROC_AUC_Score: " + str(auc_roc_score))
            #     opened_file.write("\n")
            #     opened_file.write("\n")


    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title("ROC Curve")
    ax.legend()
    if extensionLevel == "full":
        extensionLevel_str = extensionLevel
    else:
        extensionLevel_str = str(extensionLevel)
    plot_path = "./plots/" + dataset_name + "-Chordalysis_ROC_Curve-" + str(number_of_trees) + "-" + str(
        subsample_size) + "-" + extensionLevel_str + ".jpg"
    fig.savefig(plot_path)
    plt.close(fig)
