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



filenames = ["annthyroid", "cardio", "foresttype", "http","ionosphere","mammography" ,"mnist", "optdigits","pendigits","satellite","satimage-2", "shuttle","smtp_v7","speech", "thyroid"]

result_summary_path = "./datasets/Feature_Value_Counts_Summary.txt"
with open(result_summary_path, 'a') as opened_file:
    for filename in filenames:
        opened_file.write(filename)
        opened_file.write("\n")
        data = sio.loadmat('./datasets/' + filename + '.mat')
        x_data = data["X"]
        for i in range(x_data.shape[1]):
            column_values = x_data[:,i]
            column_values_count = pd.value_counts(column_values).sort_index()
            num_of_rows = column_values_count.shape[0]
            opened_file.write("V" + str(i) + ": " + str(num_of_rows))
            opened_file.write("\n")
        opened_file.write("--------------------------------")
        opened_file.write("\n")
