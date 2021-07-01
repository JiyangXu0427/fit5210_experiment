import numpy as np
import scipy.stats as sts
import pandas
import pandas as pd
import eif_old as eif_old_class
import scipy.io as sio
import os
import sklearn.metrics as skm
from sklearn.preprocessing import KBinsDiscretizer
import math


# filenames = ["foresttype",,"speech"]
filenames = ["annthyroid", "cardio",  "ionosphere","mammography" ,"satellite", "shuttle", "thyroid","smtp","satimage-2","pendigits"]

filename_list = []
feature_index_list = []
range_list = []

for filename in filenames:
    data = sio.loadmat('./datasets/' + filename + '.mat')
    pd_x = pd.DataFrame(data["X"])
    # print(pd_x.shape[1])
    # print(pd_x.columns)
    for feature_name in pd_x.columns:
        feature_values = pd_x.iloc[:, feature_name]
        # print(feature_values)
        # feature_values = pd_x[feature]
        # print(feature_values)
        feature_range = max(feature_values) - min(feature_values)

        filename_list.append(filename)
        feature_index_list.append(feature_name)
        range_list.append(feature_range)


pd_result = pd.DataFrame({"dataset":filename_list,"feature_index":feature_index_list,"range":range_list})
pd_result.to_excel("./EIF_SIF_Result/Dataset_feature_range.xlsx")