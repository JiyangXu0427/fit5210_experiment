import numpy as np
import scipy.stats as sts
import pandas
import pandas as pd
import eif_old as eif_old_class
import scipy.io as sio
import os
import sklearn.metrics as skm



# filenames = ["annthyroid", "cardio", "foresttype", "ionosphere","mammography" ,"satellite", "shuttle", "thyroid"]
# filenames = ["smtp","satimage-2","pendigits","speech"]
filenames = ["foresttype","http"]

for filename in filenames:
    data = sio.loadmat('./datasets/' + filename + '.mat')
    pd_x = pd.DataFrame(data["X"])
    pd_y = pd.DataFrame(data["y"])
    pd_x["label"] = data["y"]
    pd_x.to_csv("./datasets/"+filename+".csv",index=False)
