import numpy as np
import scipy.stats as sts
import pandas
import pandas as pd
import eif_old as eif_old_class
import scipy.io as sio
import os
import sklearn.metrics as skm
from sklearn.preprocessing import KBinsDiscretizer

def kbin_discre(filename, n_bin):
    data = sio.loadmat('./datasets/' + filename + '.mat')
    pd_x = pd.DataFrame(data["X"])
    pd_y = pd.DataFrame(data["y"])
    np_x = np.array(pd_x)
    np_y = np.array(pd_y)
    np_x_discretized = np_x
    print(filename)

    enc = KBinsDiscretizer(n_bins=n_bin, encode='ordinal',strategy="kmeans")

    if filename == "cardio":
        # continuous_feature = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,19,20]
        continuous_feature = [0,1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]
        np_x_discretized[:, continuous_feature] = enc.fit_transform(np_x[:, continuous_feature])
        # continuous_feature = [0]
        # continuous_feature = [True, True, True, True, True, False, True, True, True, True, True, True, True, True, True,False, True, True, True, True,False]

    elif filename == "ionosphere":
        continuous_feature = [ 1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,27,28,29,30,31,32]
        np_x_discretized[:, continuous_feature] = enc.fit_transform(np_x[:, continuous_feature])
    else:
        np_x_discretized = enc.fit_transform(np_x)

    pd_x_disc = pd.DataFrame(np_x_discretized)
    pd_x_disc.to_csv("./datasets/" + filename + "_discretized_"+str(n_bin) +"BIN_withoutLabel.csv", index=False)
    pd_x_disc["label"] = pd_y
    pd_x_disc.to_csv("./datasets/"+filename+"_discretized_"+str(n_bin) +"BIN_withLabel.csv",index=False)


# filenames = ["annthyroid", "cardio", "foresttype", "ionosphere","mammography" ,"satellite", "shuttle", "thyroid","smtp","satimage-2","pendigits","speech"]
filenames  =["foresttype","http"]
n_bins = [10,15]
# filenames = ["cardio"]
for n_bin in n_bins:
    for filename in filenames:
        kbin_discre(filename,n_bin)

########################################################
# from MDLP import MDLP_Discretizer
    # discretizer = MDLP_Discretizer(features=continuous_feature)
    # discretizer.fit(np_x, np_y)
    # np_x_discretized = discretizer.transform(np_x)


################################################################################
    # from mdlp.discretization import MDLP
    # if filename == "cardio":
    #     # continuous_feature = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,19,20]
    #     # continuous_feature = [0,1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]
    #     continuous_feature = [0]
    #     # continuous_feature = [True, True, True, True, True, False, True, True, True, True, True, True, True, True, True,False, True, True, True, True,False]
    #     transformer = MDLP(continuous_features=continuous_feature)
    # elif filename == "ionosphere":
    # continuous_feature = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    #                       27, 28, 29, 30, 31, 32]

#     transformer = MDLP(continuous_features=continuous_feature)
    # else:
    #     transformer = MDLP()
    #
    # X_disc = transformer.fit_transform(np_x, np_y)
    # pd_x_disc = pd.DataFrame(X_disc)
    # pd_x_disc.to_csv("./datasets/" + filename + "_discretized_withoutLabel.csv", index=False)
    # pd_x_disc["label"] = np_y
    # pd_x_disc.to_csv("./datasets/"+filename+"_discretized_withLabel.csv",index=False)
    #
