import pandas as pd
import numpy as np
import scipy.interpolate as itpl
import scipy.io as sio

# read data
# dataset_name = "annthyroid"
# dataset_name = "cardio"
# dataset_name = "ionosphere"
# dataset_name = "mnist"
# dataset_name = "satellite"
# dataset_name = "shuttle"
# dataset_name = "thyroid"

# dataset_names = ["annthyroid", "cardio", "foresttype", "ionosphere","mammography" ,"satellite", "shuttle", "thyroid","smtp","satimage-2","pendigits","speech"]
dataset_names =["foresttype","http"]
adjust_values = [0.0625, 0.25, 4, 16, 1]
for adjust_value in adjust_values:
    for dataset_name in dataset_names:
        dataset_folder_path = './datasets/'
        dataset_path = dataset_folder_path + dataset_name +'.mat'
        data = sio.loadmat(dataset_path)
        x_data = np.array(data["X"])
        y_data = np.array(data["y"])
        dataset_total_col_num = x_data.shape[1]

        file_path_cdf_x = dataset_folder_path + dataset_name + "-" + str(adjust_value) +"-CDF_points_x.csv"
        file_path_cdf_y = dataset_folder_path + dataset_name + "-" + str(adjust_value) + "-CDF_points_y.csv"
        cfd_training_datas_x = pd.read_csv(file_path_cdf_x,index_col=0)
        cfd_training_datas_y = pd.read_csv(file_path_cdf_y,index_col=0)
        training_data_total_col_num = cfd_training_datas_x.shape[1]

        counter = 0
        copula_data_index = 0
        result_df = pd.DataFrame()

        while counter < dataset_total_col_num:

            column_to_be_transformed = x_data[:, counter]

            if dataset_name == "cardio":
                if counter == 5 or counter == 15 or counter == 20:
                    result_df["v" + str(counter)] = column_to_be_transformed
                    counter = counter + 1
                    continue

            if dataset_name == "ionosphere":
                if counter == 0 :
                    result_df["v" + str(counter)] = column_to_be_transformed
                    counter = counter + 1
                    continue

            selected_column_x = cfd_training_datas_x.iloc[:,copula_data_index]
            selected_column_x = np.array(selected_column_x)

            selected_column_y = cfd_training_datas_y.iloc[:,copula_data_index]
            selected_column_y = np.array(selected_column_y)

            cfd_func = itpl.interp1d(selected_column_x,selected_column_y,kind='cubic')
            cfd_result_for_dataset = cfd_func(column_to_be_transformed)

            result_df["v"+str(counter)] = cfd_result_for_dataset

            copula_data_index = copula_data_index + 1
            counter = counter + 1

        result_df["label"] = y_data
        result_file_path = dataset_folder_path + dataset_name + "-" + str(adjust_value) + "_copula_data.xlsx"
        result_df.to_excel(result_file_path,index=False)




