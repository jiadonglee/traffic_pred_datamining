import numpy as np
import pandas as pd
import sys
import os
import math
import time
from datetime import datetime
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =======================Clean Data ===========================
def Clean_data(my_df):

    Data = df.drop([11], axis=1)
    Data.columns = ['date', 'time', 'direction', 'type', 'linkID', 'length',
                    'travelTime', 'volumn', 'speed', 'occupancy', 'congestionLevel']
    Data = Data.drop(columns=['date'])
    Data = Data.dropna()
    # Data = Data[:1000]

    #  Vectorize the parameters
    Data.length = np.float64(Data.length)
    Data.travelTime = np.float64(Data.travelTime)
    Data.volumn = np.float64(Data.volumn)
    Data.speed = np.float64(Data.speed)
    Data.occupancy = np.float64(Data.occupancy)

    Data = Data[Data['length'] >= 0]
    Data = Data[Data['travelTime'] >= 0]
    Data = Data[Data['volumn'] >= 0]
    Data = Data[Data['speed'] >= 0]

    Data = Data.reset_index()
    occpcy_mode = np.float64(Data[Data['occupancy'] >= 0].occupancy.mode())
    for j in range(len(Data)):
        if Data.loc[j, 'occupancy'] < 0.:
            Data.loc[j, 'occupancy'] = occpcy_mode
    Data.occupancy = preprocessing.scale(Data.occupancy.values.reshape(-1, 1))
    Data = Data.drop(columns=['index'])

    ID = list(Data.linkID.unique())
    Time = list(Data.time.unique())
    Data['direction'] = Data['direction'].map({'EAST_BOUND': 0, 'WEST_BOUND': 1,
                                            'SOUTH_BOUND': 2, 'NORTH_BOUND': 3, 'STH_BOUND': 4, 'UNKNOWN_DIRECTION_TYPE': 5}).astype(int)
    Data['type'] = Data['type'].map({'FREEWAY': 0, 'RAMP': 1, 'ARTERIAL': 2, 'LOCAL_ROAD': 3,
                                    'FREEWAY_REVERSIBLE': 4, 'FREEWAY_EXPRESS': 5}).astype(int)
    Data['congestionLevel'] = Data['congestionLevel'].map({'NON_CONGESTION': 0, 'LIGHT_CONGESTION': 1, 'MEDIUM_CONGESTION': 2,
                                                        'HEAVY_CONGESTION': 3, 'UNKNOWN_CONGESTION_LEVEL': 4}).astype(int)
    Data['linkID'] = Data['linkID'].map(
        dict(zip(ID, np.arange(len(ID))))).astype(int)
    Data['time'] = Data['time'].map(
        dict(zip(Time, np.arange(len(Time))))).astype(int)

    Data = Data.drop(Data[Data['congestionLevel'] == 4].index)


    length_norm = np.array(Data.length).reshape(-1, 1)
    trvTime_norm = np.array(Data.travelTime).reshape(-1, 1)
    volumn_norm = np.array(Data.volumn).reshape(-1, 1)
    speed_norm = np.array(Data.speed).reshape(-1, 1)
    occu_norm = np.array(Data.occupancy).reshape(-1, 1)

    est_length = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform').fit(length_norm)
    est_trvTime = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform').fit(trvTime_norm)
    est_volumn = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform').fit(volumn_norm)
    est_speed = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit(speed_norm)
    est_occu = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit(occu_norm)

    length_norm = est_length.transform(length_norm)
    trvTime_norm = est_trvTime.transform(trvTime_norm)
    volumn_norm = est_volumn.transform(volumn_norm)
    speed_norm = est_speed.transform(speed_norm)
    occu_norm = est_occu.transform(occu_norm)

    Data.length = length_norm
    Data.travelTime = trvTime_norm
    Data.volumn = volumn_norm
    Data.speed = speed_norm
    Data.occupancy = occu_norm

    return Data

#=================================

# =======================Load train Data=================================
dir_name = "/Users/jordan/Project/DataMining/traffic_dataset/train_set/"
df = pd.DataFrame()
for file_path in os.listdir(dir_name):
    os.chdir(dir_name + file_path)
    for file_name in os.listdir():
        new_df = pd.read_csv(file_name, header=None, dtype=np.unicode_)
        df = df.append(new_df)
# train_data = Clean_data(df)
#======================Load test Data====================================
test_name = "/Users/jordan/Project/DataMining/traffic_dataset/train_set/"
test_df = pd.DataFrame()
for file_path in os.listdir(dir_name):
    os.chdir(dir_name + file_path)
    for file_name in os.listdir():
        new_df = pd.read_csv(file_name, header=None, dtype=np.unicode_)
        test_df = df.append(new_df)

# test_data = Clean_data(test_df)

os.chdir("/Users/jordan/Project/DataMining/src/")
