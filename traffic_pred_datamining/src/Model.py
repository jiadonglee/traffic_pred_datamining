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

train_df = pd.read_csv("../traffic_dataset/train_traffic.csv")
test_df = pd.read_csv("../traffic_dataset/test_traffic.csv")
X_train = train_df.iloc[:, 0:9]
y_train = train_df['congestionLevel']

X_test = test_df.iloc[:, 0:9]
y_true = test_df['congestionLevel']
# Model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_true, y_pred)
print("accuracy score is:{0}".format(acc))
