# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

dir_train = 'traffic_dataset/train_set/0707/'

data = pd.read_csv(dir_train + '0707_seg_1.txt')