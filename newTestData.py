# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:36:03 2023

@author: rafae
"""

import pandas as pd
from utils.model_utils import  load_model, getPredictions, savePickl
from utils.plots import plotConfusionMatrices
import os
import pickle
import numpy as np

modelName = 'fineTunedNoDup'

#Read oldtest set from csb
testSetPrev = pd.read_csv(os.path.join ("data/rawData", modelName, "test.csv"))

#Get label dictionary
possible_labels = testSetPrev.cellType.unique()
label_dict = {}
for possible_label in possible_labels:
    label_dict[possible_label] = np.unique(testSetPrev['label'][testSetPrev['cellType'] == possible_label])[0]


#Get indices of test set from the main dataframe
sequenceDF = pd.read_csv('data/ATACpeaksPerCell.csv')
df3 = pd.merge(testSetPrev.reset_index() ,sequenceDF.reset_index(), on = ['peaks','cellType'])
df3 = df3.drop_duplicates(subset=['index_x'])

del sequenceDF

#Create new test set with those indices
sequenceDF = pd.read_csv('data/ATACpeaksPerCellAllPeaks.csv')
testSet = sequenceDF.iloc[df3.index_y, ]
#Add labels
testSet['label'] = testSet.cellType.replace(label_dict)

testSet.to_csv('data/rawData/fineTunedNoDup/testAllPeaks.csv', index = False)

