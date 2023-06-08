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

#Read test set from csb
testSet = pd.read_csv(os.path.join ("data/rawData", modelName, "test.csv"))
 
#Get label dictionary
possible_labels = testSet.cellType.unique()
label_dict = {}
for possible_label in possible_labels:
    label_dict[possible_label] = np.unique(testSet['label'][testSet['cellType'] == possible_label])[0]

#Set output path and class number
outputDir = os.path.join("outputs", "models", modelName)
numClasses = len(np.unique(testSet['label']))

# #Load finetuned model and tokenizer from GENALM model
model_config = {
    "model_HF_path": outputDir, #path to model
    "model_HF_path_tokenizer": 'AIRI-Institute/gena-lm-bigbird-base-t2t', #path to tokenizer
    "num_classes": numClasses }
model, tokenizer, device = load_model(model_config, return_model=True)
model.eval()


# #Predict test set and calculate loss
loss, predictions, accuracy = getPredictions (testSet, model, tokenizer, device, batchSize = 8)

# #Save results 
results = {"loss": float(loss),
           "accuracy": accuracy,
           "true": testSet.iloc[:, -1].values.tolist(),
           "predictions": np.array(predictions)}


pathToResults = os.path.join("outputs","results", modelName)
savePickl (pathToResults, 'results', results)


#Load results from pickle
with open(os.path.join(pathToResults, 'results'), 'rb') as f:
    results = pickle.load(f)

#Plot confusion matrix
plotConfusionMatrices (results, pathToResults, label_dict)
