# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:36:03 2023

@author: rafae
"""

import pandas as pd
from utils.model_utils import addLabels, savePickl, load_model
import os


modelName = 'fineTuned'

sequenceDF = pd.read_csv('data/ATACpeaksPerCell.csv')
#Add numerical labels
sequenceDF, label_dict = addLabels(sequenceDF)

#Set random seed for sampling
#setSeed (seed_val = 10)

#Split into training and test
#trainSet, testSet, valSet = splitTrainTestVal (sequenceDF,
#                                               testSize = 0.2, 
#                                               valSize = 0.2)


def encodings(dataset, tokenizer):
    text = dataset['peaks'].tolist()
    y_train = dataset['label'].values
    
    encodings = tokenizer.batch_encode_plus(
        text,
        max_length=4096,  # max len of BigBird
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    
    return encodings



### TRAINING
numClasses = len(label_dict.keys())
outputDir = os.path.join("outputs", "models", modelName)
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#Load model
model_config = {
    "model_HF_path": 'AIRI-Institute/gena-lm-bigbird-base-t2t',
    "model_HF_path_tokenizer":'AIRI-Institute/gena-lm-bigbird-base-t2t', 
    "num_classes":  numClasses}
model, tokenizer, device = load_model(model_config, checkGPU = False, return_model=True)


#Encode the test, train and validation sets
trainEncodings = encodings(sequenceDF, tokenizer)


dictPadding = {'cellType': [], 
               'padding' : [], 
               'maxLenToken': [],
               'avgLenToken': [],
               'minLenToken': []}

for i in range(len(trainEncodings._encodings)):
    dictPadding['padding'].append (trainEncodings._encodings[i].tokens.count('[PAD]'))
    tokens = trainEncodings._encodings[i].tokens
    tokens = [i for i in tokens if i != 'PAD']
    dictPadding['maxLenToken'].append(max(map(len, tokens)))
    dictPadding['minLenToken'].append(min(map(len, tokens)))
    dictPadding['avgLenToken'].append(sum(map(len, tokens))/float(len(tokens)))
    dictPadding['cellType'].append( sequenceDF.iloc[i]['cellType'])

pathToData = os.path.join ("data/rawData/", modelName)
savePickl (pathToData, 'dictPadding', dictPadding)

