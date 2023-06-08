# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:55:55 2023

@author: rafae
"""

import pandas as pd
import os
import numpy as np
from utils.model_utils import load_model
from bertviz import model_view

modelName = 'fineTunedNoDup'

#Read test set from csb
testSet = pd.read_csv(os.path.join ("data/rawData", modelName, "test.csv"))

#Set output path and class number
outputDir = os.path.join("outputs", "models", modelName)
numClasses = len(np.unique(testSet['label']))

# #Load finetuned model and tokenizer from GENALM model
model_config = {
    "model_HF_path": outputDir, #path to model
    "model_HF_path_tokenizer": 'AIRI-Institute/gena-lm-bigbird-base-t2t', #path to tokenizer
    "num_classes": numClasses }
model, tokenizer, device = load_model(model_config, return_model=True)


input_text = testSet.iloc[0]['peaks']
inputs = tokenizer.encode(input_text, return_tensors='pt')
inputs = inputs.to(device)
outputs = model(inputs, output_attentions=True)  # Run model
attention = outputs[-1] 
#attention = attention.unsqueeze(0) # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view

