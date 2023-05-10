# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:36:03 2023

@author: rafae
"""

import pandas as pd
from utils.model_utils import setSeed, addLabels, splitTrainTestVal, load_model
from utils.model_utils import encode, ClassificationTrainer, compute_metrics
import os
from transformers import TrainingArguments

sequenceDF = pd.read_csv('data/ATACpeaksPerCell.csv')

#Add numerical labels
sequenceDF, label_dict = addLabels(sequenceDF)

#Set random seed for sampling
setSeed (seed_val = 10)

#Split into training and test
trainSet, testSet, valSet = splitTrainTestVal (sequenceDF,
                                               testSize = 0.2, 
                                               valSize = 0.2)

### TRAINING
numClasses = len(label_dict.keys())

#Load model
model_config = {
    "model_HF_path": 'AIRI-Institute/gena-lm-bigbird-base-t2t',
    "model_HF_path_tokenizer":'AIRI-Institute/gena-lm-bigbird-base-t2t', 
    "num_classes":  numClasses}
model, tokenizer, device = load_model(model_config, return_model=True)


trainSet = trainSet.iloc[:4000, ]
valSet = valSet.iloc[:1000, ]
testSet = testSet.iloc[:1000, ]


#Encode the test, train and validation sets
trainEncodings = encode(trainSet, tokenizer)
valEncodings = encode(valSet, tokenizer)
testEncodings = encode(testSet, tokenizer)


#Define training parameters
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 10

outputDir = os.path.join("outputs/models/fineTuned")
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

training_args = TrainingArguments(
    output_dir=outputDir,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    weight_decay=0.01
)


trainer = ClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=trainEncodings,
    eval_dataset=valEncodings,
    compute_metrics=compute_metrics,
)


#Start training
trainer.train()
