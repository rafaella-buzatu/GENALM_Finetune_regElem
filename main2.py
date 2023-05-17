# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:36:03 2023

@author: rafae
"""

import pandas as pd
from utils.model_utils_2 import setSeed, addLabels, splitTrainTestVal, load_model
from utils.model_utils_2 import encode, ClassificationTrainer, compute_metrics
from utils.model_utils_2 import predictMultiple, savePickl
from utils.plots import plotLoss, plotConfusionMatrices
import os
from transformers import TrainingArguments
import pickle

modelName = 'fineTunedSubset'

sequenceDF = pd.read_csv('data/ATACpeaksPerCell.csv')
sequenceDF = sequenceDF.drop(sequenceDF[sequenceDF['cellType'] == 'Non-Neuronal'].index).reset_index()
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

#Remove input df variable
del sequenceDF

#Load model
model_config = {
    "model_HF_path": 'AIRI-Institute/gena-lm-bigbird-base-t2t',
    "model_HF_path_tokenizer":'AIRI-Institute/gena-lm-bigbird-base-t2t', 
    "num_classes":  numClasses}
model, tokenizer, device = load_model(model_config, return_model=True)


#Encode the test, train and validation sets
trainEncodings = encode(trainSet, tokenizer)
valEncodings = encode(valSet, tokenizer)
testEncodings = encode(testSet, tokenizer)


#Define training parameters
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 10

outputDir = os.path.join("outputs", "models", modelName)
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
    metric_for_best_model="auc",
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

#Save training log
trainingLog = trainer.state.log_history
savePickl (outputDir, 'trainingLog', trainingLog)

#Plot loss during training
plotLoss (trainingLog, outputDir)

#Evaluate the model on the test dataset
trainer.eval_dataset=testEncodings
trainer.evaluate()


#Save the model
model.save_pretrained(outputDir)

# #Load finetuned model and tokenizer from DNABERT_6
model_config = {
    "model_HF_path": outputDir, #path to model
    "model_HF_path_tokenizer": 'AIRI-Institute/gena-lm-bigbird-base-t2t', #path to tokenizer
    "num_classes": numClasses }
model, tokenizer, device = load_model(model_config, return_model=True)
model.eval()


# #Predict test set and calculate loss
loss, predictions = predictMultiple (testSet, model, tokenizer, device)

# #Save results 
results = {"loss": float(loss),
           "true": testSet.iloc[:, -1],
           "predictions": predictions.detach().numpy()}


pathToResults = os.path.join("outputs","results", modelName)
savePickl (pathToResults, 'results', results)


#Load results from pickle
with open(os.path.join(pathToResults, 'results'), 'rb') as f:
    results = pickle.load(f)

#Plot confusion matrix
plotConfusionMatrices (results, pathToResults, label_dict)