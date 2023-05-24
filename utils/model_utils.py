# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:46:29 2023

@author: rafae
"""

import torch
from transformers import (
    AutoTokenizer,
    BigBirdForSequenceClassification,
    Trainer
)
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import shuffle
from collections import Counter
from operator import itemgetter
import pandas as pd
import os
import pickle
import tqdm

def savePickl (pathToFolder, fileName, objectName):
    
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
        
    filehandler = open(os.path.join(pathToFolder,fileName), 'wb') 
    pickle.dump(objectName, filehandler)
    
def setSeed (seed_val):
    #Set random seeds
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def addLabels(dataset):
    
    possible_labels = dataset.cellType.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    #Replace cell type with label
    dataset['label'] = dataset.cellType.replace(label_dict)
    
    return dataset, label_dict

def  splitTrainTestVal (inputDF, testSize, valSize):
    
    #Get number of records for smallest class
    minClass, minClassCount = min(Counter(inputDF['label']).items(), key=itemgetter(1))
    
    balancedDFTest = pd.DataFrame()
    balancedDFTrain = pd.DataFrame()
    balancedDFVal = pd.DataFrame()
    
    #Balance the classes
    for label in np.unique(inputDF['label']):
        #Get only the records of a given class
        inputDFClass = inputDF[inputDF['label'] == label]
        #Subset the df 
        inputDFClass = inputDFClass.sample(n=minClassCount)
        #Get all indices
        indices = inputDFClass.reset_index().index.tolist()
        #Get test index
        testIndex = random.sample(indices, int(testSize*len(indices)))
        trainValIndex = list(set(indices) - set(testIndex))
        #Get validation indices
        valIndex = random.sample(trainValIndex, int(valSize*len(trainValIndex)))
        #Get training idices
        trainIndex = list(set(trainValIndex) - set(valIndex))
        #Subset the dataframe using the train and test indices
        trainSet = inputDFClass.iloc[trainIndex, :]
        valSet = inputDFClass.iloc[valIndex, :]
        testSet = inputDFClass.iloc[testIndex, :]
        #append to new DFs
        balancedDFTest = pd.concat ([balancedDFTest, testSet])
        balancedDFTrain = pd.concat ([balancedDFTrain, trainSet])
        balancedDFVal = pd.concat ([balancedDFVal, valSet])
        
    
    #Shuffle train and validation sets
    balancedDFTrain = shuffle(balancedDFTrain)
    balancedDFVal = shuffle(balancedDFVal, random_state = 10)

    
    return balancedDFTrain, balancedDFTest, balancedDFVal



def load_model(model_config, checkGPU = True, return_model=False):
    """
    Load the model based on the input configuration

    Parameters
    ----------
    model_config : dict
        model configuration

    return_model : bool, optional
        return model, tokenizer, device, by default False

    Returns
    -------
    model, tokenizer, device: optional

    """

    global model, device, tokenizer
    
    if (checkGPU == True):
        if torch.cuda.is_available():
            # for CUDA
            torch.cuda.empty_cache()
            device = torch.device("cuda:0")
            print("Running the model on CUDA")
    
        elif torch.backends.mps.is_available():
            # for M1
            device = torch.device("mps")
            print("Running the model on M1 CPU")

    else:
        device = torch.device("cpu")
        print("Running the model on CPU")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_HF_path_tokenizer"], do_lower_case=False
    )

    model = BigBirdForSequenceClassification.from_pretrained(
        model_config["model_HF_path"], num_labels=model_config["num_classes"]
    )

    print(f'{ model_config["model_HF_path"]} loaded')

    model.to(device)

    if return_model:
        return model, tokenizer, device



def compute_metrics(eval_preds):
    """
    Compute the metrics for the model

    Parameters
    ----------
    eval_preds : tuple
        tuple of predictions and labels

    Returns
    -------
    :dict
        dictionary of metrics
    """

    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)

    #auc = roc_auc_score (labels, predictions, multi_class="ovr", average="micro", needs_proba=True)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def encode(dataset, tokenizer):
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
    
    datasetHF = HF_dataset(encodings["input_ids"], encodings["attention_mask"], y_train)

    
    return datasetHF


def getPredictions (testSet, model, tokenizer, device, batchSize):
    
    lossTotal = []
    predictionsTotal = []
    accTotal = []
    
    for i in tqdm.tqdm(range (0, len(testSet), batchSize)):
        testSetBatch = testSet.iloc[i: i+batchSize, ]
        loss, predictions, acc = predictMultiple (testSetBatch, model, tokenizer, device)
        lossTotal.append(loss)
        predictionsTotal.extend(predictions)
        accTotal.append(acc)
        
    lossFinal = sum(lossTotal) / len(lossTotal)
    accFinal = sum(accTotal) / len(accTotal)
    
    return lossFinal, predictionsTotal, accFinal
    
    
    

def predictMultiple (testSet, model, tokenizer, device):
    
    inputs = tokenizer(testSet['peaks'].tolist(), 
                       max_length=4096, return_tensors="pt",
                       padding=True, truncation=True)
    inputs.to(device)
    labels = torch.tensor(testSet['label'].values)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]
    
    predictions = np.argmax(logits.cpu(), axis=1)
    acc = accuracy_score(labels, predictions)

    
    return loss, predictions, acc


class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        labels = inputs.pop("labels")
        labels.to(device)
        outputs = model(**inputs)
        logits = outputs[0]
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

class HF_dataset(torch.utils.data.Dataset):
    """
    A class to create a dataset for the HuggingFace transformers
    Parameters
    ----------
    input_ids : torch.tensor
        The input ids of the sequences
    attention_masks : torch.tensor
        The attention masks of the sequences
    labels : torch.tensor
        The labels of the sequences
    Returns
    -------
    torch.utils.data.Dataset
        A dataset compatible with the HuggingFace transformers
    """

    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.input_ids[index]),
            "attention_mask": torch.tensor(self.attention_masks[index]),
            "labels": torch.tensor(self.labels[index]),
        }