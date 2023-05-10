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
    
    random.seed(10)
    indices = inputDF.index.tolist()
    #Get test index
    testIndex = random.sample(indices, int(testSize*len(indices)))
    trainValIndex = list(set(indices) - set(testIndex))
    #Get validation indices
    valIndex = random.sample(trainValIndex, int(valSize*len(trainValIndex)))
    #Get training idices
    trainIndex = list(set(trainValIndex) - set(valIndex))
    #Subset the dataframe using the train and test indices
    trainSet = inputDF.iloc[trainIndex, :]
    valSet = inputDF.iloc[valIndex, :]
    testSet = inputDF.iloc[testIndex, :]
    
    #Shuffle train and validation sets
    trainSet = shuffle(trainSet, random_state = 10)
    valSet = shuffle(valSet, random_state = 10)

    
    return trainSet, testSet, valSet



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
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def encode(dataset, tokenizer):
    text = dataset['peaks'].tolist()
    y_train = np.float32(dataset['label'].values)

    encodings = tokenizer.batch_encode_plus(
        text,
        max_length=512,  # max len of BERT
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    
    datasetHF = HF_dataset(encodings["input_ids"], encodings["attention_mask"], y_train)

    
    return datasetHF


def predictMultiple (testSet, model, tokenizer, device, cellType):
    
    inputs = tokenizer(testSet['sequences'].tolist(), return_tensors="pt")
    inputs.to(device)
    labels = torch.tensor(testSet[cellType].tolist()).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]
    
    return loss, logits

def predictSingle (testSeq, model, tokenizer, device):
    
    inputs = tokenizer(testSeq['sequence'], return_tensors="pt")
    inputs.to(device)
    labels = torch.tensor(testSeq['label']).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]
    
    return loss, logits

class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        logits = logits.unsqueeze(1)
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