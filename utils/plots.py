# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:39:10 2023

@author: rafae
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.metrics import confusion_matrix

def plotConfusionMatrices(results, pathToPlots, label_dict):   

    CM = confusion_matrix(results['true'], results['predictions'])
    cm_df = pd.DataFrame(CM, index = label_dict.keys(), columns = label_dict.keys())
    plt.figure(figsize=(8, 6), dpi=200)
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(os.path.join(pathToPlots, 'confusionMatrix.png'))
    plt.show()
   
def plotLoss (trainingLog, pathToPlots):
    
    #Extract indices of evaluation and training steps from training log
    evalIndex = [ind for ind in range(len(trainingLog)) if 'eval_loss' in trainingLog[ind].keys()]
    trainIndex = [ind for ind in range(len(trainingLog)) if 'loss' in trainingLog[ind].keys()]

    #Extract values of losses
    evalLosses = [trainingLog[i]['eval_loss'] for i in evalIndex]
    trainLosses = [trainingLog[i]['loss'] for i in trainIndex]

    #Extract epoch numbers
    evalEpochs = [trainingLog[i]['epoch'] for i in evalIndex]
    trainEpochs = [trainingLog[i]['epoch'] for i in trainIndex]

    #Plots
    figure(figsize=(8, 6), dpi=100)
    plt.plot(evalEpochs, evalLosses, label = 'validation loss')
    plt.plot (trainEpochs, trainLosses, label = 'training loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(pathToPlots, 'trainingLoss.png'))
    plt.show()