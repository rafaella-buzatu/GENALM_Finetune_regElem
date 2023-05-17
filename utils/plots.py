# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:39:10 2023

@author: rafae
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import figure
import matplotlib.offsetbox as offsetbox
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plotConfusionMatrices(results, pathToPlots, label_dict):   
    
    f, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.ravel()
    for i in range(len(label_dict.keys())):
        indexClass = results[results['true']==i].index
        disp = ConfusionMatrixDisplay(confusion_matrix(results[true][indexClass],
                                                       results[predictions][indexClass]),
                                    display_labels=[0, i])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(label_dict[i])
        if i<10:
            disp.ax_.set_xlabel('')
        if i%5!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
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