# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:48:04 2023

@author: rafae
"""

import pandas as pd
import os

sequenceDF = pd.read_csv('data/ATACpeaksPerCell.csv')

grouped = sequenceDF.groupby(['cellType'])
del sequenceDF
NN = grouped.get_group("Non-Neuronal")
GB = grouped.get_group("GABAergic")
GL = grouped.get_group("Glutamatergic")
del grouped

DNAregions = pd.read_csv(os.path.join ('data/DNAregions.xlsx'))


def getPeaksPerc (NN, pathToSave, fileName, DNAregions):
    NNpeaks = [NN.iloc[i]['peaks'].split() for i in range(len(NN))]
    NNpeaksFlat = [item for sublist in NNpeaks for item in sublist]
    del NNpeaks
    
    from collections import Counter
    countsNN = (Counter(NNpeaksFlat))
    
    
    peakCounts = pd.DataFrame()
    for peak in countsNN.keys():
        peakLoc = DNAregions[DNAregions.DNAseq == peak]['region'].iloc[0]
        
        dic = {'peak': peakLoc,
               'percentageInPeaks': countsNN[peak]/len(NNpeaksFlat) }
        peakCounts = pd.concat([peakCounts,pd.DataFrame([dic])])
        
        
    peakCounts.to_csv(os.path.join(pathToSave, fileName), index = False)

getPeaksPerc(NN, 'data', 'NNpeaksinDataset.csv', DNAregions)
getPeaksPerc(GL, 'data', 'GLpeaksinDataset.csv', DNAregions)
getPeaksPerc(GB, 'data', 'GBpeaksinDataset.csv', DNAregions)

NN = pd.read_csv(os.path.join ("data/NNpeaksinDataset.csv"))
GL = pd.read_csv(os.path.join ("data/GLpeaksinDataset.csv"))
GB = pd.read_csv(os.path.join ("data/GBpeaksinDataset.csv"))


