# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:34:29 2023

@author: rafae
"""

import pandas as pd
import os
import numpy as np
import random
from utils.attention import getFeatures, writeListToBed, getSequence, plot_heatmap
from utils.model_utils import load_model

modelName = 'fineTunedNoDup'

#Set output path and class number
outputDir = os.path.join("outputs", "models", modelName)
numClasses = 3
#Save in text documents
pathToResults =  os.path.join("outputs", "results", modelName) 

# #Load finetuned model and tokenizer from GENALM model
model_config = {
    "model_HF_path": outputDir, #path to model
    "model_HF_path_tokenizer": 'AIRI-Institute/gena-lm-bigbird-base-t2t', #path to tokenizer
    "num_classes": numClasses }
model, tokenizer, device = load_model(model_config, checkGPU = False, return_model=True)
model.eval()


DNAregions = pd.read_csv('data/DNAregions.xlsx')


chromosomesToCheck= {'GABAergic': ['chr5:76952593-76954930']}

subsetPositions = ['chr5:76953480-76953582',
                    'chr5:76952738-76952840',
                    'chr5:76953035-76953137']


# chromosomesToCheck= {'GABAergic': ['chr1:633543-634316',
#                                    'chr3:93470467-93471024',
#                                    'chr5:76952593-76954930',
#                                    'chr13:110306246-110308993',
#                                    'chr5:100900736-100903868',
#                                    'chr2:17877787-17879898',
#                                    'chr11:16605771-16608164',
#                                    'chr7:8433246-8434867',
#                                    'chr13:37868657-37871376',
#                                    'chr2:170815757-170818430'],
#                      'Glutamatergic': ['chr3:93470467-93471024',
#                                        'chr18:12253588-12256062',
#                                        'chr1:633543-634316',
#                                        'chr1:18071366-18074416',
#                                        'chr9:135166103-135168844',
#                                        'chr3:194621781-194624359',
#                                        'chr7:45407218-45409817',
#                                        'chr9:127679563-127681320',
#                                        'chr18:51857753-51859599',
#                                        'chr11:119416959-119418818']}


combineRegions = False
pathToBed =  os.path.join("outputs", "results", modelName, 'overlapTFs', 'bedFiles')
if not os.path.exists(pathToBed):
    os.mkdir(pathToBed)

pathToPlots = os.path.join("outputs", "results", modelName, 'regionPlots')

for key in chromosomesToCheck.keys():
    for i in range(len(chromosomesToCheck[key])):
        region = chromosomesToCheck[key][i]
        
        chromosome = region.split(':')[0]
        start = region.split(':')[1].split('-')[0]
        end = region.split(':')[1].split('-')[1]
        
        sequence = getSequence(chromosome, start, end, refGenome = 'hg38')
        
        heatmap = getFeatures(sequence, DNAregions, model, tokenizer, device, 90, returnHeatmap = True)
        
        #subsetPositions = [region]
        #extract positions from subset sequences
        for subset in subsetPositions:
            startSub = subset.split(':')[1].split('-')[0]
            endSub = subset.split(':')[1].split('-')[1]
            sequenceSub = getSequence(chromosome, startSub, endSub, refGenome = 'hg38')
            
            heatmapSubset = heatmap[int(startSub)-int(start) : int(endSub)-int(start)]
            positions = [j+int(startSub) for j in range(len(heatmapSubset)) if heatmapSubset[j]>np.percentile(heatmap,90)]
            
            filename = subset.replace(":", "-" ) + '.bed'
            writeListToBed (pathToBed, filename, positions, chromosome)
            
            #Plot it
            #plotName = subset.replace(":", "-" )
            #plot_heatmap(heatmapSubset, sequenceSub, subset, pathToPlots, plotName)
            
            #Shuffle these scores 
            for k in range(100):
                n = random.random()
                random.Random(n).shuffle(heatmapSubset)
                positions = [j+int(startSub) for j in range(len(heatmapSubset)) if heatmapSubset[j]>np.percentile(heatmap,90)]
                
                #Write positions to bed file
                filename = subset.replace(":", "-" )+ '_random' + str(k) + '.bed'
                writeListToBed (pathToBed, filename, positions, chromosome)
                
                #Plot it
                #plotName = subset.replace(":", "-" )+ '_random' + str(k)
                #plot_heatmap(heatmapSubset, sequenceSub, subset, pathToPlots, plotName)

