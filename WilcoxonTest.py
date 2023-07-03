# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:17:15 2023

@author: rafae
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

overlaps = pd.read_excel('outputs/results/fineTunedNoDup/overlapTFs/overlappingDF.xlsx')

differences = pd.DataFrame(columns= ['region', 'overlapOriginal', 'overlapShuffled', 'difference'])

listDifferences = []
for region in np.unique(overlaps['region']):

    overlapInitial = overlaps[(overlaps['region'] == region) & (overlaps['randomized'] == 0)][ 'overlapCount'].values[0]
    avgRandomized = np.mean(overlaps[(overlaps['region'] == region) & (overlaps['randomized'] == 1)][ 'overlapCount'])
    difference = overlapInitial- avgRandomized
    listDifferences.append(difference)
    dictForDF = {'region' : region, 
                'overlapOriginal' : overlapInitial, 
                'overlapShuffled' : avgRandomized, 
                'difference' : difference}
    df_dictionary = pd.DataFrame([dictForDF])
    differences= pd.concat([differences, df_dictionary], ignore_index=True)
    
print (differences)

res = wilcoxon(listDifferences, alternative = 'greater')
print ('statistic =', res.statistic, '\t', 'p-value =', res.pvalue)

if (res.pvalue <0.05):
    print ('Difference is significant')
else:
    print('Difference is not significant')