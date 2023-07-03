import pandas as pd
from utils.model_utils import load_model
import os
import numpy as np
import tqdm
from utils.attention import getFeatures, writeListToFile, writeListToMEME
from collections import Counter

modelName = 'fineTunedNoDup'
#Read test data
testSet = pd.read_csv(os.path.join ("data/rawData", modelName, "test.csv"))
DNAregions = pd.read_csv('data/DNAregions.xlsx')

#Get label dictionary
possible_labels = testSet.cellType.unique()
label_dict = {}
for possible_label in possible_labels:
    label_dict[possible_label] = np.unique(testSet['label'][testSet['cellType'] == possible_label])[0]

#Set output path and class number
outputDir = os.path.join("outputs", "models", modelName)
numClasses = len(np.unique(testSet['label']))
#Save in text documents
pathToResults =  os.path.join("outputs", "results", modelName) 

# #Load finetuned model and tokenizer from GENALM model
model_config = {
    "model_HF_path": outputDir, #path to model
    "model_HF_path_tokenizer": 'AIRI-Institute/gena-lm-bigbird-base-t2t', #path to tokenizer
    "num_classes": numClasses }
model, tokenizer, device = load_model(model_config, checkGPU = False, return_model=True)
model.eval()


#Subset 100 random samples from each test set class
testSet = testSet.groupby('cellType').apply(lambda x: x.sample(n=100, random_state=5)).reset_index(drop=True)

#Create dictionary to append motifs for each cell type
dictFeatures = {key:[] for key in testSet['cellType'].unique()}

for cell in tqdm.tqdm(range(len(testSet))):
    #get cellType of cell
    cellType = testSet.iloc[cell]['cellType']
    #Get features with highest attention score for each cell
    #pathToPlots = os.path.join(pathToResults, 'regionPlots', cellType)
    features = getFeatures(testSet.iloc[cell]['peaks'], DNAregions, model, tokenizer, device, 90)#, plot = True, pathToPlots = pathToPlots)
    #Append to cell-type specific dictionary entry
    dictFeatures[cellType].append(features)

for key in dictFeatures.keys():
    #Flatten
    dictFeatures[key] = [k for i in dictFeatures[key] for k in i]
    #Count occurence of each subsequence
    counter = Counter(dictFeatures[key])
    #Only keep sequences of length >=5 and with counts more than 75%
    sequencesToKeep = [keyc for keyc in counter.keys() if len(keyc)>=5 and counter[keyc]>np.percentile(list(counter.values()), 75)]
    dictFeatures[key] = sequencesToKeep
    
#Save in text documents
pathToResults =  os.path.join("outputs", "results", modelName) 

#for key in dictFeatures.keys():
#    writeListToFile(pathToResults, 'features' + key + '.txt', dictFeatures[key])

GB = open(os.path.join(pathToResults, 'featuresGABAergic.txt'), "r").read().split("\n")
NN = open(os.path.join(pathToResults, 'featuresNon-Neuronal.txt'), "r").read().split("\n")
GL = open(os.path.join(pathToResults, 'featuresGlutamatergic.txt'), "r").read().split("\n")


onlyGB = list(set(list(set(GB) - set(NN)))-set(GL))
#onlyNN = list(set(list(set(NN) - set(GB)))-set(GL))
onlyGL = list(set(list(set(GL) - set(NN)))-set(GB))

#writeListToFile(pathToResults, 'featuresOnlyGABA.txt', onlyGB)
writeListToMEME(pathToResults, 'GABAtoMEME.meme', onlyGB)
writeListToMEME(pathToResults, 'GLUTtoMEME.meme', onlyGL)

#q-value: corrected p-value
#Extract significant results for GABAergic and Glutamatergic
GBmeme = pd.read_excel(os.path.join(pathToResults, 'tomtomGABA.xlsx'))
GBmeme = GBmeme[GBmeme['q-value'] <= 0.05]

GLmeme = pd.read_excel(os.path.join(pathToResults, 'tomtomGLUT.xlsx'))
GLmeme = GLmeme[GLmeme['q-value'] <= 0.05]

#commonGL_GB = set(GLmeme['Target_ID'].tolist()).intersection(GBmeme['Target_ID'].tolist())

#Extract unique target IDs for each cell type
onlyGB_TFs = set(GBmeme['Target_ID'].tolist()) - set((GLmeme['Target_ID'].tolist()))
onlyGL_TFs = set(GLmeme['Target_ID'].tolist()) - set((GBmeme['Target_ID'].tolist()))
