# GENALM_Finetune_regElem

### Finetune model 1
*getInput.R*: The script containing all necessary commands to generate the input dataset that was used to train and test the model. <br>
*da.R*: Script for the differential accesibility analysis for all 3 cell types. <br>
*main.py*: The script containing all the necessary commands to run the triaining and test cycle on the previously generated dataset from getInput.R. <br>
*mainTest.py*: The script containing all the necessary commands to run a test cycle. <br>
<br>
### Second test set
*newTestData.py*: Extract all peaks from the previous test dataset (without DAR intersection)<br>
*mainTestNewData.py*: The script containing all the necessary commands to run a test cycle on the new dataset from newTestData.py. <br>

<br>
### Finetune model 2
*getInputTest2.R*: The script containing all necessary commands to generate the input dataset when splitting DARs between training and test. <br>
*mainTest2.py*: The script containing all the necessary commands to run the triaining and test cycle on a previously generated dataset from getInput2.R. <br>
<br>

### Visualise important tokens
*generateFeatures.py*: A script to extract most important tokens.<br>
*generateBedRegions.py*: A script to generate a bed file with the positions of highest cummulative attention scores.<br>
*checkOverlap.R*: A script to check the overlap between the bed fiels and annotated TF binding sites.<br>
*WilcoxonTest.py*: Wilcoxon sum-ranked test on the difference between overlapping bases.<br>

