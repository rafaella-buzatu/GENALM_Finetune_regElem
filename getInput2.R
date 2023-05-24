library(writexl)
library(matrixStats)
library(readxl)

#### DEFINE FUNCTIONS

#' library(BSgenome.Hsapiens.UCSC.hg38)
#' 
#' addDNAsequences <- function (regionData){
#'   #' Adds the corresponding DNA sequences to each row of the input dataframe
#'   #' 
#'   #' regionData = A dataframe with information about each DNA region.
#'   
#'   regionData ['DNAseq'] <- NA
#'   #Define progress bar
#'   pb = txtProgressBar(min = 0, max = nrow(regionData), style = 3, width = 50) 
#'   
#'   for (row in 1:nrow(regionData)) {
#'     dnaSeq <- as.character(Biostrings::getSeq(BSgenome.Hsapiens.UCSC.hg38, 
#'                                               unlist(regionData[row, 'seqnames']), 
#'                                               unlist(regionData[row, 'start']), 
#'                                               unlist(regionData[row, 'end']) ))
#'     regionData[row, 'DNAseq'] = dnaSeq
#'     setTxtProgressBar(pb, row)
#'   }
#'   close(pb)
#'   
#'   return (regionData)
#' }

# subsetPeaks <- function (ATACseqData, subsetBy){
#   
#   #Empty vector to store variance per region
#   rowSD <- c()
#   pb = txtProgressBar(min = 0, max = nrow (ATACseqData), style = 3, width = 50) 
#   #Calculate variance per region and fill in vector
#   for (i in seq(1, nrow(ATACseqData), 3457)){
#     rowSD <- c(rowSD, rowSds(as.matrix(ATACseqData[i:(i+3456),])))
#     setTxtProgressBar(pb, i)
#   }
#   close(pb)
#   
#   #Subset based on variance 
#   ATACsubset <- ATACseqData[which(rowSD> summary(rowSD)[subsetBy]), ]
#   
#   return (ATACsubset)
# }

### SCRIPT

#Read input files with peaks
ATACseqData <- readRDS('data/SNAREseqData/Zhang_BICCN-H_20190523-20190611_huMOp_Final_AC_Peaks.RDS')
metadata <- read.delim('data/SNAREseqData/Zhang_BICCN-H_20190523-20190611_huMOp_Final_Sample_Metadata.txt')

#subset peaks 
#ATACseqData = subsetPeaks (ATACseqData, 'Median')
#Read from file
#ATACseqData <- readRDS('data/ATACpeakSubsets/ATACsubsetMedian.RDS')

#Read files with differentially accesible peaks
#da_peaks_NN = read_xlsx('data/da/da_peaks_NN.xlsx')
#da_peaks_GB = read_xlsx('data/da/da_peaks_GB.xlsx')
#da_peaks_GL = read_xlsx('data/da/da_peaks_GL.xlsx')

#daPeaks <- list('Non-Neuronal' = da_peaks_NN,
#                 'GABAergic' = da_peaks_GB,
#                 'Glutamatergic' = da_peaks_GL)

### Get matrix with all DNA regions

#DNAregions = data.frame(matrix(nrow = nrow(ATACseqData), ncol = 4 ))
#colnames(DNAregions) = c('seqnames', 'start', 'end', 'width')

#pb = txtProgressBar(min = 0, max = nrow (ATACseqData), style = 3, width = 50) 

#for (row in 1:nrow(ATACseqData)){
#  #Fill in the dataframe with the location information
#  location = row.names(ATACseqData)[row]
#  DNAregions [row, 'region'] = location
#  DNAregions [row, 'seqnames'] = strsplit(location, ":")[[1]][1]
#  DNAregions [row, 'start'] = as.numeric(strsplit(strsplit(location, ":")[[1]][2], '-')[[1]][1])
#  DNAregions [row, 'end'] = as.numeric(strsplit(strsplit(location, ":")[[1]][2], '-')[[1]][2])
#  DNAregions [row, 'width'] = DNAregions[row, 'end'] - DNAregions [ row,'start'] +1
  
#  setTxtProgressBar(pb, row)
  
#}
#close(pb)

#Remove peaks in X, Y chromosomes
#DNAregions <- subset (DNAregions, DNAregions$seqnames!= 'chrX' & DNAregions$seqnames!= 'chrY')

#Add sequences of regions
#DNAregions = addDNAsequences(DNAregions)

#Save dataframe
#if (!dir.exists(file.path ('data'))){
#  dir.create(file.path ('data'))
#}
#write.csv(DNAregions, file.path('data/DNAregions.xlsx'), row.names=FALSE)


##Read file with DNA sequences of all peaks
DNAregions <- read.csv('data/DNAregions.xlsx')


#normalize reads by length of region
#Get regions that are not in chr X or Y
rowsToKeep = Filter(function(x) !any(grepl("chrX", x)), row.names(ATACseqData))
rowsToKeep = Filter(function(x) !any(grepl("chrY", x)), rowsToKeep)

normalizedReads = ATACseqData[rowsToKeep, ]/ unlist(DNAregions['width'])

#### Get input dataframe

columns = c('peaks', 'cellType')
inputDF = data.frame(matrix(nrow = as.numeric(ncol(ATACseqData)),ncol = 2))
colnames(inputDF) = columns

pb = txtProgressBar(min = 0, max = ncol (ATACseqData), style = 3, width = 50) 

for (cell in 1:ncol (ATACseqData)){
  #Get peaks which have reads for the particular cell
  regionsWithReads = row.names(ATACseqData)[which (ATACseqData[, cell]> quantile (normalizedReads[ , cell], probs = 0.9))]
  regionsWithReads = strsplit(regionsWithReads, " ")
  
  #Extract cell type from metadata
  cellType = metadata[metadata$sample_name == colnames(ATACseqData)[cell], c('level1')]
  #Get differentially accessible peaks for that cell type
  #daRegions = strsplit(unlist( daPeaks[cellType][[1]]['peak']), " ")
  
  #Subset the regions so that only the differentially accesible peaks are left
  #peaksToKeep = intersect(regionsWithReads, daRegions)
  peaksToKeep = regionsWithReads
  #Remove regions in X and Y chromosomes
  peaksToKeep <- Filter(function(x) !any(grepl("chrX", x)), peaksToKeep)
  peaksToKeep <- Filter(function(x) !any(grepl("chrY", x)), peaksToKeep)
  
  #Extract DNA sequences of peaks in a string
  DNAsequences = paste (DNAregions[DNAregions$region %in% peaksToKeep, 'DNAseq'], collapse = " ")
  
  #Fill in the dataframe
  inputDF[cell, 'peaks'] = DNAsequences
  inputDF[cell, 'cellType'] = cellType
  
  setTxtProgressBar(pb, cell)
}
close(pb)

#Remove missing values
inputDF<-subset(inputDF, inputDF$peak!="")


#Save dataframe
if (!dir.exists(file.path ('data'))){
  dir.create(file.path ('data'))
}
write.csv(inputDF, file.path('data/ATACpeaksPerCellAllPeaks.csv'), row.names=FALSE)

