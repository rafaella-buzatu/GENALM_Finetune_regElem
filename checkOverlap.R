library(rtracklayer)
library(stringr)
library(writexl)
library(readxl)
library(R.utils)

read.delim("data/hg38_sites_v1.gff", header=F, comment.char="#") -> gff

overlappingDF = data.frame((matrix(ncol = 3, nrow = 0)))
colnames(overlappingDF) <- c("region", "randomized", "overlapCount")
pathToBedFiles = 'outputs/results/fineTunedNoDup/overlapTFs/bedFiles'
files = list.files(path = pathToBedFiles)

findOverlapping <-function (bert_region, gff){
  
  overlaps = 0
  for (i in 1:length(bert_region)){
    overlapping <- (gff['V1'] == levels(bert_region[i]@seqnames@values)) & (gff['V4'] <= bert_region[i]@ranges@start) & (gff['V5'] >= bert_region[i]@ranges@start)
    overlaps = overlaps + sum(overlapping)
  }
  
  return (overlaps)
}

pb = txtProgressBar(min = 0, max = length(files), style = 3, width = 50) 
i = 1
for (pathToBedFile in files){
  bert_region = import.bed(paste(pathToBedFiles, pathToBedFile, sep = '/'))

  overlaps = findOverlapping(bert_region, gff)
  
  if (grepl('random', pathToBedFile)){
    randomized = 1
  } else {
    randomized = 0
  }
  
  newEntry <- list(region = str_split(str_split(pathToBedFile, '.bed')[[1]][1], '_')[[1]][1], 
                   randomized = randomized, 
                   overlapCount = overlaps
                   )

  ### If you want to get all the columns of overlapped regions:
  overlappingDF = rbind (overlappingDF, as.data.frame(newEntry))
  setTxtProgressBar(pb, i)
  i= i+1
}
close(pb)
write_xlsx(overlappingDF,'outputs/results/fineTunedNoDup/overlapTFs/overlappingDF.xlsx')