library(Seurat)
library(Signac)
library(writexl)
library(readxl)
library (EnsDb.Hsapiens.v86)


processDAmatrix <- function (da_peaks){
  
  da_peaks['peak'] <-rownames (da_peaks)
  for (row in 1:nrow(da_peaks)){
    da_peaks[row,'peak'] = paste (c(paste (strsplit(da_peaks[row, 'peak'], "-")[[1]][c(1, 2)], collapse = ":"),
                                    strsplit(da_peaks[row, 'peak'], "-")[[1]][3]), collapse = "-")
  }
  
  return (da_peaks)
}


#Read input data
ATACpeaks <- readRDS('data/SNAREseqData/Zhang_BICCN-H_20190523-20190611_huMOp_Final_AC_Peaks.RDS')
#RNAcounts <- readRDS('data/SNAREseqData/Zhang_BICCN-H_20190523-20190611_huMOp_Final_RNA_Counts.RDS')
metadata <- read.delim('data/SNAREseqData/Zhang_BICCN-H_20190523-20190611_huMOp_Final_Sample_Metadata.txt')
#Make cell names the row names
metadata <- data.frame(metadata, row.names = 1)

#Add ATACpeaks as chromatin assay 
chromAssay <- CreateChromatinAssay( counts = ATACpeaks,
                                    sep = c(":", "-"),
                                    genome = 'hg38')
SeuratObject <- CreateSeuratObject(counts = chromAssay, assay = "ATACseq", meta.data = metadata)


annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotations) <- 'UCSC'

# add the gene information to the object
Annotation(SeuratObject) <- annotations

DefaultAssay(SeuratObject) <- 'ATACseq'


SeuratObject <- SetIdent(SeuratObject, value = 'level1')

#Get differentially accessible peaks

### NON-NEURONAL
da_peaks_NN_GB <- FindMarkers(
  object = SeuratObject,
  ident.1 = 'Non-Neuronal',
  ident.2 = 'GABAergic',
  test.use = 'LR',
  min.pct = 0.05,
  logfc.threshold = 0
)

da_peaks_NN_GL <- FindMarkers(
  object = SeuratObject,
  ident.1 = 'Non-Neuronal',
  ident.2 = 'Glutamatergic',
  test.use = 'LR',
  min.pct = 0.05,
  logfc.threshold = 0
)

#Process dataframes to have peak locations
da_peaks_NN_GB <- processDAmatrix(da_peaks_NN_GB)
da_peaks_NN_GL <- processDAmatrix(da_peaks_NN_GL)
#Bind into one dataframe
da_peaks_NN <- rbind (da_peaks_NN_GB, da_peaks_NN_GL)
#Subset based on conditions
da_peaks_NN <- subset (da_peaks_NN, avg_log2FC > 0 & p_val_adj < 0.05)
#Save in excel
write_xlsx (da_peaks_NN, file.path('data/da/da_peaks_NN.xlsx'))


### GLUTAMATERGIC
da_peaks_GL_GB <- FindMarkers(
  object = SeuratObject,
  ident.1 = 'Glutamatergic',
  ident.2 = 'GABAergic',
  test.use = 'LR',
  min.pct = 0.05,
  logfc.threshold = 0
)

da_peaks_GL_NN <- FindMarkers(
  object = SeuratObject,
  ident.1 = 'Glutamatergic',
  ident.2 = 'Non-Neuronal',
  test.use = 'LR',
  min.pct = 0.05,
  logfc.threshold = 0
)

#Process
da_peaks_GL_NN <- processDAmatrix (da_peaks_GL_NN)
da_peaks_GL_GB <- processDAmatrix (da_peaks_GL_GB)
da_peaks_GL <- rbind (da_peaks_GL_NN, da_peaks_GL_GB)
da_peaks_GL <- subset (da_peaks_GL, avg_log2FC > 0 & p_val_adj < 0.05)
#Save
write_xlsx (da_peaks_GL, file.path('data/da/da_peaks_GL.xlsx'))


### GABAergic
da_peaks_GB_NN <- FindMarkers(
  object = SeuratObject,
  ident.1 = 'GABAergic',
  ident.2 = 'Non-Neuronal',
  test.use = 'LR',
  min.pct = 0.05,
  logfc.threshold = 0
)

da_peaks_GB_GL <- FindMarkers(
  object = SeuratObject,
  ident.1 = 'GABAergic',
  ident.2 = 'Glutamatergic',
  test.use = 'LR',
  min.pct = 0.05,
  logfc.threshold = 0
)
#Process
da_peaks_GB_NN <- processDAmatrix (da_peaks_GB_NN)
da_peaks_GB_GL <- processDAmatrix (da_peaks_GB_GL)
da_peaks_GB <- rbind (da_peaks_GB_NN, da_peaks_GB_GL)
da_peaks_GB <- subset (da_peaks_GB, avg_log2FC > 0 & p_val_adj < 0.05)
#Save
write_xlsx (da_peaks_GB, file.path('data/da/da_peaks_GB.xlsx'))


#Read Excel files
#library(readxl)
#da_peaks_NN = read_xlsx('data/da/da_peaks_NN.xlsx')
#da_peaks_GB = read_xlsx('data/da/da_peaks_GB.xlsx')
#da_peaks_GL = read_xlsx('data/da/da_peaks_GL.xlsx')
