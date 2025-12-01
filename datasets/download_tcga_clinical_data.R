if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("TCGAbiolinks")
library(TCGAbiolinks)


query <- GDCquery(
    project = "TCGA-ACC",
    data.category = "Clinical",
    data.type = "Clinical Supplement",
    data.format = "BCR Biotab"
)
GDCdownload(query)
clinical.BCRtab.all <- GDCprepare(query)
names(clinical.BCRtab.all)

