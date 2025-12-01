# Downloader for TCGA methylation studies
library(TCGAbiolinks)

path_destination <- "./data_cancer"

# all_projects <- getGDCprojects()
# all_projects <- all_projects[grepl("TCGA", all_projects$id), ]

projects <- TCGAbiolinks:::getGDCprojects()$project_id
projects <- projects[grepl('^TCGA',projects,perl=T)]

# target_projects <- c("TCGA-BRCA", "TCGA-DLBC", "TCGA-MESO", "TCGA-THYM", "TCGA-PCPG",
#              "TCGA-GBM", "TCGA-LGG", "TCGA-HNSC", "TCGA-THCA", "TCGA-SARC",
#              "TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH", "TCGA-BLCA", "TCGA-CHOL",
#              "TCGA-OV", "TCGA-LIHC", "TCGA-PRAD", "TCGA-ESCA", "TCGA-LUSC",
#              "TCGA-LUAD", "TCGA-PAAD", "TCGA-COAD", "TCGA-READ")

for (proj in projects) {
  query <- GDCquery(
    project = proj,
    data.category = "DNA Methylation",
    platform = "Illumina Human Methylation 450",
    data.type = "Methylation Beta Value"
  )

  # common.patients <- substr(getResults(query, cols = "cases"), 1, 12)

  # Download selected TCGA studies, only a certain number of patients per study
  # query <- GDCquery(
  #   project = proj,
  #   data.category = "DNA Methylation",
  #   platform = "Illumina Human Methylation 450",
  #   data.type = "Methylation Beta Value",
  #   barcode = common.patients[1:10]
  # )

  tryCatch(GDCdownload(query = query, directory = path_destination, files.per.chunk = 5),
           error = function(e) GDCdownload(query = query, method = "client", directory = path_destination))
  cat(proj, " downloaded.\n")
}

# Move beta txt-files in the parent directory and delete unnecessary directories
dir_list <- list.dirs(path = path_destination, recursive = FALSE)
for (dir in dir_list) {
  files <- list.files(path = dir, pattern = ".txt", recursive = TRUE)
  # Move the files
  for (file in files) {
    clean_file <- gsub(".*/", "", file)
    source_path <- paste(dir, file, sep = "/")
    target_path <- paste(dir, clean_file, sep = "/")
    # cat("From : ", source_path, "\n", "to : ", target_path, "\n")
    file.rename(from = source_path, to = target_path)
    # Check if the file was successfully moved
    if (file.exists(target_path)) {
      cat("File moved successfully.\n")
    } else {
      cat("File move failed.\n")
    }
  }
  # Check if there are any .txt files in the directory
  dir_delete <- list.dirs(path = dir, recursive = FALSE)
  txt_files <- list.files(dir_delete, pattern = "\\.txt$", full.names = TRUE)
  if (length(txt_files) == 0) {
    # No .txt files found, delete the directory
    unlink(dir_delete, recursive = TRUE)
    cat("Directory deleted successfully.\n")
  } else {
    cat("Directory contains .txt files and will not be deleted.\n")
  }
}
