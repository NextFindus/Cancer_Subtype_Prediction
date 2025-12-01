library(minfi)

data_dir <- Sys.getenv(c("DATA_DIR"))
files_fullpath <- list.files(data_dir, full.names=TRUE,recursive=FALSE)

files = list()
for (file_fullpath in files_fullpath) {
    if (grepl("idat.gz",file_fullpath)) {
        file_base = gsub("_Red.idat.gz","",file_fullpath)
        file_base = gsub("_Grn.idat.gz","",file_base)
        if (!file_base %in% files) {
            files = append(files, file_base)
        }
    }
}
print(files)

for (file in files) {
    rgset = read.metharray(file)
    beta_values = getBeta(rgset)
    outfile = paste(file,"_beta",".csv",sep="")
    print(outfile)
    write.csv(beta_values,outfile)
}
