import pandas as pd
import os
import shutil


metadata_file = "Sysbio/data_original/metadata.csv"  
source_dir = "Sysbio/data_original/processed/data_all" 
output_dir = "Sysbio/data_original/processed/data_sorted"    

# Load metadata
df = pd.read_csv(metadata_file)

# for datasets with multiple tissue types. Sorted into extra dir
tissue_counts = df.groupby("dataset")["tissue_type"].nunique()
multi_tissue_datasets = set(tissue_counts[tissue_counts > 1].index)

print(f"Multi-tissue datasets: {multi_tissue_datasets}")


dataset_to_tissue = df.groupby("dataset")["tissue_type"].first().to_dict()


for dataset in df["dataset"].unique():
    if dataset in multi_tissue_datasets:
        dest_dir = os.path.join(output_dir, "multi_tissue", dataset)
    else:
        tissue = dataset_to_tissue[dataset].replace(" ", "_")
        dest_dir = os.path.join(output_dir, tissue)
    
    os.makedirs(dest_dir, exist_ok=True)

    for suffix in ["train", "test"]:
        fname = f"{dataset}_{suffix}.pkl"
        src_path = os.path.join(source_dir, fname)
        dest_path = os.path.join(dest_dir, fname)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        else:
            print(f"Missing file: {src_path}")
