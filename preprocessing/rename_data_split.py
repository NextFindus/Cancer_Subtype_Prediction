#for semantics. Renames the files from dataset.pkl to dataset_train/test.pkl. Nessesary as the preprocessed datasets are already split

import os

dir_path = "Sysbio/data_original/processed/data_train"

for filename in os.listdir(dir_path):
    if filename.endswith(".pkl"):
        old_path = os.path.join(dir_path, filename)
        new_name = filename[:-4] + "_train.pkl"  
        new_path = os.path.join(dir_path, new_name)
        os.rename(old_path, new_path)
