#Counts the number of rows in all files in the directories. Used to see how much data is available per tissue



import pandas as pd
import os
import shutil

output_dir = "Sysbio/data_original/processed/data_sorted" 
count_filename = "patient_count.txt"
for root, dirs, files in os.walk(output_dir):
    total_rows = 0
    for file in files:
        if file.endswith(".pkl"):
            file_path = os.path.join(root, file)
            try:
                df_pkl = pd.read_pickle(file_path)
                total_rows += len(df_pkl)
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
    

    if total_rows > 0:
        count_path = os.path.join(root, count_filename)
        with open(count_path, "w") as f:
            f.write(str(total_rows) + "\n")
        print(f"Wrote count for {root}: {total_rows}")