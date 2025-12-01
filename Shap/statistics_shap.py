#Gives a table of percentages of overlapping genes between shap, only first 100 genes + info of overlapping genes in models or tissues or both
import pandas as pd
import csv
from functools import reduce

models = ['AltumAge', 'Transformer']
tissue_types = ['All', 'Blood', 'Brain', 'Buccal', 'Saliva', 'Skin']

shap_list = {}

#importing all gene importance data
for model in models:
    shap_list[model] = {}
    for tissue in tissue_types:
        values = pd.read_csv(f'All_models/SHAP/mean_shap_per_gene_{model}_80:20_{tissue}.csv')
        gene_series = values['UCSC_RefGene_Name'].astype(str).head(50)  # Take first 50 genes
        
        # Split gene names by ;
        genes_split = []
        for item in gene_series:
            genes_split.extend(item.split(";"))
        
        # Remove duplicates by converting to a set and back to list
        unique_genes = list(set(genes_split))
        
        shap_list[model][tissue] = unique_genes
        print(f"{model} {tissue} - {len(unique_genes)} unique genes")

# Preparing keys 
keys = [(model, tissue) for model in shap_list for tissue in shap_list[model]]
gene_sets = {(model, tissue): set(shap_list[model][tissue]) for model, tissue in keys}

#writing csv of gene overlap percentages
with open('All_models/SHAP/statistics_shap.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([''] + [f'{m}_{t}' for m, t in keys])
    
    for k1 in keys:
        row = [f'{k1[0]}_{k1[1]}']
        set1 = gene_sets[k1]
        for k2 in keys:
            set2 = gene_sets[k2]
            if set1 or set2:
                perc = len(set1.intersection(set2)) / len(set1.union(set2)) * 100
            else:
                perc = 0.0
            row.append(f'{perc:.2f}')
        writer.writerow(row)



#genes common to all tissues and models
all_gene_sets = [set(shap_list[model][tissue]) for model in shap_list for tissue in shap_list[model]]
common_genes = list(reduce(lambda a, b: a.intersection(b), all_gene_sets))

#genes common to all tissues by model
for model in models:
    gene_sets = [set(shap_list[model][tissue]) for tissue in tissue_types]
    common_genes = list(reduce(lambda a, b: a.intersection(b), gene_sets))
    filename = f'All_models/SHAP/common_genes_{model}.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(common_genes))
    
    print(f"{len(common_genes)} common genes written to {filename}")

#genes common to both models by tissue
for tissue in tissue_types:
    gene_sets = [set(shap_list[model][tissue]) for model in models]
    common_genes = list(reduce(lambda a, b: a.intersection(b), gene_sets))
    filename = f'All_models/SHAP/common_genes_in_all_models_{tissue}.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(common_genes))
    
    print(f"{len(common_genes)} common genes written to {filename}")