# Tissue-specific Age Prediction Based on DNA Methylation

This project is based on the paper 'A pan-tissue DNA-methylation epigenetic clock based on deep learning' by de Lima Camillo et al. (https://doi.org/10.1038/s41514-022-00085-y).
It includes the AltumAge model and ElasticNet from the paper and an additional transformer model, These models are compared in regards to their performance in ge prediction on tissue specific methylation datasets.
AltumAge and the transformer are further analyzed to determine the most important CpG sites and therefore genes for age prediction in the specific tissue.

R scripts and Neural Network architecture provided by Nadine Kurz.

## Overview:

### Preprocessing:
File management of the data downloaded from https://drive.google.com/drive/folders/1RH2JYmhOmsScaj_WMQfVwYjubkNTh5Oq?usp=sharing_eip&ts=60c67fb4. Includes the renaming of the files to better distinguish between train and test, determining the data availability per tissue based on the metadata and the sorting of tissues based on the metadata.

### AltumAge:
Further preprocessing of the data per tissue, ElasticNet, the transformer and AltumAge. It inlcudes the performance analysis of all models and the SHAP analysis for the transformer. Also includes trained models.
Also includes unfinished notebook for AltumAge prediction of TCGA cancer data

### Model: 
The unadjusted transformer model 

### SHAP: 
Includes the shap analysis for AltumAge (Transformer is in AltumAge_split.ipynb). Assign the CpGs to the genes, and compare the most important genes per model.

## Usage:

Specific usage information is included at the top of the .py files
