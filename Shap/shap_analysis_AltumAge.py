#exports .npy file with mean shap values per CpG for all tissues. Currently build for tensorflow AltumAge

import tensorflow as tf

import shap
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

import pickle

cuda = os.getenv("CUDA")
if cuda is None:
    cuda = "cuda:0"

save_dir = "All_models/SHAP"

#get cpgs in same order as for everything else (exported from All_models)
CPG_PATH   = "Transformer/cpgs.npy"
cpgs           = np.load(CPG_PATH, allow_pickle=True)

tissue_types=['All','Blood','Brain','Buccal','Saliva','Skin']

#getting data
for tissue in tissue_types:
    print(f'Working on {tissue} model')

    model = tf.keras.models.load_model(f'All_models/AltumAge_models/AltumAge_npj_80:20_{tissue}.h5',custom_objects={'mse': tf.keras.metrics.MeanSquaredError()})
    DATA_DIR = f"AltumAgeNew/data_by_tissue/{tissue}/"
    TRAIN_PATH = os.path.join(DATA_DIR, f"train_{tissue}.pkl")
    train_val_data = pd.read_pickle(TRAIN_PATH)

    #cleaning up missing ages
    train_data = train_val_data.dropna(subset=['age'])

    #excluding metadata
    meta_cols = ['age', 'tissue_type', 'tissue_idx', 'dataset', 'gender']
    feature_cols = cpgs
    
    #scaling
    scaler = RobustScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_data[feature_cols]),
        index=train_data.index,
        columns=feature_cols
    )

    #sets for shap analysis
    X_background=train_scaled.iloc[0:100,:].values
    X_test=train_scaled.iloc[100:200,:].values

    

    print('Doing analysis...')
    #shap for Altum Age
    explainer = shap.GradientExplainer(model, X_background)
    shap_values = explainer(X_test)

    filepath = os.path.join(save_dir, f"shap_AltumAge_80:20_{tissue}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(shap_values, f)

 