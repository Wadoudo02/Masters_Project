#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:15:43 2025

@author: wadoudcharbak
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
import pandas as pd
from utils import *


True_SMEFT_Weights = True


# Generate a single set of features
n_features = 10
n_samples = 33362
mean_features = np.zeros(n_features)
cov_features = np.eye(n_features)  # Identity covariance
X = np.random.multivariate_normal(mean_features, cov_features, size=n_samples)

# Create a DataFrame for features
df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])

# Duplicate the feature set for SM and SMEFT
df_sm = df.copy()
df_smeft = df.copy()

if True_SMEFT_Weights:
    total_lumi = 7.9804
    target_lumi = 300
    
    df_tth = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
    df_tth = df_tth[(df_tth["mass_sel"] == df_tth["mass_sel"])]  # Remove NaNs in the selected variable
    df_tth['plot_weight'] *= target_lumi / total_lumi  # Reweight to target lumi
    
    invalid_weights = df_tth["plot_weight"] <= 0
    if invalid_weights.sum() > 0:
        print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
        df_tth = df_tth[~invalid_weights]
        
    df_tth = df_tth.reset_index()

    def SMEFT_weights(proc_data, cg, ctg, name="weights", quadratic=False):
        proc_data[name] = proc_data['plot_weight'] * (1 + proc_data['a_cg'] * cg + proc_data['a_ctgre'] * ctg)
        if quadratic:
            proc_data[name] += (
                (cg ** 2) * proc_data["b_cg_cg"]
                + cg * ctg * proc_data["b_cg_ctgre"]
                + (ctg ** 2) * proc_data["b_ctgre_ctgre"]
            )
        return proc_data[name]

    Quadratic = True

    cg = 0.3
    ctg = 0.69
    
    df_sm['weight'] = df_tth["plot_weight"]  # Uniform weights for SM
    df_smeft['weight'] = SMEFT_weights(df_tth.copy(), cg=cg, ctg=ctg, name="weight", quadratic=Quadratic) # Mapping function for SMEFT

    # Normalise weights for SM
    df_sm['weight'] /= (df_sm['weight'].sum()/10000)
    
    # Normalise weights for SMEFT
    df_smeft['weight'] /= (df_smeft['weight'].sum()/10000)
    
else:
    # Assign weights
    df_sm['weight'] = 1  # Uniform weights for SM
    df_smeft['weight'] = 1 + 0.5 * df_smeft['feature_2'] ** 2  # Mapping function for SMEFT
    # Normalise weights for SM
    df_sm['weight'] /= (df_sm['weight'].sum()/10000)
    
    # Normalise weights for SMEFT
    df_smeft['weight'] /= (df_smeft['weight'].sum()/10000)
    

# Assign labels
df_sm['label'] = 0  # SM label
df_smeft['label'] = 1  # SMEFT label

# Combine the datasets
df_combined = pd.concat([df_sm, df_smeft])

# Features, labels, and weights
X = df_combined[[f"feature_{i+1}" for i in range(n_features)]]
y = df_combined['label']
weights = df_combined['weight']

# Split into training and test sets
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=0.3, random_state=42)

# Train the XGBoost classifier
clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
clf.fit(X_train, y_train, sample_weight=w_train)

# Predict probabilities
y_train_proba = clf.predict_proba(X_train)[:, 1]
y_test_proba = clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC for training data
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba, sample_weight=w_train)
roc_auc_train = auc(fpr_train, tpr_train)

# Compute ROC curve and AUC for test data
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba, sample_weight=w_test)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curves
plt.figure(figsize=(12, 8))
plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {roc_auc_train:.4f})", color="green")
plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_test:.4f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for SMEFT vs SM Classification")
plt.legend()
plt.grid()
plt.show()

# Plot classifier output histogram
plt.figure(figsize=(12, 8))
'''
plt.hist(
    y_train_proba[y_train == 1], bins=50, range=(0, 1), histtype="step", linewidth=2,
    label="Train SMEFT", color="orange", density=True
)
plt.hist(
    y_train_proba[y_train == 0], bins=50, range=(0, 1), histtype="step", linewidth=2,
    label="Train SM", color="red", density=True
)
'''
plt.hist(
    y_test_proba[y_test == 1], bins=50, range=(0, 1), histtype="step", linewidth=2,
    label="Test SMEFT", color="blue", density=True
)
plt.hist(
    y_test_proba[y_test == 0], bins=50, range=(0, 1), histtype="step", linewidth=2,
    label="Test SM", color="green", density=True
)
plt.xlabel("Classifier Output")
plt.ylabel("Density")
plt.title("Classifier Output Histogram")
plt.legend(loc="best")
plt.grid()
plt.show()


# Generate hard predictions using the default threshold of 0.5
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Compute confusion matrices for training and test sets
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

'''
# Plot confusion matrix for training data
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["SM", "SMEFT"])
disp_train.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Training Data")
plt.show()
'''
# Plot confusion matrix for test data
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["SM", "SMEFT"])
disp_test.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Test Data")
plt.show()