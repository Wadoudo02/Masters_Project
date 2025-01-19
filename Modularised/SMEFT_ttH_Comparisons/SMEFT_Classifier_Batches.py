#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:30:04 2025

@author: wadoudcharbak
"""


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from utils import *

# Plotting style
plt.style.use(hep.style.CMS)

# Constants
total_lumi = 7.9804
target_lumi = 300

plot_fraction = True
Quadratic = True

cg = 0.3
ctg = 0.69

# Features to use in the XGBoost model
features = ["deltaR", "HT", "n_jets", "delta_phi_gg"] 
features = [f"{feature}_sel" for feature in features]

# SMEFT weighting function
def add_SMEFT_weights(proc_data, cg, ctg, name="new_weights", quadratic=False):
    proc_data[name] = proc_data['plot_weight'] * (1 + proc_data['a_cg'] * cg + proc_data['a_ctgre'] * ctg)
    if quadratic:
        proc_data[name] += (
            (cg ** 2) * proc_data["b_cg_cg"]
            + cg * ctg * proc_data["b_cg_ctgre"]
            + (ctg ** 2) * proc_data["b_ctgre_ctgre"]
        )
    return proc_data

# Load and preprocess ttH data
print(" --> Loading process: ttH")
df_tth = pd.read_parquet(f"{sample_path}/ttH_processed_selected.parquet")
df_tth = df_tth[(df_tth["mass_sel"] == df_tth["mass_sel"])]  # Remove NaNs in the selected variable
df_tth['plot_weight'] *= target_lumi / total_lumi  # Reweight to target lumi
df_tth['pt_sel'] = df_tth['pt-over-mass_sel'] * df_tth['mass_sel']

invalid_weights = df_tth["plot_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    df_tth = df_tth[~invalid_weights]

# Split the dataset into two random halves
df_sm, df_smeft = train_test_split(df_tth, test_size=0.5, random_state=40)

df_smeft = add_SMEFT_weights(df_smeft, cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)

# Normalize the "plot_weight" for df_smeft
df_smeft["plot_weight"] /= df_smeft["plot_weight"].sum()

# Normalize the "plot_weight" for df_sm
df_sm["plot_weight"] /= df_sm["plot_weight"].sum()

df_smeft["plot_weight"] *= 10**4
df_sm["plot_weight"] *= 10**4

# Label SMEFT and SM data
df_smeft['label'] = 1  # SMEFT
df_sm['label'] = 0     # SM

# Combine datasets
df_combined = pd.concat([df_smeft, df_sm])

# Features and labels
X = df_combined[features]
y = df_combined['label']  # Target labels (SMEFT vs SM)
weights = df_combined["plot_weight"]

# Split into train and test sets
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=43
)

# Plot input feature distributions for train/test representativeness
print(" --> Plotting input feature distributions...")
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_combined,
        x=feature,
        hue="label",
        weights="plot_weight",
        bins=50,
        element="step",
        common_norm=False,
        kde=False,
        palette={0: "green", 1: "blue"}
    )
    plt.title(f"Feature: {feature} - Train/Test Splits")
    plt.xlabel(feature)
    plt.ylabel("Weighted Count")
    plt.legend(["SM", "SMEFT"])
    plt.show()

# Incremental training with batching
print(" --> Training XGBoost Classifier with batching...")
batch_size = 1000

def generate_batches(X, y, weights, batch_size):
    X, y, weights = shuffle(X, y, weights, random_state=42)
    for i in range(0, len(y), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size], weights[i:i + batch_size]

clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
first_batch = True

for X_batch, y_batch, w_batch in generate_batches(X_train, y_train, w_train, batch_size):
    if first_batch:
        clf.fit(X_batch, y_batch, sample_weight=w_batch)  # No xgb_model for the first batch
        first_batch = False
    else:
        clf.fit(X_batch, y_batch, sample_weight=w_batch, xgb_model=clf.get_booster())
        
    
# Evaluate the classifier on test data
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # Probabilities for ROC
accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
print(f"Classifier Accuracy: {accuracy:.4f}")

# Compute ROC curves and AUC scores
fpr, tpr, _ = roc_curve(y_test, y_proba, sample_weight=w_test)
roc_auc = auc(fpr, tpr)
print(f"Test ROC AUC: {roc_auc:.4f}")

y_train_proba = clf.predict_proba(X_train)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba, sample_weight=w_train)
roc_auc_train = auc(fpr_train, tpr_train)
print(f"Train ROC AUC: {roc_auc_train:.4f}")

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color="green", lw=2, label=f"Train ROC (AUC = {roc_auc_train:.4f})")
plt.plot(fpr, tpr, color="blue", lw=2, label=f"Test ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SMEFT vs SM Classification")
plt.legend()
plt.grid()
plt.show()


# Plot histograms of classifier output
y_train_pred_proba = clf.predict_proba(X_train)[:, 1]
plt.figure(figsize=(12, 8), dpi=300)

plt.hist(y_proba[y_test == 1], bins=50, range=(0, 1),  density=plot_fraction, weights = w_test[y_test == 1], histtype='step', linewidth=2, label=f"SMEFT $(c_g, c_{{tg}}) = ({cg}, {ctg})$")
plt.hist(y_proba[y_test == 0], bins=50, range=(0, 1),  density=plot_fraction, histtype='step', weights = w_test[y_test == 0], linewidth=2, label="SM $(c_g, c_{{tg}}) = (0, 0)$")
plt.xlabel("XGBoost Classifier Output")
plt.ylabel("Fraction of Events" if plot_fraction else "Events")

plt.legend(loc = "best")
hep.cms.label("Classifier SMEFT vs SM", com="13.6", lumi=target_lumi, ax=plt.gca())
plt.tight_layout()
plt.show()


# Generate hard predictions using the default threshold of 0.5
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Compute confusion matrices for training and test sets
cm_train = confusion_matrix(y_train, y_train_pred, sample_weight=w_train)
cm_test = confusion_matrix(y_test, y_test_pred, sample_weight=w_test)
'''
# Plot confusion matrix for training data
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["SM", "SMEFT"])
disp_train.plot(cmap="Blues", values_format=".2f")
plt.title("Confusion Matrix - Training Data")
plt.show()
'''

# Plot confusion matrix for test data
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["SM", "SMEFT"])
disp_test.plot(cmap="Blues", values_format=".2f")
plt.title("Confusion Matrix - Test Data")
plt.show()