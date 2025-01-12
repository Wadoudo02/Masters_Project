#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:25:32 2024

@author: wadoudcharbak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from utils import *

plt.style.use(hep.style.CMS)

# Constants
total_lumi = 7.9804
target_lumi = 300
plot_fraction = True  # Toggle for normalizing histograms to fractions

# Processes to plot
procs = {
    "ggH": ["ggH", "cornflowerblue"],
    "VBF": ["VBF", "red"],
    "VH": ["VH", "orange"],
    "background" : ["Background", "black"],
    "ttH": ["ttH", "mediumorchid"],
}

# Features to use in the XGBoost model
features = ["deltaR", "HT", "n_jets", "max_btag", "j0_pt", "delta_phi_gg", "j0_btagB", "j1_btagB"] 
features = [f"{feature}_sel" for feature in features]

# Load and preprocess data for all processes
dfs = {}
for proc, (label, color) in procs.items():
    print(f" --> Loading process: {proc}")
    dfs[proc] = pd.read_parquet(f"{sample_path}/{proc}_processed_selected.parquet")
    dfs[proc] = dfs[proc][(dfs[proc]["mass_sel"] == dfs[proc]["mass_sel"])]  # Remove NaNs
    dfs[proc]['plot_weight'] *= target_lumi / total_lumi  # Reweight to target lumi
    dfs[proc]['pt_sel'] = dfs[proc]['pt-over-mass_sel'] * dfs[proc]['mass_sel']

# Prepare combined dataset for XGBoost
dfs["ttH"]["label"] = 1  # Label ttH as signal
non_ttH_data = pd.concat(
    [dfs[proc] for proc in procs if proc != "ttH"], ignore_index=True
)
non_ttH_data["label"] = 0  # Label non-ttH as background

# Combine ttH and non-ttH
df_combined = pd.concat([dfs["ttH"], non_ttH_data], ignore_index=True)

# Remove rows with invalid weights (weights <= 0)
invalid_weights = df_combined["plot_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    df_combined = df_combined[~invalid_weights]

# Split into train and test sets
X = df_combined[features]
y = df_combined["label"]
weights = df_combined["plot_weight"]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=43
)

# Train the XGBoost classifier
print(" --> Training XGBoost Classifier...")
clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
clf.fit(X_train, y_train, sample_weight=w_train)

# Evaluate the classifier on test data
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # Probabilities for ROC
accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
print(f"Classifier Accuracy: {accuracy:.4f}")

# Compute ROC curve and AUC for test data
fpr, tpr, _ = roc_curve(y_test, y_proba, sample_weight=w_test)
roc_auc = auc(fpr, tpr)
print(f"Test ROC AUC: {roc_auc:.4f}")

# Compute ROC curve and AUC for training data
y_train_proba = clf.predict_proba(X_train)[:, 1]  # Probabilities for ROC
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba, sample_weight=w_train)
roc_auc_train = auc(fpr_train, tpr_train)
print(f"Train ROC AUC: {roc_auc_train:.4f}")

# Plot ROC curves for train and test data
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color="green", lw=2, label=f"Train ROC (AUC = {roc_auc_train:.4f})")
plt.plot(fpr, tpr, color="blue", lw=2, label=f"Test ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: Train vs Test")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Plot classifier output scores
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

# Histogram for ttH (signal)
mask_signal = y_test == 1
ax.hist(
    y_proba[mask_signal],
    bins=50,
    range=(0, 1),
    density=plot_fraction,
    histtype="step",
    color="mediumorchid",
    linewidth=2,
    label="ttH (Signal)",
)

# Histogram for non-ttH (background)
mask_background = y_test == 0
ax.hist(
    y_proba[mask_background],
    bins=50,
    range=(0, 1),
    density=plot_fraction,
    histtype="step",
    color="black",
    linewidth=2,
    label="non-ttH (Background)",
)

ax.set_xlabel("XGBoost Classifier Output")
ax.set_ylabel("Fraction of Events" if plot_fraction else "Events")
ax.legend(loc="best")
hep.cms.label("", com="13.6", lumi=target_lumi, lumi_format="{0:.2f}", ax=ax)
plt.tight_layout()
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

#%%


from xgboost import plot_tree


# Plot the first tree in the XGBoost model
fig, ax = plt.subplots(figsize=(20, 10), dpi = 1000)  # Set figure size
plot_tree(clf, num_trees=0, ax=ax)  # num_trees=0 selects the first tree
plt.show()