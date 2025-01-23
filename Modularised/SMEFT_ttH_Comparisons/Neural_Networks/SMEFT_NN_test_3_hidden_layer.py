#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 15:21:21 2025

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

from utils import *

import torch
import torch.nn as nn
import torch.optim as optim

# Plotting style
plt.style.use(hep.style.CMS)

# Constants
total_lumi = 7.9804
target_lumi = 300

plot_fraction = True

Quadratic = True

cg = 0.3
ctg = 0.69

PlotInputFeatures = False

LossPlotLog = True  # Toggle for log scale

seed_number = 42


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
df_sm, df_smeft = train_test_split(df_tth, test_size=0.5, random_state=seed_number)

df_smeft = add_SMEFT_weights(df_smeft, cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)

'''
# Add SMEFT weights for classification
df_smeft = add_SMEFT_weights(df_tth.copy(), cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)
df_sm = df_tth.copy()  # SM is treated as the baseline with cg=0, ctg=0
'''

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


if PlotInputFeatures:
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


# Features and labels
X = df_combined[features]
y = df_combined['label']
weights = df_combined["plot_weight"]

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)
weights_tensor = torch.tensor(weights.values, dtype=torch.float32)

# Train/test split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_tensor, y_tensor, weights_tensor, test_size=0.3, random_state=seed_number
)

# Define DataLoader for mini-batch training
train_data = torch.utils.data.TensorDataset(X_train, y_train, w_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Define neural network model with one hidden layer
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Dropout with 30%
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
        # Xavier initialisation
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.batchnorm(x)  # Batch normalisation
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return torch.sigmoid(x)

# Model, loss function, and optimiser
input_dim = X_train.shape[1]
hidden_dim = input_dim * 3  # Example: double the number of input features
model = NeuralNetwork(input_dim, hidden_dim)
criterion = nn.BCELoss(reduction='none')  # Weighted BCE loss
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)  # Reduce learning rate

# Set random seeds for reproducibility
torch.manual_seed(seed_number)
np.random.seed(seed_number)

# Training loop with epoch tracking
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y, batch_w in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        weighted_loss = (loss * batch_w).mean()

        # Backward pass and optimisation
        weighted_loss.backward()
        optimizer.step()
        epoch_loss += weighted_loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test).squeeze()
        loss_test = criterion(outputs_test, y_test)
        weighted_loss_test = (loss_test * w_test).mean().item()
    test_losses.append(weighted_loss_test)

    # Adjust learning rate
    scheduler.step()

    # Print diagnostics every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

#%%

# Plot training and test loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
if LossPlotLog:
    plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss vs. Epochs')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_proba_test = model(X_test).squeeze()
    y_proba_train = model(X_train).squeeze()

# Compute ROC and AUC for test and train
fpr_test, tpr_test, _ = roc_curve(y_test.numpy(), y_proba_test.numpy(), sample_weight=w_test.numpy())
roc_auc_test = auc(fpr_test, tpr_test)

fpr_train, tpr_train, _ = roc_curve(y_train.numpy(), y_proba_train.numpy(), sample_weight=w_train.numpy())
roc_auc_train = auc(fpr_train, tpr_train)

print(f"Test ROC AUC: {roc_auc_test:.4f}")
print(f"Train ROC AUC: {roc_auc_train:.4f}")


# Plot Histograms
plt.figure(figsize=(12, 8), dpi=300)

plt.hist(y_proba_test[y_test == 1], bins=50, range=(0, 1),  density=plot_fraction, weights = w_test[y_test == 1], histtype='step', linewidth=2, label=f"SMEFT $(c_g, c_{{tg}}) = ({cg}, {ctg})$")
plt.hist(y_proba_test[y_test == 0], bins=50, range=(0, 1),  density=plot_fraction, histtype='step', weights = w_test[y_test == 0], linewidth=2, label="SM $(c_g, c_{{tg}}) = (0, 0)$")
plt.xlabel("Neural Network Output")
plt.ylabel("Fraction of Events" if plot_fraction else "Events")

plt.legend(loc = "best")
hep.cms.label("Classifier SMEFT vs SM", com="13.6", lumi=target_lumi, ax=plt.gca())
plt.tight_layout()
plt.show()


# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color="green", lw=2, label=f"Train ROC (AUC = {roc_auc_train:.4f})")
plt.plot(fpr_test, tpr_test, color="blue", lw=2, label=f"Test ROC (AUC = {roc_auc_test:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Neural Network")
plt.legend()
plt.grid()
plt.show()


#%%



# Save the model
torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "hidden_dim": hidden_dim}, "neural_network.pth")

# Compute max and min probabilities
max_proba = y_proba_test.max()
min_proba = y_proba_test.min()

# Save to a file (e.g., JSON format)
import json

proba_data = {"max_proba": float(max_proba), "min_proba": float(min_proba)}
with open("proba_values.json", "w") as json_file:
    json.dump(proba_data, json_file)

