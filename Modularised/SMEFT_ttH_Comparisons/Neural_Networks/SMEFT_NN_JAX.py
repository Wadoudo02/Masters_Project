#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 15:57:51 2025

@author: wadoudcharbak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
from functools import partial
from utils import *

import time

start_time = time.time()

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


# Prepare data
X = df_combined[features].values
y = df_combined['label'].values
weights = df_combined["plot_weight"].values

# Train/test split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=seed_number
)

# Convert to JAX arrays
X_train = jnp.array(X_train, dtype=jnp.float32)
X_test = jnp.array(X_test, dtype=jnp.float32)
y_train = jnp.array(y_train, dtype=jnp.float32)
y_test = jnp.array(y_test, dtype=jnp.float32)
w_train = jnp.array(w_train, dtype=jnp.float32)
w_test = jnp.array(w_test, dtype=jnp.float32)

# Initialize random key
key = random.PRNGKey(seed_number)

def init_network_params(key, input_dim, hidden_dim):
    """Initialize neural network parameters using Xavier initialization"""
    # Hidden layer
    key, subkey = random.split(key)
    w1 = random.normal(subkey, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim)
    b1 = jnp.zeros(hidden_dim)
    
    # Output layer
    key, subkey = random.split(key)
    w2 = random.normal(subkey, (hidden_dim, 1)) * jnp.sqrt(2.0 / hidden_dim)
    b2 = jnp.zeros(1)
    
    return [(w1, b1), (w2, b2)]

def batch_norm(x, is_training):
    """Simple batch normalization implementation"""
    mean = jnp.mean(x, axis=0)
    var = jnp.var(x, axis=0)
    return (x - mean) / jnp.sqrt(var + 1e-5)

@partial(jit, static_argnums=(3,))
def forward(params, x, key, is_training):
    """Forward pass of the neural network"""
    # Unpack parameters
    (w1, b1), (w2, b2) = params
    
    # First layer with batch norm and ReLU
    x = jnp.dot(x, w1) + b1
    x = batch_norm(x, is_training)
    x = jax.nn.relu(x)
    
    # Dropout layer with conditional
    key, dropout_key = random.split(key)
    mask = random.bernoulli(dropout_key, 0.7, x.shape)  # 0.7 is keep probability
    x = jnp.where(is_training, x * mask / 0.7, x)
    
    # Output layer with sigmoid
    x = jnp.dot(x, w2) + b2
    return jax.nn.sigmoid(x).squeeze()

@partial(jit, static_argnums=(5,))
def loss_fn(params, x, y, weights, key, is_training):
    """Weighted binary cross-entropy loss"""
    predictions = forward(params, x, key, is_training)
    bce_loss = -weights * (y * jnp.log(predictions + 1e-7) + 
                          (1 - y) * jnp.log(1 - predictions + 1e-7))
    return jnp.mean(bce_loss)

@partial(jit, static_argnums=(6,))
def update(params, x, y, weights, key, optimizer_state, tx):
    """Single SGD update step"""
    loss_grad = grad(loss_fn)(params, x, y, weights, key)
    updates, new_optimizer_state = tx.update(loss_grad, optimizer_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_optimizer_state

# Initialize model
input_dim = X_train.shape[1]
hidden_dim = input_dim * 3
params = init_network_params(key, input_dim, hidden_dim)

# Setup optimizer with learning rate schedule
schedule_fn = optax.exponential_decay(
    init_value=0.01, 
    transition_steps=20,
    decay_rate=0.7
)
tx = optax.chain(
    optax.adam(learning_rate=schedule_fn),
)
optimizer_state = tx.init(params)

# Training loop
batch_size = 64
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Create batch indices
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_idx = perm[i:i + batch_size]
        batch_x = X_train[batch_idx]
        batch_y = y_train[batch_idx]
        batch_w = w_train[batch_idx]
        
        key, subkey = random.split(key)
        loss_grad = grad(lambda p: loss_fn(p, batch_x, batch_y, batch_w, subkey, True))(params)
        updates, optimizer_state = tx.update(loss_grad, optimizer_state)
        params = optax.apply_updates(params, updates)
    
    # Calculate training and test loss
    key, subkey1, subkey2 = random.split(key, 3)
    predictions_train = forward(params, X_train, subkey1, True)
    predictions_test = forward(params, X_test, subkey2, False)
    
    train_loss = -jnp.mean(w_train * (y_train * jnp.log(predictions_train + 1e-7) + 
                          (1 - y_train) * jnp.log(1 - predictions_train + 1e-7)))
    test_loss = -jnp.mean(w_test * (y_test * jnp.log(predictions_test + 1e-7) + 
                        (1 - y_test) * jnp.log(1 - predictions_test + 1e-7)))
    
    train_losses.append(float(train_loss))
    test_losses.append(float(test_loss))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Evaluation
key, eval_key = random.split(key)
y_proba_test = forward(params, X_test, eval_key, False)
key, eval_key = random.split(key)
y_proba_train = forward(params, X_train, eval_key, False)

# Convert to numpy for sklearn metrics
y_proba_test = np.array(y_proba_test)
y_proba_train = np.array(y_proba_train)
y_test_np = np.array(y_test)
y_train_np = np.array(y_train)
w_test_np = np.array(w_test)
w_train_np = np.array(w_train)

# Compute ROC and AUC
fpr_test, tpr_test, _ = roc_curve(y_test_np, y_proba_test, sample_weight=w_test_np)
roc_auc_test = auc(fpr_test, tpr_test)

fpr_train, tpr_train, _ = roc_curve(y_train_np, y_proba_train, sample_weight=w_train_np)
roc_auc_train = auc(fpr_train, tpr_train)

print(f"Test ROC AUC: {roc_auc_test:.4f}")
print(f"Train ROC AUC: {roc_auc_train:.4f}")


# End the timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")


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


