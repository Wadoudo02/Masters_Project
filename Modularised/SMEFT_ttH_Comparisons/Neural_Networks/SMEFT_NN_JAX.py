#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 15:21:21 2025

@author: wadoudcharbak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import flax.struct
import optax
from typing import Any, Callable

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
df_tth = df_tth[(df_tth["mass_sel"] == df_tth["mass_sel"])]  # Remove NaNs
df_tth['plot_weight'] *= target_lumi / total_lumi  # Reweight to target lumi
df_tth['pt_sel'] = df_tth['pt-over-mass_sel'] * df_tth['mass_sel']

invalid_weights = df_tth["plot_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    df_tth = df_tth[~invalid_weights]

# Split into two halves
df_sm, df_smeft = train_test_split(df_tth, test_size=0.5, random_state=seed_number)
df_smeft = add_SMEFT_weights(df_smeft, cg=cg, ctg=ctg, name="plot_weight", quadratic=Quadratic)

# Normalise plot_weight
df_smeft["plot_weight"] /= df_smeft["plot_weight"].sum()
df_sm["plot_weight"] /= df_sm["plot_weight"].sum()

df_smeft["plot_weight"] *= 10**4
df_sm["plot_weight"] *= 10**4

# Label SMEFT vs SM
df_smeft['label'] = 1  # SMEFT
df_sm['label'] = 0     # SM

# Combine
df_combined = pd.concat([df_smeft, df_sm])

if PlotInputFeatures:
    # Plot input features if needed
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

# -----------------------------------------------------------
# Everything above this line remains unchanged.
# -----------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# For HPC-style plotting
plt.style.use(hep.style.CMS)

# -------------------------------------------------------------------------
# 1. Data Loading & Preprocessing
#    (Adapt the lines below to your actual dataset)
# -------------------------------------------------------------------------
# Suppose you already have a combined DataFrame called df_combined
# with columns "deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel"
# for input features, "label" for class, and "plot_weight" for weights.

# e.g.:
# from your_code import df_combined, features  # or reuse your code above


# Train/test split
X_train_np, X_test_np, y_train_np, y_test_np, w_train_np, w_test_np = train_test_split(
    X, y, weights, test_size=0.3, random_state=42
)

# Standard-scale the features: mean=0, std=1
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np  = scaler.transform(X_test_np)

# Convert to JAX arrays
X_train = jnp.array(X_train_np, dtype=jnp.float32)
y_train = jnp.array(y_train_np, dtype=jnp.float32)
w_train = jnp.array(w_train_np, dtype=jnp.float32)

X_test  = jnp.array(X_test_np,  dtype=jnp.float32)
y_test  = jnp.array(y_test_np,  dtype=jnp.float32)
w_test  = jnp.array(w_test_np,  dtype=jnp.float32)

# -------------------------------------------------------------------------
# 2. Define the MLP Model
#    Here we build a multi-layer MLP that uses:
#      - Multiple hidden layers with BatchNorm + SiLU + Dropout
#      - Output layer with 1 neuron + Sigmoid for binary classification
# -------------------------------------------------------------------------
class DeepMLP(nn.Module):
    hidden_dims: tuple

    @nn.compact
    def __call__(self, x, train: bool = True):
        for hdim in self.hidden_dims:
            x = nn.Dense(hdim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.silu(x)  # SiLU activation
            x = nn.Dropout(rate=0.3, deterministic=not train)(x)
        # Output
        x = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x)
        return nn.sigmoid(x)

# -------------------------------------------------------------------------
# 3. Custom Train State for BN + parameters
# -------------------------------------------------------------------------
@flax.struct.dataclass
class TrainState:
    """Similar to Flax's TrainState, extended to store batch_stats."""
    step: int
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    params: Any
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState
    batch_stats: Any

    @classmethod
    def create(cls, apply_fn, params, tx, batch_stats):
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            batch_stats=batch_stats
        )

    def apply_gradients(self, grads, new_batch_stats=None):
        if new_batch_stats is None:
            new_batch_stats = self.batch_stats
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            batch_stats=new_batch_stats
        )

# -------------------------------------------------------------------------
# 4. Create Train State
#    We'll define a cosine decay schedule as an example.
# -------------------------------------------------------------------------
def create_train_state(rng, model, input_dim, hidden_dims):
    # Example: initial learning rate is 1e-3
    init_learning_rate = 1e-3
    # We'll do 100 epochs => let's define a schedule for that
    # Cosine decay from 1e-3 -> 1e-6 over 100 epochs
    scheduler = optax.cosine_decay_schedule(init_value=init_learning_rate,
                                            decay_steps=100,
                                            alpha=1e-3)  # final ~1e-6

    tx = optax.adam(scheduler)

    dummy_input = jnp.ones((1, input_dim), dtype=jnp.float32)
    variables = model.init(rng, dummy_input, train=True)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )

# -------------------------------------------------------------------------
# 5. Loss Function and train/eval steps
# -------------------------------------------------------------------------
def weighted_bce_loss(preds, labels, weights):
    """Binary cross-entropy with event-level weights."""
    eps = 1e-8
    bce = - (labels * jnp.log(preds + eps) + (1.0 - labels) * jnp.log(1.0 - preds + eps))
    return jnp.mean(bce * weights)

@jax.jit
def train_step(state, rng, x_batch, y_batch, w_batch):
    """One gradient update step."""
    def loss_fn(params, batch_stats):
        preds, updated_model_state = state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            x_batch,
            train=True,
            rngs={"dropout": rng},  # for dropout
            mutable=["batch_stats"]
        )
        loss_value = weighted_bce_loss(preds, y_batch, w_batch)
        return loss_value, updated_model_state["batch_stats"]

    (loss_value, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.batch_stats
    )
    new_state = state.apply_gradients(grads=grads, new_batch_stats=new_batch_stats)
    return new_state, loss_value

@jax.jit
def eval_step(state, x, y, w):
    """Compute the loss without updating batch stats."""
    preds = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        x,
        train=False,
        mutable=False
    )
    loss_value = weighted_bce_loss(preds, y, w)
    return loss_value, preds

# -------------------------------------------------------------------------
# 6. Data Loader for Mini-Batching
# -------------------------------------------------------------------------
def data_loader(X, Y, W, batch_size=128, shuffle=True):
    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(idx)
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = idx[start_idx:start_idx + batch_size]
        yield X[batch_indices], Y[batch_indices], W[batch_indices]

# -------------------------------------------------------------------------
# 7. Build & Train
# -------------------------------------------------------------------------
rng = jax.random.PRNGKey(seed_number)

# Example architecture: 3 hidden layers of sizes 64 -> 128 -> 64
hidden_dims = (64, 128, 64)
model = DeepMLP(hidden_dims=hidden_dims)

# Create the train state
rng, init_rng = jax.random.split(rng)
state = create_train_state(init_rng, model, X_train.shape[1], hidden_dims)

epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Training loop
    batch_losses = []
    for x_b, y_b, w_b in data_loader(X_train, y_train, w_train, batch_size=128, shuffle=True):
        rng, subkey = jax.random.split(rng)
        state, loss_val = train_step(state, subkey, x_b, y_b, w_b)
        batch_losses.append(loss_val)
    epoch_train_loss = float(np.mean(batch_losses))
    train_losses.append(epoch_train_loss)

    # Validation
    test_loss, _ = eval_step(state, X_test, y_test, w_test)
    test_losses.append(float(test_loss))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.6f}, Test Loss: {float(test_loss):.6f}")

# -------------------------------------------------------------------------
# 8. Plot Loss
# -------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(range(epochs), train_losses, label="Train Loss")
plt.plot(range(epochs), test_losses,  label="Test  Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------
# 9. Evaluate: Predictions, AUC, Plots
# -------------------------------------------------------------------------
_, y_proba_test = eval_step(state, X_test, y_test, w_test)
_, y_proba_train = eval_step(state, X_train, y_train, w_train)

# Convert from JAX to NumPy
y_proba_test = np.array(y_proba_test).squeeze()
y_proba_train = np.array(y_proba_train).squeeze()

y_test_np  = np.array(y_test)
y_train_np = np.array(y_train)
w_test_np  = np.array(w_test)
w_train_np = np.array(w_train)

# Compute ROC
fpr_test, tpr_test, _ = roc_curve(y_test_np, y_proba_test, sample_weight=w_test_np)
roc_auc_test = auc(fpr_test, tpr_test)

fpr_train, tpr_train, _ = roc_curve(y_train_np, y_proba_train, sample_weight=w_train_np)
roc_auc_train = auc(fpr_train, tpr_train)

print(f"Train ROC AUC: {roc_auc_train:.4f}")
print(f"Test  ROC AUC: {roc_auc_test:.4f}")

# Plot Network Outputs
plt.figure(figsize=(8, 6))
bins = 50
_ = plt.hist(y_proba_test[y_test_np == 1], bins=bins, range=(0, 1), alpha=0.5,
             weights=w_test_np[y_test_np == 1], label="SMEFT (Test)", histtype="step")
_ = plt.hist(y_proba_test[y_test_np == 0], bins=bins, range=(0, 1), alpha=0.5,
             weights=w_test_np[y_test_np == 0], label="SM (Test)", histtype="step")
plt.xlabel("MLP Output")
plt.ylabel("Weighted Events")
plt.title("Classifier Output Distribution")
plt.legend()
hep.cms.label("SMEFT vs SM", data=True, lumi=300, com=13.6)
plt.show()

# Plot ROC
plt.figure(figsize=(7, 6))
plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC={roc_auc_train:.3f})", color="green")
plt.plot(fpr_test,  tpr_test,  label=f"Test ROC (AUC={roc_auc_test:.3f})",  color="blue")
plt.plot([0,1],[0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()