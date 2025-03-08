#%%
from EFT import * 
from utils import *
from selection import *
from SMEFT_utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import joblib
import random

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
plt.style.use(hep.style.CMS)
from Plotter import Plotter

def get_eft_comb_df(comb_df_sm, comb_df_eft, cgs, ctgs, norm_eft=True):
    comb_df_sm["cg"] = cgs
    comb_df_sm["ctg"] = ctgs
    comb_df_sm["labels"] = 0    

    comb_df_sm.reset_index(drop=True, inplace=True)

    # rand_cg_eft = np.random.uniform(*c_g_range, size=len(comb_df_eft))
    # rand_ctg_eft = np.random.uniform(*c_tg_range, size=len(comb_df_eft))

    comb_df_eft["cg"] = cgs
    comb_df_eft["ctg"] = ctgs
    comb_df_eft["labels"] = 1
    comb_df_eft["weight"] = calc_weights(comb_df_eft, cg=cgs, ctg=ctgs, weight_col="weight")

    if norm_eft:
        comb_df_sm["weight"]/=comb_df_sm["weight"].sum()
        comb_df_sm["weight"]*=10**4

        comb_df_eft["weight"] /= comb_df_eft["weight"].sum()
        comb_df_eft["weight"] *= 10**4

    comb_df = pd.concat([comb_df_sm, comb_df_eft], axis=0, ignore_index=True)
    return comb_df
plotter = Plotter()

ttH_df = get_tth_df()

#special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel","lead_pt-over-mass_sel"]

c_g_range = (-1.5, 1.5)
c_tg_range = (-1.5,1.5)

comb_df_init = pd.concat([ttH_df[var] for var in special_features+["true_weight_sel","a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]], axis=1)
comb_df_init.rename(columns={'true_weight_sel': 'weight'}, inplace=True)
comb_df_init["og_weight"] = comb_df_init["weight"].copy()

comb_df_init = comb_df_init.dropna()
mine = True
norm_eft = True
duplicate = True
copy_first = True
transform = True

#Duplicating dataset for eft and sm
# comb_df_eft = copy.deepcopy(comb_df_init)
#comb_df_sm=copy.deepcopy(comb_df_init)

#Randomly splitting same dataset into eft and sm
comb_df_sm, comb_df_eft = train_test_split(comb_df_init, test_size=0.5, random_state=25, shuffle=True)


rand_cg_sm = np.random.uniform(*c_g_range, size=len(comb_df_sm))
rand_ctg_sm = np.random.uniform(*c_tg_range, size=len(comb_df_sm))

#%%
#Creating 5 copies of dataset all with cg = 0
datasets = []
dataset_weights = []
ctg_sets = [3, 2.5, 2,1.5,1,0.5,0,-0.5,-1,-1.5,-2, -2.5, -3]
#ctg_sets = np.linspace(-10,10, 20)
#ctg_sets = [2,1,0,-1,-2]
if duplicate:

    for ctg in ctg_sets:
        #If copy_first we split after duplicating otherwise we use same split for all
        if copy_first:
            comb_df_init_copy = copy.deepcopy(comb_df_init)
            eft_copy, sm_copy = train_test_split(comb_df_init_copy, test_size=0.5, random_state=25, shuffle=True)
        else:
            eft_copy = copy.deepcopy(comb_df_eft)
            sm_copy = copy.deepcopy(comb_df_sm)
        
        comb_df_ctg = get_eft_comb_df(sm_copy, eft_copy, cgs=0, ctgs=ctg)
        datasets.append(comb_df_ctg)
        fig, ax = plt.subplots(nrows=1, ncols = len(special_features
        ),figsize=(30, 5))
        for i in range(len(special_features)):
            var = special_features[i]
            plot_eft_hists(df=comb_df_ctg,var= var, combs=[(0,ctg)], weight_col="weight", ax = ax[i])
else:
    if copy_first:
        comb_dfs = np.array_split(comb_df_init,5)
    else:
        sm_dfs = np.array_split(comb_df_eft, 5)
        eft_dfs = np.array_split(comb_df_sm, 5)

    for ctg in ctg_sets:
        for i in range(5):
            if copy_first:
                comb_df_init_copy = copy.deepcopy(comb_dfs[i])
                eft_copy, sm_copy = train_test_split(comb_df_init_copy, test_size=0.5, random_state=25, shuffle=True)
            else:
                sm_copy = copy.deepcopy(sm_dfs[i])
                eft_copy = copy.deepcopy(eft_dfs[i])
            comb_df_ctg = get_eft_comb_df(sm_copy, eft_copy, cgs=0, ctgs=ctg)
            datasets.append(comb_df_ctg)
            fig, ax = plt.subplots(nrows=1, ncols = len(special_features
            ),figsize=(30, 5))
            for i in range(len(special_features)):
                var = special_features[i]
                plot_eft_hists(df=comb_df_ctg,var= var, combs=[(0,ctg)], weight_col="weight", ax = ax[i])

comb_df = pd.concat(datasets, axis=0, ignore_index=True)
comb_df = comb_df.sample(frac=1).reset_index(drop=True)

#%%
weights, labels, og_weights = comb_df["weight"], comb_df["labels"], comb_df["og_weight"]

#Saving coefs for testing
a_cg, a_ctgre, b_cg_cg, b_cg_ctgre, b_ctgre_ctgre = comb_df["a_cg"], comb_df["a_ctgre"], comb_df["b_cg_cg"], comb_df["b_cg_ctgre"], comb_df["b_ctgre_ctgre"]

comb_df = comb_df.drop(columns=["weight", "labels", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre", "og_weight"])
print("Final training data columns: ", comb_df.columns)


X, y, w, og_w = comb_df.values, labels.values, weights.values, og_weights.values

(X_train, X_test, y_train,
y_test, w_train, w_test,
a_cg_train, a_cg_test,
a_ctgre_train, a_ctgre_test,
b_cg_cg_train, b_cg_cg_test,
b_cg_ctgre_train, b_cg_ctgre_test,
b_ctgre_ctgre_train, b_ctgre_ctgre_test, og_w_train, og_w_test) = train_test_split(X,
                                                            y,
                                                            w,
                                                            a_cg,
                                                            a_ctgre,
                                                            b_cg_cg,
                                                            b_cg_ctgre,
                                                            b_ctgre_ctgre,og_w,
                                                            test_size=0.2,
                                                            random_state=45, shuffle=True)

# Define the ColumnTransformer to transform features
if transform:
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), np.arange(len(special_features)))  # Scale only the features
        ],
        remainder='passthrough'  # Keep other columns unchanged
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

(X_train, X_val, y_train,
 y_val, w_train, w_val)= train_test_split(X_train, y_train, w_train, test_size=0.2, random_state=42, shuffle=True)



y_train_tensor, y_test_tensor, y_val_tensor,w_train_tensor, w_test_tensor, w_val_tensor, X_train_tensor,X_test_tensor, X_val_tensor = get_tensors([y_train, y_test, y_val, w_train, w_test, w_val], [X_train, X_test, X_val])

#Input dim of 10(features + coefs) + 2(cg, ctg) -5(coefs) for parameters and buncha hidden layers.
input_dim = X_train.shape[1]
hidden_dim = [256, 64, 32, 16, 8]

#model = MergedNN(input_dim, hidden_dim, 1)
if mine:
    model = ComplexNN(input_dim, hidden_dim, 1) 

else:
    model = WadNeuralNetwork(input_dim, input_dim*3)

#criterion = nn.BCELoss(reduction='none')  # No reduction for custom weighting
criterion = WeightedBCELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Scheduler to adjust learning rate
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

loss_values = []
val_loss_values = []

# Training loop
num_epochs = 100
early_stopping_patience = 10  # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_without_improvement = 0

# Create a TensorDataset and DataLoader for training data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, w_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create a TensorDataset and DataLoader for validation data
val_dataset = TensorDataset(X_val_tensor, y_val_tensor, w_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
#%%
# Training loop
for epoch in range(num_epochs):
    # Forward pass
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch, w_batch in train_loader:
        
        logits = model(X_batch)
        loss_mean = criterion(logits, y_batch, w_batch)  #Mean loss values

        # Backward pass and optimization

        #zero the gradients of the optimizer
        optimizer.zero_grad()

        #Perform backward pass and calc gradients wrt weights
        loss_mean.backward()

        #Take step in direction of gradients and update parameters
        optimizer.step()

        epoch_loss += loss_mean.item()
    
    epoch_loss /= len(train_loader)
    loss_values.append(epoch_loss)

    # VALIDATION
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch, w_batch in val_loader:
            val_logits = model(X_batch)
            val_loss_mean = criterion(val_logits, y_batch, w_batch)  # Mean loss values
            val_loss += val_loss_mean.item()

    
    # Calculate average validation loss for the epoch
    val_loss /= len(val_loader)
    val_loss_values.append(val_loss)

    # Adjust learning rate
    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save the best model
        torch.save(model.state_dict(), 'saved_models/best_model.pth')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Weighted train Loss: {epoch_loss:.4f}, Weighted val Loss: {val_loss:.4f}")

# Load the best model
model.load_state_dict(torch.load('saved_models/best_model.pth'))

#%%
#Testing
model.eval()
#test_pairs = [(0.5,0.5), (0.5, -0.5), (0.75, 0.5), (0.5, 0.75), (0.75, 0.75)]
test_pairs = [(0, -2),(0,-1.5), (0, -1),(0,-0.5), (0, 0),(0,0.5), (0, 1),(0,1.5), (0, 2)]
#test_pairs = [(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)]
#for cg, ctg in [(i,j) for i in np.arange(0, 2, 0.5) for j in np.arange(0, 2, 0.5)]:

#Creating dataframe for just unseen test data, weights here are the SM weights for the test data
X_test_df = pd.concat([pd.DataFrame(np.hstack((
                        np.array(X_test),
                        np.array(og_w_test).reshape(-1, 1),
                        np.array(a_cg_test).reshape(-1, 1),
                        np.array(a_ctgre_test).reshape(-1, 1),
                        np.array(b_cg_cg_test).reshape(-1, 1),
                        np.array(b_cg_ctgre_test).reshape(-1, 1),
                        np.array(b_ctgre_ctgre_test).reshape(-1, 1)
                    )),
                                    columns=special_features+["cg","ctg","true_weight", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])], axis=1)
X_test_df.reset_index(drop=True, inplace=True)
for cg, ctg in test_pairs:
    print("--------cg:", cg, "ctg:", ctg, "-----------")
    #Sum of SM weights of EFT events
    sm_sum = sum(w_test[y_test==0])
    w_test[y_test==1]= calc_weights(X_test_df[y_test==1], cg=cg, ctg=ctg, weight_col="true_weight")
    if norm_eft:
        w_test[y_test==1] *= (sm_sum/sum(w_test[y_test==1]))

    #Removing random cg and ctg assigned at the start
    X_test_new = X_test[:,:-2]

    X_test_aug = np.hstack([X_test_new, np.full((X_test_new.shape[0], 1), cg), np.full((X_test_new.shape[0], 1), ctg)])  # Add parameters as input features
    X_test_tensor = torch.tensor(X_test_aug, dtype=torch.float32)

    # Evaluate the model on the test and train set
    with torch.no_grad():
        probabilities = model(X_test_tensor)
        train_proba = model(X_train_tensor)
        
        predictions = probabilities > 0.5  # Threshold at 0.5
        accuracy = (predictions.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
        print("Probabilities:", probabilities.squeeze().numpy())
        print("Predictions:", predictions.squeeze().numpy())
        print("Ground truth:", y_test_tensor.squeeze().numpy())

    y_test_np = y_test_tensor.cpu().numpy()
    predictions_np = predictions.cpu().numpy().flatten()
    train_proba_np = train_proba.cpu().numpy()

    #print(len(y_test_np), len(probabilities), len(w_test_new))
    classification_analysis(y_test_np, w_test.flatten(), probabilities.squeeze().cpu().numpy(), predictions_np, y_train, w_train, train_proba_np, ["SM", "EFT"], cg = cg, ctg = ctg)
    #classification_analysis(y_test, w_test, probabilities.squeeze(), predictions.squeeze(), y_train, w_train, train_proba.squeeze(), ["SM", "EFT"])

    # Plotting the training loss values
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Training Loss', color='blue')
    plt.plot(val_loss_values, label='Validation Loss', color='orange')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Save the trained model
#torch.save(model.state_dict(), 'saved_models/param_model.pth')

# %%
#Variation in AUC over range of ctg values
hidden_dim = [256, 64, 32, 16, 8]
input_dim = len(special_features)
auc_ctg = 0.69

if mine:
    model2 = ComplexNN(input_dim, hidden_dim, 1)
    model2.load_state_dict(torch.load("saved_models/model.pth"))
else:
    model2 = WadNeuralNetwork(input_dim, input_dim*3)
    model2.load_state_dict(torch.load("saved_models/wad_neural_network.pth"))

model2.eval()
auc_nn= []

cg_vals_gen = [-0.75, 0.5, 1.5]
ctg_vals_test = np.arange(-1.5, 1.5, 0.025)
#fig, ax = plt.subplots(figsize=(10, 5))

aucs = []
for ctg in ctg_vals_test:
    #print("Original df belongs to, cg:", cg, "ctg:", 0)
    ttH_df_set = ttH_df.copy()

    comb_df_init = pd.concat([ttH_df_set[var] for var in special_features+["true_weight_sel","a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]], axis=1)
    comb_df_init.rename(columns={'true_weight_sel': 'weight'}, inplace=True)

    comb_df_init = comb_df_init.dropna()
    #Change to ctg when model trained over ctg
    comb_df_init["cg"] = 0
    comb_df_init["ctg"] = ctg

    #get_labeled_comb_df(comb_df_init, features=special_features, c_g=cg, c_tg=0, norm_weights=False)
    comb_df_set_eft, comb_df_set_sm = train_test_split(comb_df_init, test_size=0.5, random_state=25, shuffle=True)
    comb_df_set_eft["labels"] = 1
    comb_df_set_sm["labels"] = 0
    #Change back to ctg when model trained over ctg
    comb_df_set_eft["weight"] = calc_weights(comb_df_set_eft, cg=0, ctg=ctg, weight_col="weight")
    if norm_eft:
        comb_df_set_sm["weight"]/=comb_df_set_sm["weight"].sum()
        comb_df_set_sm["weight"]*=10**4
        comb_df_set_eft["weight"]/=comb_df_set_eft["weight"].sum()
        comb_df_set_eft["weight"]*=10**4

    comb_df_set = pd.concat([comb_df_set_eft, comb_df_set_sm], axis=0, ignore_index=True)
    comb_df_shuf = comb_df_set.sample(frac=1).reset_index(drop=True)

    w, l = comb_df_shuf["weight"], comb_df_shuf["labels"]

    comb_df_shuf = comb_df_shuf.drop(columns=["weight", "labels", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])
 
    X, y, w = comb_df_shuf.values, l, w

    X = preprocessor.fit_transform(X)

    X_tensor= torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_tensor)
    probs_np=probs.squeeze().detach().numpy()
    #plt.hist(probs_np, bins=50, histtype="step", label=f"cg={cg_test}, ctg=0", density=True)
    fpr, tpr, _ = roc_curve(y, probs_np, sample_weight=w)
    auc_s = auc(fpr, tpr)
    aucs.append(auc_s)
    #print("For cg:", cg_test, "ctg:", 0, "AUC:", auc_s)

    with torch.no_grad():
        probs = model2(X_tensor[:,:-2])
    probs_np=probs.squeeze().detach().numpy()
    #plt.hist(probs_np, bins=50, histtype="step", label=f"cg={cg_test}, ctg=0", density=True)
    fpr, tpr, _ = roc_curve(y, probs_np, sample_weight=w)
    auc_s_nn = auc(fpr, tpr)
    auc_nn.append(auc_s_nn)

    #param_aucs[cg] = aucs
# ax.plot(cg_vals_test, aucs, label=f"ctg={auc_ctg}")

# ax.set_xlabel("cg")
# ax.set_ylabel("AUC")
# ax.legend()
fig, ax = plt.subplots(figsize=(7, 5))
plotter.overlay_line_plots(ctg_vals_test, [auc_nn, aucs],xlabel="cg", ylabel="AUC", title="AUC variation over cg", labels=["Basic NN, cg=0.3", "Param NN"], colors=["blue", "orange"], axes=ax)
ax.axvline(x=0.3, color="black", linestyle="--", label="Basic NN training cg = 0.3")
ax.legend(fontsize=12)

#%%
#2D variation of AUC over cg and ctg
# Define the range of values for cg and ctg
num_points = 30
wilson_range = (-1.5, 1.5)
cg_grid_test = np.linspace(*wilson_range, num_points)
ctg_grid_test = np.linspace(*wilson_range, num_points)

# Initialize a 2D array to store AUC values
auc_matrix_param = np.zeros((len(cg_grid_test), len(ctg_grid_test)))
auc_matrix_basic = np.zeros((len(cg_grid_test), len(ctg_grid_test)))

# Loop over cg and ctg values
for i, cg in enumerate(cg_grid_test):
    for j, ctg in enumerate(ctg_grid_test):
        # Copy and prepare the dataset
        ttH_df_set = ttH_df.copy()
        comb_df_init = pd.concat([ttH_df_set[var] for var in special_features + ["true_weight_sel", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]], axis=1)
        comb_df_init.rename(columns={'true_weight_sel': 'weight'}, inplace=True)
        comb_df_init = comb_df_init.dropna()
        comb_df_init["cg"] = cg
        comb_df_init["ctg"] = ctg

        # Split the dataset into EFT and SM
        comb_df_set_eft, comb_df_set_sm = train_test_split(comb_df_init, test_size=0.5, random_state=25, shuffle=True)
        comb_df_set_eft["labels"] = 1
        comb_df_set_sm["labels"] = 0

        comb_df_set_eft["weight"] = calc_weights(comb_df_set_eft, cg=cg, ctg=ctg, weight_col="weight")

        if norm_eft:
            comb_df_set_sm["weight"] /= comb_df_set_sm["weight"].sum()
            comb_df_set_sm["weight"] *= 10**4
            comb_df_set_eft["weight"] /= comb_df_set_eft["weight"].sum()
            comb_df_set_eft["weight"] *= 10**4

        
        # Combine and shuffle the dataset
        comb_df_set = pd.concat([comb_df_set_eft, comb_df_set_sm], axis=0, ignore_index=True)
        comb_df_shuf = comb_df_set.sample(frac=1).reset_index(drop=True)

        # Prepare the data for the model
        w, l = comb_df_shuf["weight"], comb_df_shuf["labels"]
        
        comb_df_shuf = comb_df_shuf.drop(columns=["weight", "labels", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])
        X, y, w = comb_df_shuf.values, l, w
        X = preprocessor.fit_transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        #print(X_tensor[:,-2], comb_df_shuf["cg"], X_tensor[:,-1],comb_df_shuf["ctg"])

        # Get model predictions and calculate AUC
        # Model2 is basic NN
        with torch.no_grad():
            probs_basic = model2(X_tensor[:,:-2])

        probs_np_basic = probs_basic.squeeze().detach().numpy()
        fpr, tpr, _ = roc_curve(y, probs_np_basic, sample_weight=w)
        auc_s_basic = auc(fpr, tpr)
        auc_matrix_basic[i, j] = auc_s_basic

        #Model is param NN
        with torch.no_grad():
            probs_param = model(X_tensor)
        
        probs_np_param = probs_param.squeeze().detach().numpy()
        fpr, tpr, _ = roc_curve(y, probs_np_param, sample_weight=w)
        auc_s_param = auc(fpr, tpr)
        auc_matrix_param[i, j] = auc_s_param

#%%
# Plotting the 2D heatmap basic
# Create tick values every 0.25 in the continuous range
tick_values = np.arange(-1.5, 1.5+0.25, 0.25)
# Find the corresponding indices in the heatmap:
tick_locs = np.linspace(0, len(ctg_grid_test)-1, len(tick_values))
CTG, CG = np.meshgrid(ctg_grid_test,cg_grid_test)  

plt.figure(figsize=(8,6))
cs = plt.contourf(CTG, CG, auc_matrix_basic, levels=20, cmap="viridis")
plt.colorbar(cs, label="NN AUC Score")
plt.xlabel(r"$c_{tg}$")
plt.ylabel(r"$c_{g}$")
#plt.title(r"2D Contour of AUC vs $(c_g, c_{tg}), Basic NN$")
plt.show()

# Plotting the 2D heatmap param
# We'll put c_{tg} on the x-axis and c_g on the y-axis.

plt.figure(figsize=(8,6))
cs = plt.contourf(CTG, CG, auc_matrix_param, levels=20, cmap="viridis")
plt.colorbar(cs, label="PNN AUC Score")
plt.xlabel(r"$c_{tg}$")
plt.ylabel(r"$c_{g}$")
#plt.title(r"2D Contour of AUC vs $(c_g, c_{tg}), Param NN$", fontsize=24)
plt.show()

# Calculate the AUC difference matrix (Param - Basic)
diff_matrix = auc_matrix_param - auc_matrix_basic

# Create the heatmap
plt.figure(figsize=(8,6))
cs = plt.contourf(CTG, CG, diff_matrix, levels=20, cmap="coolwarm")
plt.colorbar(cs, label=r"$\Delta$AUC Score (PNN - NN)")
plt.xlabel(r"$c_{tg}$")
plt.ylabel(r"$c_{g}$")
#plt.title(r"2D Contour of difference in AUC (Param NN - Basic NN) vs $(c_g, c_{tg})$")
plt.show()
# %%
#Likelihood ratio analysis

ttH_df = get_tth_df()

#special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel","lead_pt-over-mass_sel"]

comb_df_init = pd.concat([ttH_df[var] for var in special_features+["true_weight_sel","a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]], axis=1)
comb_df_init.rename(columns={'true_weight_sel': 'weight'}, inplace=True)

comb_df_init = comb_df_init.dropna()
#Randomly splitting same dataset into eft and sm
comb_df_sm, comb_df_eft = train_test_split(comb_df_init, test_size=0.5, random_state=25, shuffle=True)

true_ctg = 3
ctg_range = np.linspace(-10, 10, 200)
likelihood_ratios = []
sm_copy = copy.deepcopy(comb_df_sm)
eft_copy = copy.deepcopy(comb_df_eft)

comb_df_anal = get_eft_comb_df(sm_copy, eft_copy, cgs=0, ctgs=true_ctg, norm_eft=True)

w, l = comb_df_anal["weight"], comb_df_anal["labels"]

comb_df_anal = comb_df_anal.drop(columns=["weight", "labels", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])

X_true, y, w = comb_df_anal.values, l, w
X_true = preprocessor.transform(X_true)
X_tensor = torch.tensor(X_true, dtype=torch.float32)
with torch.no_grad():
    probs_true = model(X_tensor)
probs_np_true = probs_true.squeeze().detach().numpy()

print(comb_df_anal.columns)
for ctg in ctg_range:
    comb_df_anal["ctg"] = ctg
    X, y, w = comb_df_anal.values, l, w
    X = preprocessor.transform(X)

    #Doing liklihoods on just unseen test data.
    # X = np.hstack([X_test[:,:-2], np.full((X_test.shape[0], 1), 0), np.full((X_test.shape[0], 1), ctg)])
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_tensor)
    probs_np = probs.squeeze().detach().numpy()

    # plt.hist(probs_np, bins = 40, weights=w, label = f"NN output ctg = {ctg}", histtype="step", linewidth=2)
    # # plt.hist(probs_np[l==0], bins = 40, weights=w[l==0], label = "SM", histtype="step", linewidth=2)
    # # plt.hist(probs_np[l==1], bins = 40, weights=w[l==1], label = f"EFT (ctg = {true_ctg})", histtype="step", linewidth=2)
    # plt.hist(probs_np_true, bins = 40, weights=w, label = f"NN output (ctg = {true_ctg})", color="grey", alpha = 0.5)
    # plt.ylim(0,2000)
    # plt.xlabel(f"Probability (evaluated at ctg = {ctg:.2f})")
    # plt.ylabel("Frequency")
    # plt.legend(loc = "best")
    # plt.savefig(f"prob_dist_evol/with_true/ctg_{ctg:.2f}.png")
    # plt.show()

    likelihood = 0
    for i in range(len(probs_np)):
        likelihood += ((np.log(probs_np[i])- np.log(1-probs_np[i]))* w[i])
        # likelihood*=(probs_np[i]/(1-probs_np[i]))
        # print(likelihood)
        
    #log_ratios = np.log(probs_np) - np.log(1 - probs_np)
    #log_l_ratios = -1*np.sum(log_ratios)  # Use sum instead of np.prod
    likelihood_ratios.append(-1*likelihood)

plt.plot(ctg_range, likelihood_ratios)
plt.xlabel("ctg")
plt.ylabel("Log Likelihood Ratios")
plt.title("Log Likelihood Ratios vs ctg")
plt.show()

# Assuming likelihood_ratios and ctg_range are NumPy arrays
likelihood_ratios = np.array(likelihood_ratios)
mask = (-5 < ctg_range) & (ctg_range < 5)  # Creates a boolean mask

# Apply mask to filter values
filtered_likelihoods = likelihood_ratios[mask]
filtered_ctg_range = ctg_range[mask]
min_index = np.argmin(filtered_likelihoods)  # Get index of min value
print("Minimum at", filtered_ctg_range[min_index])
#%%
# from moviepy import ImageSequenceClip
# import os

# # List sorted images from a directory
# image_files = [os.path.join("prob_dist_evol/with_true", img)
#                       for img in os.listdir("prob_dist_evol/with_true")
#                       if img.endswith(".png")]

# # Create a clip from the images at 10 fps
# clip = ImageSequenceClip(image_files, fps=10)

# # Write to a file
# clip.write_videofile("prob_dist_evol/output_with_true_comb.mp4", codec="libx264")
#%%
#Plotting NN output for SM and EFT

ctg_range = [-3,-2,-1,0,1,2,3]
fig, ax = plt.subplots(nrows=len(ctg_range), ncols = len(ctg_range),figsize=(50, 50))
for i in range(len(ctg_range)):
    true_ctg = ctg_range[i]
    comb_df_sm, comb_df_eft = train_test_split(comb_df_init, test_size=0.5, random_state=25, shuffle=True)

    sm_copy = copy.deepcopy(comb_df_sm)
    eft_copy = copy.deepcopy(comb_df_eft)

    comb_df_anal = get_eft_comb_df(sm_copy, eft_copy, cgs=0, ctgs=true_ctg, norm_eft=True)

    w, l = comb_df_anal["weight"], comb_df_anal["labels"]

    comb_df_anal = comb_df_anal.drop(columns=["weight", "labels", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])

    
    for j in range(len(ctg_range)):
        ctg = ctg_range[j]
        comb_df_anal["ctg"] = ctg
        X, y, w = comb_df_anal.values, l, w
        X = preprocessor.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs_true = model(X_tensor)
        probs_np_true = probs_true.squeeze().detach().numpy()
        ax[i][j].hist(probs_np_true, bins = 40, weights=w, label = f"NN output (ctg = {true_ctg:.2f})", color="grey", alpha = 0.5)
        ax[i][j].legend(loc = "best")
        ax[i][j].set_xlabel(f"Probability (evaluated at ctg = {ctg:.2f})")
        ax[i][j].set_ylabel("Frequency")
        ax[i][j].set_ylim(0,3000)