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

plotter = Plotter()

ttH_df = get_tth_df()

#special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel","lead_pt-over-mass_sel"]

c_g_range = (-1, 1)
c_tg_range = (-1,1)

comb_df_init = pd.concat([ttH_df[var] for var in special_features+["true_weight_sel","a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]], axis=1)
comb_df_init.rename(columns={'true_weight_sel': 'weight'}, inplace=True)

comb_df_init = comb_df_init.dropna()
mine = False
norm_eft = True

#Duplicating dataset for eft and sm
# comb_df_eft = copy.deepcopy(comb_df_init)
#comb_df_sm=copy.deepcopy(comb_df_init)

#Randomly splitting same dataset into eft and sm
comb_df_sm, comb_df_eft = train_test_split(comb_df_init, test_size=0.5, random_state=25, shuffle=True)


rand_cg_sm = np.random.uniform(*c_g_range, size=len(comb_df_sm))
rand_ctg_sm = np.random.uniform(*c_tg_range, size=len(comb_df_sm))

comb_df_sm["cg"] = rand_cg_sm
comb_df_sm["ctg"] = rand_ctg_sm
comb_df_sm["labels"] = 0    

comb_df_sm.reset_index(drop=True, inplace=True)

rand_cg_eft = np.random.uniform(*c_g_range, size=len(comb_df_eft))
rand_ctg_eft = np.random.uniform(*c_tg_range, size=len(comb_df_eft))

comb_df_eft["cg"] = rand_cg_eft
comb_df_eft["ctg"] = rand_ctg_eft
comb_df_eft["labels"] = 1
comb_df_eft["weight"] = calc_weights(comb_df_eft, cg=rand_cg_eft, ctg=rand_ctg_eft, weight_col="weight")

if norm_eft:
    comb_df_sm["weight"]/=comb_df_sm["weight"].sum()
    comb_df_sm["weight"]*=10**4

    comb_df_eft["weight"] /= comb_df_eft["weight"].sum()
    comb_df_eft["weight"] *= 10**4

comb_df = pd.concat([comb_df_sm, comb_df_eft], axis=0, ignore_index=True)

weights, labels = comb_df["weight"], comb_df["labels"]

#Saving coefs for testing
a_cg, a_ctgre, b_cg_cg, b_cg_ctgre, b_ctgre_ctgre = comb_df["a_cg"], comb_df["a_ctgre"], comb_df["b_cg_cg"], comb_df["b_cg_ctgre"], comb_df["b_ctgre_ctgre"]

comb_df = comb_df.drop(columns=["weight", "labels", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])
# print("Final training data columns: ", comb_df.columns)


X, y, w = comb_df.values, labels.values, weights.values

(X_train, X_test, y_train,
y_test, w_train, w_test,
a_cg_train, a_cg_test,
a_ctgre_train, a_ctgre_test,
b_cg_cg_train, b_cg_cg_test,
b_cg_ctgre_train, b_cg_ctgre_test,
b_ctgre_ctgre_train, b_ctgre_ctgre_test) = train_test_split(X,
                                                            y,
                                                            w,
                                                            a_cg,
                                                            a_ctgre,
                                                            b_cg_cg,
                                                            b_cg_ctgre,
                                                            b_ctgre_ctgre,
                                                            test_size=0.2,
                                                            random_state=45, shuffle=True)

# Define the ColumnTransformer to transform features
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), np.arange(len(special_features)))  # Scale only the features
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


#To allow recalculation of weights when it 
# coefs_test = pd.DataFrame(X_test[:,-7:-2], columns=["a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])

# X_test = np.hstack((X_test[:,:-7], X_test[:,-2:]))
# X_train = np.hstack((X_train[:,:-7], X_train[:,-2:]))
# _____________________________________________________


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
optimizer = optim.Adam(model.parameters(), lr=0.01)

#Scheduler to adjust learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)


loss_values = []
val_loss_values = []

# Training loop
num_epochs = 100


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
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Weighted train Loss: {epoch_loss:.4f}, Weighted val Loss: {val_loss:.4f}")

#%%
#Testing
model.eval()
#for cg, ctg in [(i,j) for i in np.arange(0, 2, 0.5) for j in np.arange(0, 2, 0.5)]:
X_test_df = pd.concat([pd.DataFrame(np.hstack((
                        np.array(X_test),
                        np.array(w_test).reshape(-1, 1),
                        np.array(a_cg_test).reshape(-1, 1),
                        np.array(a_ctgre_test).reshape(-1, 1),
                        np.array(b_cg_cg_test).reshape(-1, 1),
                        np.array(b_cg_ctgre_test).reshape(-1, 1),
                        np.array(b_ctgre_ctgre_test).reshape(-1, 1)
                    )),
                                    columns=special_features+["cg","ctg","plot_weight", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])], axis=1)
X_test_df.reset_index(drop=True, inplace=True)
for cg, ctg in [(0.5,0.5), (0.5, -0.5), (0.75, 0.5), (0.5, 0.75), (0.75, 0.75)]:
    print("--------cg:", cg, "ctg:", ctg, "-----------")
    eft_sum = sum(w_test[y_test==1])
    w_test[y_test==1]= calc_weights(X_test_df[y_test==1], cg=cg, ctg=ctg)
    if norm_eft:
        w_test[y_test==1] *= eft_sum/sum(w_test[y_test==1])

    #w_test_new = X_test_df["plot_weight"]

    #print(len(w_test_new))
   # w_test_new = calc_weights(pd.concat(pd.DataFrame(np.hstack((X_test,
                                                # w_test)),
                                                # columns=special_features+["plot_weight"]),coefs_test),
                                                # cg=cg, ctg=ctg)
    #X_test_new = X_test[:,:-5]
    #X_train_new = X_train[:,:-5]

    #Removing random cg and ctg assigned at the start
    X_test_new = X_test[:,:-2]

    X_test_aug = np.hstack([X_test_new, np.full((X_test_new.shape[0], 1), cg), np.full((X_test_new.shape[0], 1), ctg)])  # Add parameters as input features
    X_test_tensor = torch.tensor(X_test_aug, dtype=torch.float32)
    
    #X_train_aug = np.hstack([X_train_new, np.full((X_train_new.shape[0], 1), cg), np.full((X_train_new.shape[0], 1), ctg)])  # Add parameters as input features
    #X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
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
    classification_analysis(y_test_np, w_test.flatten(), probabilities.squeeze().cpu().numpy(), predictions_np, y_train, w_train, train_proba_np, ["SM", "EFT"])
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
torch.save(model.state_dict(), 'saved_models/wad_param_model.pth')

# %%
#Variation in AUC over range of cg values

cg_vals_gen = [-0.75, 0.5, 1.5]
cg_vals_test = np.arange(-1.5, 1.5, 0.1)
fig, ax = plt.subplots(figsize=(10, 5))
param_aucs = {}
aucs = []
for cg in cg_vals_test:
    #print("Original df belongs to, cg:", cg, "ctg:", 0)
    ttH_df_set = ttH_df.copy()

    comb_df_init = pd.concat([ttH_df_set[var] for var in special_features+["true_weight_sel","a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]], axis=1)
    comb_df_init.rename(columns={'true_weight_sel': 'weight'}, inplace=True)

    comb_df_init = comb_df_init.dropna()
    comb_df_init["cg"] = cg
    comb_df_init["ctg"] = 0

    #get_labeled_comb_df(comb_df_init, features=special_features, c_g=cg, c_tg=0, norm_weights=False)
    comb_df_set_eft, comb_df_set_sm = train_test_split(comb_df_init, test_size=0.5, random_state=25, shuffle=True)
    comb_df_set_eft["labels"] = 1
    comb_df_set_sm["labels"] = 0
    comb_df_set_eft["weight"] = calc_weights(comb_df_set_eft, cg=cg, ctg=0, weight_col="weight")
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

    #param_aucs[cg] = aucs
ax.plot(cg_vals_test, aucs, label=f"cg={cg}, ctg=0")

ax.set_xlabel("cg")
ax.set_ylabel("AUC")
ax.legend()
#%%
#2D variation of AUC over cg and ctg
# Define the range of values for cg and ctg
cg_vals_test = np.arange(-1.5, 1.5, 0.3)
ctg_vals_test = np.arange(-1.5, 1.5, 0.3)

# Initialize a 2D array to store AUC values
auc_matrix = np.zeros((len(cg_vals_test), len(ctg_vals_test)))

# Loop over cg and ctg values
for i, cg in enumerate(cg_vals_test):
    for j, ctg in enumerate(ctg_vals_test):
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

        # Get model predictions and calculate AUC
        with torch.no_grad():
            probs = model(X_tensor)
        probs_np = probs.squeeze().detach().numpy()
        fpr, tpr, _ = roc_curve(y, probs_np, sample_weight=w)
        auc_s = auc(fpr, tpr)
        auc_matrix[i, j] = auc_s

# Plotting the 2D heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(auc_matrix, xticklabels=np.round(ctg_vals_test, 2), yticklabels=np.round(cg_vals_test, 2), cmap="viridis", ax=ax)
ax.set_xlabel("ctg")
ax.set_ylabel("cg")
ax.set_title("AUC variation over cg and ctg")
plt.show()
#%%
#Testing basic NN on data with weights calculated on new cg vals
#This model is trained for cg=0.3, ctg=0
hidden_dim = [256, 64, 32, 16, 16,8]
input_dim = len(special_features)

if mine:
    model2 = ComplexNN(input_dim, hidden_dim, 1)
    model2.load_state_dict(torch.load("saved_models/model2.pth"))
else:
    model2 = WadNeuralNetwork(input_dim, input_dim*3)
    model2.load_state_dict(torch.load("saved_models/wad_neural_network2.pth"))

model2.eval()
auc_nn= []
for cg in cg_vals_test:
    comb_df_init = pd.concat([ttH_df_set[var] for var in special_features+["true_weight_sel","a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]], axis=1)
    comb_df_init.rename(columns={'true_weight_sel': 'weight'}, inplace=True)

    comb_df_init = comb_df_init.dropna()

    comb_df_set_eft, comb_df_set_sm = train_test_split(comb_df_init, test_size=0.5, random_state=25, shuffle=True)
    comb_df_set_eft["labels"] = 1
    comb_df_set_sm["labels"] = 0
    comb_df_set_eft["weight"] = calc_weights(comb_df_set_eft, cg=cg, ctg=0, weight_col="weight")

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
        probs = model2(X_tensor)
    probs_np=probs.squeeze().detach().numpy()
    #plt.hist(probs_np, bins=50, histtype="step", label=f"cg={cg_test}, ctg=0", density=True)
    fpr, tpr, _ = roc_curve(y, probs_np, sample_weight=w)
    auc_s_nn = auc(fpr, tpr)
    auc_nn.append(auc_s_nn)


fig, ax = plt.subplots(figsize=(7, 5))
plotter.overlay_line_plots(cg_vals_test, [auc_nn, aucs],xlabel="cg", ylabel="AUC", title="AUC variation over cg", labels=["Basic NN, cg=0.3", "Param NN"], colors=["blue", "orange"], axes=ax)
ax.axvline(x=0.3, color="black", linestyle="--", label="Basic NN training cg = 0.3")
ax.legend(fontsize=12)
#%%


