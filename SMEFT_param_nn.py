#%%
from EFT import * 
from utils import *
from selection import *
from SMEFT_utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

c_g_range = [-0.5, 0.5]
c_tg_range = [-0.5, 1]
do_grid_search = False

ttH_df = get_tth_df()

#special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel","lead_pt-over-mass_sel"]

#Df with all SM events on top and all EFT events on bottom
# comb_df=get_labeled_comb_df(ttH_df, "rand", special_features+["a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"], 0, 0, norm_weights=True)
# #Dropping all rows with nans
comb_df = pd.concat([ttH_df[var] for var in special_features+["true_weight_sel","a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"]], axis=1)
comb_df.rename(columns={'true_weight_sel': 'weight'}, inplace=True)

comb_df = comb_df.dropna()

rand_cg = np.random.uniform(*c_g_range, size=len(comb_df))
rand_ctg = np.random.uniform(*c_tg_range, size=len(comb_df))

comb_df["cg"] = rand_cg
comb_df["ctg"] = rand_ctg
comb_df["labels"] = 0

comb_df.reset_index(drop=True, inplace=True)

# comb_df_eft = comb_df.copy()
# rand_cg_eft = np.random.uniform(*c_g_range, size=len(comb_df_eft))
# rand_ctg_eft = np.random.uniform(*c_tg_range, size=len(comb_df_eft))

# comb_df_eft["cg"] = rand_cg_eft
# comb_df_eft["ctg"] = rand_ctg_eft
# comb_df_eft["labels"] = 1
# comb_df_eft["weight"] = calc_weights(comb_df, cg=rand_cg, ctg=rand_ctg, weight_col="weight")

# comb_df = pd.concat([comb_df, comb_df_eft], axis=0, ignore_index=True)

#Randomly assigning EFT events and changing the weight for EFT events
for index, row in comb_df.iterrows():
    if random.choice([True, False]):
        comb_df.at[index, "weight"] = calc_weights(row, cg=row["cg"], ctg=row["ctg"], weight_col="weight")
        comb_df.at[index, "labels"] = 1

weights = comb_df["weight"]
labels = comb_df["labels"]

# # Filter rows where labels are 1
# eft_weights = weights[labels == 1]

# # Normalize the weights for these rows
# normalized_weights = eft_weights / eft_weights.sum()

# # Update the DataFrame with the normalized weights
# weights[labels == 1] = normalized_weights

weights_eft = weights.loc[labels == 1]
weights.loc[labels == 1] = (weights_eft / weights_eft.sum())*10**4

# Normalize the weights for rows where labels are 0
weights_sm = weights.loc[labels == 0]
weights.loc[labels == 0] = (weights_sm / weights_sm.sum())*10**4

comb_df = comb_df.drop(columns=["weight", "labels", "a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])
# print("Final training data columns: ", comb_df.columns)

(X_train_init, X_test, y_train,
y_test, w_train_init, w_test) = train_test_split(comb_df,
                                                labels,
                                                weights,
                                                test_size=0.2,
                                                random_state=45, shuffle=True)

# Define the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), special_features)
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

X_train_init = preprocessor.fit_transform(X_train_init)
X_test = preprocessor.transform(X_test)



(X_train, X_test,
 y_train, y_test,
 w_train, w_test, weights) = make_np_arr(X_train_init,
                                                X_test,
                                                y_train,
                                                y_test,
                                                w_train_init,
                                                w_test,
                                                weights)


#_____________________________________________________
#coefs_test = pd.DataFrame(X_test[:,-7:-2], columns=["a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre"])

# X_test = np.hstack((X_test[:,:-7], X_test[:,-2:]))
# X_train = np.hstack((X_train[:,:-7], X_train[:,-2:]))
#_____________________________________________________

#w_train_init=w_train_init.reshape(-1,1)

# w_train = np.array(calc_weights(pd.DataFrame(np.hstack((X_train_init,w_train_init)),
#                                             columns=special_features+["a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre","plot_weight"]),
#                                             cg=rand_cg, ctg=rand_ctg))

#Removing last 5 coefficient columns and adding cg and ctg
# X_train = np.hstack([X_train_init[:,:-5], rand_cg.reshape(-1,1), rand_ctg.reshape(-1,1)])  # Add parameters as input features

(X_train, X_val, y_train,
 y_val, w_train, w_val)= train_test_split(X_train, y_train, w_train, test_size=0.2, random_state=42, shuffle=True)


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_val = scaler.transform(X_val)


#Making sure everything is np array, only necessayr becasue of some version mismatch.


w_train = w_train.reshape(-1, 1)
w_val = w_val.reshape(-1, 1)
w_test = w_test.reshape(-1, 1)

pairs = [(0.5, 0.5), (0.5, 0.75), (0.75, 0.75), (0.25, 0.25), (0.25, 0.75), (0.5, 0.25), (0.75, 0.25), (0.25, 0.5), (0.75, 0.5)]

y_train_tensor, y_test_tensor, y_val_tensor,w_train_tensor, w_test_tensor, w_val_tensor, X_train_tensor,X_test_tensor, X_val_tensor = get_tensors([y_train, y_test, y_val, w_train, w_test, w_val], [X_train, X_test, X_val])

#Input dim of 10(features + coefs) + 2(cg, ctg) -5(coefs) for parameters and buncha hidden layers.
input_dim = X_train.shape[1]
hidden_dim = [256, 64, 32, 16, 16, 8]

#model = LogisticRegression(input_dim)
#model = ComplexNN(input_dim, hidden_dim, 1) 
model = WadNeuralNetwork(input_dim, input_dim*4)

#criterion = nn.BCELoss(reduction='none')  # No reduction for custom weighting
criterion = WeightedBCELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_values = []
val_loss_values = []

# Training loop
num_epochs = 30

#cg, ctg = pairs[np.random.choice(len(pairs))]

# w_val_new = calc_weights(pd.DataFrame(np.hstack((X_val,
#                                             w_val)),
#                                             columns=special_features+["a_cg", "a_ctgre", "b_cg_cg", "b_cg_ctgre", "b_ctgre_ctgre","plot_weight"]),
#                                             cg=rand_cg, ctg=rand_ctg)

# w_train_tensor = torch.tensor(w_train_new, dtype=torch.float32)
# w_val_tensor = torch.tensor(w_val_new, dtype=torch.float32)

#Removing coeffiicent columns
# X_train_new = X_train[:,:-5]
# X_val_new = X_val[:,:-5]



# Convert to PyTorch tensors
# X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
# X_val_tensor = torch.tensor(X_val_aug, dtype=torch.float32)

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

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Weighted train Loss: {epoch_loss:.4f}, Weighted val Loss: {val_loss:.4f}")

#%%
#Testing
model.eval()
#for cg, ctg in [(i,j) for i in np.arange(0, 2, 0.5) for j in np.arange(0, 2, 0.5)]:
# X_test_df = pd.concat([pd.DataFrame(np.hstack((X_test,
#                                     w_test)),
#                                     columns=special_features+["cg","ctg","plot_weight"]), coefs_test], axis=1)
# X_test_df.reset_index(drop=True, inplace=True)
#for cg, ctg in [(-0.5, -0.5),(-0.75,-0.5),(0.75, 0.75), (0.5,0.5), (0.5, 0.75)]:
    #_____________________________________________________
#     print("--------cg:", cg, "ctg:", ctg, "-----------")
#     for idx, row in X_test_df.iterrows():
#         if y_test[idx] == 1:
#             X_test_df.at[idx, "plot_weight"] = calc_weights(row, cg=cg, ctg=ctg)
#     w_test_new = X_test_df["plot_weight"]

#     print(len(w_test_new))
#    # w_test_new = calc_weights(pd.concat(pd.DataFrame(np.hstack((X_test,
#                                                 # w_test)),
#                                                 # columns=special_features+["plot_weight"]),coefs_test),
#                                                 # cg=cg, ctg=ctg)
#     #X_test_new = X_test[:,:-5]
#     #X_train_new = X_train[:,:-5]

#     #Removing random cg and ctg assigned at the start
#     X_test_new = X_test[:,:-2]

#     X_test_aug = np.hstack([X_test_new, np.full((X_test_new.shape[0], 1), cg), np.full((X_test_new.shape[0], 1), ctg)])  # Add parameters as input features
#     X_test_tensor = torch.tensor(X_test_aug, dtype=torch.float32)
    #_______________________________________________
    # X_train_aug = np.hstack([X_train_new, np.full((X_train_new.shape[0], 1), cg), np.full((X_train_new.shape[0], 1), ctg)])  # Add parameters as input features
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
#torch.save(model.state_dict(), 'saved_models/model.pth')

# %%
