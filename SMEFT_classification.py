#%%
from EFT import *
from utils import *
from selection import *
from SMEFT_utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


c_g = 0.3
c_tg = 0.69
do_grid_search = False

ttH_df = pd.read_parquet(f"{new_sample_path}/ttH_processed_selected_with_smeft_cut_mupcleq90.parquet")

ttH_df = ttH_df[(ttH_df["mass_sel"] == ttH_df["mass_sel"])]
ttH_df['plot_weight'] *= target_lumi / total_lumi 
#ttH_df = ttH_df.dropna()

invalid_weights = ttH_df["plot_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    ttH_df = ttH_df[~invalid_weights]


#special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel"] 
#comb_df = pd.concat([ttH_df[var] for var in special_features], axis=1)
EFT_weights = np.asarray(calc_weights(ttH_df, cg=c_g, ctg=c_tg))
EFT_weights = (EFT_weights/np.sum(EFT_weights))*10000
EFT_labels = np.ones(len(EFT_weights))
SM_weights = np.asarray(ttH_df["plot_weight"])
SM_weights = (SM_weights/np.sum(SM_weights))*10000
SM_labels = np.zeros(len(SM_weights))



comb_df_SM = pd.concat([ttH_df[var] for var in special_features], axis=1)
comb_df_SM["weight"] = SM_weights

comb_df_EFT = comb_df_SM.copy()#pd.concat([ttH_df[var] for var in special_features], axis=1)
comb_df_EFT["weight"] = EFT_weights

#EFT=1, SM=0
labels = np.concatenate([EFT_labels, SM_labels])
comb_df = pd.concat([comb_df_EFT, comb_df_SM], axis = 0)
comb_df["labels"] = labels

#Dropping all rows with nans
comb_df = comb_df.dropna()

weights = comb_df["weight"]
labels = comb_df["labels"]
comb_df = comb_df.drop(columns=["weight", "labels"])
print("Final training data columns: ", comb_df.columns)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(comb_df,
                                                                     labels,
                                                                     weights,
                                                                     test_size=0.2,
                                                                     random_state=50, shuffle=True)
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X_train, y_train, w_train, test_size=0.2, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

#Making sure everything is np array, only necessayr becasue of some version mismatch.
X_train, X_test, X_val, y_train, y_test,y_val, w_train, w_test, w_val, weights = make_np_arr(X_train,
                                                                 X_test,
                                                                     X_val,
                                                                   y_train,
                                                                     y_test,
                                                                        y_val,
                                                                       w_train,
                                                                         w_test,
                                                                          w_val, weights)

#%%
#Training nn instead

y_train_tensor, y_test_tensor, y_val_tensor,w_train_tensor, w_test_tensor, w_val_tensor, X_train_tensor,X_test_tensor, X_val_tensor = get_tensors([y_train, y_test, y_val, w_train, w_test, w_val], [X_train, X_test, X_val])

input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)

#criterion = nn.BCELoss(reduction='none')  # No reduction for custom weighting
criterion = WeightedBCELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_values = []
val_loss_values = []

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    model.train() 
    logits = model(X_train_tensor)
    loss_mean = criterion(logits, y_train_tensor, w_train_tensor)  #Mean loss values

    # Backward pass and optimization
    optimizer.zero_grad()
    loss_mean.backward()
    optimizer.step()

    # VALIDATION
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_loss_mean = criterion(val_logits, y_val_tensor, w_val_tensor)  # Mean loss values
    loss_values.append(loss_mean.item())
    val_loss_values.append(val_loss_mean.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Weighted Loss: {loss_mean.item():.4f}")

# Evaluate the model on the test and train set
with torch.no_grad():
    probabilities = model(X_test_tensor)
    train_proba = model(X_train_tensor)
    
    predictions = probabilities > 0.5  # Threshold at 0.5
    accuracy = (predictions.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    print("Probabilities:", probabilities.squeeze().numpy())
    print("Predictions:", predictions.squeeze().numpy())
    print("Ground truth:", y_test_tensor.squeeze().numpy())

classification_analysis(y_test, w_test, probabilities.squeeze(), predictions.squeeze(), y_train, w_train, train_proba.squeeze(), ["SM", "EFT"])

# Plotting the training loss values
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss', color='blue')
plt.plot(val_loss_values, label='Validation Loss', color='orange')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()