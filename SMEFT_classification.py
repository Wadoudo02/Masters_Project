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
print(f"--> Remaining rows = {len(ttH_df)}")

#special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel","lead_pt-over-mass_sel"] 
# EFT_weights = np.asarray(calc_weights(ttH_df, cg=c_g, ctg=c_tg))
# EFT_weights = (EFT_weights/np.sum(EFT_weights))*10000
# EFT_labels = np.ones(len(EFT_weights))
# SM_weights = np.asarray(ttH_df["plot_weight"])
# SM_weights = (SM_weights/np.sum(SM_weights))*10000
# SM_labels = np.zeros(len(SM_weights))


# comb_df_SM = pd.concat([ttH_df[var] for var in special_features], axis=1)
# comb_df_SM["weight"] = SM_weights

# comb_df_EFT = comb_df_SM.copy()#pd.concat([ttH_df[var] for var in special_features], axis=1)
# comb_df_EFT["weight"] = EFT_weights

# #EFT=1, SM=0
# labels = np.concatenate([EFT_labels, SM_labels])
# comb_df = pd.concat([comb_df_EFT, comb_df_SM], axis = 0)
# comb_df["labels"] = labels

comb_df=get_labeled_comb_df(ttH_df, "dup", special_features, c_g, c_tg)

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
(X_train, X_test, X_val,
y_train, y_test,y_val,
w_train, w_test, w_val, weights) = make_np_arr(X_train,
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
hidden_dim = [256, 64, 32, 16, 16, 8]

#model = LogisticRegression(input_dim)
model = ComplexNN(input_dim, hidden_dim, 1) 

#criterion = nn.BCELoss(reduction='none')  # No reduction for custom weighting
criterion = WeightedBCELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_values = []
val_loss_values = []

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # Forward pass
    model.train() 
    logits = model(X_train_tensor)
    loss_mean = criterion(logits, y_train_tensor, w_train_tensor)  #Mean loss values

    # Backward pass and optimization

    #zero the gradients of the optimizer
    optimizer.zero_grad()

    #Perform backward pass and calc gradients wrt weights
    loss_mean.backward()

    #Take step in direction of gradients and update parameters
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

y_test_np = y_test_tensor.cpu().numpy()
predictions_np = predictions.cpu().numpy().flatten()
train_proba_np = train_proba.cpu().numpy()

classification_analysis(y_test_np, w_test, probabilities.squeeze().cpu().numpy(), predictions_np, y_train, w_train, train_proba_np, ["SM", "EFT"])
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
# %%
#Extract relevant columns from overall df
cats = [0,0.4, 0.5, 0.6, 0.7,1]

dfs = get_dfs(new_sample_path)
for proc, df in dfs.items():
    new_df = pd.concat([df[feature] for feature in special_features]+[df["mass_sel"],df["plot_weight"]], axis=1)
    dfs[proc] = torch.tensor(new_df.to_numpy(), dtype=torch.float32)
#%%
#Get predictions for all events
dfs_preds = {}
dfs_cats = {}

for proc, df in dfs.items():
    print(proc, df)
    if proc=="ttH":
        continue
    mass,weight = df[:,-2],df[:,-1]
    df = df[:,:-2]
    #new_df = torch.tensor(df,dtype=torch.float32)
    probs = model(df)

    dfs_preds[proc] = [probs, mass,weight]

#%%
#Split events by category
for proc, df in dfs.items():
    if proc=="ttH":
        continue
    preds,mass, weights = dfs_preds[proc]
    dfs_cats[proc]={"events":[], "mass":[],"weights":[]}
    for i in range(1, len(cats)):
        bools = ((cats[i-1]<preds) & (preds<cats[i])).squeeze()
        dfs_cats[proc]["events"].append(df[bools.numpy()])
        dfs_cats[proc]["weights"].append(weights[bools.numpy()])
        dfs_cats[proc]["mass"].append(mass[bools.numpy()])

#Over number of cats
fig, ax = plt.subplots(ncols=4,figsize=(10, 5))    
for i in range(1,len(dfs_cats["ggH"])):
    for proc, df in dfs_cats.items():
        print(proc, "cat: ", i)
        print("events: ", df["events"][i].shape, "Weights: ", df["weights"][i].shape)
        ax[i].hist(dfs_cats[proc]["mass"][i],weights=dfs_cats[proc]["weights"][i], bins=50, alpha=0.5, label=proc, )




# %%
