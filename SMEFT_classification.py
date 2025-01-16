#%%
from EFT import *
from utils import *
from selection import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
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


special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]

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
                                                                     random_state=45, shuffle=True)
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

#Initial best params for xgb which are overwritten if grid search performed.
best_params = {"gamma": 0, "learning_rate": 0.2, "max_depth": 3, "n_estimators": 50}

#%%
#Grid paramter scan for xgb
if do_grid_search:
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss"
    )
    # Define the grid of parameters to search
    param_grid = {
        'max_depth': [3, 5, 7],
        'gamma': [0, 1, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',       # Use an appropriate metric (e.g., AUC)
        cv=3,                    # 3-fold cross-validation
        verbose=3,
        n_jobs=-1                # Parallelize computation
    )

    # Fit the model
    grid_search.fit(X_train, y_train, sample_weight=w_train)

    # Get the best parameters
    print("Best Parameters:", grid_search.best_params_)
    best_params = grid_search.best_params_
#%%
# Train the xgb model with the best parameters
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="logloss",
    **best_params
)

xgb_model.fit(X_train, y_train, sample_weight=w_train)
#%%
#Classification analysis for xgb
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)
y_proba_train = xgb_model.predict_proba(X_train)

classification_analysis(y_test,w_test,y_proba,y_pred,y_train ,y_proba_train, ["SM", "EFT"] )

# Feature importance
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(xgb_model, ax=ax, importance_type='weight')

tick_locs = ax.get_yticks()
ax.set_yticks(tick_locs)
ax.set_yticklabels(special_features[:len(tick_locs)])
ax.set_ylabel('Features')

# %%
#Evaluating over all data 
y_proba = xgb_model.predict_proba(comb_df.to_numpy())

sm_probs = y_proba[:, 1][labels == 0]  # Probabilities for SM (true label 0)
eft_probs = y_proba[:, 1][labels == 1]  # Probabilities for EFT (true label 1)

# Separate weights based on true labels
sm_weights = weights[labels == 0]
eft_weights = weights[labels == 1]

# Plot histograms for SM and EFT
plt.figure(figsize=(10, 6))
plt.hist(sm_probs, weights=sm_weights, bins=30, alpha=0.7, color='blue', label="SM (True Label 0)")
plt.hist(eft_probs, weights=eft_weights, bins=30, alpha=0.7, color='orange', label="EFT (True Label 1)")
plt.xlabel('Predicted Probability of EFT (Positive Class)')
plt.ylabel('Weighted Frequency')
plt.title('Histogram of Predicted Probabilities (BDT Output)')
plt.grid(True)
plt.legend()
plt.show()
#%%
#Trying to see seperation over predicted features

SM_features = X_test[y_pred==0]
SM_w = w_test[y_pred==0]

EFT_features = X_test[y_pred==1]
EFT_w = w_test[y_pred==1]

# for col in range(len(special_features)):
#     plt.hist(SM_features[:,col],weights=SM_w, bins=50, alpha=0.5, label="SM")
#     plt.hist(EFT_features[:,col],weights=EFT_w, bins=50, alpha=0.5, label="EFT")
#     plt.xlabel(special_features[col])
#     plt.ylabel("Events")
#     plt.legend()
#     plt.show()

#%%
#Training nn instead
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (N, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
w_train_tensor = torch.tensor(w_train, dtype=torch.float32).unsqueeze(1)
w_test_tensor = torch.tensor(w_test, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
w_val_tensor = torch.tensor(w_val, dtype=torch.float32).unsqueeze(1)

input_dim = X_train.shape[1]
model = nn.Sequential(nn.Linear(input_dim, 1))

criterion = nn.BCEWithLogitsLoss(reduction='none')  # No reduction for custom weighting

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

loss_values = []
val_loss_values = []

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    model.train() 
    logits = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)  # Get raw loss values
    weighted_loss = loss * w_train_tensor  # Apply sample weights
    loss_mean = weighted_loss.mean()  # Mean loss across batch

    # Backward pass and optimization
    optimizer.zero_grad()
    loss_mean.backward()
    optimizer.step()

    # VALIDATION
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_loss = criterion(val_logits, y_val_tensor)  # Compute raw loss
        weighted_val_loss = val_loss * w_val_tensor  # Apply weights
        val_loss_mean = weighted_val_loss.mean()  # Average loss
    loss_values.append(loss_mean.item())
    val_loss_values.append(val_loss_mean.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Weighted Loss: {loss_mean.item():.4f}")

# Evaluate the model on the test set
with torch.no_grad():
    logits = model(X_test_tensor)
    probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
    predictions = probabilities > 0.5  # Threshold at 0.5
    accuracy = (predictions.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    print("Predictions:", predictions.squeeze().numpy())
    print("Ground truth:", y_test_tensor.squeeze().numpy())
    print(f"Accuracy on test set: {accuracy:.4f}")

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


