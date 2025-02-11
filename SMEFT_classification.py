#%%
from EFT import * 
from utils import *
from selection import *
from SMEFT_utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
plt.style.use(hep.style.CMS)

c_g = 0.3
c_tg = 0
do_grid_search = False

ttH_df = get_tth_df()

#special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "pt-over-mass_sel"]#,"lead_pt-over-mass_sel"] 

comb_df=get_labeled_comb_df(ttH_df, "rand", special_features, c_g, c_tg)

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
joblib.dump(scaler, "saved_models/scaler.pkl")

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
#Training nn

y_train_tensor, y_test_tensor, y_val_tensor,w_train_tensor, w_test_tensor, w_val_tensor, X_train_tensor,X_test_tensor, X_val_tensor = get_tensors([y_train, y_test, y_val, w_train, w_test, w_val], [X_train, X_test, X_val])

#Input dim of 4 and buncha hidden layers.
input_dim = X_train.shape[1]
hidden_dim = [256, 64, 32, 16, 16, 8]

#model = LogisticRegression(input_dim)
model = ComplexNN(input_dim, hidden_dim, 1) 
#model = WadNeuralNetwork(input_dim, input_dim*3)
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
# checkpoints = torch.load("saved_models/wad_neural_network.pth")
# model = WadNeuralNetwork(checkpoints["input_dim"], checkpoints["hidden_dim"])
# model.load_state_dict(checkpoints["model_state"])
# model.eval()
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

# Save the trained model
#torch.save(model.state_dict(), 'saved_models/model2.pth')
#torch.save(model.state_dict(), 'saved_models/wad_neural_network.pth')
# %%