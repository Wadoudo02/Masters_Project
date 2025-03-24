#%%
from EFT import * 
from utils import *
from selection import *
from SMEFT_utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

c_g = 0
c_tg = 0.69
grid_search = False


ttH_df = get_tth_df()

#special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]
special_features = ["deltaR_sel", "HT_sel", "n_jets_sel", "delta_phi_gg_sel", "pt-over-mass_sel"]#,"lead_pt-over-mass_sel"] 

comb_df=get_labeled_comb_df(ttH_df, "rand", special_features, c_g, c_tg, norm_weights=True)

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
if grid_search:
    # Example parameter grid to search over
    param_grid = {
        'lr': [0.001, 0.01, 0.1],
        'hidden_dim': [[64, 32, 16, 8], [128, 64, 32, 16, 8]],
        'num_epochs': [50, 100]
    }

    # Assume X_train and y_train are your training data (e.g., from your earlier split)
    input_dim = X_train.shape[1]

    # Create the estimator instance with fixed parameters that won’t change in the grid search
    torch_clf = TorchClassifier(input_dim=input_dim, batch_size=32, verbose=True)

    # Create the grid search object
    grid_search = GridSearchCV(
        estimator=torch_clf,
        param_grid=param_grid,
        cv=3,                  # 3-fold cross validation
        scoring='accuracy',    # You can choose another scoring metric if needed
        verbose=2
    )

    # Run grid search (this may take a while depending on your grid and data size)
    grid_search.fit(X_train, y_train, sample_weight=w_train)

    print("Best parameters: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    lr, hidden_dim, num_epochs = grid_search.best_params_['lr'], grid_search.best_params_['hidden_dim'], grid_search.best_params_['num_epochs']
#%%
#Training nn
if not grid_search:
    lr, hidden_dim, num_epochs = 0.01, [64, 32, 16, 8], 100

y_train_tensor, y_test_tensor, y_val_tensor,w_train_tensor, w_test_tensor, w_val_tensor, X_train_tensor,X_test_tensor, X_val_tensor = get_tensors([y_train, y_test, y_val, w_train, w_test, w_val], [X_train, X_test, X_val])

#Input dim of 4 and buncha hidden layers.
input_dim = X_train.shape[1]
hidden_dim = [256, 64, 32, 16, 8]
#hidden_dim = [64, 32, 16, 8]

#model = LogisticRegression(input_dim)
model = ComplexNN(input_dim, hidden_dim, 1) 
#model = WadNeuralNetwork(input_dim, input_dim*3)
#model = MergedNN(input_dim, hidden_dims=hidden_dim, output_dim=1)
#criterion = nn.BCELoss(reduction='none')  # No reduction for custom weighting
criterion = WeightedBCELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_values = []
val_loss_values = []

# Training loop
#num_epochs = 100

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

classification_analysis(y_test_np, w_test, probabilities.squeeze().cpu().numpy(), predictions_np, y_train, w_train, train_proba_np, ["SM", "EFT"], cg=c_g, ctg = c_tg)
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
torch.save(model.state_dict(), 'saved_models/model_ctg.pth')
#torch.save(model.state_dict(), 'saved_models/mergedNN.pth')
# %%
#2 plots 1 of pt seperation and other of nn seperation
fig = plt.figure(figsize=(10, 20))
outer_grid = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15)

# Top nested 1x2 subplot
inner_grid_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1], height_ratios=[3, 1], hspace=0.2)
ax_top = fig.add_subplot(inner_grid_top[0])
ax_ratio_top = fig.add_subplot(inner_grid_top[1], sharex=ax_top)
plot_classifier_output(train_proba_np.squeeze(), y_train.squeeze(), w_train.squeeze(), ax=ax_top, ax_ratio=ax_ratio_top, cg=c_g, ctg = c_tg)
for x in [0.3, 0.4, 0.5, 0.6, 0.7]:
    ax_top.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    ax_ratio_top.axvline(x=x, color='gray', linestyle='--', alpha=0.5)

# Bottom nested 1x2 subplot
inner_grid_bottom = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[0], height_ratios=[3, 1], hspace=0.2)
ax_bottom = fig.add_subplot(inner_grid_bottom[0])
ax_ratio_bottom = fig.add_subplot(inner_grid_bottom[1], sharex=ax_bottom)
plot_eft_hists(df=ttH_df, var="pt", combs=[(c_g, c_tg)], ax=ax_bottom, ax_ratio=ax_ratio_bottom, density = False)
for pt in [0, 60, 120, 200, 300]:
    ax_bottom.axvline(x=pt, color='gray', linestyle='--', alpha=0.5)
    ax_ratio_bottom.axvline(x=pt, color='gray', linestyle='--', alpha=0.5)
#%%

# Create a figure with two rows (main plot and ratio plot)
fig = plt.figure(figsize=(10, 20))
outer_grid = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15)

# Create a nested grid for the classifier plot (top subplot of outer_grid)
inner_grid_top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1],
                                                  height_ratios=[3, 1], hspace=0.2)
ax_top = fig.add_subplot(inner_grid_top[0])
ax_ratio_top = fig.add_subplot(inner_grid_top[1], sharex=ax_top)

# Plot the classifier output using your helper function
plot_classifier_output(train_proba_np.squeeze(), y_train.squeeze(), w_train.squeeze(),
                         ax=ax_top, ax_ratio=ax_ratio_top, cg=c_g, ctg=c_tg)

# Draw vertical reference lines on both axes
for x in [0.4, 0.5, 0.6, 0.7, 0.8]:
    ax_top.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    ax_ratio_top.axvline(x=x, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


#%%
# Create a figure with 2 rows: main plot and ratio plot
c_tg = 10
c_g = 100
fig, (ax_main, ax_ratio) = plt.subplots(
    nrows=2, ncols=1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

plot_eft_hists(
    df=ttH_df, 
    var="pt", 
    combs=[(c_g, c_tg)], 
    ax=ax_main, 
    ax_ratio=ax_ratio, 
    density=True,
    weight_col="true_weight_sel"
)
# eft_scale = (1.0+ ttH_df["a_cg"]*c_g + ttH_df["a_ctgre"]*c_tg)+(ttH_df["b_cg_cg"]*(c_g**2)+ttH_df["b_cg_ctgre"]*(c_tg*c_g) + ttH_df["b_ctgre_ctgre"]*(c_tg**2))
# print(eft_scale)
# plt.hist(ttH_df["pt"],weights= ttH_df["true_weight_sel"], label="SM", alpha=0.5, bins = 50, density=True)
# plt.hist(ttH_df["pt"],weights= (ttH_df["true_weight_sel"]*eft_scale), label="EFT", alpha=0.5, bins = 50, density=True)

#Draw vertical reference lines on both axes
for pt in [0, 60, 120, 200, 300]:
    ax_main.axvline(x=pt, color='gray', linestyle='--', alpha=0.5)
    ax_ratio.axvline(x=pt, color='gray', linestyle='--', alpha=0.5)

# Label axes
ax_main.set_ylabel("Count")
ax_ratio.set_ylabel("Ratio")
ax_ratio.set_xlabel("pt")

plt.tight_layout()
plt.show()
#%%
#All input features
c_g = 0.3
c_tg = 0.69
# Create a figure with 2 rows and 5 columns
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(30, 7), sharex=False,gridspec_kw={'height_ratios': [3, 1]})
# Optional: share x-axis for the ratio row if it makes sense (e.g., sharex='col')

# Define the vertical reference lines that you want to draw on each subplot
#vlines = [0, 60, 120, 200, 300]

for i, var in enumerate(special_features):
    ax_main = axes[0, i]
    ax_ratio = axes[1, i]
    
    # Plot the EFT hist and ratio for this variable using your helper function
    plot_eft_hists(df=ttH_df, 
                   var=var, 
                   combs=[(c_g, c_tg)], 
                   ax=ax_main, 
                   ax_ratio=ax_ratio, 
                   density=True,
                   weight_col="true_weight_sel",
                   fontsize=24)
    
    # Draw the same vertical reference lines on both the main and ratio axes
    # for pt in vlines:
    #     ax_main.axvline(x=pt, color='gray', linestyle='--', alpha=0.5)
    #     ax_ratio.axvline(x=pt, color='gray', linestyle='--', alpha=0.5)
    
    # Optional: Set a title for each main axis with the variable name
    #ax_main.set_title(var, fontsize=16)
    
    # Label only the leftmost column with the y-axis label to reduce clutter
    # if i == 0:
    #     ax_main.set_ylabel("Count", fontsize=14)
    #     ax_ratio.set_ylabel("Ratio", fontsize=14)
    
    # Label the bottom row with x-axis labels
    # ax_ratio.set_xlabel(var + " (units)", fontsize=14)
    ax_ratio.legend(loc="best")

plt.tight_layout(pad=0.5)
plt.show()
