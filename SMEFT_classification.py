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

c_g = 0.3
c_tg = 0.69

ttH_df = pd.read_parquet(f"{new_sample_path}/ttH_processed_selected_with_smeft_cut_mupcleq90.parquet")

ttH_df = ttH_df[(ttH_df["mass_sel"] == ttH_df["mass_sel"])]
ttH_df['plot_weight'] *= target_lumi / total_lumi 

invalid_weights = ttH_df["plot_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    ttH_df = ttH_df[~invalid_weights]

#all_features = get_features()
#plot_SMEFT_features(all_features)
special_features = ["lead_pt_sel", "HT_sel", "cosDeltaPhi_sel" ,"pt-over-mass_sel", "deltaR_sel", "min_delta_R_j_g_sel", "delta_phi_jj_sel", "sublead_pt-over-mass_sel", "delta_eta_gg_sel", "lead_pt-over-mass_sel", "delta_phi_gg_sel"]

comb_df = pd.concat([ttH_df[var] for var in special_features], axis=1)
EFT_weights = np.array(calc_weights(ttH_df, cg=c_g, ctg=c_tg))
SM_weights = np.array(ttH_df["plot_weight"])

# 0 = SM, 1 = EFT
labels = [1 if EFT_weights[i] > SM_weights[i] else 0 for i in range(len(EFT_weights))]

weights = np.array([EFT_weights[i] if labels[i] == 1 else SM_weights[i] for i in range(len(labels))])

x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(comb_df,
                                                                     labels,
                                                                     weights,
                                                                     test_size=0.2,
                                                                     random_state=42)

#%%

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
scaler.transform(x_test)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
w_train = np.asarray(w_train)

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
    verbose=1,
    n_jobs=-1                # Parallelize computation
)

# Fit the model
grid_search.fit(x_train, y_train)

# Get the best parameters
print("Best Parameters:", grid_search.best_params_)

#%%
xgb_model.fit(x_train, y_train, sample_weight=w_train)

#%%

y_pred = xgb_model.predict(x_test)
y_proba = xgb_model.predict_proba(x_test)
y_proba_train = xgb_model.predict_proba(x_train)
print(classification_report(y_test, y_pred, target_names=["SM", "EFT"]))

accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
print(f"Classifier Accuracy: {accuracy:.4f}")


fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")

fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train[:, 1], pos_label=1)
roc_auc_train = auc(fpr_train, tpr_train)
print(f"ROC AUC (Train): {roc_auc_train:.4f}")


# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot(fpr_train, tpr_train, color="green", lw=2, label=f"Train ROC Curve (AUC = {roc_auc_train:.4f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost SMEFT Classifier")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# Feature importance
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(xgb_model, ax=ax, importance_type='weight')
#ax.set_yticklabels(special_features)
tick_locs = ax.get_yticks()
ax.set_yticks(tick_locs)
ax.set_yticklabels(special_features[:len(tick_locs)])
ax.set_ylabel('Features')

#Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["SM", "EFT"], yticklabels=["SM", "EFT"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()







# %%
