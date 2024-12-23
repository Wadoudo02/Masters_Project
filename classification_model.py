#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from selection import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
#%%
#If categorisation is binary or multiclass

dfs = get_dfs(sample_path)
#%%
'''
Labels:
0 - Background
1 - ttH
2 - ggH
3 - VBF
4 - VH
'''
binary_cat = True
for i, proc in enumerate(procs.keys()):
    if i==0:
        continue
    if binary_cat:
        if proc == "ttH":
            dfs[proc]["class_label"] = 1
        else:
            dfs[proc]["class_label"] = 0
    else:
        dfs[proc]["class_label"] = i-1

target_names_all = ['Background', 'ttH', 'ggH', 'VBF', 'VH']
target_names_no_bg = ['ttH', 'ggH', 'VBF', 'VH']
target_names_bin = ["ttH", "non-ttH"]

if binary_cat:
    target_names = target_names_bin
else:
    target_names = target_names_no_bg

combined_df = pd.concat([dfs[proc] for proc in target_names_no_bg])

print(combined_df["class_label"].unique())
invalid_weights = combined_df["plot_weight"] <= 0
if invalid_weights.sum() > 0:
    print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
    combined_df = combined_df[~invalid_weights]
#%%
'''
Relevant features:
- n_jets_sel
- max_b_tag_score
- second_max_b_tag_score
- HT_sel
- Delta R
'''
features = ["deltaR_sel", "HT_sel", "n_jets_sel", "max_b_tag_score", "j0_pt_sel", "delta_phi_gg_sel", "j0_btagB_sel", "j1_btagB_sel"]

rel_data = combined_df[features]
labels = combined_df["class_label"]
weights = combined_df["true_weight_sel"]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(rel_data, labels, weights, test_size=0.2, random_state=42)

#%%

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# xgb_model = xgb.XGBClassifier(
#     objective="multi:softmax",
#     num_class=len(np.unique(labels)),
#     max_depth=6,
#     eta=0.3,
#     min_child_weight=1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     n_estimators=100,
#     random_state=42
# )

xgb_model = xgb.XGBClassifier(
    use_label_encoder=False, eval_metric="logloss"
)

# Train model
xgb_model.fit(X_train, y_train, sample_weight=w_train)
#%%


# Evaluation
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)
y_proba_train = xgb_model.predict_proba(X_train)
print(classification_report(y_test, y_pred, target_names=target_names))

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
plt.title("ROC Curve for XGBoost Classifier")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# Feature importance
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(xgb_model, ax=ax, importance_type='weight')
ax.set_yticklabels(features)
ax.set_ylabel('Features')

#Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# %%
