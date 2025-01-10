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
do_grid_search = False

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

#comb_df = pd.concat([ttH_df[var] for var in special_features], axis=1)
EFT_weights = np.asarray(calc_weights(ttH_df, cg=c_g, ctg=c_tg))
EFT_weights = (EFT_weights/np.sum(EFT_weights))*10000
EFT_labels = np.ones(len(EFT_weights))
SM_weights = np.asarray(ttH_df["plot_weight"])
SM_weights = (SM_weights/np.sum(SM_weights))*10000
SM_labels = np.zeros(len(SM_weights))



comb_df_SM = pd.concat([ttH_df[var] for var in special_features], axis=1)
comb_df_SM["weight"] = SM_weights

comb_df_EFT = pd.concat([ttH_df[var] for var in special_features], axis=1)
comb_df_EFT["weight"] = EFT_weights

#EFT=1, SM=0
labels = np.concatenate([EFT_labels, SM_labels])
comb_df = pd.concat([comb_df_EFT, comb_df_SM], axis = 0)

# 0 = SM, 1 = EFT
#labels = [1 if EFT_weights[i] > SM_weights[i] else 0 for i in range(len(EFT_weights))]

#weights = np.array([EFT_weights[i] if labels[i] == 1 else SM_weights[i] for i in range(len(labels))])
#weights = EFT_weights
#weights = SM_weights
weights = comb_df["weight"]

x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(comb_df,
                                                                     labels,
                                                                     weights,
                                                                     test_size=0.2,
                                                                     random_state=45)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Making sure everything is np array, only necessayr becasue of some version mismatch.
x_train, x_test, y_train, y_test, w_train, w_test, weights = make_np_arr(x_train,
                                                                 x_test,
                                                                   y_train,
                                                                     y_test,
                                                                       w_train,
                                                                         w_test, weights)

#Initial best params which are overwritten if grid search performed.
best_params = {"gamma": 0, "learning_rate": 0.2, "max_depth": 3, "n_estimators": 50}

#%%
#Grid paramter scan
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
    grid_search.fit(x_train, y_train, sample_weight=w_train)

    # Get the best parameters
    print("Best Parameters:", grid_search.best_params_)
    best_params = grid_search.best_params_
#%%
# Train the model with the best parameters
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="logloss",
    **best_params
)

xgb_model.fit(x_train, y_train, sample_weight=w_train)

#%%

y_pred = xgb_model.predict(x_test)
y_proba = xgb_model.predict_proba(x_test)
y_proba_train = xgb_model.predict_proba(x_train)

classification_analysis(y_test,w_test,y_proba,y_pred,y_train ,y_proba_train, ["SM", "EFT"] )

# Feature importance
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(xgb_model, ax=ax, importance_type='weight')

tick_locs = ax.get_yticks()
ax.set_yticks(tick_locs)
ax.set_yticklabels(special_features[:len(tick_locs)])
ax.set_ylabel('Features')

# %%

SM_features = x_test[y_pred==0]
SM_w = w_test[y_pred==0]

EFT_features = x_test[y_pred==1]
EFT_w = w_test[y_pred==1]

for col in range(len(special_features)):
    plt.hist(SM_features[:,col],weights=SM_w, bins=50, alpha=0.5, label="SM")
    plt.hist(EFT_features[:,col],weights=EFT_w, bins=50, alpha=0.5, label="EFT")
    plt.xlabel(special_features[col])
    plt.ylabel("Events")
    plt.legend()
    plt.show()

