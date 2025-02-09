import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns

import torch
import torch.nn as nn

from EFT import *
class WadNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WadNeuralNetwork, self).__init__()
        self.hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, 1)
        
        # Xavier initialisation
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.zeros_(self.hidden.bias)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return torch.sigmoid(x)
class ComplexNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_prob=0.3):
        super(ComplexNN, self).__init__()
        layers = []
        current_dim = input_dim
    
        for dim in hidden_dim:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.BatchNorm1d(dim))  # Batch normalization
            layers.append(nn.LeakyReLU(negative_slope=0.01))  # LeakyReLU activation
            layers.append(nn.Dropout(p=dropout_prob))  # Dropout
            current_dim = dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # One output for binary classification

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Apply sigmoid for probabilities

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
    def forward(self, logits, targets, weights):
        loss = self.bce_loss(logits, targets)
        weighted_loss = loss * weights
        return weighted_loss.mean()
    

'''
Takes in np arrays that are 1D (converts to 2d)
and 2D and returns a list of tensors. Returns 1d first then 2d.
'''
def get_tensors(oned, twod):
    res = []
    for arg in oned:
        res.append(torch.tensor(arg, dtype=torch.float32).unsqueeze(1))
    for arg in twod:
        res.append(torch.tensor(arg, dtype=torch.float32))
    return res

def plot_classifier_output(y_probs, y_true, ws, ax):
    sm_probs = y_probs[y_true == 0].squeeze()  # Probabilities for SM (true label 0)
    eft_probs = y_probs[y_true == 1].squeeze()  # Probabilities for EFT (true label 1)
    ax.hist(sm_probs, weights=ws[y_true == 0], histtype="step",bins=30, alpha=0.7, color='blue', label="SM (cg=0, ctg=0)", linewidth=2)
    ax.hist(eft_probs, weights=ws[y_true == 1], histtype="step", bins=30, alpha=0.7, color='orange', label="EFT (cg=0.3, ctg=0.69)", linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Predicted Probabilities')
    ax.set_ylabel('Weighted Frequency')
    ax.legend(loc="best")

'''
Type can be: dup (duplicate), rand (random), rand_eft (random with eft weights), rand_SM (random with SM weights)
'''
def get_labeled_comb_df(ttH_df, type, features, c_g, c_tg, norm_weights = True):
        
        if type=="dup":
            EFT_weights = np.asarray(calc_weights(ttH_df, cg=c_g, ctg=c_tg))
            EFT_weights = (EFT_weights/np.sum(EFT_weights))*10000
            EFT_labels = np.ones(len(EFT_weights))
            SM_weights = np.asarray(ttH_df["plot_weight"])
            SM_weights = (SM_weights/np.sum(SM_weights))*10000
            SM_labels = np.zeros(len(SM_weights))


            comb_df_SM = pd.concat([ttH_df[var] for var in features], axis=1)
            comb_df_SM["weight"] = SM_weights

            comb_df_EFT = comb_df_SM.copy()#pd.concat([ttH_df[var] for var in special_features], axis=1)
            comb_df_EFT["weight"] = EFT_weights

            #EFT=1, SM=0
            labels = np.concatenate([EFT_labels, SM_labels])
            comb_df = pd.concat([comb_df_EFT, comb_df_SM], axis = 0)
            comb_df["labels"] = labels
        
        elif type[:4]=="rand":
            EFT_weights = np.asarray(calc_weights(ttH_df, cg=c_g, ctg=c_tg))
            if norm_weights:
                EFT_weights = (EFT_weights/np.sum(EFT_weights))*10000
            ttH_df["EFT_weight"] = EFT_weights
            if norm_weights:
                ttH_df["plot_weight"] = (ttH_df["plot_weight"]/np.sum(ttH_df["plot_weight"]))*10000

            comb_df = pd.concat([ttH_df[var] for var in features]+[ttH_df["plot_weight"], ttH_df["EFT_weight"]], axis=1)
            comb_df_SM, comb_df_EFT = train_test_split(comb_df, test_size=0.5)
            
            if type[4:]=="EFT":
                comb_df_SM.drop(columns=["plot_weight"], inplace=True)
                comb_df_SM.rename(columns={'EFT_weight': 'weight'}, inplace=True)
                comb_df_EFT.drop(columns=["plot_weight"], inplace=True)
                comb_df_EFT.rename(columns={'EFT_weight': 'weight'}, inplace=True)
            elif type[4:]=="SM":
                comb_df_EFT.drop(columns=["EFT_weight"], inplace=True)
                comb_df_EFT.rename(columns={'plot_weight': 'weight'}, inplace=True)
                comb_df_SM.drop(columns=["EFT_weight"], inplace=True)
                comb_df_SM.rename(columns={'plot_weight': 'weight'}, inplace=True)
            else:
                comb_df_SM.drop(columns=["EFT_weight"], inplace=True)
                comb_df_SM.rename(columns={'plot_weight': 'weight'}, inplace=True)
                comb_df_EFT.drop(columns=["plot_weight"], inplace=True)
                comb_df_EFT.rename(columns={'EFT_weight': 'weight'}, inplace=True)

            comb_df_SM["labels"] = np.zeros(len(comb_df_SM["weight"]))
            comb_df_EFT["labels"] = np.ones(len(comb_df_EFT["weight"]))

            comb_df = pd.concat([comb_df_EFT, comb_df_SM], axis = 0)
    

        return comb_df

def classification_analysis(y_test,w_test, y_proba, y_pred, y_train, w_train,y_proba_train, target_names):
    classification_report(y_test, y_pred, target_names=target_names, sample_weight=w_test)

    accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
    print(f"Classifier Accuracy: {accuracy:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1, sample_weight=w_test)
    unique_fpr, indices = np.unique(fpr, return_index=True)
    unique_tpr = tpr[indices]
    #fpr, tpr = np.unique(fpr, return_index=True)[0], np.unique(tpr, return_index=True)[0]
    roc_auc = auc(unique_fpr, unique_tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train, pos_label=1, sample_weight=w_train)
    #fpr_train, tpr_train = np.unique(fpr_train, return_index=True)[0], np.unique(tpr_train, return_index=True)[0]  # Ensure unique values
    unique_fpr_train, indices_train = np.unique(fpr_train, return_index=True)
    unique_tpr_train = tpr_train[indices_train]
    roc_auc_train = auc(unique_fpr_train, unique_tpr_train)
    print(f"ROC AUC (Train): {roc_auc_train:.4f}")


    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot(fpr_train, tpr_train, color="green", lw=2, label=f"Train ROC Curve (AUC = {roc_auc_train:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


    #Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, sample_weight=w_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt=".0f", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot the histogram of the predicted probabilities
    weights_proba_gt = np.sum(w_test[y_proba > 0.5])

    # Sum the weights of events predicted as EFT (positive class)
    weights_predicted_as_EFT = np.sum(w_test[y_pred == 1])

    # Print the sums
    print(f"Sum of weights for events with prob > 0.5: {weights_proba_gt}")
    print(f"Sum of weights for events predicted as EFT: {weights_predicted_as_EFT}")

    # Plot histograms for SM and EFT
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Classifier Output over test')
    plot_classifier_output(y_proba.squeeze(), y_test.squeeze(), w_test, ax)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Classifier Output over train')
    plot_classifier_output(y_proba_train.squeeze(), y_train.squeeze(), w_train.squeeze(), ax)


def eval_in_batches(model, X, batch_size=1000):
    model.eval()
    with torch.no_grad():
        y_prob = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_prob.append(model(X_batch).squeeze().numpy())
        y_prob = np.concatenate(y_prob, axis=0)
    return y_prob

def get_tth_df():
    ttH_df = pd.read_parquet(f"{new_sample_path}/ttH_processed_selected_with_smeft_cut_mupcleq90.parquet")
    #tth_df = get_selection(ttH_df, "ttH")
    ttH_df = ttH_df[(ttH_df["mass_sel"] == ttH_df["mass_sel"])]
    ttH_df['plot_weight'] *= target_lumi / total_lumi
    ttH_df['true_weight_sel'] = ttH_df['plot_weight']/10  # Remove x10 multiplier
    #ttH_df = ttH_df.dropna()

    invalid_weights = ttH_df["plot_weight"] <= 0
    init_yield = ttH_df["plot_weight"].sum()
    if invalid_weights.sum() > 0:
        print(f" --> Removing {invalid_weights.sum()} rows with invalid weights.")
        ttH_df = ttH_df[~invalid_weights]
    
    new_yield = ttH_df["plot_weight"].sum()

    #Making sure yield after removing neg weights is same as yield after
    ttH_df["plot_weight"] = (ttH_df["plot_weight"] / new_yield) * init_yield

    print(f"--> Remaining rows = {len(ttH_df)}")
    ttH_df["EFT_weight"] = np.asarray(calc_weights(ttH_df))
    return ttH_df
