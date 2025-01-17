import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns

import torch
import torch.nn as nn

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
    sm_probs = y_probs[y_true == 0].squeeze().numpy()  # Probabilities for SM (true label 0)
    eft_probs = y_probs[y_true == 1].squeeze().numpy()  # Probabilities for EFT (true label 1)
    ax.hist(sm_probs, weights=ws[y_true == 0], bins=30, alpha=0.7, color='blue', label="SM (True Label 0)")
    ax.hist(eft_probs, weights=ws[y_true == 1], bins=30, alpha=0.7, color='orange', label="EFT (True Label 1)")
    ax.set_xlabel('Predicted Probabilities')
    ax.set_ylabel('Weighted Frequency')
    ax.legend()


def classification_analysis(y_test,w_test, y_proba, y_pred, y_train, w_train,y_proba_train, target_names):
    classification_report(y_test, y_pred, target_names=target_names, sample_weight=w_test)

    accuracy = accuracy_score(y_test, y_pred, sample_weight=w_test)
    print(f"Classifier Accuracy: {accuracy:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1, sample_weight=w_test)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train, pos_label=1, sample_weight=w_train)
    roc_auc_train = auc(fpr_train, tpr_train)
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