#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:33:35 2025

@author: wadoudcharbak
"""

# -------------------------------------------------------------------------
#                   DEFINE OUR NEURAL NETWORK
# -------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
        # Xavier initialisation
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return torch.sigmoid(x)


# -------------------------------------------------------------------------
#            MODEL INITIALISATION, LOSS FUNCTION, OPTIMISER
# -------------------------------------------------------------------------
input_dim = X_train_t.shape[1]        # e.g. 6
hidden_dim = input_dim * 4            # arbitrary choice
model = NeuralNetwork(input_dim, hidden_dim)

criterion = nn.BCELoss(reduction="none")  # We'll apply event weights manually
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)


# -------------------------------------------------------------------------
#                   DEFINE OUR NEURAL NETWORK
# -------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super().__init__()

        # First hidden layer
        self.hidden1 = nn.Linear(input_dim, hidden_dim1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim1)

        # Second hidden layer
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim2)

        # Final output layer
        self.output = nn.Linear(hidden_dim2, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        # First hidden layer
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Second hidden layer
        x = self.hidden2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.output(x)
        return torch.sigmoid(x)


# -------------------------------------------------------------------------
#            MODEL INITIALISATION, LOSS FUNCTION, OPTIMISER
# -------------------------------------------------------------------------
input_dim = X_train_t.shape[1]   # e.g. 6
hidden_dim1 = input_dim * 4      # for example
hidden_dim2 = input_dim * 2      # or another suitable size

model = NeuralNetwork(input_dim, hidden_dim1, hidden_dim2)
criterion = nn.BCELoss(reduction="none")  # We'll apply event weights manually
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)


# -------------------------------------------------------------------------
#                   DEFINE OUR NEURAL NETWORK
# -------------------------------------------------------------------------
class NeuralNetwork3Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super().__init__()
        
        # First hidden layer
        self.hidden1 = nn.Linear(input_dim, hidden_dim1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim1)
        
        # Second hidden layer
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim2)
        
        # Third hidden layer
        self.hidden3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.batchnorm3 = nn.BatchNorm1d(hidden_dim3)
        
        # Final output layer
        self.output = nn.Linear(hidden_dim3, 1)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Xavier initialisation
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)
        nn.init.xavier_uniform_(self.hidden3.weight)
        nn.init.zeros_(self.hidden3.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        # First hidden layer
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.hidden2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Third hidden layer
        x = self.hidden3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.output(x)
        return torch.sigmoid(x)


# -------------------------------------------------------------------------
#            MODEL INITIALISATION, LOSS FUNCTION, OPTIMISER
# -------------------------------------------------------------------------
input_dim = X_train_t.shape[1]  # e.g., 6 or however many input features
hidden_dim1 = input_dim * 4
hidden_dim2 = input_dim * 3
hidden_dim3 = input_dim * 2

model = NeuralNetwork3Layer(input_dim, hidden_dim1, hidden_dim2, hidden_dim3)
criterion = nn.BCELoss(reduction="none")  # We'll apply event weights manually
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
