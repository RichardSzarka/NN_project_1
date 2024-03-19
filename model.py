import torch.nn as nn


# Patience class for implementing early stopping
class Patience:
    def __init__(self, patience=100):
        self.patience = patience         # The number of epochs to wait for improvement before stopping
        self.init_patience = patience    # Initial patience value, used for reset
        self.best_loss = float("inf")    # Best loss value seen so far

    # Determine if training should stop based on validation loss
    def should_i_stop(self, val_loss):
        # If the current validation loss is better (lower) than the best loss
        if val_loss < self.best_loss:
            self.patience = self.init_patience
            self.best_loss = val_loss
            return False, True        # Continue training
        else:
            self.patience -= 1
            if self.patience != 0:
                return False, False   # Continue training
            else:
                return True, False    # Stop training


# Main building block for MLP with optional batch normalization
class Linear_Block(nn.Module):
    def __init__(self, in_features, out_features, batch_norm=True):
        super().__init__()
        if batch_norm:
            self.layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU()
            )
        else:
            self.layer = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU()
            )

    def forward(self, x):
        return self.layer(x)


# Multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, binary=False, batch_norm=False):
        super().__init__()
        self.binary = binary
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

        # Defining the layers of the network
        self.lb1 = Linear_Block(13, 32, batch_norm=batch_norm)
        self.lb2 = Linear_Block(32, 16, batch_norm=batch_norm)
        self.lb3 = Linear_Block(16, 8, batch_norm=batch_norm)

        # Final layer differing based on binary classification or not
        if binary:
            self.linear_final = nn.Linear(8, 1)
        else:
            self.linear_final = nn.Linear(8, 7)

    # Forward pass of the network
    def forward(self, x):
        # Passing the input through each linear block
        x = self.lb1(x)
        x = self.lb2(x)
        x = self.lb3(x)
        x = self.linear_final(x)

        # Sigmoid for binary classification
        if self.binary:
            x = self.sigmoid(x)

        return x
