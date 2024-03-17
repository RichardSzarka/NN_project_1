import torch.nn as nn


class Patience():
    def __init__(self, patience=100):
        self.patience = patience
        self.init_patience = patience
        self.best_loss = float("inf")

    def should_i_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.patience = self.init_patience
            self.best_loss = val_loss
            return False, True
        else:
            self.patience -= 1
            if self.patience != 0:
                return False, False
            else:
                return True, False


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


class MLP(nn.Module):
    def __init__(self, binary=False, batch_norm=False):
        super().__init__()
        self.binary = binary
        self.sigmoid = nn.Sigmoid()

        self.lb1 = Linear_Block(13, 32, batch_norm=batch_norm)
        self.lb2 = Linear_Block(32, 16, batch_norm=batch_norm)
        self.lb3 = Linear_Block(16, 8, batch_norm=batch_norm)

        if binary:
            self.linear_final = nn.Linear(8, 1)
        else:
            self.linear_final = nn.Linear(8, 7)

    def forward(self, x):

        x = self.lb1(x)
        x = self.lb2(x)
        x = self.lb3(x)
        x = self.linear_final(x)

        if self.binary:
            x = self.sigmoid(x)

        return x
