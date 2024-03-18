from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def init_datasets(path="dataset.csv", binary=False, EDA=False, balance=False):
    df = pd.read_csv(path)

    if EDA:
        df = df.sample(frac=1).reset_index(drop=True)

    train, valid = train_test_split(df, test_size=0.2, random_state=30)

    if not EDA and balance and binary:
        class_1 = df[df['Class'] == 1]
        class_2 = df[df['Class'] == 2]
        n = len(class_2) - len(class_1)
        class_1 = class_1.sample(n=n + len(class_1), random_state=42)

        train = pd.concat([class_1, class_2]).sample(frac=1, random_state=42).reset_index(drop=True)

    train = SteelDataset(train, binary=binary, EDA=EDA)
    valid = SteelDataset(valid, binary=binary, EDA=EDA)

    return train, valid


class SteelDataset(Dataset):
    def __init__(self, raw_data, binary=False, EDA=False):
        self.raw_data = raw_data
        self.raw_data = self.raw_data.drop(columns=["V12"])
        self.x = self.raw_data.iloc[:, :-7]

        if binary:
            self.y = self.raw_data.iloc[:, -1:]

        else:
            self.y = self.raw_data.iloc[:, -7:]

        self.y[self.y.columns[-1]] = self.y[self.y.columns[-1]] - 1

        self.x = self.x.apply(lambda column: self.Z_score(column, dont_normalize=["V13"]), axis=0)

        if not EDA:
            self.x = self.x[['V14', 'V11', 'V25', 'V22', 'V16', 'V15', 'V20', 'V4', 'V3', 'V26', 'V8', 'V1', 'V2']]

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        return (torch.tensor(self.x.iloc[index].values, dtype=torch.float32),
                torch.tensor(self.y.iloc[index].values, dtype=torch.float32))

    def Z_score(self, column, dont_normalize=["V13"]):
        if column.name in dont_normalize:
            return column

        mean_value = np.mean(column)
        std_dev = np.std(column)

        if std_dev != 0:
            z_score_normalized = (column - mean_value) / std_dev
        else:
            z_score_normalized = column

        return z_score_normalized

init_datasets(path="dataset.csv", binary=False, EDA=False, balance=False)