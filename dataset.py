from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def init_datasets(path="dataset.csv", binary=False, EDA=False, balance=False):
    """

    Initializes train/validation split according to the predefined competitive mode

    """
    df = pd.read_csv(path)

    # If EDA (Exploratory Data Analysis) is True, shuffle the dataset for feature selection tree-based method
    if EDA:
        df = df.sample(frac=1).reset_index(drop=True)

    # Split the dataset into training and validation sets
    train, valid = train_test_split(df, test_size=0.2, random_state=30)

    # If not performing EDA, and balancing and binary classification are enabled,
    # balance the classes in the dataset
    if not EDA and balance and binary:
        class_1 = df[df['Class'] == 1]
        class_2 = df[df['Class'] == 2]
        n = len(class_2) - len(class_1)
        class_1 = class_1.sample(n=n + len(class_1), random_state=42)

        train = pd.concat([class_1, class_2]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Initialize the training and validation torch datasets
    train = SteelDataset(train, binary=binary, EDA=EDA)
    valid = SteelDataset(valid, binary=binary, EDA=EDA)

    return train, valid


class SteelDataset(Dataset):
    """

    Custom Dataset class for Steel dataset

    """
    def __init__(self, raw_data, binary=False, EDA=False):
        # Initialize dataset
        self.raw_data = raw_data
        self.raw_data = self.raw_data.drop(columns=["V12"])

        # All columns up to 7th from last are predictors
        self.x = self.raw_data.iloc[:, :-7]

        # y is different for binary/multiclass task
        if binary:
            self.y = self.raw_data.iloc[:, -1:]
        else:
            self.y = self.raw_data.iloc[:, -7:]

        # Adjust class labels from 1|2 to 0|1
        self.y[self.y.columns[-1]] = self.y[self.y.columns[-1]] - 1

        # Normalize all columns using z-score
        self.x = self.x.apply(lambda column: self.Z_score(column, dont_normalize=["V13"]), axis=0)

        # Select specific features if not performing EDA, found by feature selection method
        if not EDA:
            self.x = self.x[['V14', 'V11', 'V25', 'V22', 'V16', 'V15', 'V20', 'V4', 'V3', 'V26', 'V8', 'V1', 'V2']]

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        return (torch.tensor(self.x.iloc[index].values, dtype=torch.float32),
                torch.tensor(self.y.iloc[index].values, dtype=torch.float32))

    def Z_score(self, column, dont_normalize=["V13"]):

        # Skip normalization for specified columns
        if column.name in dont_normalize:
            return column

        # Calculate the Z-score for normalization
        mean_value = np.mean(column)
        std_dev = np.std(column)

        # Normalize the column if the standard deviation is non-zero
        if std_dev != 0:
            z_score_normalized = (column - mean_value) / std_dev
        else:
            z_score_normalized = column

        return z_score_normalized


init_datasets(path="dataset.csv", binary=False, EDA=False, balance=False)
